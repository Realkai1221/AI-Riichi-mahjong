"""Numerical fixtures must be launched through the A2 cumulative budget wrapper."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from riichi_ai.model import PolicyNetwork, efficiency_model_config
from riichi_ai.package_a_stability import (
    ContextCapture, DEFAULT_SAFETY, SafetyMonitor, activation_diagnostics,
    probability_rows, smoothed_targets, stability_loss, train_stability,
)
from riichi_ai.stage2b_training import train_stage2b
from riichi_ai.training import collate_training, _cpu_state
from test_training_v2 import records, SMALL


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def labels_fixture():
    return {"q": torch.tensor([[.5, .5, 0., 0.], [1., 0., 0., 0.], [.5, .5, 0., 0.]]),
            "best": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool),
            "optional": torch.tensor([True, False, True]), "rank": torch.tensor([0, 1, 2])}, torch.tensor(
                [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)


def test_smoothing_excludes_padding_and_forced_and_keeps_original_labels():
    labels, mask = labels_fixture()
    saved = {k: v.clone() for k, v in labels.items()}
    smooth = smoothed_targets(labels, mask, .01)
    assert torch.equal(smooth[0], torch.tensor([.495, .495, .01, 0.]))
    assert torch.equal(smooth[1:], labels["q"][1:])
    assert torch.equal(smooth[~mask], torch.zeros_like(smooth[~mask]))
    assert torch.allclose(smooth.sum(-1), torch.ones(3))
    assert all(torch.equal(v, saved[k]) for k, v in labels.items())
    assert smoothed_targets(labels, mask, 0) is labels["q"]
    for epsilon in (-.01, 1., float("nan")):
        with pytest.raises(ValueError): smoothed_targets(labels, mask, epsilon)


def test_smoothing_loss_preserves_unsmoothed_metrics_and_frozen_probe_gradient():
    labels, mask = labels_fixture()
    raw = torch.tensor([[0., 1., 2., 0.], [1., 0., 0., 0.], [0., 2., 0., 0.]], requires_grad=True)
    logits = raw.masked_fill(~mask, torch.finfo(torch.float32).min)
    ranks = torch.zeros((3, 4), requires_grad=True)
    control, control_parts = stability_loss(logits, ranks, labels, mask, 0)
    smooth, smooth_parts = stability_loss(logits, ranks, labels, mask, .01)
    assert smooth != control
    assert all(torch.equal(v, smooth_parts[k]) for k, v in control_parts.items())
    smooth.backward()
    assert ranks.grad is None
    assert torch.count_nonzero(raw.grad[1]) == 0
    assert torch.count_nonzero(raw.grad[~mask]) == 0
    assert torch.isfinite(raw.grad).all()


def test_diagnostics_exclude_padding_and_separate_common_offset():
    labels, mask = labels_fixture()
    logits = torch.tensor([[100., 102., 104., -3.4e38], [100., -3.4e38, -3.4e38, -3.4e38],
                           [100., 102., -3.4e38, -3.4e38]])
    batch = {"candidate_mask": mask}
    report = activation_diagnostics(torch.tensor([[3., 4.]] * 3), logits, torch.zeros((3, 4)), batch, labels)
    assert report["padding_excluded"] == 6
    assert report["legal_logit_abs"]["maximum"] == 104
    assert report["centered_legal_logit_abs"]["maximum"] == 2
    assert report["context_l2"]["mean"] == 5
    assert report["best_minus_nonbest_max_gap"]["mean"] == -2
    assert report["frozen_rank_probe_ce"]["mean"] == pytest.approx(math.log(4))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_context_hook_changes_neither_forward_backward_nor_rng(device, record_property):
    if device == "mps" and not torch.backends.mps.is_available(): pytest.skip("MPS unavailable")
    torch.manual_seed(123)
    model = PolicyNetwork(efficiency_model_config(SMALL)).to(device)
    batch, labels = collate_training(records(True)[:4], device)
    capture = ContextCapture(model)
    rng = torch.get_rng_state().clone()
    accelerator_rng = torch.mps.get_rng_state().clone() if device == "mps" else None
    logits, ranks = model(batch)
    report = activation_diagnostics(capture.value, logits, ranks, batch, labels)
    loss, _ = stability_loss(logits, ranks, labels, batch["candidate_mask"], 0)
    loss.backward()
    grads = {name: p.grad.clone() if p.grad is not None else None for name, p in model.named_parameters()}
    capture.close()
    model.zero_grad(set_to_none=True)
    plain_logits, plain_ranks = model(batch)
    plain_loss, _ = stability_loss(plain_logits, plain_ranks, labels, batch["candidate_mask"], 0)
    plain_loss.backward()
    assert torch.equal(logits, plain_logits) and torch.equal(ranks, plain_ranks)
    assert torch.equal(loss, plain_loss)
    first_plain_grads = {name: p.grad.clone() if p.grad is not None else None
                         for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    repeat_logits, repeat_ranks = model(batch)
    repeat_loss, _ = stability_loss(repeat_logits, repeat_ranks, labels, batch["candidate_mask"], 0)
    repeat_loss.backward()
    assert torch.equal(plain_logits, repeat_logits) and torch.equal(plain_ranks, repeat_ranks)
    assert torch.equal(plain_loss, repeat_loss)
    second_plain_grads = {name: p.grad for name, p in model.named_parameters()}
    for label, left, right in (("hook_to_plain", grads, first_plain_grads),
                               ("plain_to_repeat_plain", first_plain_grads, second_plain_grads)):
        maximum = max(float((left[name] - right[name]).abs().max().cpu())
                      for name in left if left[name] is not None and right[name] is not None)
        record_property(f"{device}_{label}_gradient_max_abs", maximum)
        for name, value in left.items():
            if value is None:
                assert right[name] is None
            else:
                assert right[name] is not None
                if device == "cpu":
                    assert torch.equal(value, right[name]), (label, name, maximum)
                else:
                    # Existing CPU/MPS parity tolerance; the repeated plain
                    # backward measures accelerator variation without a hook.
                    assert torch.allclose(value, right[name], atol=5e-5, rtol=5e-4), (label, name, maximum)
    record_property("gradient_atol", 0. if device == "cpu" else 5e-5)
    record_property("gradient_rtol", 0. if device == "cpu" else 5e-4)
    assert torch.equal(rng, torch.get_rng_state())
    if accelerator_rng is not None: assert torch.equal(accelerator_rng, torch.mps.get_rng_state())
    assert report["context_rms"]["maximum"] >= 0


def test_cpu_mps_smoothed_forward_backward_parity():
    if not torch.backends.mps.is_available(): pytest.skip("MPS unavailable")
    torch.manual_seed(521)
    cpu = PolicyNetwork(efficiency_model_config(SMALL))
    mps = deepcopy(cpu).to("mps")
    left, labels = collate_training(records(True)[:4], "cpu")
    right, labels_mps = collate_training(records(True)[:4], "mps")
    c_logits, c_ranks = cpu(left)
    m_logits, m_ranks = mps(right)
    c_loss, _ = stability_loss(c_logits, c_ranks, labels, left["candidate_mask"], .01)
    m_loss, _ = stability_loss(m_logits, m_ranks, labels_mps, right["candidate_mask"], .01)
    c_loss.backward(); m_loss.backward()
    assert torch.allclose(c_logits, m_logits.cpu(), atol=5e-5, rtol=5e-4)
    assert torch.allclose(c_loss, m_loss.cpu(), atol=5e-5, rtol=5e-4)
    for name, p in cpu.named_parameters():
        right_grad = dict(mps.named_parameters())[name].grad
        if p.grad is None: assert right_grad is None
        else: assert torch.allclose(p.grad, right_grad.cpu(), atol=5e-5, rtol=5e-4)


def test_control_reproduces_original_updates_stream_and_frozen_rank_head(tmp_path):
    data = records(True)
    args = dict(seed=71, steps=4, batch_size=4, device="cpu", max_seconds=30,
                raw_config=SMALL, validate_every=2, validation_per_stratum=4)
    original = train_stage2b(data, tmp_path / "original", validation=data, assisted=True,
                            rank_weight=0, checkpoint_every=4, **args)
    control = train_stability(data, tmp_path / "control", validation=data, epsilon=0,
                             diagnostic_every=1, diagnostic_per_stratum=2, **args)
    assert control["complete"] and original["failure"] is None
    assert control["sampled_index_digest"] == original["sampled_index_digest"]
    old = torch.load(original["checkpoint"], weights_only=True)
    new = torch.load(control["terminal_checkpoint"], weights_only=True)
    assert all(torch.equal(value, new["model_state_dict"][name]) for name, value in old["model_state_dict"].items())
    assert control["frozen_rank_head_unchanged"]
    assert control["initial_state_sha256"] == original["initialization"]["state_sha256"]
    assert control["initial_fixed_diagnostics"]["context_rms"]["count"] == 4
    assert "optimizer_state_dict" in new and "rng_state" in new
    assert control["terminal_probability_metrics"]["complete"]
    assert all(x["parameters"]["rank_head"]["update_l2"] == 0 for x in control["diagnostics"])
    with pytest.raises(FileExistsError):
        train_stability(data, tmp_path / "control", validation=data, epsilon=0, **args)


def scale_fixture(value):
    return {population: {key: {"mean": value, "maximum": value} for key in SafetyMonitor.SCALE_METRICS}
            for population in ("sampled", "fixed")}


def test_scale_alarm_warmup_floor_persistence_and_reset():
    monitor = SafetyMonitor({"warmup_steps": 1, "persistence": 3})
    assert not monitor.update_scale(1, scale_fixture(.01))
    assert not monitor.update_scale(2, scale_fixture(11))
    assert not monitor.update_scale(3, scale_fixture(1))
    assert not monitor.update_scale(4, scale_fixture(11))
    assert not monitor.update_scale(5, scale_fixture(11))
    assert monitor.update_scale(6, scale_fixture(11))
    assert monitor.scale_alarm
    assert all(e["threshold"] == 10 for e in monitor.events)
    assert not any(e["metric"].split(":")[2] == "legal_logit_abs" for e in monitor.events)
    assert all(v == 1 for v in monitor.references.values())


def test_gradient_alarm_persistence_does_not_depend_on_smoothed_loss():
    monitor = SafetyMonitor({"gradient_persistence": 3})
    assert not monitor.update_loss(1, 1., 51.)
    assert not monitor.update_loss(2, 1., 51.)
    assert not monitor.update_loss(3, 1., 1.)
    assert not monitor.update_loss(4, 1., 51.)
    assert not monitor.update_loss(5, 1., 51.)
    assert monitor.update_loss(6, 1., 51.)
    assert monitor.tripped and not monitor.scale_alarm
    with pytest.raises(FloatingPointError): monitor.update_loss(7, float("nan"), 1.)


def test_entropy_and_call_mass_keep_forced_decisions_separate():
    from riichi_ai.semantics import ACTION_KINDS
    kinds = [ACTION_KINDS.index(k) + 1 for k in ("DISCARD", "CHI", "ANKAN")]
    logits = torch.tensor([[0., 0., 0.], [0., -3.4e38, -3.4e38]])
    batch = {"candidate_mask": torch.tensor([[True] * 3, [True, False, False]]),
             "action_tokens": torch.tensor([[[k] for k in kinds], [[kinds[0]], [0], [0]]])}
    labels = {"best": torch.tensor([[True, False, False], [True, False, False]])}
    values, counts, calls, kans = probability_rows(logits, batch, labels)
    assert counts == [3, 1] and calls == [True, False] and kans == [True, False]
    assert values[0].tolist() == pytest.approx([math.log(3), 1., 1/3, 2/3, 2/3, 1/3])
    assert values[1].tolist() == pytest.approx([0., 0., 1., 0., 0., 0.])


def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/package_a_stability.py"
    spec = importlib.util.spec_from_file_location("a2_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_safety_keys_and_failed_budget_accounting(tmp_path):
    cli = launcher()
    _, config, _ = cli.config_at(Path(__file__).resolve().parents[1] / "configs/package_a.json")
    assert set(config["safety"]) == set(DEFAULT_SAFETY)
    ledger = cli.Ledger(tmp_path / "ledger.json", "fixed", 10)
    with pytest.raises(RuntimeError, match="fixture"):
        with ledger.operation("failed", 2): raise RuntimeError("fixture")
    report = ledger.snapshot()
    assert report["operations"][0]["status"] == "failed"
    assert report["charged_seconds"] >= 0
    with ledger.operation("still_active", 3):
        assert ledger.snapshot()["charged_seconds"] >= 3
        with pytest.raises(RuntimeError, match="active"):
            with ledger.operation("duplicate", 2): pass
    with pytest.raises(ValueError, match="identity"):
        cli.Ledger(ledger.path, "changed", 10).snapshot()


def test_safety_stopped_attempt_is_not_durable_or_selected(tmp_path):
    data = records(True)
    report = train_stability(data, tmp_path / "stopped", validation=data, seed=71, epsilon=.01,
        steps=4, batch_size=4, device="cpu", max_seconds=30, raw_config=SMALL,
        validate_every=2, validation_per_stratum=4, diagnostic_every=2,
        diagnostic_per_stratum=2, safety={"gradient_threshold": 1e-20, "gradient_persistence": 1})
    assert report["completed_steps"] == 1 and report["stop_reason"] == "safety_alarm"
    assert not report["complete"] and not report["durability_passed"]
    assert report["selected_checkpoint"] is None
    assert Path(report["terminal_checkpoint"]).exists()
    assert report["safety"]["events"][0]["abort"]


def test_hidden_worker_requires_matching_live_lease(tmp_path):
    cli = launcher()
    ledger = cli.Ledger(tmp_path / "ledger.json", "fixed", 10)
    with pytest.raises(RuntimeError, match="matching"):
        cli.require_worker_lease(ledger, "absent", 1., "probe")
    with ledger.operation("probe", 3):
        token = ledger.snapshot()["operations"][-1]["token"]
        assert cli.require_worker_lease(ledger, token, 2., "probe")["name"] == "probe"
        for supplied, allowance, name in (("wrong", 2., "probe"), (token, 4., "probe"), (token, 2., "formal")):
            with pytest.raises(RuntimeError, match="matching"):
                cli.require_worker_lease(ledger, supplied, allowance, name)
