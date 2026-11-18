"""Numerical resume fixtures: execute only within the continuation lease."""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from riichi_ai.package_a_continuation import (
    continue_stability, optimizer_without_lr, restore_safety, safety_state, state_digest,
)
from riichi_ai.package_a_stability import SafetyMonitor, train_stability
from riichi_ai.training import _restore_rng, _rng_state
from test_training_v2 import records, SMALL


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("continuation-fixture")
    data = records(True)
    args = dict(validation=data, seed=71, epsilon=.01, batch_size=4, device="cpu",
                max_seconds=30, raw_config=SMALL, validate_every=2,
                validation_per_stratum=4, diagnostic_every=2, diagnostic_per_stratum=2,
                safety={"warmup_steps": 2})
    split = train_stability(data, root / "split", steps=4, **args)
    full = train_stability(data, root / "full", steps=8, **args)
    assert split["complete"] and full["complete"]
    yield data, split, full
    torch.set_num_threads(previous)


def run(source, destination, **kwargs):
    data, split, _ = source
    return continue_stability(data, destination, validation=data,
        checkpoint_path=split["terminal_checkpoint"], source_report=split,
        lr=kwargs.pop("lr", .0003), additional_steps=kwargs.pop("additional_steps", 4),
        device="cpu", max_seconds=kwargs.pop("max_seconds", 30),
        validation_per_stratum=4, diagnostic_per_stratum=2,
        expected_start_step=4, **kwargs)


def test_unchanged_rate_split_resume_matches_uninterrupted_exactly(source, tmp_path):
    report = run(source, tmp_path / "resumed")
    full = source[2]
    assert report["complete"] and report["validation_schedule_complete"]
    assert report["completed_steps"] == 8 and report["additional_steps"] == 4
    assert report["sampled_index_digest"] == full["sampled_index_digest"]
    assert report["history"] == full["history"][4:]
    assert report["diagnostics"] == full["diagnostics"][2:]
    assert report["safety"] == full["safety"]
    assert report["initial_safety_state"]["references"] == source[1]["safety"]["references"]
    left = torch.load(report["terminal_checkpoint"], weights_only=True)
    right = torch.load(full["terminal_checkpoint"], weights_only=True)
    for key in ("model_state_dict", "optimizer_state_dict", "rng_state"):
        assert state_digest(left[key]) == state_digest(right[key]), key
    assert left["safety_state"] == report["safety_state"]
    assert report["frozen_rank_head_unchanged"]
    assert report["best_step"] > 4
    with pytest.raises(FileExistsError):
        run(source, tmp_path / "resumed")


def test_second_resume_uses_saved_counters_and_preserves_full_stream(source, tmp_path):
    first = run(source, tmp_path / "first", additional_steps=2)
    data = source[0]
    second = continue_stability(data, tmp_path / "second", validation=data,
        checkpoint_path=first["terminal_checkpoint"], source_report=first,
        lr=.0003, additional_steps=2, device="cpu", max_seconds=30,
        validation_per_stratum=4, diagnostic_per_stratum=2, expected_start_step=6)
    full = torch.load(source[2]["terminal_checkpoint"], weights_only=True)
    final = torch.load(second["terminal_checkpoint"], weights_only=True)
    assert second["complete"]
    assert second["sampled_index_digest"] == source[2]["sampled_index_digest"]
    for key in ("model_state_dict", "optimizer_state_dict", "rng_state"):
        assert state_digest(final[key]) == state_digest(full[key]), key
    assert second["initial_safety_state"] == first["safety_state"]


def test_lower_lr_changes_only_lr_at_restore_and_preserves_sampling(source, tmp_path):
    control = run(source, tmp_path / "control", additional_steps=2)
    lower = run(source, tmp_path / "lower", additional_steps=2, lr=.0001)
    for key in ("initial_state_sha256", "initial_optimizer_state_sha256", "initial_optimizer_without_lr_sha256",
                "initial_rng_state_sha256", "initial_safety_state_sha256", "initial_sampled_index_digest",
                "sampled_index_digest", "sampled_prefixes"):
        assert lower[key] == control[key], key
    left = torch.load(lower["terminal_checkpoint"], weights_only=True)
    right = torch.load(control["terminal_checkpoint"], weights_only=True)
    assert left["optimizer_state_dict"]["param_groups"][0]["lr"] == .0001
    assert right["optimizer_state_dict"]["param_groups"][0]["lr"] == .0003
    assert state_digest(left["model_state_dict"]) != state_digest(right["model_state_dict"])
    assert {k: v for k, v in lower["settings"].items() if k != "lr"} == {k: v for k, v in control["settings"].items() if k != "lr"}
    assert all(int(value["step"]) == 6 for value in left["optimizer_state_dict"]["state"].values())


def test_historical_monitor_replay_keeps_nonzero_streak():
    monitor = SafetyMonitor({"warmup_steps": 1, "gradient_threshold": 5., "gradient_persistence": 3})
    history = []
    diagnostics = []
    for step, norm in ((1, 1.), (2, 6.), (3, 7.)):
        row = {"step": step, "policy_ce": 1., "gradient_norm": norm}
        history.append(row)
        monitor.update_loss(step, row["policy_ce"], norm)
        diag = {"step": step, **{population: {metric: {"mean": 1., "maximum": 1.}
                for metric in monitor.SCALE_METRICS} for population in ("sampled", "fixed")}}
        diagnostics.append(diag)
        monitor.update_scale(step, diag)
    report = {"history": history, "diagnostics": diagnostics, "settings": {"diagnostic_every": 1},
              "safety": monitor.report()}
    restored = restore_safety(report, {"step": 3})
    assert restored.streaks["gradient_norm"] == 2
    assert restored.update_loss(4, 1., 8.)
    assert restored.events[-1]["abort"]
    broken = deepcopy(report)
    broken["history"].pop(0)
    with pytest.raises(ValueError, match="complete ordered"):
        restore_safety(broken, {"step": 3})
    broken = deepcopy(report)
    broken["safety"]["references"]["scale:fixed:context_rms:mean"] = 42
    with pytest.raises(ValueError, match="disagrees"):
        restore_safety(broken, {"step": 3})


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_all_rng_families_restore_exact_draws(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    device = torch.device(device)
    sampler = np.random.default_rng(271)
    torch.manual_seed(17)
    np.random.seed(18)
    random.seed(19)
    saved = _rng_state(sampler, device)

    def draw():
        return (random.random(), np.random.random(4), sampler.integers(1000, size=4),
                torch.rand(4), torch.rand(4, device=device).cpu())
    expected = draw()
    _restore_rng(saved, sampler, device)
    assert state_digest(saved) == state_digest(_rng_state(sampler, device))
    actual = draw()
    assert expected[0] == actual[0]
    assert np.array_equal(expected[1], actual[1]) and np.array_equal(expected[2], actual[2])
    assert torch.equal(expected[3], actual[3]) and torch.equal(expected[4], actual[4])


def test_mismatched_data_and_selected_only_checkpoints_are_rejected(source, tmp_path):
    data, split, _ = source
    changed = deepcopy(data)
    changed[0]["rank_target"] = (changed[0]["rank_target"] + 1) % 4
    with pytest.raises(ValueError, match="Training data manifest changed"):
        continue_stability(changed, tmp_path / "changed", validation=data,
            checkpoint_path=split["terminal_checkpoint"], source_report=split, lr=.0003,
            additional_steps=1, device="cpu", max_seconds=30)
    with pytest.raises(ValueError, match="terminal optimizer"):
        continue_stability(data, tmp_path / "selected", validation=data,
            checkpoint_path=split["selected_checkpoint"], source_report=split, lr=.0003,
            additional_steps=1, device="cpu", max_seconds=30)
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        run(source, tmp_path / "hash", expected_checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="Frozen row identities"):
        run(source, tmp_path / "indices", fixed_indices=[0])


@pytest.mark.parametrize("constraint", ["deadline", "memory"])
def test_resource_stops_save_non_success_without_updates(source, tmp_path, constraint):
    options = {"max_seconds": .001} if constraint == "deadline" else {"memory_limit_bytes": 1}
    result = run(source, tmp_path / constraint, **options)
    assert not result["complete"] and not result["durability_passed"]
    assert result["additional_steps"] == 0
    assert result["stop_reason"] == ("time_budget" if constraint == "deadline" else "memory_budget")
    assert Path(result["terminal_checkpoint"]).exists()
    assert result["selected_checkpoint"] is None
    saved = torch.load(result["terminal_checkpoint"], weights_only=True)
    original = torch.load(source[1]["terminal_checkpoint"], weights_only=True)
    assert state_digest(saved["model_state_dict"]) == state_digest(original["model_state_dict"])


def test_storage_guard_failure_is_recorded_before_any_update(source, tmp_path):
    def stop():
        raise OSError("fixture artifact ceiling")
    output = tmp_path / "storage"
    with pytest.raises(OSError, match="artifact ceiling"):
        run(source, output, artifact_guard=stop)
    report = json.loads((output / "report.json").read_text())
    assert not report["complete"] and report["additional_steps"] == 0
    assert report["failure"]["type"] == "OSError"
    assert Path(report["terminal_checkpoint"]).exists()


def test_mps_optimizer_and_model_split_resume_match_uninterrupted(tmp_path, record_property):
    """Exercise actual accelerator AdamW restore, allowing bounded float32 drift."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    atol, rtol = 2e-6, 2e-5
    record_property("mps_resume_atol", atol)
    record_property("mps_resume_rtol", rtol)
    data = records(True)
    args = dict(validation=data, seed=71, epsilon=.01, batch_size=4, device="mps",
                max_seconds=30, raw_config=SMALL, validate_every=2,
                validation_per_stratum=4, diagnostic_every=2, diagnostic_per_stratum=2,
                safety={"warmup_steps": 2})
    split = train_stability(data, tmp_path / "mps-split", steps=4, **args)
    full = train_stability(data, tmp_path / "mps-full", steps=8, **args)
    assert split["complete"] and full["complete"]
    resumed = continue_stability(data, tmp_path / "mps-resumed", validation=data,
        checkpoint_path=split["terminal_checkpoint"], source_report=split,
        lr=.0003, additional_steps=4, device="mps", max_seconds=30,
        validation_per_stratum=4, diagnostic_per_stratum=2, expected_start_step=4)
    assert resumed["complete"] and resumed["completed_steps"] == 8
    left = torch.load(resumed["terminal_checkpoint"], weights_only=True)
    right = torch.load(full["terminal_checkpoint"], weights_only=True)
    assert left["step"] == right["step"] == 8
    assert state_digest(left["rng_state"]) == state_digest(right["rng_state"])
    assert resumed["sampled_index_digest"] == full["sampled_index_digest"]
    assert resumed["sampled_prefixes"] == full["sampled_prefixes"][2:]
    initial = torch.load(split["terminal_checkpoint"], weights_only=True)
    original_monitor = restore_safety(split, initial)
    assert resumed["initial_safety_state"] == safety_state(original_monitor)
    assert resumed["safety"]["references"] == split["safety"]["references"]
    assert resumed["safety_state"]["streaks"] == dict(restore_safety(full, right).streaks)
    assert not resumed["safety"]["tripped"] and not resumed["safety"]["scale_alarm"]
    maximum = 0.
    for name, value in left["model_state_dict"].items():
        other = right["model_state_dict"][name]
        maximum = max(maximum, float((value - other).abs().max()))
        assert torch.allclose(value, other, atol=atol, rtol=rtol), name
    record_property("mps_resume_model_max_abs", maximum)
    assert left["optimizer_state_dict"]["param_groups"] == right["optimizer_state_dict"]["param_groups"]
    maximum = 0.
    for parameter, values in left["optimizer_state_dict"]["state"].items():
        for name, value in values.items():
            other = right["optimizer_state_dict"]["state"][parameter][name]
            if name == "step":
                assert torch.equal(value, other) and int(value) == 8
            elif isinstance(value, torch.Tensor):
                maximum = max(maximum, float((value - other).abs().max()))
                assert torch.allclose(value, other, atol=atol, rtol=rtol), (parameter, name)
            else:
                assert value == other
    record_property("mps_resume_optimizer_max_abs", maximum)
