"""Small numerical fixtures, not a training campaign or strength benchmark."""

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from riichi_ai.features import encode_view
from riichi_ai.model import NeuralPolicy, PolicyNetwork
from riichi_ai.player import PlayerAdapter
from riichi_ai.semantics import semantic_actions
from riichi_ai.simulator import create_env
from riichi_ai.training import (
    check_cpu_mps_parity, check_resume, collate_training, evaluate_model, losses,
    tiny_overfit, train,
)


SMALL = {"suit_width": 16, "history_width": 8, "hidden_width": 32,
         "context_width": 16, "candidate_width": 16}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def records():
    result = []
    for index, seed in enumerate((131, 133, 137, 139)):
        _, observations = create_env(seed)
        view = PlayerAdapter(0).build(observations[0])
        view = replace(view, legal_actions=tuple(a for a in view.legal_actions if a.kind == "DISCARD"))
        actions = semantic_actions(view)
        assert len(actions) > 3
        # Deliberately known synthetic targets: two unique-best and two
        # multiple-best distributions. These are optimizer fixtures, not
        # claims that these moves are strategically correct teacher labels.
        best = np.zeros(len(actions), dtype=np.bool_)
        best[[0] if index % 2 == 0 else [1, 2]] = True
        result.append({"encoded": encode_view(view, actions), "policy_target": best.astype(np.float32) / best.sum(),
                       "best_mask": best, "chosen_index": int(np.flatnonzero(best)[0]),
                       "rank_target": index, "optional": True})
    return result


def test_optional_policy_loss_entropy_and_forced_rank_target():
    example = records()[1]
    forced = {**example, "encoded": {**example["encoded"], "action_tokens": example["encoded"]["action_tokens"][:1]},
              "policy_target": np.ones(1, np.float32), "best_mask": np.ones(1, bool),
              "chosen_index": 0, "optional": False}
    batch, labels = collate_training([example, forced])
    logits = torch.zeros(batch["candidate_mask"].shape, requires_grad=True)
    ranks = torch.zeros((2, 4), requires_grad=True)
    total, components = losses(logits, ranks, labels)
    count = len(example["policy_target"])
    assert float(components["policy_ce"].detach()) == pytest.approx(math.log(count))
    assert float(components["policy_kl"].detach()) == pytest.approx(math.log(count) - math.log(2))
    assert float(components["rank_ce"].detach()) == pytest.approx(math.log(4))
    total.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert torch.count_nonzero(ranks.grad[1]) > 0


def test_uniform_prediction_metrics_have_known_values():
    data = records()
    model = PolicyNetwork(SMALL)
    for parameter in model.parameters():
        parameter.data.zero_()
    report = evaluate_model(model, data, batch_size=3)
    assert report["complete"] and report["illegal_choices"] == 0
    assert report["tie_agreement"] == report["exact_agreement"] == 0.5
    assert report["rank_nll"] == pytest.approx(math.log(4))
    assert report["rank_brier"] == pytest.approx(0.75)
    assert report["rank_accuracy"] == pytest.approx(0.25)
    assert report["random_legal_tie_agreement"] == pytest.approx(np.mean([
        sample["best_mask"].sum() / len(sample["best_mask"]) for sample in data]))
    assert report["per_action_kind"]["DISCARD"]["optional_rows"] == 4


def test_cpu_resume_matches_uninterrupted_losses_and_weights_exactly(tmp_path):
    report = check_resume(records(), tmp_path, steps=6, config=SMALL, max_seconds=30)
    assert report["passed"]
    assert report["maximum_parameter_difference"] == 0
    assert report["maximum_loss_difference"] == 0
    NeuralPolicy.from_checkpoint(tmp_path / "resumed" / "checkpoint.pt")


def test_resume_rejects_changed_validation_and_training_content(tmp_path):
    data = records()
    arguments = dict(steps=1, batch_size=4, config=SMALL, validate_every=1,
                     checkpoint_every=5, validation_max_samples=4, max_seconds=10)
    first = train(data, tmp_path / "first", validation=data, **arguments)
    changed = [dict(sample) for sample in data]
    changed[0]["rank_target"] = 3
    with pytest.raises(ValueError, match="validation data differs"):
        train(data, tmp_path / "bad_validation", validation=changed,
              resume_from=first["checkpoint"], **arguments)
    with pytest.raises(ValueError, match="data/configuration differs"):
        train(changed, tmp_path / "bad_training", validation=data,
              resume_from=first["checkpoint"], **arguments)


def test_one_hot_training_uses_soft_validation_and_saves_selected_model(tmp_path):
    data = records()
    report = train(data, tmp_path, validation=data, steps=2, batch_size=4, config=SMALL,
                   target_mode="one_hot", validate_every=1, max_seconds=10)
    assert report["failure"] is None and report["completed_steps"] == 2
    assert report["training_config"]["target_mode"] == "one_hot"
    assert all(item["target_mode"] == "soft" for item in report["validations"])
    assert report["best_validation"] is not None
    NeuralPolicy.from_checkpoint(report["best_checkpoint"])


def test_resource_stop_and_invalid_targets_are_explicit(tmp_path):
    data = records()
    report = train(data, tmp_path, steps=2, config=SMALL, memory_limit_bytes=1, max_seconds=10)
    assert report["completed_steps"] == 0 and report["stop_reason"] == "memory_budget"
    bad = {**data[0], "policy_target": np.zeros_like(data[0]["policy_target"])}
    with pytest.raises(ValueError, match="Invalid semantic target"):
        collate_training([bad])


def test_tiny_soft_fit_learns_nontrivial_tied_and_unique_targets(tmp_path):
    report = tiny_overfit(records(), tmp_path, sample_count=4, config=SMALL,
                          max_steps=1000, max_seconds=20)
    assert report["tiny_fit_passed"], report.get("best_validation")
    assert report["best_validation"]["policy_kl"] <= 0.02
    assert report["best_validation"]["tie_agreement"] == 1
    assert sorted(report["data_manifest"]["tie_sizes"]) == [1, 1, 2, 2]


def test_mps_check_reports_unavailable_without_a_fabricated_pass(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    result = check_cpu_mps_parity(records(), config=SMALL)
    assert result["status"] == "unavailable" and not result["passed"]
