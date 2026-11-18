"""Matched experimental and metric fixtures; no strength-training campaign."""
from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from riichi_ai.features import encode_view, FEATURE_SCHEMA_V2
from riichi_ai.model import PolicyNetwork, efficiency_model_config, matched_initialization, NeuralPolicy
from riichi_ai.player import PlayerAdapter
from riichi_ai.semantics import semantic_actions
from riichi_ai.simulator import create_env
from riichi_ai.stage2b_training import (
    BudgetLedger, EqualRowMixture, evaluate_stage2b, select_recipes, stratified_indices, train_stage2b,
    complete_validation_schedule,
)
from riichi_ai.training import check_resume

SMALL = {"suit_width": 16, "history_width": 8, "hidden_width": 32,
         "context_width": 16, "candidate_width": 16}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def records(assisted=False):
    _, observations = create_env(131)
    view = PlayerAdapter(0).build(observations[0])
    view = replace(view, legal_actions=tuple(a for a in view.legal_actions if a.kind == "DISCARD"))
    actions = semantic_actions(view)[:2]
    view = replace(view, legal_actions=tuple(action.representative for action in actions))
    actions = semantic_actions(view)
    encoded = encode_view(view, actions, feature_schema=FEATURE_SCHEMA_V2 if assisted else "actor-features-v1")
    result = []
    for index in range(16):
        # Synthetic diagnostic preferences: correct, +1 shanten, -3 ukeire,
        # red-only error. Teacher has each pattern 3x; student is all correct.
        pattern = index % 4 if index < 12 else 0
        keys = np.asarray([[0, 0, 2, -10, 0], [0, 0, 2, -10, 0]], np.int64)
        best = np.asarray([True, False])
        keys[1, 2] += 1
        if pattern:
            keys[1] = [0, 0, 2, -10, 0]
            keys[0, {1: 2, 2: 3, 3: 4}[pattern]] += {1: 1, 2: 3, 3: 1}[pattern]
            best = np.asarray([False, True])
        result.append({"encoded": encoded, "policy_target": best.astype(np.float32),
                       "best_mask": best, "chosen_index": int(best.argmax()), "rank_target": index % 4,
                       "optional": True, "teacher_preference_keys": keys,
                       "stratum": "teacher" if index < 12 else "student", "match_id": str(index // 4),
                       "seat": index % 4, "step": index})
    return result


def test_equal_stratum_metrics_and_known_error_severities():
    model = PolicyNetwork(SMALL)
    for parameter in model.parameters():
        parameter.data.zero_()
    report = evaluate_stage2b(model, records(), batch_size=3)
    assert report["complete"] and report["illegal_choices"] == 0
    assert report["discard_tie_agreement"] == 7 / 16
    assert report["equal_stratum_means"]["discard_tie_agreement"] == .625
    assert report["equal_stratum_means"]["shanten_worsening_rate"] == .125
    assert report["strata"]["teacher"]["shanten_increments"]["histogram"] == {"1": 3}
    assert report["strata"]["teacher"]["same_shanten_lost_ukeire"]["histogram"] == {"3": 3}
    assert report["red_only_errors"] == 3
    assert report["rank_nll"] == pytest.approx(math.log(4))
    assert report["rank_brier"] == pytest.approx(.75)
    assert report["rank_top_confidence_ece_10_bins"] == 0


def test_stratified_fixed_indices_are_identical_for_raw_and_assisted():
    raw, assisted = records(), records(True)
    left = stratified_indices(raw, per_stratum=3, seed=72)
    assert left == stratified_indices(assisted, per_stratum=3, seed=72)
    assert len(left) == 6 and sum(raw[index]["stratum"] == "teacher" for index in left) == 3


def test_matched_training_sampler_rank_zero_and_checkpoint_compatibility(tmp_path):
    raw, assisted, manifest = matched_initialization(71, SMALL)
    assert manifest["maximum_common_difference"] == 0
    for name in raw:
        assert torch.equal(raw[name], assisted[name])
    arguments = dict(steps=2, batch_size=4, seed=71, raw_config=SMALL,
                     validate_every=1, checkpoint_every=2, max_seconds=15,
                     validation_per_stratum=4)
    raw_report = train_stage2b(records(), tmp_path / "raw", validation=records(), assisted=False, **arguments)
    feature_report = train_stage2b(records(True), tmp_path / "feature", validation=records(True),
                                  assisted=True, rank_weight=0, **arguments)
    assert raw_report["failure"] is feature_report["failure"] is None
    assert raw_report["sampled_index_digest"] == feature_report["sampled_index_digest"]
    checkpoint = torch.load(feature_report["checkpoint"], weights_only=True)
    rank_names = [name for name in assisted if name.startswith("rank_head.")]
    assert rank_names
    assert all(torch.equal(assisted[name], checkpoint["model_state_dict"][name]) for name in rank_names)
    assert feature_report["model_config"]["feature_schema"] == FEATURE_SCHEMA_V2
    NeuralPolicy.from_checkpoint(feature_report["best_checkpoint"])


def test_v2_cpu_resume_exact_and_resume_binds_diagnostic_labels(tmp_path):
    data = records(True)
    numerical = check_resume(data, tmp_path / "check", steps=4,
                             config=efficiency_model_config(SMALL), max_seconds=20)
    assert numerical["passed"] and numerical["maximum_parameter_difference"] == 0
    arguments = dict(steps=1, batch_size=4, seed=71, raw_config=SMALL,
                     validate_every=1, checkpoint_every=2, max_seconds=15,
                     validation_per_stratum=4, assisted=True)
    first = train_stage2b(data, tmp_path / "first", validation=data, **arguments)
    modified = deepcopy(data)
    modified[0]["teacher_preference_keys"][0, 3] -= 1
    with pytest.raises(ValueError, match="validation data differs"):
        train_stage2b(data, tmp_path / "bad", validation=modified,
                      resume_from=first["checkpoint"], **arguments)


def test_custom_stratified_selection_resume_is_exact(tmp_path):
    data = records(True)
    arguments = dict(batch_size=4, seed=71, raw_config=SMALL, validate_every=1,
                     checkpoint_every=2, max_seconds=15, validation_per_stratum=4, assisted=True)
    full = train_stage2b(data, tmp_path / "full", validation=data, steps=4, **arguments)
    part = train_stage2b(data, tmp_path / "part", validation=data, steps=2, **arguments)
    resumed = train_stage2b(data, tmp_path / "resumed", validation=data, steps=4,
                            resume_from=part["checkpoint"], **arguments)
    assert full["failure"] is part["failure"] is resumed["failure"] is None
    left = torch.load(full["checkpoint"], weights_only=True)
    right = torch.load(resumed["checkpoint"], weights_only=True)
    assert all(torch.equal(value, right["model_state_dict"][name]) for name, value in left["model_state_dict"].items())
    assert full["sampled_index_digest"] == resumed["sampled_index_digest"]
    assert full["best_step"] == resumed["best_step"]
    assert full["best_validation"]["checkpoint_selection_key"] == resumed["best_validation"]["checkpoint_selection_key"]


def test_recipe_selection_averages_both_seeds_and_keeps_declared_primary():
    def run(recipe, seed, score):
        return {"recipe": recipe, "seed": seed, "eligible": True,
                "full_validation": {"complete": True, "checkpoint_selection_key": [score, -.1, -.2]}}
    runs = [run("a", 1, .2), run("a", 2, 1.), run("b", 1, .55), run("b", 2, .55), run("c", 1, 1)]
    report = select_recipes(runs, required_seeds=[1, 2], primary_seed=1, recipe_order=["a", "b", "c"])
    assert report["winner"]["recipe"] == "a"
    assert report["winner"]["primary"]["seed"] == 1
    assert report["incomplete_recipes"] == ["c"]


def test_equal_mixture_has_exact_minibatch_allocation_and_reproducible_sampling():
    original, student = records()[:12], records()[12:]
    mixed = EqualRowMixture(original, student)
    left = mixed.sample_indices(np.random.default_rng(81), 256)
    right = mixed.sample_indices(np.random.default_rng(81), 256)
    assert np.array_equal(left, right)
    assert sum(left < len(original)) == sum(left >= len(original)) == 128
    assert mixed.dataset_identity()["weights"] == [.5, .5]
    with pytest.raises(ValueError, match="even"):
        mixed.sample_indices(np.random.default_rng(81), 255)


def test_final_and_intermediate_validation_must_complete_before_promotion():
    report = {"validations": [{"step": 2, "complete": True}, {"step": 4, "complete": True}]}
    assert complete_validation_schedule(report, 4, 2)
    report["validations"][-1]["complete"] = False
    assert not complete_validation_schedule(report, 4, 2)
    report["validations"] = [{"step": 4, "complete": True}]
    assert not complete_validation_schedule(report, 4, 2)
    report["validations"] = [{"step": 2, "complete": True}, {"step": 3, "complete": True}]
    assert complete_validation_schedule(report, 3, 2)


def test_ledger_failures_external_charges_identity_and_crashed_reservations(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json", limit_seconds=10, identity="frozen")
    ledger.account("external failed audit", 3, failed=True)
    with pytest.raises(RuntimeError, match="fixture"):
        with ledger.operation("failed fixture", allowance_seconds=2):
            raise RuntimeError("fixture")
    report = ledger.snapshot()
    assert report["charged_seconds"] >= 3 and report["operations"][1]["status"] == "failed"
    with ledger.operation("active", allowance_seconds=4):
        assert ledger.snapshot()["charged_seconds"] >= 7
    with pytest.raises(ValueError, match="identity"):
        BudgetLedger(ledger.path, limit_seconds=10, identity="changed").snapshot()
    with pytest.raises(ValueError, match="Reserve"):
        with ledger.operation("invalid reserve", reserve_seconds=-1):
            pass
