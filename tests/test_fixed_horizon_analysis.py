"""Saved-data horizon/provenance guards; no models or simulated matches."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location("fixed_horizon_analysis", SCRIPTS / "analyze_stage3e.py")
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def receipt(name):
    return {"name": name, "config_sha256": "campaign", "implementation": {"source": "sha"},
            "status": "completed", "failure": None, "started_unix": 1, "finished_unix": 2}


def budget_receipt():
    result = receipt("eval:31")
    names = [f"batch:{i}" for i in range(4, 32)] + [f"eval:{i}" for i in range(32)]
    result["budget"] = {"identity": "campaign", "limit_seconds": 1200, "operations": [
        {"name": name, "status": "completed", "failure": None} for name in names]}
    return result


@pytest.mark.parametrize("mutation", ["missing_training", "missing_evaluation", "duplicate", "reordered", "active", "failed", "wrong_campaign"])
def test_final_receipt_requires_closed_complete_training_then_evaluation(mutation):
    item = budget_receipt()
    analysis.validate_closed_campaign_budget(item, "campaign")
    operations = item["budget"]["operations"]
    if mutation == "missing_training":
        del operations[:28]
    elif mutation == "missing_evaluation":
        operations.pop()
    elif mutation == "duplicate":
        operations.insert(0, deepcopy(operations[0]))
    elif mutation == "reordered":
        operations[0], operations[1] = operations[1], operations[0]
    elif mutation in ("active", "failed"):
        operations[10]["status"] = mutation
    else:
        item["budget"]["identity"] = "other"
    with pytest.raises(ValueError, match="Fixed training/evaluation"):
        analysis.validate_closed_campaign_budget(item, "campaign")


def valid_record(role):
    training = role == "training"
    planned = {"match_index": 576 if training else 1024, "seed": 7, "sampling_seed": 9, "learner_seat": 0}
    batch_index, block, arm = (4, None, "learner") if training else (2000, 0, "candidate")
    row = {"training_eligible": training, "eligible": True, "policy_id": "fixed-policy", "seat": 0,
           "batch_index": batch_index, "match_index": planned["match_index"], "fallback": False, "overridden": False}
    record = {**planned, "schema": "stage3c-frozen-base-call-heads-v1", "completed": True, "terminated": True,
        "terminal_result_verified": True, "failure": None, "truncated": False, "interrupted": False,
        "collection_role": role, "training_eligible": training, "learner_rows": [row],
        "collector_schema": "stage3e-continuation-lifecycle-v1", "campaign_sha256": "campaign",
        "protocol_sha256": "behavior", "campaign_plan_sha256": "plan", "campaign_phase": role,
        "campaign_block": block, "campaign_arm": arm, "campaign_batch_index": batch_index,
        "batch_index": batch_index, "policy_id": "fixed-policy"}
    arguments = dict(campaign="campaign", behavior="behavior", plan_sha="plan", role=role, batch_index=batch_index,
                     block=block, arm=arm, planned=planned, state={"policy_id": "fixed-policy"})
    return record, arguments


@pytest.mark.parametrize("role", ["training", "evaluation"])
@pytest.mark.parametrize("mutation", ["partial", "match_role", "row_eligibility", "opponent_row", "behavior_protocol", "campaign", "batch", "old_collector"])
def test_summary_excludes_partial_foreign_or_wrong_role_data(role, mutation):
    record, arguments = valid_record(role)
    analysis.validate_record(record, **arguments)
    if mutation == "partial":
        record["completed"] = False
    elif mutation == "match_role":
        record["collection_role"] = "evaluation" if role == "training" else "training"
    elif mutation == "row_eligibility":
        record["learner_rows"][0]["training_eligible"] = not record["training_eligible"]
    elif mutation == "opponent_row":
        record["learner_rows"][0]["seat"] = 1
    elif mutation == "behavior_protocol":
        record["protocol_sha256"] = "campaign"
    elif mutation == "campaign":
        record["campaign_sha256"] = "previous"
    elif mutation == "batch":
        record["campaign_batch_index"] += 1
    else:
        record["collector_schema"] = "stage3d-evaluation-lifecycle-v1"
    with pytest.raises(ValueError):
        analysis.validate_record(record, **arguments)


def valid_batch():
    before = {"updates": 4, "optimizer_steps_executed": 4, "retired_batches": list(range(4)),
        "base_weights_sha256": "base", "immutable_base_sha256": "base", "rank_head_sha256": "rank",
        "optimizer_config": {"learning_rate": 1e-4}, "model_config": {"model": "fixed"},
        "head_config": {"heads": "fixed"}, "reference_sha256": "reference", "protocol_sha256": "behavior"}
    after = {**before, "updates": 5, "optimizer_steps_executed": 5, "retired_batches": list(range(5))}
    boundary = {"path": "old.pt", "sha256": "old", "updates": 4}
    planned = [{"match_index": i} for i in range(576, 592)]
    update = {"before": before, "after": after, "batch_index": 4, "planned_matches": planned,
        "completed_matches": 16, "loss_denominator_matches": 16, "optimizer_steps": 1, "opponent_rows_in_loss": 0,
        "probability_replay_max_absolute_error": 0, "selected_log_probability_replay_max_error": 0,
        "distribution_diagnostic_complete": True, "distribution_diagnostic_rows": 25,
        "optional_learner_rows": 25, "gradients_finite": True, "parameters_finite": True,
        "base_unchanged": True, "rank_head_unchanged": True}
    batch = {"schema": "stage3e-batch-v1", "complete": True, "batch_index": 4,
        "config_sha256": "campaign", "behavior_protocol_sha256": "behavior", "plan_sha256": "plan",
        "planned_matches": planned, "input_state": before, "input_boundary": boundary,
        "implementation": {"source": "sha"}, "termination_requested": False, "matches": [{}] * 16,
        "output_state": after, "update": update, "boundary_roundtrip_verified": True,
        "output_boundary": {**after, "path": "new.pt", "sha256": "new"}}
    arguments = dict(index=4, campaign="campaign", behavior="behavior", plan_sha="plan", planned=planned,
                     state=deepcopy(before), boundary=boundary, implementation={"source": "sha"})
    return batch, receipt("batch:4"), arguments


@pytest.mark.parametrize("mutation", ["skipped_update", "reset_optimizer", "changed_lr", "incomplete_diagnostic", "lost_rows", "replay_error", "frozen_rank", "active_receipt", "wrong_checkpoint"])
def test_training_summary_requires_exact_continuation_and_complete_diagnostics(mutation):
    batch, closed, arguments = valid_batch()
    analysis.validate_training_batch(batch, closed, **arguments)
    if mutation == "skipped_update":
        batch["output_state"]["updates"] = 6
    elif mutation == "reset_optimizer":
        batch["input_state"]["optimizer_steps_executed"] = 0
    elif mutation == "changed_lr":
        batch["output_state"]["optimizer_config"] = {"learning_rate": 1e-3}
    elif mutation == "incomplete_diagnostic":
        batch["update"]["distribution_diagnostic_complete"] = False
    elif mutation == "lost_rows":
        batch["update"]["distribution_diagnostic_rows"] = 24
    elif mutation == "replay_error":
        batch["update"]["probability_replay_max_absolute_error"] = 1e-9
    elif mutation == "frozen_rank":
        batch["output_state"]["rank_head_sha256"] = "changed"
    elif mutation == "active_receipt":
        closed["status"] = "active"
    else:
        batch["output_boundary"]["updates"] = 7
    with pytest.raises(ValueError):
        analysis.validate_training_batch(batch, closed, **arguments)


def test_fixed_eval_block_rejects_swapped_baseline_or_unclosed_receipt():
    states = {"candidate": {"updates": 32}, "baseline": {"updates": 4}}
    boundaries = {"candidate": {"sha256": "candidate"}, "baseline": {"sha256": "baseline"}}
    planned = {"pairs": []}
    block = {"schema": "stage3e-evaluation-block-v1", "complete": True, "block": 0,
        "config_sha256": "campaign", "behavior_protocol_sha256": "behavior", "plan_sha256": "plan",
        "planned": planned, "states_before": deepcopy(states), "states_after": deepcopy(states),
        "boundaries": boundaries, "implementation": {"source": "sha"}, "termination_requested": False,
        "matches": [{}] * 8}
    args = dict(index=0, campaign="campaign", behavior="behavior", plan_sha="plan", planned=planned,
                states=states, boundaries=boundaries, implementation={"source": "sha"})
    analysis.validate_closed_block(block, receipt("eval:0"), **args)
    with pytest.raises(ValueError):
        analysis.validate_closed_block(block, receipt("eval:0") | {"status": "active"}, **args)
    block["states_after"]["baseline"]["updates"] = 0
    with pytest.raises(ValueError, match="Fixed evaluation"):
        analysis.validate_closed_block(block, receipt("eval:0"), **args)


def probability_row(step, probabilities):
    return {"step": step, "player_view_sha256": f"view-{step}", "encoded_sha256": f"encoded-{step}",
        "semantic_keys": ["pass", "pon"], "uniform_draw": .4, "probabilities": probabilities,
        "call_candidates": 1, "call_mass": probabilities[1], "gate_mass": probabilities[1],
        "selected_semantic_key": "pass"}


def test_probability_change_uses_only_shared_visible_prefix():
    first = {"learner_rows": [probability_row(0, [.95, .05]), probability_row(2, [.94, .06]), probability_row(4, [.1, .9])]}
    second = {"learner_rows": [probability_row(0, [.95, .05]), probability_row(2, [.95, .05]), probability_row(4, [.9, .1])]}
    second["learner_rows"][-1]["player_view_sha256"] = "different-after-divergence"
    values = analysis.common_state_probability_rows(first, second, {"first_divergent_step": 2})
    assert [value["step"] for value in values] == [0, 2]
    assert values[1]["probability_l1_change"] == pytest.approx(.02)
    with pytest.raises(ValueError, match="identical player-visible"):
        analysis.common_state_probability_rows(first, second, {"first_divergent_step": None})
