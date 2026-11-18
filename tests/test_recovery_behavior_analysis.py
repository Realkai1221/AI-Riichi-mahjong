"""Saved-data inclusion/provenance checks; no model imports or simulated games."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location("recovery_behavior_analysis", SCRIPTS / "analyze_stage3d.py")
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def valid_record():
    planned = {"match_index": 320, "seed": 7, "sampling_seed": 9, "learner_seat": 0}
    record = {**planned, "completed": True, "terminated": True, "terminal_result_verified": True,
        "failure": None, "truncated": False, "interrupted": False, "collection_role": "evaluation",
        "training_eligible": False, "learner_rows": [{"training_eligible": False}],
        "campaign_sha256": "campaign", "protocol_sha256": "behavior",
        "campaign_plan_sha256": "plan", "campaign_block": 0, "campaign_arm": "candidate",
        "batch_index": 1000, "policy_id": "fixed-candidate"}
    arguments = dict(campaign="campaign", behavior="behavior", plan_sha="plan", block=0,
                     arm="candidate", planned=planned, state={"policy_id": "fixed-candidate"})
    return record, arguments


@pytest.mark.parametrize("key,value", [("completed", False), ("failure", {"type": "TimeoutError"}),
    ("collection_role", "training"), ("training_eligible", True), ("campaign_sha256", "old-campaign"),
    ("protocol_sha256", "campaign"), ("campaign_block", 1), ("seed", 8), ("policy_id", "other")])
def test_partial_historical_or_eligible_match_rejected(key, value):
    record, arguments = valid_record()
    analysis.validate_record(record, **arguments)
    record[key] = value
    with pytest.raises(ValueError):
        analysis.validate_record(record, **arguments)


def test_row_eligibility_cannot_bypass_match_exclusion():
    record, arguments = valid_record()
    record["learner_rows"][0]["training_eligible"] = True
    with pytest.raises(ValueError, match="Training-eligible"):
        analysis.validate_record(record, **arguments)


def test_complete_block_requires_closed_receipt():
    state, implementation, planned = {"candidate": {"update": 4}}, {"a": "sha"}, {"pairs": []}
    block = {"schema": "stage3d-evaluation-block-v1", "complete": True, "block": 0,
        "config_sha256": "campaign", "plan_sha256": "plan", "planned": planned,
        "states_before": state, "states_after": state, "implementation": implementation,
        "termination_requested": False, "matches": [{}] * 8}
    receipt = {"name": "eval:0", "config_sha256": "campaign", "implementation": implementation,
               "status": "completed", "failure": None, "started_unix": 1, "finished_unix": 2}
    args = dict(index=0, campaign="campaign", plan_sha="plan", planned=planned,
                states=state, implementation=implementation)
    analysis.validate_closed_block(block, receipt, **args)
    for change in ({"status": "active"}, {"status": "failed"}, {"failure": {"type": "SourceGuard"}},
                   {"implementation": {"a": "changed"}}):
        with pytest.raises(ValueError):
            analysis.validate_closed_block(block, receipt | change, **args)


@pytest.mark.parametrize("status", ["active", "failed"])
def test_successful_final_receipt_cannot_hide_unclosed_budget_prefix(status):
    receipt = {"name": "eval:31", "config_sha256": "campaign", "status": "completed", "failure": None,
        "budget": {"identity": "campaign", "operations": [
            {"name": f"eval:{index}", "status": "completed", "failure": None} for index in range(32)]}}
    analysis.validate_closed_evaluation_budget(receipt, "campaign")
    receipt["budget"]["operations"][-1]["status"] = status
    with pytest.raises(ValueError, match="incomplete, failed or duplicated"):
        analysis.validate_closed_evaluation_budget(receipt, "campaign")


def row(step, probabilities):
    return {"step": step, "player_view_sha256": f"visible-{step}", "encoded_sha256": f"encoded-{step}",
        "semantic_keys": ["discard", "pon"], "uniform_draw": .4, "probabilities": probabilities,
        "call_candidates": 1, "call_mass": probabilities[1], "gate_mass": probabilities[1],
        "selected_semantic_key": "discard"}


def test_probability_comparison_excludes_states_after_action_divergence():
    first = {"learner_rows": [row(0, [.95, .05]), row(2, [.94, .06]), row(4, [.1, .9])]}
    second = {"learner_rows": [row(0, [.95, .05]), row(2, [.95, .05]), row(4, [.9, .1])]}
    second["learner_rows"][-1]["player_view_sha256"] = "different-state-after-divergence"
    values = analysis.common_state_probability_rows(first, second, {"first_divergent_step": 2})
    assert [value["step"] for value in values] == [0, 2]
    assert values[0]["maximum_absolute_probability_change"] == 0
    assert values[1]["probability_l1_change"] == pytest.approx(.02)
    assert values[1]["call_mass_change"] == pytest.approx(.01)
    summary = analysis.summarize_probability_rows(values)["call_opportunity_rows"]
    assert summary["rows"] == 2 and summary["changed_probability_rows"] == 1
    with pytest.raises(ValueError, match="identical player-visible"):
        analysis.common_state_probability_rows(first, second, {"first_divergent_step": None})


def test_common_prefix_requires_matching_sampler_and_encoding():
    first = {"learner_rows": [row(0, [.94, .06])]}
    for field in ("encoded_sha256", "uniform_draw", "semantic_keys"):
        second = deepcopy(first)
        second["learner_rows"][0][field] = "changed"
        with pytest.raises(ValueError, match="identical player-visible"):
            analysis.common_state_probability_rows(first, second, {"first_divergent_step": None})
