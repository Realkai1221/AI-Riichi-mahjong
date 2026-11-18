"""Saved-data summary invariants, without model or simulator imports."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("analyze_stage3c", Path(__file__).resolve().parents[1] / "scripts/analyze_stage3c.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture_record():
    row = {"action_kinds": ["CHI", "CHI", "PON", "RON", "PASS"], "selected_kind": "PON",
        "call_candidates": 3, "forced": False, "gate_mass": .05, "call_mass": .05,
        "post_call_state": False, "open_hand_state": False, "hand_id": 1,
        "win_mass_without_gate": 1., "win_mass_actual": .95, "win_mass_lost": .05,
        "requested_call_resolved": True}
    counts = {kind: int(kind == "PON") for kind in analysis.CALL_KINDS}
    return {"completed": True, "terminal_result_verified": True, "learner_rows": [row],
        "requested_calls": counts.copy(), "declared_calls": counts.copy(), "resolved_calls": counts.copy(),
        "post_call_rows": 0, "open_hand_rows": 0, "learner_seat": 0, "ranks": [1, 2, 3, 4],
        "scores": [40000, 30000, 20000, 10000], "learner_utility": 1.5, "hands": 1, "steps": 1,
        "seed": 1, "sampling_seed": 2, "match_index": 3,
        "requests": [{"choices": {"0": {"physical_token": "pon-a"}}}]}


def test_overlapping_opportunities_distinguish_rows_candidates_requests_and_wins():
    summary = analysis.summarize([fixture_record()])
    assert summary["call_opportunity_rows"] == 1
    assert summary["multiple_call_opportunity_rows"] == 1
    assert summary["call_kinds"]["CHI"]["opportunity_rows"] == 1
    assert summary["call_kinds"]["CHI"]["candidate_count"] == 2
    assert summary["call_kinds"]["CHI"]["requested"] == 0
    assert summary["call_kinds"]["PON"]["resolved"] == 1
    assert summary["win_call_competition"]["rows"] == 1
    assert summary["win_call_competition"]["selected_nonwin"] == 1
    assert summary["win_call_competition"]["selected_call_resolved"] == 1
    assert summary["win_call_competition"]["sum_win_mass_lost"] == .05


def test_rejects_impossible_resolved_count_or_mismatched_group_count():
    record = fixture_record()
    record["resolved_calls"]["CHI"] = 1
    with pytest.raises(ValueError, match="Inconsistent"):
        analysis.summarize([record])
    record = fixture_record()
    record["learner_rows"][0]["call_candidates"] = 2
    with pytest.raises(ValueError, match="membership"):
        analysis.summarize([record])


def test_robbed_kan_declaration_remains_distinct_from_resolution():
    record = fixture_record()
    record["learner_rows"][0]["action_kinds"] = ["KAKAN", "RON", "PASS"]
    record["learner_rows"][0]["selected_kind"] = "KAKAN"
    record["learner_rows"][0]["call_candidates"] = 1
    record["learner_rows"][0]["requested_call_resolved"] = False
    for field in ("requested_calls", "declared_calls", "resolved_calls"):
        record[field] = {kind: int(kind == "KAKAN" and field != "resolved_calls") for kind in analysis.CALL_KINDS}
    summary = analysis.summarize([record])
    assert summary["call_kinds"]["KAKAN"]["declared"] == 1
    assert summary["call_kinds"]["KAKAN"]["resolved"] == 0


def test_pair_divergence_ignores_policy_identity_and_rejects_inconsistent_equal_histories():
    candidate = fixture_record()
    baseline = deepcopy(candidate)
    baseline["match_index"] = 4
    baseline["requests"][0]["choices"]["0"]["policy_id"] = "different-but-same-action"
    comparison = analysis.compare_pair(candidate, baseline, 0, 0)
    assert comparison["identical_action_sequence"]
    baseline["requests"][0]["choices"]["0"]["physical_token"] = "pass"
    comparison = analysis.compare_pair(candidate, baseline, 0, 0)
    assert comparison["first_divergent_step"] == 0
    baseline = deepcopy(candidate)
    baseline["scores"][0] += 1000
    with pytest.raises(ValueError, match="Identical paired action"):
        analysis.compare_pair(candidate, baseline, 0, 0)
