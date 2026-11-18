"""Evidence-quality guards for a fixed training-only diagnostic; no model loads."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/stage3f_diagnose.py"
SPEC = importlib.util.spec_from_file_location("stage3f_diagnosis_tests", PATH)
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def candidate(index, win=False, chosen=False):
    return {"batch_index": 4, "match_index": 576, "row_index": index,
            "encoded_sha256": "fixed", "strata": {"win_vs_call": win, "selected_multiple_call": chosen}}


def record():
    return {"collection_role": "training", "campaign_phase": "training", "training_eligible": True,
        "completed": True, "terminated": True, "terminal_result_verified": True, "failure": None,
        "truncated": False, "interrupted": False, "batch_index": 4, "policy_id": "behavior",
        "learner_utility": .5, "learner_seat": 2, "match_index": 576,
        "learner_rows": [{"row_index": 0, "batch_index": 4, "match_index": 576, "policy_id": "behavior",
            "seat": 2, "training_eligible": True, "eligible": True, "fallback": False, "overridden": False}]}


def plan():
    return {"schema": diagnosis.SCHEMA, "row_limit": 256, "full_match_gradient_batch": 4,
        "max_seconds": 240, "threads": 1, "device": "cpu", "optimizer_steps": 0, "new_matches": 0,
        "evaluation_rows": 0, "selected_rows": [candidate(1)], "full_match_ids": list(range(576, 592)),
        "matches": {str(index): {"batch_index": 4} for index in range(576, 592)}}


def test_deterministic_selection_preserves_all_rare_strata_and_order_independence():
    source = [candidate(index, win=index < 16, chosen=12 <= index < 95) for index in range(1000)]
    first = diagnosis.select_rows(source)
    second = diagnosis.select_rows(list(reversed(source)))
    assert first == second
    assert len(first) == 256
    assert len({diagnosis.row_identity(item) for item in first}) == 256
    assert all(item in first for item in source if any(item["strata"].values()))
    assert sum(item["strata"]["win_vs_call"] for item in first) == 16
    assert sum(item["strata"]["selected_multiple_call"] for item in first) == 83


def test_selection_remainder_does_not_depend_on_rewards_or_model_outputs():
    source = [candidate(index) for index in range(300)]
    changed = deepcopy(source)
    for index, item in enumerate(changed):
        item["utility"] = -1.5 if index % 2 else 1.5
        item["hypothetical_output"] = index * 10
    assert [diagnosis.row_identity(item) for item in diagnosis.select_rows(source)] == [
        diagnosis.row_identity(item) for item in diagnosis.select_rows(changed)]


@pytest.mark.parametrize("limit", [0, 257, -1, 1.5, True])
def test_invalid_limit_rejected(limit):
    with pytest.raises(ValueError, match="Selection limit"):
        diagnosis.select_rows([candidate(0)], limit)


def test_required_strata_never_silently_truncated():
    with pytest.raises(ValueError, match="Required strata"):
        diagnosis.select_rows([candidate(index, win=True) for index in range(257)])


def test_duplicate_rows_rejected():
    with pytest.raises(ValueError, match="Duplicate row"):
        diagnosis.select_rows([candidate(0), candidate(0)])


@pytest.mark.parametrize("key,value", [
    ("collection_role", "evaluation"), ("campaign_phase", "evaluation"), ("training_eligible", False),
    ("completed", False), ("terminated", False), ("terminal_result_verified", False),
    ("failure", "failed"), ("truncated", True), ("interrupted", True), ("batch_index", 5),
    ("policy_id", "wrong"), ("learner_utility", 1), ("learner_seat", 4)])
def test_evaluation_partial_wrong_behavior_and_wrong_utility_rejected(key, value):
    valid = record()
    diagnosis.validate_record(valid, 4, "behavior")
    valid[key] = value
    with pytest.raises(ValueError):
        diagnosis.validate_record(valid, 4, "behavior")


@pytest.mark.parametrize("key,value", [("row_index", 1), ("batch_index", 5), ("match_index", 99),
    ("policy_id", "wrong"), ("seat", 1), ("training_eligible", False), ("eligible", False),
    ("fallback", True), ("overridden", True)])
def test_frozen_opponent_or_altered_rows_rejected(key, value):
    value_record = record()
    value_record["learner_rows"][0][key] = value
    with pytest.raises(ValueError, match="Wrong-seat"):
        diagnosis.validate_record(value_record, 4, "behavior")


@pytest.mark.parametrize("key,value", [("max_seconds", 300), ("threads", 2), ("device", "mps"),
    ("optimizer_steps", 1), ("new_matches", 1), ("evaluation_rows", 1),
    ("full_match_gradient_batch", 5), ("row_limit", 257)])
def test_scope_expansion_rejected(key, value):
    valid = plan()
    diagnosis.validate_plan(valid)
    valid[key] = value
    with pytest.raises(ValueError, match="scope changed"):
        diagnosis.validate_plan(valid)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_batch", "selected_duplicate", "too_many"])
def test_complete_fixed_batch_and_selected_row_ceiling_required(mutation):
    value = plan()
    if mutation == "missing":
        value["full_match_ids"].pop()
    elif mutation == "duplicate":
        value["full_match_ids"][-1] = value["full_match_ids"][0]
    elif mutation == "wrong_batch":
        value["matches"]["576"]["batch_index"] = 5
    elif mutation == "selected_duplicate":
        value["selected_rows"] *= 2
    else:
        value["selected_rows"] = [candidate(index) for index in range(257)]
    with pytest.raises(ValueError):
        diagnosis.validate_plan(value)


def test_input_mutation_detected_before_model_load(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"record": 1}')
    item = diagnosis.artifact(source)
    diagnosis.verify_artifact(item)
    source.write_text('{"record": 2}')
    with pytest.raises(ValueError, match="hash mismatch"):
        diagnosis.verify_artifact(item)


def test_no_overwrite_evidence(tmp_path):
    output = tmp_path / "evidence.json"
    diagnosis.write_new(output, {"valid": True})
    with pytest.raises(FileExistsError):
        diagnosis.write_new(output, {"valid": False})
    assert json.loads(output.read_text()) == {"valid": True}


def test_visible_context_uses_explicit_hand_ordinal_and_no_hidden_log_fields():
    class Matrix:
        def __getitem__(self, key):
            i, j = key
            return .25 if (i, j) == (0, 0) else 0
    source = {"auditor_log": [
        {"type": "start_kyoku", "bakaze": "S", "kyoku": 4, "honba": 1, "oya": 3,
         "tehais": ["secret"], "wall": "secret"},
        {"type": "start_kyoku", "bakaze": "S", "kyoku": 4, "honba": 2, "oya": 3,
         "tehais": ["different secret"], "wall": "secret"}]}
    row = {"seat": 2, "hand_id": 2, "own_meld_count": 1, "own_open_meld_count": 1,
           "semantic_keys": ["RON", "CHI"], "selected_semantic_key": "CHI", "selected_kind": "CHI",
           "probabilities": [.95, .05], "uniform_draw": .99, "requested_call_resolved": True}
    values = [.5, .6, .4, .3] + [0.] * 41
    result = diagnosis.visible_context(source, row, {"globals": values, "tiles": Matrix()})
    assert result["hand_ordinal"] == 2 and result["honba"] == 2
    assert result["current_scores_absolute_seats"] == [20000, 15000, 25000, 30000]
    assert result["own_concealed_tiles"] == [{"tile": "1m", "count": 1, "red_count": 0}]
    assert "secret" not in json.dumps(result) and "wall" not in result and "tehais" not in result
