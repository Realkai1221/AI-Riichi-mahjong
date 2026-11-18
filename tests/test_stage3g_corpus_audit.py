"""Tamper rejection for behavior provenance; synthetic rows are not game evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from riichi_ai import stage3g_collect as collector

SPEC = importlib.util.spec_from_file_location("stage3g_independent_corpus_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_stage3g_corpus.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    plan = collector.derive_plan(321, 0, "development")
    plan["seat_behaviors"] = list(collector.BEHAVIORS)
    rngs = [random.Random(seed) for seed in plan["sampling_seeds"]]
    policies = []
    for seat, name in enumerate(collector.BEHAVIORS):
        details = {"synthetic": True}
        if name == "explorer":
            details.update(collector_schema=collector.SCHEMA, sampler=collector.SAMPLER,
                epsilon=collector.EXPLORATION, win_rule="first legal semantic RON/TSUMO, probability 1",
                component_scope="sampled independently at each non-win explorer decision",
                uniform_scope="legal semantic non-win candidates")
        policies.append({"seat": seat, "behavior": name, "behavior_id": seat,
            "policy_id": name + ":" + audit.digest(details), "details": details,
            "sampling_seed": plan["sampling_seeds"][seat]})
    record = {"schema": collector.SCHEMA, "plan": plan, "policies": policies,
        "completed": True, "failure": None, "training_eligible": False, "ppo_eligible": False,
        "sampling_rng_start": [rng.getstate() for rng in rngs], "rows": [], "requests": []}
    for step, seats in enumerate(([0, 1, 2, 3], [3], [3])):
        hashes = {str(seat): audit.digest([step, seat]) for seat in seats}
        pre_id = audit.digest({"step": step, "views": hashes})
        batch = []
        for seat in seats:
            kinds = ["PASS", "PON"] if step == 0 else ["PASS", "RON"] if step == 1 else ["PASS"]
            if seat == 3:
                selection = collector.explore([SimpleNamespace(kind=kind) for kind in kinds],
                    [.95, .05] if len(kinds) == 2 else [1.], rngs[seat])
            else:
                selection = {"selected_index": 0, "probabilities": [1., 0.], "probability": 1.,
                    "selected_component": collector.BEHAVIORS[seat], "component_probability": 1.,
                    "conditional_probability": 1., "joint_probability": 1., "component_draw": None, "uniform_draw": None}
            index = selection["selected_index"]
            tokens = [f"physical-{kind}" for kind in kinds]
            keys = [f"semantic-{kind}" for kind in kinds]
            row = {"row_index": len(record["rows"]), "step": step, "seat": seat,
                "policy_id": policies[seat]["policy_id"], "behavior_id": seat,
                "pre_response_id": pre_id, "player_view_sha256": hashes[str(seat)],
                "training_eligible": False, "ppo_eligible": False, "semantic_keys": keys,
                "action_kinds": kinds, "representative_tokens": tokens, "physical_candidates": tokens,
                "selected_kind": kinds[index], "selected_semantic_key": keys[index],
                "selected_physical_token": tokens[index], **selection}
            record["rows"].append(row)
            batch.append(row["row_index"])
        record["requests"].append({"step": step, "submitted": True, "rows": batch,
                                   "pre_response_id": pre_id, "player_views": hashes})
    record["sampling_rng_end"] = [rng.getstate() for rng in rngs]
    return record


def test_complete_synthetic_behavior_replays_exact_rng_and_components():
    record = fixture()
    counts = audit.audit_behavior(record)
    assert counts["rows"] == 6 and counts["win_heuristic_rows"] == 1
    assert sum(counts.get(name, 0) for name in ("semantic_uniform", "gated_base")) == 2


@pytest.mark.parametrize("mutation", [
    "seat_seed", "rng_start", "rng_end", "component_draw", "action_draw", "marginal",
    "selected_propensity", "component_propensity", "conditional_propensity", "joint_propensity",
    "component_identity", "selected_semantic", "selected_physical", "policy_identity",
    "epsilon", "row_eligible", "match_eligible", "request_state", "row_state", "missing_row",
    "duplicate_row", "unsubmitted", "non_normalized_base", "uniform_physical_bias",
])
def test_behavior_tampering_is_rejected(mutation):
    record = fixture()
    row = record["rows"][3]
    if mutation == "seat_seed":
        record["plan"]["sampling_seeds"][3] += 1
    elif mutation == "rng_start":
        record["sampling_rng_start"][3] = random.Random(444).getstate()
    elif mutation == "rng_end":
        record["sampling_rng_end"][3] = random.Random(444).getstate()
    elif mutation == "component_draw":
        row["component_draw"] = .0
    elif mutation == "action_draw":
        row["uniform_draw"] = .0
    elif mutation == "marginal":
        row["probabilities"] = [.5, .5]
        row["probability"] = .5
    elif mutation == "selected_propensity":
        row["probability"] *= .9
    elif mutation == "component_propensity":
        row["component_probability"] = .8
    elif mutation == "conditional_propensity":
        row["conditional_probability"] = .8
    elif mutation == "joint_propensity":
        row["joint_probability"] = .8
    elif mutation == "component_identity":
        row["selected_component"] = "another component"
    elif mutation == "selected_semantic":
        row["selected_semantic_key"] = "unknown"
    elif mutation == "selected_physical":
        row["selected_physical_token"] = "unknown"
    elif mutation == "policy_identity":
        row["policy_id"] = record["policies"][0]["policy_id"]
    elif mutation == "epsilon":
        record["policies"][3]["details"]["epsilon"] = .1
        record["policies"][3]["policy_id"] = "explorer:" + audit.digest(record["policies"][3]["details"])
        for candidate in record["rows"]:
            if candidate["seat"] == 3:
                candidate["policy_id"] = record["policies"][3]["policy_id"]
    elif mutation == "row_eligible":
        row["ppo_eligible"] = True
    elif mutation == "match_eligible":
        record["training_eligible"] = True
    elif mutation == "request_state":
        record["requests"][0]["pre_response_id"] = "a" * 64
    elif mutation == "row_state":
        row["player_view_sha256"] = "a" * 64
    elif mutation == "missing_row":
        record["requests"][0]["rows"].pop()
    elif mutation == "duplicate_row":
        record["requests"][0]["rows"].append(3)
    elif mutation == "unsubmitted":
        record["requests"][0]["submitted"] = False
    elif mutation == "non_normalized_base":
        row["base_probabilities"] = [.4, .4]
    elif mutation == "uniform_physical_bias":
        row["uniform_probabilities"] = [.75, .25]
    with pytest.raises(ValueError):
        audit.audit_behavior(record)


@pytest.mark.parametrize("values,draw,expected", [([0., 1.], 0., 1), ([.25, .75], .25, 1),
    ([.25, .75], .249, 0), ([1.], None, 0)])
def test_independent_cdf_preserves_zero_probability_and_boundary_semantics(values, draw, expected):
    assert audit.cdf_choice(np.asarray(values), draw) == expected


def test_win_branch_cannot_invent_rng_consumption():
    record = fixture()
    record["rows"][4]["component_draw"] = .5
    with pytest.raises(ValueError, match="Deterministic branch"):
        audit.audit_behavior(record)


def test_coverage_distinguishes_opportunities_from_chosen_and_resolved_experience():
    rows = [
        {"behavior_id": 3, "action_kinds": ["PASS", "CHI", "CHI"], "selected_kind": "CHI",
         "probabilities": [.8, .1, .1], "probability": .1, "requested_call_resolved": True,
         "post_call_state": False, "open_hand_state": False},
        {"behavior_id": 3, "action_kinds": ["DISCARD", "DISCARD"], "selected_kind": "DISCARD",
         "probabilities": [.25, .75], "probability": .75, "requested_call_resolved": None,
         "post_call_state": True, "open_hand_state": True},
        {"behavior_id": 3, "action_kinds": ["PASS"], "selected_kind": "PASS",
         "probabilities": [1.], "probability": 1., "requested_call_resolved": None,
         "post_call_state": True, "open_hand_state": True},
    ]
    coverage = audit.describe_coverage(rows, {"history_prefix_length": [10, 150, 160]})["explorer"]
    assert coverage["rows"] == 3
    assert coverage["semantic_optional_rows"] == 2 and coverage["semantic_forced_rows"] == 1
    assert coverage["legal_candidates"] == coverage["positive_probability_candidates"] == 6
    assert coverage["optional_rows_with_full_positive_support"] == 2
    assert coverage["call_opportunity_rows"] == coverage["multiple_call_candidate_rows"] == 1
    assert coverage["selected_calls_among_multiple_call_candidates"] == coverage["resolved_calls_among_multiple_call_candidates"] == 1
    assert coverage["requested_calls"] == coverage["resolved_calls"] == {"CHI": 1}
    assert coverage["candidate_action_opportunities"]["CHI"] == 1  # Two candidates, one opportunity row.
    assert coverage["candidate_family_opportunities"]["pass"] == 2
    assert coverage["chosen_action_families"] == {"call": 1, "discard": 1, "pass": 1}
    assert coverage["post_call_rows"] == 2 and coverage["optional_post_call_rows"] == 1
    assert coverage["history_overflow_rows"] == 2 and coverage["maximum_history_prefix_length"] == 160
    assert coverage["actual_call_mass_on_call_opportunities"]["count"] == 1
    assert coverage["actual_call_mass_on_call_opportunities"]["sum"] == .2
    merged = audit.empty_coverage()
    audit.merge_coverage(merged, coverage)
    audit.merge_coverage(merged, coverage)
    audit.finalize_coverage(merged)
    assert merged["rows"] == 6 and merged["maximum_history_prefix_length"] == 160
    assert sum(merged["selected_behavior_probability"]["histogram"]) == 6
    assert merged["selected_behavior_probability"]["mean"] == pytest.approx((.1 + .75 + 1) / 3)
    assert merged["actual_call_mass_on_call_opportunities"]["mean"] == .2


def call_fixture(resolved=True):
    declaration = {"actor": 0, "kind": "KAKAN", "event_index": 12, "resolved": resolved}
    rows = [{"seat": 0, "step": 0, "selected_kind": "KAKAN", "requested_call": True,
             "requested_call_resolved": resolved, "call_resolution_evidence": [declaration]},
            {"seat": 1, "step": 0, "selected_kind": "PON", "requested_call": True,
             "requested_call_resolved": False, "call_resolution_evidence": []},
            {"seat": 2, "step": 0, "selected_kind": "PASS", "requested_call": False,
             "requested_call_resolved": None}]
    return rows, [{"auditor_event_start": 12, "auditor_event_end": 14}], [declaration]


@pytest.mark.parametrize("resolved", [True, False])
def test_call_reconciliation_accepts_replacement_or_robbed_kan_and_rejected_other_request(resolved):
    rows, requests, declarations = call_fixture(resolved)
    audit.reconcile_calls(rows, requests, declarations, 20)


@pytest.mark.parametrize("mutation", ["unresolved", "hidden_request", "missing_declaration", "duplicate_attribution",
    "false_resolution", "noncall_resolution", "interval_outside_log", "wrong_actor"])
def test_call_reconciliation_rejects_inconsistent_evidence(mutation):
    rows, requests, declarations = deepcopy(call_fixture())
    if mutation == "unresolved":
        declarations[0]["resolved"] = None
        rows[0]["requested_call_resolved"] = None
    elif mutation == "hidden_request":
        rows[0]["requested_call"] = False
    elif mutation == "missing_declaration":
        rows[0]["call_resolution_evidence"] = []
        rows[0]["requested_call_resolved"] = False
    elif mutation == "duplicate_attribution":
        rows.append(deepcopy(rows[0]))
    elif mutation == "false_resolution":
        rows[1]["requested_call_resolved"] = True
    elif mutation == "noncall_resolution":
        rows[2]["requested_call_resolved"] = True
    elif mutation == "interval_outside_log":
        requests[0]["auditor_event_end"] = 30
    elif mutation == "wrong_actor":
        rows[0]["seat"] = 1
    with pytest.raises(ValueError):
        audit.reconcile_calls(rows, requests, declarations, 20)


def test_only_training_explorer_optional_rows_are_offline_eligible():
    record = fixture()
    record['plan']['role'] = 'training'
    record['training_eligible'] = True
    for row in record['rows']:
        row['training_eligible'] = row['behavior_id'] == 3 and len(row['semantic_keys']) > 1
    assert audit.audit_behavior(record)['rows'] == 6
    record['rows'][0]['training_eligible'] = True
    with pytest.raises(ValueError, match='Offline row eligibility'):
        audit.audit_behavior(record)

def test_preflight_cannot_be_marked_training_eligible():
    record = fixture()
    record['plan']['role'] = 'preflight'
    record['training_eligible'] = True
    with pytest.raises(ValueError, match='Offline match eligibility'):
        audit.audit_behavior(record)
