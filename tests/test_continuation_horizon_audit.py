"""Independent rejection tests for fixed-horizon continuation evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("stage3e_independent_audit", ROOT / "scripts/audit_stage3e.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def record_fixture(role="training", completed=True):
    planned = {"match_index": 576, "seed": 17, "sampling_seed": 42, "learner_seat": 0}
    eligible = role == "training" and completed
    record = {**planned, "campaign_sha256": "a" * 64, "protocol_sha256": audit.BEHAVIOR_PROTOCOL,
        "collector_schema": "stage3e-continuation-lifecycle-v1", "collection_role": role,
        "training_eligible": eligible, "learner_rows": [{"training_eligible": eligible}],
        "completed": completed, "interrupted": not completed,
        "failure": None if completed else {"type": "CollectionCancelled"},
        "learner_utility": .5 if completed else None}
    return record, planned


@pytest.mark.parametrize("role", ["training", "evaluation"])
@pytest.mark.parametrize("completed", [False, True])
def test_only_completed_training_rows_are_eligible(role, completed):
    record, planned = record_fixture(role, completed)
    audit.audit_campaign_record(record, "a" * 64, audit.BEHAVIOR_PROTOCOL,
                                planned, audit.Checks(), role=role)


@pytest.mark.parametrize("role,completed", [("training", False), ("evaluation", False), ("evaluation", True)])
@pytest.mark.parametrize("row", [False, True])
def test_incomplete_or_evaluation_eligibility_rejected(role, completed, row):
    record, planned = record_fixture(role, completed)
    (record["learner_rows"][0] if row else record)["training_eligible"] = True
    with pytest.raises(ValueError):
        audit.audit_campaign_record(record, "a" * 64, audit.BEHAVIOR_PROTOCOL,
                                    planned, audit.Checks(), role=role)


def test_restored_behavior_identity_cannot_be_rebound_to_new_campaign():
    record, planned = record_fixture()
    record["protocol_sha256"] = "a" * 64
    with pytest.raises(ValueError):
        audit.audit_campaign_record(record, "a" * 64, audit.BEHAVIOR_PROTOCOL,
                                    planned, audit.Checks(), role="training")


def continuation_fixture():
    historical = json.loads((ROOT / "reports/stage3c/batch-3.json").read_text())
    before, boundary = historical["output_state"], historical["output_boundary"]
    after = deepcopy(before)
    after.update(updates=5, optimizer_steps_executed=5, retired_batches=list(range(5)),
                 weights_sha256="a" * 64, heads_sha256="b" * 64, optimizer_sha256="c" * 64)
    after["policy_id"] = f"{after['schema']}:update-5:{after['weights_sha256']}"
    planned = [{"match_index": i} for i in range(16)]
    batch = {"input_state": deepcopy(before), "input_boundary": deepcopy(boundary),
        "batch_index": 4, "planned_matches": deepcopy(planned),
        "output_state": deepcopy(after), "output_boundary": {"path": "/new/batch-4.pt", "sha256": "d" * 64, **after},
        "optimizer_steps_executed_delta": 1, "completed_update_count_delta": 1,
        "final_in_memory_state": deepcopy(after), "update": {"after": deepcopy(after)},
        "boundary_roundtrip_verified": True, "sources_unchanged": True}
    return batch, before, boundary, planned


def test_absolute_update_four_continues_to_five_without_reset():
    batch, before, boundary, planned = continuation_fixture()
    after, saved = audit.audit_continuation_link(batch, before, boundary, planned, 4, audit.Checks())
    assert after["updates"] == 5 and saved["sha256"] == "d" * 64


@pytest.mark.parametrize("mutation", [
    lambda batch: batch["input_state"].__setitem__("optimizer_sha256", "0" * 64),
    lambda batch: batch["input_boundary"].__setitem__("sha256", "0" * 64),
    lambda batch: batch.__setitem__("batch_index", 5),
    lambda batch: batch.__setitem__("completed_update_count_delta", 2),
    lambda batch: batch.__setitem__("boundary_roundtrip_verified", False),
    lambda batch: batch["planned_matches"].reverse(),
])
def test_changed_optimizer_skipped_update_or_batch_substitution_rejected(mutation):
    batch, before, boundary, planned = continuation_fixture()
    mutation(batch)
    with pytest.raises(ValueError):
        audit.audit_continuation_link(batch, before, boundary, planned, 4, audit.Checks())


@pytest.mark.parametrize("field,value", [("optimizer_sha256", None),
                                       ("retired_batches", list(range(4))),
                                       ("rank_head_sha256", "0" * 64)])
def test_consistently_falsified_nonadvancing_or_frozen_state_rejected(field, value):
    batch, before, boundary, planned = continuation_fixture()
    value = before[field] if value is None else value
    for state in (batch["output_state"], batch["final_in_memory_state"], batch["update"]["after"]):
        state[field] = value
    with pytest.raises(ValueError):
        audit.audit_continuation_link(batch, before, boundary, planned, 4, audit.Checks())


def lifecycle_fixture():
    record, planned = record_fixture("evaluation")
    record.update(campaign_plan_sha256="c" * 64, campaign_block=0, campaign_arm="candidate", campaign_phase="evaluation", campaign_batch_index=2000)
    state = {"weights_sha256": "d" * 64, "optimizer_sha256": "e" * 64}
    evidence = {"json_path": "/example/match.json", "json_sha256": "f" * 64,
                "npz_path": "/example/match.npz", "npz_sha256": "1" * 64}
    start_path, start_sha = Path("/example/match-start.json"), "2" * 64
    start = {"schema": "stage3e-match-start-v1", "campaign_sha256": "a" * 64,
        "behavior_protocol_sha256": audit.BEHAVIOR_PROTOCOL, "plan_sha256": "c" * 64,
        "block": 0, "arm": "candidate", "phase": "evaluation", "collection_batch_index": 2000,
        "planned_match": planned, "state": state, "started_unix": 100.}
    finish = {"schema": "stage3e-match-finish-v1", "campaign_sha256": "a" * 64,
        "behavior_protocol_sha256": audit.BEHAVIOR_PROTOCOL, "start_path": str(start_path),
        "start_sha256": start_sha, "block": 0, "arm": "candidate", "phase": "evaluation", "match_index": 576,
        "completed": True, "interrupted": False, "failure": None,
        "evidence": deepcopy(evidence), "finished_unix": 102.}
    kwargs = dict(campaign="a" * 64, plan_sha="c" * 64, block=0, arm="candidate",
        planned=deepcopy(planned), state=deepcopy(state), record=record, evidence=evidence,
        start_path=start_path, start_sha=start_sha, checked=audit.Checks())
    return start, finish, kwargs


def test_durable_lifecycle_binds_fixed_admission_and_both_saved_artifacts():
    start, finish, kwargs = lifecycle_fixture()
    audit.audit_lifecycle(start, finish, **kwargs)


@pytest.mark.parametrize("target,field,value", [
    ("start", "campaign_sha256", "wrong"),
    ("start", "behavior_protocol_sha256", "a" * 64),
    ("start", "collection_batch_index", 0),
    ("start", "arm", "baseline"),
    ("finish", "start_sha256", "wrong"),
    ("finish", "match_index", 321),
    ("finish", "finished_unix", 99.),
    ("finish", "completed", False),
    ("finish", "evidence", {"json_path": "/wrong"}),
])
def test_durable_lifecycle_malformed_provenance_rejected(target, field, value):
    start, finish, kwargs = lifecycle_fixture()
    {"start": start, "finish": finish}[target][field] = value
    with pytest.raises(ValueError):
        audit.audit_lifecycle(start, finish, **kwargs)


def test_start_cannot_substitute_state_or_seed_plan():
    start, finish, kwargs = lifecycle_fixture()
    start["state"]["optimizer_sha256"] = "changed"
    with pytest.raises(ValueError):
        audit.audit_lifecycle(start, finish, **kwargs)
    start, finish, kwargs = lifecycle_fixture()
    start["planned_match"]["sampling_seed"] = 43
    with pytest.raises(ValueError):
        audit.audit_lifecycle(start, finish, **kwargs)




def closure_fixture(tmp_path, monkeypatch):
    data, reports = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    reports.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"data_root": str(data), "reports_root": str(reports),
                                       "artifact_bytes": 1500000000}))
    input_path = tmp_path / "verified-input.json"
    input_path.write_text('{"saved":"unchanged"}\n')
    active = {"id": 0, "name": "check:independent-audit", "token": "reserved-token",
              "reserved_seconds": 20., "started_unix": 100., "status": "active"}
    closed = {**active, "status": "completed", "failure": None,
              "elapsed_seconds": 1., "exceeded_reservation": False}
    ledger_path = data / "budget-ledger.json"
    ledger_path.write_text(json.dumps({"operations": [closed]}))
    prefix = {"schema": "stage3e-independent-audit-v1", "valid": True,
        "config_sha256": audit.digest(config_path), "auditor_sha256": audit.digest(Path(audit.__file__)),
        "budget": {"all_closed": False, "preparation_seconds": 16.},
        "active_audit_operation": active, "closed_ledger_prefix": [],
        "evidence_inputs": [{"path": str(input_path), "sha256": audit.digest(input_path)}],
        "trajectory_file_membership": [], "lifecycle_file_membership": [], "checkpoint_file_membership": [], "created_utc": "prior"}
    prefix_path = reports / "audit-prefix.json"
    prefix_path.write_text(json.dumps(prefix))
    # Budget arithmetic has separate full-audit coverage. These tests isolate
    # hash continuity and the exact one-operation closure transition.
    monkeypatch.setattr(audit, "audit_budget", lambda *args: ({"all_closed": True,
        "closed_operations": 1, "closed_charged_seconds": 1., "current_audit_lease": None,
        "current_audit_reservation_seconds": 0, "failed_operations": 0}, {}))
    args = (prefix_path, audit.digest(prefix_path), config_path, audit.digest(config_path))
    return args, input_path, ledger_path


def test_prefix_closure_retains_full_audit_without_numeric_replay(tmp_path, monkeypatch):
    args, _, _ = closure_fixture(tmp_path, monkeypatch)
    result = audit.finalize_prefix(*args)
    assert result["closure"]["full_match_numeric_arithmetic_repeated"] is False
    assert result["closure"]["evidence_hashes_revalidated"] == 1
    assert result["prefix_audit"] == {"path": str(args[0]), "sha256": args[1]}
    assert result["budget"]["all_closed"] is True
    assert result["budget"]["preparation_seconds"] == 16.
    assert result["active_audit_operation"] is None


def test_prefix_closure_rejects_changed_input(tmp_path, monkeypatch):
    args, input_path, _ = closure_fixture(tmp_path, monkeypatch)
    input_path.write_text('{"saved":"changed"}\n')
    with pytest.raises(ValueError, match="evidence input changed"):
        audit.finalize_prefix(*args)


def test_prefix_closure_rejects_extra_operation(tmp_path, monkeypatch):
    args, _, ledger_path = closure_fixture(tmp_path, monkeypatch)
    ledger = json.loads(ledger_path.read_text())
    ledger["operations"].append({**ledger["operations"][0], "id": 1, "name": "check:extra"})
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="one reserved audit operation"):
        audit.finalize_prefix(*args)


def test_incomplete_training_cannot_retain_terminal_utility():
    record, planned = record_fixture("training", False)
    record["learner_utility"] = .5
    with pytest.raises(ValueError):
        audit.audit_campaign_record(record, "a" * 64, audit.BEHAVIOR_PROTOCOL,
                                    planned, audit.Checks(), role="training")


def test_prefix_closure_rejects_added_checkpoint_even_if_original_inputs_unchanged(tmp_path, monkeypatch):
    args, _, _ = closure_fixture(tmp_path, monkeypatch)
    (tmp_path / "data" / "extra.pt").write_bytes(b"unrecorded checkpoint")
    with pytest.raises(ValueError, match="Evidence membership changed"):
        audit.finalize_prefix(*args)


def budget_fixture(tmp_path, name):
    data, reports = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    reports.mkdir()
    campaign = "a" * 64
    config = {"prepare_allowance_seconds": 20, "per_batch_allowance_seconds": 45,
              "per_block_allowance_seconds": 35, "test_allowance_seconds": 45,
              "memory_bytes": 1000, "artifact_bytes": 1000}
    operation = {"id": 0, "name": name, "status": "completed", "elapsed_seconds": 2.,
                 "reserved_seconds": 20., "exceeded_reservation": False, "failure": None}
    raw = {"schema": "package-a2-budget-v1", "identity": campaign, "limit_seconds": 1200,
           "operations": [operation]}
    saved = {**raw, "charged_seconds": 2., "remaining_seconds": 1198.}
    marker = {"name": name, "config_sha256": campaign, "status": "completed", "failure": None,
              "protected_reserve_seconds": 500 if name.startswith("batch:") else 100,
              "budget": saved, "watchdog": {"sampled_peak_tree_rss_bytes": 300,
                                            "artifact_bytes_peak": 100, "exit_code": 0}}
    filename = "attempt-4.json" if name.startswith("batch:") else "eval-attempt-0.json"
    (data / "budget-ledger.json").write_text(json.dumps(raw))
    (reports / "budget.json").write_text(json.dumps(saved))
    marker_path = reports / filename
    marker_path.write_text(json.dumps(marker))
    return (config, campaign, reports, data, audit.Checks(), False), marker_path


@pytest.mark.parametrize("name", ["batch:4", "eval:0"])
def test_phase_reserves_are_bound_to_saved_admission(tmp_path, name):
    args, _ = budget_fixture(tmp_path, name)
    budget, _ = audit.audit_budget(*args)
    assert budget["closed_charged_seconds"] == 2. and budget["all_closed"] is True


@pytest.mark.parametrize("name", ["batch:4", "eval:0"])
def test_phase_cannot_claim_only_shutdown_reserve(tmp_path, name):
    args, marker_path = budget_fixture(tmp_path, name)
    marker = json.loads(marker_path.read_text())
    marker["protected_reserve_seconds"] = 60
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="Operation marker differs"):
        audit.audit_budget(*args)


def update_fixture():
    batch, before, _, planned = continuation_fixture()
    matches = [{"completed": True, "optional_rows": 1, "forced_rows": 0,
                "learner_utility": .5, "stored_log_probability_sum": -1.} for _ in range(16)]
    update = {"schema": audit.SCHEMA, "before": before, "after": batch["output_state"],
        "batch_index": 4, "planned_matches": planned, "optimizer_steps": 1, "completed_matches": 16,
        "loss_denominator_matches": 16, "baseline": 0., "discount": 1., "opponent_rows_in_loss": 0,
        "optional_learner_rows": 16, "forced_learner_rows": 0, "match_utilities": [.5] * 16,
        "match_log_probability_sums": [-1.] * 16, "stored_behavior_match_log_probability_sums": [-1.] * 16,
        "stored_behavior_loss": .5, "behavior_reconstructed_loss": .5, "differentiable_loss": .5,
        "probability_replay_max_absolute_error": 0., "probability_replay_atol": 0., "probability_replay_rtol": 0.,
        "selected_log_probability_replay_max_error": 0., "selected_log_probability_replay_atol": 0.,
        "gradient_clip": 1., "gradients_finite": True, "parameters_finite": True,
        "base_unchanged": True, "rank_head_unchanged": True, "gradient_norm_before_clip": 1.,
        "parameter_max_absolute_change": .1, "parameter_l2_change": .1,
        "distribution_max_absolute_change": .01,
        "gradient_components": {"gate": {"finite": True, "l2_norm": 1., "nonzero_elements": 1,
                                         "hidden_layer_l2_norm": 0.},
                                "call_head": {"finite": True, "l2_norm": 0., "nonzero_elements": 0,
                                              "hidden_layer_l2_norm": 0.}},
        "parameter_component_l2_change": {"gate": .1, "call": 0.},
        "distribution_diagnostic_complete": True, "distribution_diagnostic_rows": 16,
        "numerical_no_op": False}
    return update, matches, before, planned


def test_finite_zero_conditional_call_gradient_is_coverage_not_integrity_failure():
    update, matches, before, planned = update_fixture()
    result = audit.audit_continuation_update(update, matches, before, planned, audit.Checks())
    assert result["gradient_nonzero"] == {"gate": True, "call_head": False}
    assert result["parameter_component_changed"] == {"gate": True, "call": False}


def test_finite_zero_parameter_or_distribution_change_is_reported_honestly():
    update, matches, before, planned = update_fixture()
    for field in ("gradient_norm_before_clip", "parameter_max_absolute_change", "parameter_l2_change",
                  "distribution_max_absolute_change"):
        update[field] = 0.
    for component in update["gradient_components"].values():
        component.update(l2_norm=0., nonzero_elements=0)
    update["parameter_component_l2_change"] = {"gate": 0., "call": 0.}
    update["after"]["heads_sha256"] = before["heads_sha256"]
    update["after"]["weights_sha256"] = before["weights_sha256"]
    update["after"]["policy_id"] = f"{before['schema']}:update-5:{before['weights_sha256']}"
    update["numerical_no_op"] = True
    result = audit.audit_continuation_update(update, matches, before, planned, audit.Checks())
    assert result["heads_changed"] is False and result["numerical_no_op"] is True
    assert not any(result["gradient_nonzero"].values())


@pytest.mark.parametrize("mutation", [
    lambda update: update.__setitem__("probability_replay_max_absolute_error", 1e-15),
    lambda update: update.__setitem__("selected_log_probability_replay_max_error", 1e-15),
    lambda update: update.__setitem__("differentiable_loss", .6),
    lambda update: update["gradient_components"]["call_head"].__setitem__("l2_norm", float("nan")),
    lambda update: update["gradient_components"]["call_head"].__setitem__("finite", False),
    lambda update: update["gradient_components"]["call_head"].__setitem__("nonzero_elements", -1),
    lambda update: update.__setitem__("parameter_l2_change", float("inf")),
    lambda update: update["parameter_component_l2_change"].__setitem__("call", -1.),
    lambda update: update.__setitem__("numerical_no_op", True),
])
def test_nonfinite_gradient_replay_error_or_inconsistent_noop_still_rejected(mutation):
    update, matches, before, planned = update_fixture()
    mutation(update)
    with pytest.raises(ValueError):
        audit.audit_continuation_update(update, matches, before, planned, audit.Checks())
