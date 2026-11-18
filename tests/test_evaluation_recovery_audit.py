"""Independent rejection checks for recovery identity and inference evidence."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("stage3d_independent_audit", ROOT / "scripts/audit_stage3d.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def record_fixture():
    planned = {"match_index": 320, "seed": 17, "sampling_seed": 42, "learner_seat": 0}
    record = {**planned, "campaign_sha256": "a" * 64, "protocol_sha256": audit.BEHAVIOR_PROTOCOL,
        "collector_schema": "stage3d-evaluation-lifecycle-v1", "collection_role": "evaluation",
        "training_eligible": False, "learner_rows": [{"training_eligible": False}],
        "completed": True, "interrupted": False, "failure": None, "learner_utility": .5}
    return record, planned


def test_campaign_preserves_behavior_identity_and_evaluation_role():
    record, planned = record_fixture()
    audit.audit_campaign_record(record, "a" * 64, audit.BEHAVIOR_PROTOCOL, planned, audit.Checks())


@pytest.mark.parametrize("mutation", [
    lambda value: value.__setitem__("campaign_sha256", "b" * 64),
    lambda value: value.__setitem__("protocol_sha256", "a" * 64),
    lambda value: value.__setitem__("collection_role", "training"),
    lambda value: value.__setitem__("training_eligible", True),
    lambda value: value["learner_rows"][0].__setitem__("training_eligible", True),
    lambda value: value.__setitem__("match_index", 0),
    lambda value: value.__setitem__("interrupted", True),
    lambda value: value.__setitem__("failure", {"type": "CollectionCancelled"}),
])
def test_wrong_campaign_or_lifecycle_provenance_rejected(mutation):
    record, planned = record_fixture()
    mutation(record)
    with pytest.raises(ValueError):
        audit.audit_campaign_record(record, "a" * 64, audit.BEHAVIOR_PROTOCOL, planned, audit.Checks())


def test_incomplete_record_cannot_claim_terminal_utility():
    record, planned = record_fixture()
    record.update(completed=False, interrupted=True, failure={"type": "CollectionCancelled"})
    with pytest.raises(ValueError):
        audit.audit_campaign_record(record, "a" * 64, audit.BEHAVIOR_PROTOCOL, planned, audit.Checks())
    record["learner_utility"] = None
    audit.audit_campaign_record(record, "a" * 64, audit.BEHAVIOR_PROTOCOL, planned, audit.Checks())


def test_campaign_and_behavior_are_distinct_identities():
    record, planned = record_fixture()
    with pytest.raises(ValueError):
        audit.audit_campaign_record(record, audit.BEHAVIOR_PROTOCOL, audit.BEHAVIOR_PROTOCOL,
                                    planned, audit.Checks())


def lifecycle_fixture():
    record, planned = record_fixture()
    record.update(campaign_plan_sha256="c" * 64, campaign_block=0, campaign_arm="candidate")
    state = {"weights_sha256": "d" * 64, "optimizer_sha256": "e" * 64}
    evidence = {"json_path": "/example/match.json", "json_sha256": "f" * 64,
                "npz_path": "/example/match.npz", "npz_sha256": "1" * 64}
    start_path, start_sha = Path("/example/match-start.json"), "2" * 64
    start = {"schema": "stage3d-match-start-v1", "campaign_sha256": "a" * 64,
        "behavior_protocol_sha256": audit.BEHAVIOR_PROTOCOL, "plan_sha256": "c" * 64,
        "block": 0, "arm": "candidate", "collection_batch_index": 1000,
        "planned_match": planned, "state": state, "started_unix": 100.}
    finish = {"schema": "stage3d-match-finish-v1", "campaign_sha256": "a" * 64,
        "behavior_protocol_sha256": audit.BEHAVIOR_PROTOCOL, "start_path": str(start_path),
        "start_sha256": start_sha, "block": 0, "arm": "candidate", "match_index": 320,
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


def test_explicit_membership_ignores_unrelated_additions_and_rejects_mutations(tmp_path):
    (tmp_path / "runtime.py").write_text("frozen-runtime\n")
    pins = {"runtime.py": audit.digest(tmp_path / "runtime.py")}
    audit.audit_file_membership(tmp_path, ["runtime.py"], pins, audit.Checks())
    (tmp_path / "new-analysis.py").write_text("unrelated\n")
    audit.audit_file_membership(tmp_path, ["runtime.py"], pins, audit.Checks())
    (tmp_path / "runtime.py").write_text("changed-runtime\n")
    with pytest.raises(ValueError):
        audit.audit_file_membership(tmp_path, ["runtime.py"], pins, audit.Checks())
    (tmp_path / "runtime.py").unlink()
    with pytest.raises(ValueError):
        audit.audit_file_membership(tmp_path, ["runtime.py"], pins, audit.Checks())


@pytest.mark.parametrize("members", [["runtime.py", "runtime.py"], [], ["../escape.py"]])
def test_invalid_explicit_membership_rejected(tmp_path, members):
    (tmp_path / "runtime.py").write_text("frozen-runtime\n")
    with pytest.raises(ValueError):
        audit.audit_file_membership(tmp_path, members, {"runtime.py": audit.digest(tmp_path / "runtime.py")},
                                    audit.Checks())


def endpoint_fixture():
    differences = [0.] * 32
    width = 6 * math.sqrt(math.log(40) / 64)
    matches = [{"arm": arm, "rank": 2, "completed": True}
               for _ in range(128) for arm in ("candidate", "baseline")]
    result = {"blocks": 32, "matches": 256, "block_differences": differences,
        "mean_utility_difference": 0., "halfwidth": width, "confidence_interval": [-width, width],
        "candidate_promoted": False, "candidate_mean_rank": 2., "baseline_mean_rank": 2.}
    return result, differences, matches


def test_fixed_block_level_interval():
    result, differences, matches = endpoint_fixture()
    mean, interval = audit.audit_fixed_endpoint(result, differences, matches, audit.Checks())
    assert mean == 0 and interval[1] > 1.4


@pytest.mark.parametrize("mutation", [
    lambda result: result.__setitem__("blocks", 8),
    lambda result: result.__setitem__("matches", 64),
    lambda result: result.__setitem__("halfwidth", .01),
    lambda result: result.__setitem__("confidence_interval", [-.01, .01]),
    lambda result: result.__setitem__("candidate_promoted", True),
    lambda result: result.__setitem__("candidate_mean_rank", 1.),
])
def test_invalid_endpoint_inference_rejected(mutation):
    result, differences, matches = endpoint_fixture()
    mutation(result)
    with pytest.raises(ValueError):
        audit.audit_fixed_endpoint(result, differences, matches, audit.Checks())


def test_incomplete_prefix_cannot_receive_fixed_endpoint_interval():
    result, differences, matches = endpoint_fixture()
    with pytest.raises(ValueError):
        audit.audit_fixed_endpoint(result, differences[:8], matches[:64], audit.Checks())


def closure_fixture(tmp_path, monkeypatch):
    data, reports = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    reports.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"data_root": str(data), "reports_root": str(reports),
                                       "artifact_bytes": 500000000}))
    input_path = tmp_path / "verified-input.json"
    input_path.write_text('{"saved":"unchanged"}\n')
    active = {"id": 0, "name": "check:independent-audit", "token": "reserved-token",
              "reserved_seconds": 20., "started_unix": 100., "status": "active"}
    closed = {**active, "status": "completed", "failure": None,
              "elapsed_seconds": 1., "exceeded_reservation": False}
    ledger_path = data / "budget-ledger.json"
    ledger_path.write_text(json.dumps({"operations": [closed]}))
    prefix = {"schema": "stage3d-independent-audit-v1", "valid": True,
        "config_sha256": audit.digest(config_path), "auditor_sha256": audit.digest(Path(audit.__file__)),
        "budget": {"all_closed": False, "preparation_seconds": 16.},
        "active_audit_operation": active, "closed_ledger_prefix": [],
        "evidence_inputs": [{"path": str(input_path), "sha256": audit.digest(input_path)}],
        "trajectory_file_membership": [], "lifecycle_file_membership": [], "created_utc": "prior"}
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
