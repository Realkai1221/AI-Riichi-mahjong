"""Independent stdlib saved-evidence audit of Stage 3E fixed-horizon continuation.

The campaign identity is separate from the retained Stage 3C behavior identity.
Raw NPY records, terminal accounting, RNG draws and actor provenance are checked
using the preserved independent Stage 3C/3A auditors. No learner is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from audit_stage3c import CALLS, audit_boundary, audit_match, audit_state, SCHEMA
from audit_stage3d import audit_file_membership, audit_fixed_endpoint
from stage3a_audit import (Checks, REFERENCE, audit_encoded, canonical,
                          checked_artifact, close, digest, finite, read)

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_PROTOCOL = "1fb067d7b76a02eb286c9ab560678a78f8623cc6f59a75cbfa64016fe2936776"


def audit_campaign_record(record, campaign, behavior, planned, checked, *, role="evaluation"):
    """Bind campaign identity without rebinding the restored behavior protocol."""
    checked.require(campaign != behavior and behavior == BEHAVIOR_PROTOCOL,
                    "Campaign identity substituted for behavior identity", "provenance")
    eligible = role == "training" and record["completed"] is True
    checked.require(role in ("training", "evaluation")
        and record["campaign_sha256"] == campaign and record["protocol_sha256"] == behavior
        and record["collector_schema"] == "stage3e-continuation-lifecycle-v1"
        and record["collection_role"] == role and record["training_eligible"] is eligible,
        "Continuation trajectory role or behavior protocol differs", "provenance")
    checked.require(all(record[key] == value for key, value in planned.items()),
                    "Continuation trajectory differs from fixed match plan", "provenance")
    checked.require(all(row["training_eligible"] is eligible for row in record["learner_rows"]),
                    "Learner row has incorrect completed-training eligibility", "provenance")
    if record["completed"]:
        checked.require(record["interrupted"] is False and record["failure"] is None,
                        "Completed record contains interruption or failure", "lifecycle")
    else:
        checked.require(record["learner_utility"] is None
            and (record["failure"] is not None or record.get("truncated") is True),
            "Incomplete record has terminal reward or hides failure/truncation", "lifecycle")


def audit_continuation_link(batch, before, boundary, planned, index, checked):
    """Reject optimizer reset, skipped/duplicate updates, or endpoint substitution."""
    checked.require(index in range(4, 32) and before["updates"] == index
        and before["optimizer_steps_executed"] == index
        and before["retired_batches"] == list(range(index)),
        "Continuation did not resume its exact absolute update boundary", "update")
    checked.require(batch["input_state"] == before and batch["input_boundary"] == boundary
        and batch["batch_index"] == index and batch["planned_matches"] == planned,
        "Continuation input model, optimizer, or planned batch changed", "update")
    after = batch["output_state"]
    checked.require(batch["optimizer_steps_executed_delta"] == batch["completed_update_count_delta"] == 1
        and batch["final_in_memory_state"] == after == batch["update"]["after"]
        and batch["boundary_roundtrip_verified"] is True and batch["sources_unchanged"] is True,
        "Continuation saved update boundary or reload mismatch", "checkpoint")
    audit_state(after, index + 1, BEHAVIOR_PROTOCOL, checked, before)
    checked.require(after["optimizer_sha256"] != before["optimizer_sha256"],
        "Continuation optimizer state did not advance", "update")
    return after, batch["output_boundary"]


def audit_continuation_update(update, matches, before, planned, checked):
    checked.require(len(matches) == len(planned) == 16 and all(match["completed"] for match in matches),
                    "Update lacks exactly sixteen completed planned matches", "update")
    checked.require(update["schema"] == SCHEMA and update["before"] == before
        and update["batch_index"] == before["updates"] and update["planned_matches"] == planned
        and update["optimizer_steps"] == 1 and update["completed_matches"] == 16
        and update["loss_denominator_matches"] == 16, "Update membership/normalization mismatch", "objective")
    checked.require(update["baseline"] == 0 and update["discount"] == 1
        and update["opponent_rows_in_loss"] == 0, "Complete-match learner-only objective changed", "objective")
    optional = sum(match["optional_rows"] for match in matches)
    checked.require(update["optional_learner_rows"] == optional
        and update["forced_learner_rows"] == sum(match["forced_rows"] for match in matches),
        "Objective row counts mismatch", "objective")
    utilities = [match["learner_utility"] for match in matches]
    sums = [match["stored_log_probability_sum"] for match in matches]
    checked.require(update["match_utilities"] == utilities, "Utility labels differ", "objective")
    for field in ("match_log_probability_sums", "stored_behavior_match_log_probability_sums"):
        checked.require(len(update[field]) == 16 and all(close(a, b) for a, b in zip(update[field], sums)),
                        "Behavior log-probability sums differ", "objective")
    loss = -math.fsum(utility * total for utility, total in zip(utilities, sums)) / 16
    for field in ("stored_behavior_loss", "behavior_reconstructed_loss", "differentiable_loss"):
        checked.require(close(update[field], loss), f"Objective reconstruction differs: {field}", "objective")
    for field in ("probability_replay_max_absolute_error", "probability_replay_atol", "probability_replay_rtol",
                  "selected_log_probability_replay_max_error", "selected_log_probability_replay_atol"):
        checked.require(update[field] == 0., "Replay is not exact", "replay")
    audit_state(update["after"], before["updates"] + 1, before["protocol_sha256"], checked, before)
    checked.require(update["gradient_clip"] == 1. and update["gradients_finite"] is True
        and update["parameters_finite"] is True and update["base_unchanged"] is True
        and update["rank_head_unchanged"] is True, "Update finite/frozen guards fail", "update")
    for field in ("gradient_norm_before_clip", "parameter_max_absolute_change", "parameter_l2_change",
                  "distribution_max_absolute_change"):
        checked.require(finite(update[field]) and update[field] >= 0, "Nonfinite or negative learning diagnostic", "update")
    for component in ("gate", "call_head"):
        values = update["gradient_components"][component]
        checked.require(values["finite"] is True and finite(values["l2_norm"]) and values["l2_norm"] >= 0
            and type(values["nonzero_elements"]) is int and values["nonzero_elements"] >= 0
            and (values["l2_norm"] == 0) is (values["nonzero_elements"] == 0)
            and finite(values["hidden_layer_l2_norm"]) and values["hidden_layer_l2_norm"] >= 0,
            f"Invalid gradient diagnostics for {component}", "update")
    checked.require(update["distribution_diagnostic_complete"] is True
        and update["distribution_diagnostic_rows"] == optional,
        "Post-update diagnostic is incomplete", "update")
    checked.require(update["numerical_no_op"] is (update["parameter_max_absolute_change"] == 0
        or update["distribution_max_absolute_change"] == 0), "No-op diagnostic contradicts recorded changes", "update")
    checked.require(all(finite(update["parameter_component_l2_change"][component])
        and update["parameter_component_l2_change"][component] >= 0 for component in ("gate", "call")),
        "Nonfinite or negative component parameter changes", "update")
    heads_changed = update["after"]["heads_sha256"] != before["heads_sha256"]
    checked.require(not (update["parameter_max_absolute_change"] > 0) or heads_changed,
                    "Positive parameter change retained the same head hash", "update")
    return {"batch_index": before["updates"], "optional_rows": optional,
        "loss": loss, "heads_changed": heads_changed, "base_unchanged": True,
        "numerical_no_op": update["numerical_no_op"],
        "gradient_nonzero": {component: update["gradient_components"][component]["nonzero_elements"] > 0
                             for component in ("gate", "call_head")},
        "parameter_component_changed": {component: update["parameter_component_l2_change"][component] > 0
                                        for component in ("gate", "call")},
        "coverage_interpretation": "Finite zero gradients or changes are coverage diagnostics, not integrity failures."}



def audit_budget(config, campaign, reports, data_root, checked, allow_active):
    raw = read(data_root / "budget-ledger.json")
    saved = read(reports / "budget.json")
    checked.require(raw["schema"] == "package-a2-budget-v1" and raw["identity"] == campaign
        and raw["limit_seconds"] == 1200, "Campaign budget identity mismatch", "budget")
    operations = raw["operations"]
    active = [operation for operation in operations if operation["status"] == "active"]
    checked.require(not active or (allow_active and len(active) == 1 and active[0] == operations[-1]
        and active[0]["name"].startswith("check:") and "audit" in active[0]["name"]),
        "Numerical work is not closed or the current explicit audit", "budget")
    closed = operations[:-1] if active else operations
    checked.require(saved["operations"] == closed and saved["identity"] == campaign,
                    "Saved budget is not the closed durable ledger prefix", "budget")
    charged, peaks, sizes, markers = 0., [], [], {}
    for index, operation in enumerate(closed):
        name = operation["name"]
        checked.require(operation["id"] == index and operation["status"] in ("completed", "failed")
            and finite(operation["elapsed_seconds"]) and operation["elapsed_seconds"] >= 0,
            "Malformed numerical operation", "budget")
        if name == "prepare":
            marker_path, maximum = reports / "prepare-attempt.json", config["prepare_allowance_seconds"]
        elif re.fullmatch(r"batch:(?:[4-9]|[12][0-9]|3[01])", name):
            marker_path, maximum = reports / f"attempt-{name.split(':')[1]}.json", config["per_batch_allowance_seconds"]
        elif re.fullmatch(r"eval:(?:[0-9]|[12][0-9]|3[01])", name):
            marker_path, maximum = reports / f"eval-attempt-{name.split(':')[1]}.json", config["per_block_allowance_seconds"]
        elif re.fullmatch(r"check:[a-zA-Z0-9_-]+", name):
            marker_path, maximum = reports / f"checks-{name.split(':')[1]}.json", config["test_allowance_seconds"]
        else:
            raise ValueError("Numerical operation outside fixed continuation scope")
        checked.require(name not in markers and 0 < operation["reserved_seconds"] <= maximum
            and operation["reserved_seconds"] <= 1200 - charged - 60 + 1e-8,
            "Repeated operation or reservation exceeded cap/shutdown reserve", "budget")
        if name.startswith("batch:"):
            checked.require(operation["reserved_seconds"] <= 1200 - charged - 60 - 400 - 40 + 1e-8,
                            "Training admission consumed protected evaluation/audit reserve", "budget")
        expected_reserve = 500 if name.startswith("batch:") else 100 if name.startswith("eval:") else 60
        checked.require(operation["reserved_seconds"] <= 1200 - charged - expected_reserve + 1e-8,
                        "Operation consumed a protected phase reserve", "budget")
        charged += operation["elapsed_seconds"]
        checked.require(operation["exceeded_reservation"] is False
            and operation["elapsed_seconds"] <= operation["reserved_seconds"],
            "Numerical operation exceeded reservation", "budget")
        marker = read(marker_path)
        checked.require(marker["protected_reserve_seconds"] == expected_reserve
            and marker["name"] == name and marker["config_sha256"] == campaign
            and marker["status"] == operation["status"] and marker["failure"] == operation["failure"],
            "Operation marker differs from durable ledger", "budget")
        checked.require(marker["budget"]["operations"] == closed[:index + 1]
            and close(marker["budget"]["charged_seconds"], charged),
            "Operation budget snapshot differs from ledger prefix", "budget")
        watchdog = marker["watchdog"]
        checked.require(watchdog["sampled_peak_tree_rss_bytes"] < config["memory_bytes"]
            and watchdog["artifact_bytes_peak"] < config["artifact_bytes"],
            "Sampled memory or artifact cap exceeded", "budget")
        if operation["status"] == "completed":
            checked.require(watchdog["exit_code"] == 0 and operation["failure"] is None,
                            "Completed operation has failure evidence", "budget")
        else:
            checked.require(operation["failure"] is not None, "Failed operation hides reason", "budget")
        peaks.append(watchdog["sampled_peak_tree_rss_bytes"])
        sizes.append(watchdog["artifact_bytes_peak"])
        markers[name] = marker
    checked.require(charged <= 1200 and close(saved["charged_seconds"], charged)
        and close(saved["remaining_seconds"], 1200 - charged), "Cumulative budget arithmetic differs", "budget")
    if active:
        checked.require(0 < active[0]["reserved_seconds"] <= 1200 - charged - 60,
                        "Current audit reservation consumes shutdown reserve", "budget")
    return {"limit_seconds": 1200, "closed_charged_seconds": charged,
        "closed_operations": len(closed), "all_closed": not active,
        "current_audit_lease": active[0]["name"] if active else None,
        "current_audit_reservation_seconds": active[0]["reserved_seconds"] if active else 0,
        "failed_operations": sum(operation["status"] == "failed" for operation in closed),
        "sampled_peak_tree_rss_bytes": max(peaks, default=0), "artifact_bytes_peak": max(sizes, default=0)}, markers


def audit_lifecycle(start, finish, *, campaign, plan_sha, block, arm, planned,
                    state, record, evidence, start_path, start_sha, checked,
                    role="evaluation", collection_batch=None):
    collection_batch = 2000 + block if collection_batch is None else collection_batch
    checked.require(start["schema"] == "stage3e-match-start-v1"
        and start["campaign_sha256"] == campaign and start["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL
        and start["plan_sha256"] == plan_sha and start["block"] == block and start["arm"] == arm
        and start["phase"] == role and start["collection_batch_index"] == collection_batch
        and start["planned_match"] == planned and start["state"] == state
        and finite(start["started_unix"]), "Durable start differs from fixed admission", "lifecycle")
    checked.require(finish["schema"] == "stage3e-match-finish-v1"
        and finish["campaign_sha256"] == campaign and finish["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL
        and finish["start_path"] == str(start_path) and finish["start_sha256"] == start_sha
        and finish["phase"] == role and finish["block"] == block and finish["arm"] == arm
        and finish["match_index"] == planned["match_index"]
        and finish["completed"] is record["completed"] and finish["interrupted"] is record["interrupted"]
        and finish["failure"] == record["failure"] and finite(finish["finished_unix"])
        and finish["finished_unix"] >= start["started_unix"],
        "Durable completion does not correspond to admitted record", "lifecycle")
    checked.require(finish["evidence"] == {key: evidence[key] for key in
        ("json_path", "json_sha256", "npz_path", "npz_sha256")},
        "Durable completion does not bind both persisted artifacts", "lifecycle")
    checked.require(record["campaign_plan_sha256"] == plan_sha and record["campaign_block"] == block
        and record["campaign_arm"] == arm and record["campaign_phase"] == role
        and record["campaign_batch_index"] == collection_batch,
        "Raw record differs from campaign lifecycle identity", "provenance")


def fixed_seed_plan():
    def seed(kind, index):
        return int.from_bytes(hashlib.sha256(f"riichi-ai-stage3e-20261021:{kind}:{index}".encode()).digest()[:8], "big")
    batches = [{"batch_index": index, "matches": [
        {"match_index": 576 + (index - 4) * 16 + offset,
         "seed": seed("training-wall", (index - 4) * 16 + offset),
         "sampling_seed": seed("training-sampling", (index - 4) * 16 + offset),
         "learner_seat": offset % 4} for offset in range(16)]} for index in range(4, 32)]
    evaluation = [{"block": block, "wall_seed": seed("evaluation-wall", block), "pairs": [
        {"learner_seat": seat, "sampling_seed": seed("evaluation-sampling", block * 4 + seat),
         "candidate_match_index": 1024 + block * 8 + seat * 2,
         "baseline_match_index": 1025 + block * 8 + seat * 2} for seat in range(4)]} for block in range(32)]
    return batches, evaluation


def seeds_of(config):
    return ([match[key] for batch in config.get("batches", []) for match in batch["matches"]
             for key in ("seed", "sampling_seed")]
        + [block["wall_seed"] for block in config["evaluation"]]
        + [pair["sampling_seed"] for block in config["evaluation"] for pair in block["pairs"]])


def audit(config_path, expected_sha, *, allow_active=False):
    checked = Checks()
    config_path = Path(config_path).resolve()
    campaign = digest(config_path)
    checked.require(campaign == expected_sha, "Campaign differs from externally pinned SHA", "identity")
    config = read(config_path)
    exact = {"schema": "stage3e-continuation-config-v1", "limit_seconds": 1200,
        "preparation_limit_seconds": 60, "memory_bytes": 32 * 1024**3, "artifact_bytes": 1500000000,
        "reserve_seconds": 60, "protected_evaluation_seconds": 400, "protected_audit_seconds": 40,
        "per_batch_allowance_seconds": 45, "per_block_allowance_seconds": 35,
        "prepare_allowance_seconds": 20, "test_allowance_seconds": 45, "device": "cpu", "torch_threads": 1,
        "starting_updates": 4, "maximum_updates": 32, "maximum_batches": 28, "matches_per_batch": 16,
        "training_matches": 448, "evaluation_matches": 256, "evaluation_blocks": 32, "maximum_matches": 704,
        "learning_rate": .0001, "weight_decay": 0., "gradient_clip": 1., "microbatch_size": 1,
        "replay_atol": 0., "replay_rtol": 0., "replay_log_atol": 0., "max_steps": 100000,
        "behavior_protocol_sha256": BEHAVIOR_PROTOCOL, "reference_sha256": REFERENCE}
    checked.require(all(config.get(key) == value for key, value in exact.items()),
                    "Authorized fixed continuation scope or limits changed", "scope")
    constraints = {"book_training": False, "test_split_loading": False, "external_acquisition": False,
        "paid_compute": False, "policy_promotion": False, "failed_work_replacement": False,
        "old_evaluation_pooling": False, "opponent_actions_eligible": False, "baseline": "zero",
        "entropy_bonus": 0., "reward_shaping": False, "replay_reuse": False, "base_frozen": True,
        "rank_head_frozen": True, "best_checkpoint_selection": False}
    checked.require(config["constraints"] == constraints, "Fixed continuation constraints changed", "scope")
    batches, evaluation = fixed_seed_plan()
    checked.require(config["batches"] == batches and config["evaluation"] == evaluation,
                    "Fixed fresh seed/seat/arm plan changed", "scope")
    prior_manifest_path = ROOT / "reports/stage3d/manifest.json"
    checked.require(digest(prior_manifest_path) == "30234190420f681c5f7ec857288da04df102e9dc030749aa6c950cd8b4f89058",
                    "Historical Stage 3D inventory identity differs", "identity")
    prior_pins = {entry["path"]: entry["sha256"] for entry in read(prior_manifest_path)["files"]
                  if entry["scope"] == "project"}
    older_manifest_path = ROOT / "reports/stage3c/manifest.json"
    checked.require(digest(older_manifest_path) == "c4f7090dbce4067bb93bb21ad5751dbc830c84e2708e0bee8503e8cb0ac9c8ce",
                    "Historical Stage 3C inventory identity differs", "identity")
    older_pins = {entry["path"]: entry["sha256"] for entry in read(older_manifest_path)["files"]
                  if entry["scope"] == "project"}
    prior_paths = ("configs/stage3c.json", "configs/stage3d.json", "reports/stage3c/plan.json",
                   "reports/stage3c/batch-3.json", "reports/stage3d/plan.json")
    for path in prior_paths:
        pins = older_pins if "stage3c" in path else prior_pins
        checked.require(digest(ROOT / path) == pins[path], "Historical protocol or endpoint changed", "identity")
    old_config = read(ROOT / "configs/stage3c.json")
    old_seeds = set(seeds_of(old_config)) | set(seeds_of(read(ROOT / "configs/stage3d.json")))
    fresh = seeds_of(config)
    checked.require(len(fresh) == len(set(fresh)) == 1056 and not set(fresh) & old_seeds,
                    "Continuation seeds collide or reuse historical campaign seeds", "scope")
    data_root, reports = (ROOT / config["data_root"]).resolve(), (ROOT / config["reports_root"]).resolve()
    reference = (ROOT / config["reference_checkpoint"]).resolve()
    checked.require(digest(reference) == REFERENCE, "Original opponent checkpoint changed", "identity")
    snapshot_path = reports / "previous-stage3d-snapshot.json"
    snapshot = read(snapshot_path)
    checked.require(snapshot["valid"] is True and snapshot["verified_files"] == 1216
        and snapshot["snapshot_commit"] == "7d3380f2d9c595b4da6d99871ede8ccd92380608"
        and snapshot["manifest_sha256"] == digest(prior_manifest_path),
        "Historical Stage 3D snapshot prerequisite differs", "identity")
    prior_plan = read(ROOT / "reports/stage3c/plan.json")
    prior_final = read(ROOT / "reports/stage3c/batch-3.json")
    previous_plan = read(ROOT / "reports/stage3d/plan.json")
    initial = prior_final["output_state"]
    initial_boundary = {"path": prior_final["output_boundary"]["path"],
        "sha256": "4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967", "updates": 4}
    checked.require(config["initial_boundary"] == initial_boundary
        and prior_final["output_boundary"]["sha256"] == initial_boundary["sha256"],
        "Continuation starting endpoint substituted", "checkpoint")
    audit_state(initial, 4, BEHAVIOR_PROTOCOL, checked, prior_plan["initial_state"])
    audit_boundary(prior_final["output_boundary"], initial, Path(old_config["data_root"]).resolve(), checked)
    plan_path = reports / "plan.json"
    plan, plan_sha = read(plan_path), digest(plan_path)
    checked.require(plan["schema"] == "stage3e-plan-v1" and plan["config_sha256"] == campaign
        and plan["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL and plan["batches"] == batches
        and plan["evaluation"] == evaluation and plan["constraints"] == constraints
        and plan["initial_boundary"] == initial_boundary and plan["initial_state"] == initial
        and plan["frozen_before_formal_collection"] is True
        and plan["seed_audit"] == {"fresh_seed_count": 1056, "stage3c_stage3d_disjoint": True},
        "Frozen continuation plan, starting model or optimizer differs", "identity")
    expected_members = sorted(set(previous_plan["implementation"]) | {"scripts/stage3e.py",
        "src/riichi_ai/stage3e.py", "tests/test_stage3e_controller.py", "tests/test_stage3e_runtime.py"})
    checked.require(config["source_paths"] == expected_members,
                    "Continuation membership omitted historical runtime or new checks", "scope")
    implementation = audit_file_membership(ROOT, config["source_paths"], plan["implementation"], checked)
    checked.require(all(implementation[key] == value for key, value in previous_plan["implementation"].items()),
                    "Historical policy/runtime implementation changed", "identity")
    checked.require(plan["versions"] == {"torch": "2.14.1", "numpy": "2.5.4", "riichienv": "0.4.10",
        "mahjong": "2.0.0", "psutil": "7.2.2"}, "Recorded versions differ from pinned stack", "identity")
    opponent = plan["opponent"]
    checked.require(opponent == prior_plan["opponent"] and opponent["checkpoint_sha256"] == REFERENCE
        and opponent["weights_sha256"] == initial["base_weights_sha256"]
        and opponent["decision_rule"] == "stable_semantic_argmax", "Original frozen opponents changed", "identity")
    budget, markers = audit_budget(config, campaign, reports, data_root, checked, allow_active)
    checked.require(markers["prepare"]["status"] == "completed"
        and markers["prepare"]["implementation"] == implementation, "Preparation did not pass", "scope")
    preflight_path, xml_path = reports / "checks-release-preflight.json", reports / "release-preflight.xml"
    preflight, xml = read(preflight_path), ET.parse(xml_path).getroot()
    cases = list(xml.iter("testcase"))
    classes = {case.get("classname", "").split(".")[-1] for case in cases}
    checked.require(preflight["status"] == "completed" and preflight["implementation"] == implementation
        and preflight["finished_unix"] <= markers["prepare"]["started_unix"]
        and {"test_stage3e_controller", "test_stage3e_runtime", "test_stage3c_postcall"} <= classes
        and not any(list(xml.iter(kind)) for kind in ("error", "failure", "skipped")),
        "Required continuation/runtime/postcall checks did not pass before prepare", "scope")
    checked.require(plan["preflight_receipt"] == {"receipt_sha256": digest(preflight_path),
        "junit_sha256": digest(xml_path), "passing_cases": len(cases)},
        "Frozen preflight receipt differs", "identity")
    batch_names = [name for name in markers if name.startswith("batch:")]
    eval_names = [name for name in markers if name.startswith("eval:")]
    checked.require(0 < len(batch_names) <= 28 and batch_names == [f"batch:{i}" for i in range(4, 4 + len(batch_names))],
                    "Training attempt order is not the fixed absolute prefix", "scope")
    failed_batches = [name for name in batch_names if markers[name]["status"] == "failed"]
    checked.require(not failed_batches or (failed_batches == batch_names[-1:] and not eval_names),
                    "Campaign continued or retried after training failure", "scope")
    checked.require(len(eval_names) <= 32 and eval_names == [f"eval:{i}" for i in range(len(eval_names))]
        and (not eval_names or (len(batch_names) == 28 and not failed_batches)),
        "Evaluation preceded fixed update-32 endpoint or skipped/retried a block", "scope")
    failed_eval = [name for name in eval_names if markers[name]["status"] == "failed"]
    checked.require(not failed_eval or failed_eval == eval_names[-1:], "Evaluation continued after failure", "scope")
    formal = [name for name in markers if name.startswith(("batch:", "eval:"))]
    checked.require(formal == batch_names + eval_names, "Training/evaluation operations interleaved", "scope")
    operations = read(data_root / "budget-ledger.json")["operations"]
    first_batch = next(operation["id"] for operation in operations if operation["name"] == "batch:4")
    prep_seconds = sum(operation["elapsed_seconds"] for operation in operations[:first_batch])
    checked.require(prep_seconds <= 60, "Preparation cap exceeded before formal play", "budget")
    all_raw_files, lifecycle_files, partials, checkpoint_files = [], [], [], []
    last_finish_time = None

    def saved_match(evidence, planned, state, marker, *, role, batch_index, arm, block=None):
        nonlocal last_finish_time
        checked.require(evidence["arm"] == arm and evidence["learner_seat"] == planned["learner_seat"],
                        "Fixed arm/seat collection order differs", "provenance")
        record_path = checked_artifact(evidence["json_path"], evidence["json_sha256"], data_root, checked)
        arrays = checked_artifact(evidence["npz_path"], evidence["npz_sha256"], data_root, checked)
        record = read(record_path)
        audit_campaign_record(record, campaign, BEHAVIOR_PROTOCOL, planned, checked, role=role)
        start_path = checked_artifact(evidence["lifecycle_start_path"], evidence["lifecycle_start_sha256"], data_root, checked)
        finish_path = checked_artifact(evidence["lifecycle_finish_path"], evidence["lifecycle_finish_sha256"], data_root, checked)
        start, finish = read(start_path), read(finish_path)
        audit_lifecycle(start, finish, campaign=campaign, plan_sha=plan_sha, block=block,
            arm=arm, planned=planned, state=state, record=record, evidence=evidence,
            start_path=start_path, start_sha=evidence["lifecycle_start_sha256"], checked=checked,
            role=role, collection_batch=batch_index)
        checked.require(marker["started_unix"] <= start["started_unix"] <= finish["finished_unix"] <= marker["finished_unix"],
                        "Match lifecycle lies outside operation", "lifecycle")
        checked.require(last_finish_time is None or start["started_unix"] >= last_finish_time,
                        "Match admitted before preceding artifact completion", "lifecycle")
        last_finish_time = finish["finished_unix"]
        lifecycle_files.extend((str(start_path), str(finish_path)))
        all_raw_files.extend((str(record_path), str(arrays)))
        if not record["completed"]:
            audit_encoded(arrays, record["learner_rows"], checked)
            partials.append({"role": role, "batch_index": batch_index, "block": block,
                "match_index": planned["match_index"], "interrupted": record["interrupted"], "failure": record["failure"],
                "training_eligible": False, "included_in_endpoint": False,
                "encoded_archive_checked": True, "full_terminal_audit_applicable": False})
            return None
        return audit_match(evidence, planned, state, opponent, BEHAVIOR_PROTOCOL, data_root, checked,
                           collection_role=role, collection_batch=batch_index)

    state, boundary, training, update_summaries = initial, initial_boundary, [], []
    failed_training_progress = []
    for offset, name in enumerate(batch_names):
        index, planned = 4 + offset, batches[offset]["matches"]
        batch, marker = read(reports / f"batch-{index}.json"), markers[name]
        complete = batch["complete"]
        checked.require(type(complete) is bool and (marker["status"] == "completed") is complete
            and marker["implementation"] == implementation and batch["schema"] == "stage3e-batch-v1"
            and batch["config_sha256"] == campaign and batch["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL
            and batch["plan_sha256"] == plan_sha and batch["implementation"] == implementation
            and batch["planned_matches"] == planned and batch["batch_index"] == index
            and batch["input_state"] == state and batch["input_boundary"] == boundary
            and batch["opponent"] == opponent and 0 <= len(batch["matches"]) <= 16,
            "Training chain identity or completion mismatch", "update")
        matches = []
        for position, evidence in enumerate(batch["matches"]):
            match = saved_match(evidence, planned[position], state, marker,
                role="training", batch_index=index, arm="learner")
            if match is None:
                checked.require(not complete and position == len(batch["matches"]) - 1,
                                "Partial training trajectory was followed or accepted", "update")
            else:
                matches.append(match)
        training.extend(matches)
        if complete:
            checked.require(len(matches) == 16 and batch["termination_requested"] is False,
                            "Complete batch lacks sixteen full matches or was interrupted", "update")
            update_summaries.append(audit_continuation_update(batch["update"], matches, state, planned, checked))
            state, boundary = audit_continuation_link(batch, state, boundary, planned, index, checked)
            audit_boundary(boundary, state, data_root, checked)
            checkpoint_files.append(str(Path(boundary["path"]).resolve()))
        else:
            checked.require(index == 3 + len(batch_names) and marker["failure"] is not None,
                            "Incomplete training was not final failed attempt", "update")
            # An interrupted update must retain its actual final state/delta;
            # no such boundary is eligible for the fixed-endpoint evaluation.
            final_state = batch.get("final_in_memory_state")
            checked.require(final_state is not None
                and type(batch["optimizer_steps_executed_delta"]) is int
                and type(batch["completed_update_count_delta"]) is int
                and batch["optimizer_steps_executed_delta"] in (0, 1)
                and batch["completed_update_count_delta"] in (0, 1)
                and final_state["optimizer_steps_executed"] - state["optimizer_steps_executed"]
                    == batch["optimizer_steps_executed_delta"]
                and final_state["updates"] - state["updates"] == batch["completed_update_count_delta"],
                "Failed update hid in-memory optimizer/update progress", "update")
            failed_training_progress.append({"batch_index": index,
                "optimizer_steps_executed_delta": batch["optimizer_steps_executed_delta"],
                "completed_update_count_delta": batch["completed_update_count_delta"],
                "final_in_memory_state": final_state, "saved_boundary": batch.get("output_boundary"),
                "eligible_for_fixed_endpoint": False})
            if batch.get("output_boundary") is not None:
                saved_boundary = batch["output_boundary"]
                checked_artifact(saved_boundary["path"], saved_boundary["sha256"], data_root, checked)
                checkpoint_files.append(str(Path(saved_boundary["path"]).resolve()))
    training_complete = len(update_summaries) == 28 and not failed_batches
    states = {"candidate": state, "baseline": initial}
    boundaries = {"candidate": boundary, "baseline": initial_boundary}
    evaluated, differences = [], []
    for index, planned_block in enumerate(evaluation[:len(eval_names)]):
        block, marker = read(reports / f"eval-{index}.json"), markers[f"eval:{index}"]
        complete = block["complete"]
        checked.require(type(complete) is bool and (marker["status"] == "completed") is complete
            and marker["implementation"] == implementation and block["schema"] == "stage3e-evaluation-block-v1"
            and block["block"] == index and block["config_sha256"] == campaign
            and block["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL and block["plan_sha256"] == plan_sha
            and block["implementation"] == implementation and block["planned"] == planned_block
            and block["boundaries"] == boundaries and block["states_before"] == states
            and 0 <= len(block["matches"]) <= 8, "Evaluation endpoint or block identity differs", "evaluation")
        if complete:
            checked.require(len(block["matches"]) == 8 and block["states_after"] == states
                and block["termination_requested"] is False, "Completed evaluation changed states or stopped", "evaluation")
        else:
            checked.require(index == len(eval_names) - 1 and marker["failure"] is not None
                and block["mean_utility_difference"] is None,
                "Incomplete block asserted endpoint or work continued", "evaluation")
        outcomes, pairs, block_matches = [], [], []
        for pair_index, pair in enumerate(planned_block["pairs"]):
            arms = {}
            for arm_index, arm in enumerate(("candidate", "baseline")):
                position = pair_index * 2 + arm_index
                if position >= len(block["matches"]):
                    continue
                planned = {"match_index": pair[f"{arm}_match_index"], "seed": planned_block["wall_seed"],
                    "sampling_seed": pair["sampling_seed"], "learner_seat": pair["learner_seat"]}
                match = saved_match(block["matches"][position], planned, states[arm], marker,
                    role="evaluation", batch_index=2000 + index, arm=arm, block=index)
                if match is None:
                    checked.require(not complete and position == len(block["matches"]) - 1,
                                    "Incomplete evaluation was followed or counted", "lifecycle")
                    continue
                arms[arm] = match
                evaluated.append({"arm": arm, **match})
                block_matches.append(match)
                outcomes.append({"arm": arm, "learner_seat": pair["learner_seat"],
                    "utility": match["learner_utility"], "rank": match["rank"], "scores": match["scores"]})
            if len(arms) == 2:
                a, b = arms["candidate"], arms["baseline"]
                common = min(len(a["ordinal_wall_fingerprints"]), len(b["ordinal_wall_fingerprints"]))
                checked.require(common > 0 and a["ordinal_wall_fingerprints"][:common] == b["ordinal_wall_fingerprints"][:common],
                                "Paired physical wall prefixes differ", "evaluation")
                pairs.append({"learner_seat": pair["learner_seat"], "common_ordinal_walls": common,
                    "utility_difference": a["learner_utility"] - b["learner_utility"]})
        for match in block_matches[1:]:
            other = block_matches[0]
            common = min(len(match["ordinal_wall_fingerprints"]), len(other["ordinal_wall_fingerprints"]))
            checked.require(common > 0 and match["ordinal_wall_fingerprints"][:common] == other["ordinal_wall_fingerprints"][:common],
                            "Rotated matches sharing a seed have different physical wall prefixes", "evaluation")
        if complete:
            mean = math.fsum(pair["utility_difference"] for pair in pairs) / 4
            checked.require(block["outcomes"] == outcomes and block["pairs"] == pairs
                and len(pairs) == 4 and block["mean_utility_difference"] == mean,
                "Fixed paired-block arithmetic differs", "evaluation")
            differences.append(mean)
        else:
            checked.require(block["outcomes"] == outcomes[:len(block["outcomes"])]
                and len(block["outcomes"]) <= len(outcomes)
                and block["pairs"] == pairs[:len(block["pairs"])] and len(block["pairs"]) <= len(pairs),
                "Incomplete evaluation summary is not a persisted prefix", "evaluation")
    experiment_complete = training_complete and len(differences) == 32 and not failed_eval
    mean = interval = None
    if experiment_complete:
        result = read(reports / "evaluation.json")
        checked.require(result["schema"] == "stage3e-evaluation-v1" and result["old_evaluation_pooled"] is False
            and result["training_updates"] == 28 and result["evaluation_optimizer_updates"] == 0,
            "Fixed endpoint pooled old data or updated in evaluation", "evaluation")
        mean, interval = audit_fixed_endpoint(result, differences, evaluated, checked)
    else:
        checked.require(not (reports / "evaluation.json").exists(), "Incomplete campaign has a fixed endpoint", "evaluation")
    checked.require(len(all_raw_files) == len(set(all_raw_files)) and len(lifecycle_files) == len(set(lifecycle_files))
        and len(checkpoint_files) == len(set(checkpoint_files)), "Repeated raw/checkpoint/lifecycle evidence", "identity")
    actual_raw = {str(path.resolve()) for folder in ("evaluation", "trajectories")
                  for path in (data_root / folder).rglob("*") if path.is_file()}
    actual_lifecycle = {str(path.resolve()) for path in (data_root / "lifecycle").rglob("*") if path.is_file()}
    actual_checkpoints = {str(path.resolve()) for path in data_root.rglob("*") if path.is_file() and path.suffix in (".pt", ".pth")}
    checked.require(actual_raw == set(all_raw_files), "Unreported raw artifacts", "scope")
    checked.require(actual_checkpoints == set(checkpoint_files), "Unreported or missing checkpoints", "scope")
    # Complete campaigns have exactly one start/finish pair for each saved match.
    # A process death before persistence must retain one explicit unmatched start.
    extra_starts = actual_lifecycle - set(lifecycle_files)
    interrupted_start = []
    if extra_starts:
        checked.require(not experiment_complete and len(extra_starts) == 1 and (failed_batches or failed_eval),
                        "Unreported lifecycle evidence outside final failed attempt", "lifecycle")
        path = Path(next(iter(extra_starts)))
        start = read(path)
        last_name = formal[-1]
        if last_name.startswith("batch:"):
            index = int(last_name.split(":")[1]); final_report = read(reports / f"batch-{index}.json")
            planned = batches[index - 4]["matches"][len(final_report["matches"])]
            role, arm, block_index, collection_batch, start_state = "training", "learner", None, index, final_report["input_state"]
        else:
            block_index = int(last_name.split(":")[1]); final_report = read(reports / f"eval-{block_index}.json")
            offset = len(final_report["matches"])
            pair, arm = evaluation[block_index]["pairs"][offset // 2], ("candidate", "baseline")[offset % 2]
            planned = {"match_index": pair[f"{arm}_match_index"], "seed": evaluation[block_index]["wall_seed"],
                "sampling_seed": pair["sampling_seed"], "learner_seat": pair["learner_seat"]}
            role, collection_batch, start_state = "evaluation", 2000 + block_index, states[arm]
        checked.require(path.name == f"match-{planned['match_index']}-start.json" and start["schema"] == "stage3e-match-start-v1"
            and start["campaign_sha256"] == campaign and start["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL
            and start["plan_sha256"] == plan_sha and start["planned_match"] == planned and start["state"] == start_state
            and start["phase"] == role and start["block"] == block_index and start["arm"] == arm
            and start["collection_batch_index"] == collection_batch,
            "Unfinished start is not the next fixed admission", "lifecycle")
        interrupted_start.append({"path": str(path), "sha256": digest(path), "match_index": planned["match_index"],
            "classification": "admitted without durable trajectory completion; never counted or replaced"})
    actual_bytes = sum(path.stat().st_size for directory in (data_root, reports, ROOT / "Books/stage3e")
                       for path in directory.rglob("*") if path.is_file())
    checked.require(actual_bytes < config["artifact_bytes"], "Current artifacts exceed cap", "budget")
    call_keys = ("requested_calls", "declared_calls", "completed_calls", "robbed_kans")
    totals = {arm: {key: {kind: sum(match[key][kind] for match in evaluated if match["arm"] == arm)
        for kind in CALLS} for key in call_keys} for arm in ("candidate", "baseline")}
    training_calls = {key: {kind: sum(match[key][kind] for match in training) for kind in CALLS} for key in call_keys}
    helper_paths = [ROOT / f"scripts/{name}" for name in ("audit_stage3c.py", "audit_stage3d.py", "stage3a_audit.py")]
    input_paths = {config_path, plan_path, preflight_path, xml_path, prior_manifest_path, older_manifest_path, snapshot_path,
        reference, Path(initial_boundary["path"]), Path(__file__).resolve(),
        ROOT / "tests/test_continuation_horizon_audit.py", *helper_paths}
    input_paths.update(ROOT / path for path in prior_paths)
    input_paths.update(ROOT / path for path in implementation)
    input_paths.update(Path(path) for path in all_raw_files + lifecycle_files + checkpoint_files)
    input_paths.update(Path(path) for path in extra_starts)
    input_paths.update(reports / f"batch-{index}.json" for index in range(4, 4 + len(batch_names)))
    input_paths.update(reports / f"eval-{index}.json" for index in range(len(eval_names)))
    for name in markers:
        if name == "prepare": input_paths.add(reports / "prepare-attempt.json")
        elif name.startswith("batch:"): input_paths.add(reports / f"attempt-{name.split(':')[1]}.json")
        elif name.startswith("eval:"): input_paths.add(reports / f"eval-attempt-{name.split(':')[1]}.json")
        else: input_paths.add(reports / f"checks-{name.split(':')[1]}.json")
    if experiment_complete: input_paths.add(reports / "evaluation.json")
    raw_ledger = read(data_root / "budget-ledger.json")
    closed_ledger = [operation for operation in raw_ledger["operations"] if operation["status"] != "active"]
    active_ledger = [operation for operation in raw_ledger["operations"] if operation["status"] == "active"]
    return {"schema": "stage3e-independent-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True, "experiment_complete": experiment_complete, "engineering_evidence_pass": experiment_complete,
        "config_sha256": campaign, "behavior_protocol_sha256": BEHAVIOR_PROTOCOL, "plan_sha256": plan_sha,
        "implementation": implementation, "auditor_sha256": digest(Path(__file__)),
        "evidence_inputs": [{"path": str(path.resolve()), "sha256": digest(path)} for path in sorted(input_paths)],
        "trajectory_file_membership": sorted(actual_raw), "lifecycle_file_membership": sorted(actual_lifecycle),
        "checkpoint_file_membership": sorted(actual_checkpoints),
        "closed_ledger_prefix": closed_ledger, "active_audit_operation": active_ledger[0] if active_ledger else None,
        "audit_helpers": {str(path.relative_to(ROOT)): digest(path) for path in helper_paths},
        "checks": dict(sorted(checked.counts.items())), "check_count": sum(checked.counts.values()),
        "budget": {**budget, "preparation_seconds": prep_seconds, "current_artifact_bytes": actual_bytes},
        "training_complete": training_complete, "training_batches_attempted": len(batch_names),
        "training_matches": len(training), "training_learner_rows": sum(match["learner_rows"] for match in training),
        "training_calls": training_calls, "updates": update_summaries,
        "starting_updates": 4, "candidate_updates": state["updates"], "optimizer_updates": len(update_summaries),
        "optimizer_updates_scope": "Successful audited batches only; failed-attempt optimizer progress is reported separately.",
        "failed_training_progress": failed_training_progress,
        "observed_optimizer_steps_executed": len(update_summaries)
            + sum(item["optimizer_steps_executed_delta"] for item in failed_training_progress),
        "observed_completed_update_count": len(update_summaries)
            + sum(item["completed_update_count_delta"] for item in failed_training_progress),
        "checkpoint_writes": len(checkpoint_files), "evaluation_blocks": len(differences),
        "evaluation_blocks_attempted": len(eval_names), "evaluation_matches": len(evaluated),
        "persisted_partial_matches": partials, "admitted_without_durable_completion": interrupted_start,
        "complete_block_differences": differences, "mean_utility_difference": mean, "confidence_interval": interval,
        "calls": totals, "evaluation_learner_rows": sum(match["learner_rows"] for match in evaluated),
        "states": states, "evaluation_optimizer_updates": 0, "candidate_promoted": False, "old_evaluation_pooled": False,
        "limitations": ["Saved evidence integrity, not competitive-strength certification.",
            "Checkpoint hashes and worker state manifests are checked; model forwards and gradients remain worker evidence.",
            "Physical wall commitments are compared; unseen physical walls are not reconstructed from hashes.",
            "Memory maxima are sampled process-tree RSS.",
            "If current_audit_lease is present, its completion is absent from the closed-prefix budget.",
            "A failed attempt without any durable worker report requires separate failure forensics; this checker does not certify it."]}


def finalize_prefix(prefix_path, expected_prefix_sha, config_path, expected_config_sha):
    """Close a successful audit lease using hashes and budget bookkeeping only.

    Match probabilities, encoded tensor arithmetic and terminal accounting are
    not rerun here. Their prior checked results are retained only if all of their
    saved inputs and audit code remain byte-identical.
    """
    checked = Checks()
    prefix_path, config_path = Path(prefix_path).resolve(), Path(config_path).resolve()
    checked.require(digest(prefix_path) == expected_prefix_sha,
                    "Preserved full-audit prefix differs from externally pinned hash", "closure")
    result = read(prefix_path)
    checked.require(result["schema"] == "stage3e-independent-audit-v1" and result["valid"] is True
        and result["config_sha256"] == expected_config_sha == digest(config_path)
        and result["auditor_sha256"] == digest(Path(__file__))
        and result["active_audit_operation"] is not None and result["budget"]["all_closed"] is False,
        "Expected the unchanged result of a leased full audit", "closure")
    config = read(config_path)
    data_root, reports = (ROOT / config["data_root"]).resolve(), (ROOT / config["reports_root"]).resolve()
    seen = set()
    for item in result["evidence_inputs"]:
        path = Path(item["path"])
        checked.require(path.is_absolute() and path not in seen and path.is_file()
            and digest(path) == item["sha256"], "Full-audit evidence input changed before closure", "closure")
        seen.add(path)
    raw_paths = sorted(str(path.resolve()) for directory in ("evaluation", "trajectories")
                       for path in (data_root / directory).rglob("*") if path.is_file())
    lifecycle_paths = sorted(str(path.resolve()) for path in (data_root / "lifecycle").rglob("*") if path.is_file())
    checkpoints = sorted(str(path.resolve()) for path in data_root.rglob("*")
                         if path.is_file() and path.suffix in (".pt", ".pth"))
    checked.require(raw_paths == result["trajectory_file_membership"]
        and lifecycle_paths == result["lifecycle_file_membership"]
        and checkpoints == result["checkpoint_file_membership"],
        "Evidence membership changed after full audit", "closure")
    operations = read(data_root / "budget-ledger.json")["operations"]
    previous, active = result["closed_ledger_prefix"], result["active_audit_operation"]
    checked.require(len(operations) == len(previous) + 1 and operations[:-1] == previous,
                    "Ledger changed beyond closure of the one reserved audit operation", "closure")
    final = operations[-1]
    checked.require(all(final[key] == active[key] for key in ("id", "name", "token", "reserved_seconds", "started_unix"))
        and final["status"] == "completed" and final["failure"] is None
        and final["exceeded_reservation"] is False,
        "Full-audit lease did not close successfully with the same identity", "closure")
    budget, _ = audit_budget(config, expected_config_sha, reports, data_root, checked, False)
    actual_bytes = sum(path.stat().st_size for directory in (data_root, reports, ROOT / "Books/stage3e")
                       for path in directory.rglob("*") if path.is_file())
    checked.require(actual_bytes < config["artifact_bytes"], "Artifact cap exceeded at closure", "closure")
    result["full_audit_created_utc"] = result["created_utc"]
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["budget"] = {**budget, "preparation_seconds": result["budget"]["preparation_seconds"],
                        "current_artifact_bytes": actual_bytes}
    result["closed_ledger_prefix"] = operations
    result["active_audit_operation"] = None
    result["prefix_audit"] = {"path": str(prefix_path), "sha256": expected_prefix_sha}
    result["closure"] = {"valid": True, "evidence_hashes_revalidated": len(seen),
        "checks": dict(sorted(checked.counts.items())), "check_count": sum(checked.counts.values()),
        "full_match_numeric_arithmetic_repeated": False,
        "method": "Unchanged full-audit code and input hashes plus exact successful closure of its sole active lease; budget bookkeeping only."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/stage3e.json"))
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-active-audit", action="store_true")
    parser.add_argument("--finalize-prefix")
    parser.add_argument("--expected-prefix-sha256")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve prior audit evidence; use a new output")
    if args.finalize_prefix:
        if not args.expected_prefix_sha256 or args.allow_active_audit:
            parser.error("Prefix closure requires its pinned SHA and cannot allow active work")
        result = finalize_prefix(args.finalize_prefix, args.expected_prefix_sha256,
                                 args.config, args.expected_config_sha256)
    else:
        result = audit(args.config, args.expected_config_sha256, allow_active=args.allow_active_audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(canonical({"valid": result["valid"], "checks": result["check_count"],
        "engineering_evidence_pass": result["engineering_evidence_pass"]}))


if __name__ == "__main__":
    main()
