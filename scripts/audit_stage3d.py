"""Independent stdlib saved-evidence audit of Stage 3D evaluation recovery.

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

from audit_stage3c import CALLS, audit_boundary, audit_match, audit_state
from stage3a_audit import (Checks, REFERENCE, audit_encoded, canonical,
                          checked_artifact, close, digest, finite, read)

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_PROTOCOL = "1fb067d7b76a02eb286c9ab560678a78f8623cc6f59a75cbfa64016fe2936776"


def audit_file_membership(root, members, expected, checked):
    """Hash exactly the declared files; unrelated new files are outside scope."""
    checked.require(isinstance(members, list) and len(members) == len(set(members))
        and members == sorted(members) and members,
        "Explicit implementation membership is empty, repeated or unordered", "identity")
    checked.require(set(expected) == set(members),
                    "Frozen hashes differ from explicit file membership", "identity")
    root = Path(root).resolve()
    actual = {}
    for member in members:
        path = root / member
        checked.require(isinstance(member, str) and not Path(member).is_absolute()
            and path.resolve().is_relative_to(root) and path.is_file(),
            "Implementation member is missing or escapes project", "identity")
        actual[member] = digest(path)
    checked.require(actual == expected, "Frozen implementation member changed", "identity")
    return actual


def audit_campaign_record(record, campaign, behavior, planned, checked):
    """Retain historical policy identity while binding new campaign identity."""
    checked.require(campaign != behavior and behavior == BEHAVIOR_PROTOCOL,
                    "Campaign identity substituted for behavior identity", "provenance")
    checked.require(record["campaign_sha256"] == campaign and record["protocol_sha256"] == behavior
        and record["collector_schema"] == "stage3d-evaluation-lifecycle-v1"
        and record["collection_role"] == "evaluation"
        and record["training_eligible"] is False,
        "Recovery trajectory role or behavior protocol differs", "provenance")
    checked.require(all(record[key] == value for key, value in planned.items()),
                    "Recovery trajectory differs from fixed match plan", "provenance")
    checked.require(all(row["training_eligible"] is False for row in record["learner_rows"]),
                    "Evaluation row is training eligible", "provenance")
    if record["completed"]:
        checked.require(record["interrupted"] is False and record["failure"] is None,
                        "Completed record contains interruption or failure", "lifecycle")
    else:
        checked.require(record["learner_utility"] is None
            and (record["failure"] is not None or record.get("truncated") is True),
            "Incomplete record has terminal reward or hides failure/truncation", "lifecycle")


def audit_fixed_endpoint(result, differences, matches, checked):
    checked.require(len(differences) == 32 and len(matches) == 256
        and all(finite(value) and -3 <= value <= 3 for value in differences),
        "Fixed endpoint lacks 32 blocks and 256 complete matches", "evaluation")
    mean = math.fsum(differences) / 32
    halfwidth = 6 * math.sqrt(math.log(40) / 64)
    interval = [max(-3., mean - halfwidth), min(3., mean + halfwidth)]
    checked.require(result["blocks"] == 32 and result["matches"] == 256
        and result["block_differences"] == differences
        and close(result["mean_utility_difference"], mean, 1e-14)
        and close(result["halfwidth"], halfwidth, 1e-14)
        and len(result["confidence_interval"]) == 2
        and all(close(a, b, 1e-14) for a, b in zip(result["confidence_interval"], interval))
        and result["candidate_promoted"] is False,
        "Fixed-sample interval, membership or promotion differs", "evaluation")
    for arm in ("candidate", "baseline"):
        selected = [match for match in matches if match["arm"] == arm]
        checked.require(len(selected) == 128 and all(match["completed"] for match in selected),
                        "Arm does not contain 128 complete matches", "evaluation")
        checked.require(close(result[f"{arm}_mean_rank"], math.fsum(match["rank"] for match in selected) / 128),
                        "Arm mean rank differs", "evaluation")
    return mean, interval


def audit_budget(config, campaign, reports, data_root, checked, allow_active):
    raw = read(data_root / "budget-ledger.json")
    saved = read(reports / "budget.json")
    checked.require(raw["schema"] == "package-a2-budget-v1" and raw["identity"] == campaign
        and raw["limit_seconds"] == 400, "Campaign budget identity mismatch", "budget")
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
        elif re.fullmatch(r"eval:(?:[0-9]|[12][0-9]|3[01])", name):
            marker_path, maximum = reports / f"eval-attempt-{name.split(':')[1]}.json", config["per_block_allowance_seconds"]
        elif re.fullmatch(r"check:[a-zA-Z0-9_-]+", name):
            marker_path, maximum = reports / f"checks-{name.split(':')[1]}.json", config["test_allowance_seconds"]
        else:
            raise ValueError("Numerical operation outside fixed evaluation-only scope")
        checked.require(name not in markers and 0 < operation["reserved_seconds"] <= maximum
            and operation["reserved_seconds"] <= 400 - charged - 30 + 1e-8,
            "Repeated operation or reservation exceeded cap/shutdown reserve", "budget")
        charged += operation["elapsed_seconds"]
        checked.require(operation["exceeded_reservation"] is False
            and operation["elapsed_seconds"] <= operation["reserved_seconds"],
            "Numerical operation exceeded reservation", "budget")
        marker = read(marker_path)
        checked.require(marker["name"] == name and marker["config_sha256"] == campaign
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
    checked.require(charged <= 400 and close(saved["charged_seconds"], charged)
        and close(saved["remaining_seconds"], 400 - charged), "Cumulative budget arithmetic differs", "budget")
    if active:
        checked.require(0 < active[0]["reserved_seconds"] <= 400 - charged - 30,
                        "Current audit reservation consumes shutdown reserve", "budget")
    return {"limit_seconds": 400, "closed_charged_seconds": charged,
        "closed_operations": len(closed), "all_closed": not active,
        "current_audit_lease": active[0]["name"] if active else None,
        "current_audit_reservation_seconds": active[0]["reserved_seconds"] if active else 0,
        "failed_operations": sum(operation["status"] == "failed" for operation in closed),
        "sampled_peak_tree_rss_bytes": max(peaks, default=0), "artifact_bytes_peak": max(sizes, default=0)}, markers


def audit_lifecycle(start, finish, *, campaign, plan_sha, block, arm, planned,
                    state, record, evidence, start_path, start_sha, checked):
    checked.require(start["schema"] == "stage3d-match-start-v1"
        and start["campaign_sha256"] == campaign and start["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL
        and start["plan_sha256"] == plan_sha and start["block"] == block and start["arm"] == arm
        and start["collection_batch_index"] == 1000 + block
        and start["planned_match"] == planned and start["state"] == state
        and finite(start["started_unix"]), "Durable start differs from frozen match admission", "lifecycle")
    checked.require(finish["schema"] == "stage3d-match-finish-v1"
        and finish["campaign_sha256"] == campaign and finish["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL
        and finish["start_path"] == str(start_path) and finish["start_sha256"] == start_sha
        and finish["block"] == block and finish["arm"] == arm and finish["match_index"] == planned["match_index"]
        and finish["completed"] is record["completed"] and finish["interrupted"] is record["interrupted"]
        and finish["failure"] == record["failure"] and finite(finish["finished_unix"])
        and finish["finished_unix"] >= start["started_unix"],
        "Durable completion does not correspond to admitted record", "lifecycle")
    expected_evidence = {key: evidence[key] for key in ("json_path", "json_sha256", "npz_path", "npz_sha256")}
    checked.require(finish["evidence"] == expected_evidence,
                    "Durable completion does not bind both persisted artifacts", "lifecycle")
    checked.require(record["campaign_plan_sha256"] == plan_sha and record["campaign_block"] == block
        and record["campaign_arm"] == arm, "Raw record differs from campaign lifecycle identity", "provenance")


def audit(config_path, expected_sha, *, allow_active=False):
    checked = Checks()
    config_path = Path(config_path).resolve()
    campaign = digest(config_path)
    checked.require(campaign == expected_sha, "Campaign differs from externally pinned SHA", "identity")
    config = read(config_path)
    exact = {"schema": "stage3d-evaluation-config-v1", "limit_seconds": 400,
        "preparation_limit_seconds": 45, "memory_bytes": 32 * 1024**3, "artifact_bytes": 500000000,
        "reserve_seconds": 30, "per_block_allowance_seconds": 30, "prepare_allowance_seconds": 20,
        "test_allowance_seconds": 40, "device": "cpu", "torch_threads": 1, "evaluation_blocks": 32,
        "maximum_matches": 256, "max_steps": 100000, "behavior_protocol_sha256": BEHAVIOR_PROTOCOL,
        "reference_sha256": REFERENCE}
    checked.require(all(config.get(key) == value for key, value in exact.items()),
                    "Authorized recovery scope or limits changed", "scope")
    constraints = {"optimizer_updates": 0, "checkpoint_writes": 0, "book_training": False,
        "test_split_loading": False, "external_acquisition": False, "paid_compute": False,
        "policy_promotion": False, "failed_block_replacement": False, "old_evaluation_pooling": False}
    checked.require(config["constraints"] == constraints, "Evaluation-only scope changed", "scope")
    def seed(kind, index):
        return int.from_bytes(hashlib.sha256(f"riichi-ai-stage3d-20261021:{kind}:{index}".encode()).digest()[:8], "big")
    evaluation = [{"block": block, "wall_seed": seed("evaluation-wall", block), "pairs": [
        {"learner_seat": seat, "sampling_seed": seed("evaluation-sampling", block * 4 + seat),
         "candidate_match_index": 320 + block * 8 + seat * 2,
         "baseline_match_index": 321 + block * 8 + seat * 2} for seat in range(4)]} for block in range(32)]
    checked.require(config["evaluation"] == evaluation, "Fixed fresh seed/seat/arm plan changed", "scope")
    old_config_path = ROOT / "configs/stage3c.json"
    checked.require(digest(old_config_path) == BEHAVIOR_PROTOCOL, "Historical Stage 3C protocol changed", "identity")
    old_config = read(old_config_path)
    old_seeds = {match[key] for batch in old_config["batches"] for match in batch["matches"]
                 for key in ("seed", "sampling_seed")}
    old_seeds |= {block["wall_seed"] for block in old_config["evaluation"]}
    old_seeds |= {pair["sampling_seed"] for block in old_config["evaluation"] for pair in block["pairs"]}
    fresh = [block["wall_seed"] for block in evaluation]
    fresh += [pair["sampling_seed"] for block in evaluation for pair in block["pairs"]]
    checked.require(len(fresh) == len(set(fresh)) == 160 and not set(fresh) & old_seeds,
                    "Recovery seeds collide or reuse stopped campaign seeds", "scope")
    data_root, reports = (ROOT / config["data_root"]).resolve(), (ROOT / config["reports_root"]).resolve()
    reference = (ROOT / config["reference_checkpoint"]).resolve()
    checked.require(digest(reference) == REFERENCE, "Original opponent checkpoint changed", "identity")
    history_manifest_path = ROOT / "reports/stage3c/manifest.json"
    checked.require(digest(history_manifest_path) == "c4f7090dbce4067bb93bb21ad5751dbc830c84e2708e0bee8503e8cb0ac9c8ce",
                    "Historical Stage 3C inventory identity differs", "identity")
    history_manifest = read(history_manifest_path)
    history_pins = {entry["path"]: entry["sha256"] for entry in history_manifest["files"] if entry["scope"] == "project"}
    for path in ("configs/stage3c.json", "reports/stage3c/plan.json", "reports/stage3c/batch-3.json"):
        checked.require(digest(ROOT / path) == history_pins[path],
                        "Historical endpoint/protocol metadata changed", "identity")
    snapshot = read(reports / "previous-stage3c-snapshot.json")
    checked.require(snapshot["valid"] is True and snapshot["verified_files"] == 383
        and snapshot["snapshot_commit"] == "fbba75bc5e2301b67f1addd68844d3b4589cbfdf"
        and snapshot["manifest_sha256"] == digest(history_manifest_path),
        "Historical Stage 3C snapshot prerequisite differs", "identity")
    prior_plan, prior_final = read(ROOT / "reports/stage3c/plan.json"), read(ROOT / "reports/stage3c/batch-3.json")
    states = {"candidate": prior_final["output_state"], "baseline": prior_plan["initial_state"]}
    prior_boundaries = {"candidate": prior_final["output_boundary"], "baseline": prior_plan["initial_boundary"]}
    boundary_shas = {"candidate": "4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967",
                     "baseline": "3baca74ddcb011ea7346396621c709e7499835c2cf4b9e6e201688aa52e75bf2"}
    boundaries = {arm: {"path": boundary["path"], "sha256": boundary_shas[arm], "updates": states[arm]["updates"]}
                  for arm, boundary in prior_boundaries.items()}
    checked.require(config["boundaries"] == boundaries, "Recovery endpoints changed", "checkpoint")
    old_data_root = Path(old_config["data_root"]).resolve()
    for arm, updates in (("candidate", 4), ("baseline", 0)):
        audit_state(states[arm], updates, BEHAVIOR_PROTOCOL, checked, states["baseline"])
        checked.require(prior_boundaries[arm]["sha256"] == boundary_shas[arm],
                        "Historical endpoint file identity differs", "checkpoint")
        audit_boundary(prior_boundaries[arm], states[arm], old_data_root, checked)
    plan_path = reports / "plan.json"
    plan, plan_sha = read(plan_path), digest(plan_path)
    checked.require(plan["schema"] == "stage3d-plan-v1" and plan["config_sha256"] == campaign
        and plan["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL and plan["evaluation"] == evaluation
        and plan["constraints"] == constraints and plan["boundaries"] == boundaries
        and plan["states"] == states and plan["frozen_before_formal_collection"] is True
        and plan["old_stage3c_seeds_disjoint"] is True,
        "Frozen recovery plan identity or endpoint differs", "identity")
    expected_members = sorted(set(prior_plan["implementation"]) | {"scripts/stage3d.py",
        "src/riichi_ai/stage3d.py", "tests/test_stage3d_controller.py", "tests/test_stage3d_lifecycle.py"})
    checked.require(config["source_paths"] == expected_members,
                    "Recovery source membership omitted historical runtime or new checks", "scope")
    implementation = audit_file_membership(ROOT, config["source_paths"], plan["implementation"], checked)
    checked.require(all(implementation[key] == value for key, value in prior_plan["implementation"].items()),
                    "Historical policy/runtime implementation changed", "identity")
    checked.require(plan["versions"] == {"torch": "2.14.1", "numpy": "2.5.4", "riichienv": "0.4.10",
        "mahjong": "2.0.0", "psutil": "7.2.2"}, "Recorded versions differ from pinned stack", "identity")
    opponent = plan["opponent"]
    checked.require(opponent == prior_plan["opponent"] and opponent["checkpoint_sha256"] == REFERENCE
        and opponent["weights_sha256"] == states["baseline"]["base_weights_sha256"]
        and opponent["decision_rule"] == "stable_semantic_argmax", "Original frozen opponents changed", "identity")
    budget, markers = audit_budget(config, campaign, reports, data_root, checked, allow_active)
    checked.require(markers["prepare"]["status"] == "completed"
        and markers["prepare"]["implementation"] == implementation, "Preparation did not pass", "scope")
    preflight_path = reports / "checks-release-preflight.json"
    preflight = read(preflight_path)
    xml_path = reports / "release-preflight.xml"
    xml = ET.parse(xml_path).getroot()
    cases = list(xml.iter("testcase"))
    classes = {case.get("classname", "").split(".")[-1] for case in cases}
    checked.require(preflight["status"] == "completed" and preflight["implementation"] == implementation
        and preflight["finished_unix"] <= markers["prepare"]["started_unix"]
        and {"test_stage3d_controller", "test_stage3d_lifecycle"} <= classes
        and not any(list(xml.iter(kind)) for kind in ("error", "failure", "skipped")),
        "Required source-freeze/lifecycle checks did not pass before prepare", "scope")
    checked.require(plan["preflight_receipt"] == {"receipt_sha256": digest(preflight_path),
        "junit_sha256": digest(xml_path), "passing_cases": len(cases)},
        "Frozen preflight receipt differs", "identity")
    eval_names = [name for name in markers if name.startswith("eval:")]
    checked.require(0 < len(eval_names) <= 32 and eval_names == [f"eval:{i}" for i in range(len(eval_names))],
                    "Evaluation attempt order is not a fixed prefix", "scope")
    failed = [name for name in eval_names if markers[name]["status"] == "failed"]
    checked.require(not failed or failed == eval_names[-1:], "Evaluation continued or retried after failure", "scope")
    operations = read(data_root / "budget-ledger.json")["operations"]
    first_eval = next(operation["id"] for operation in operations if operation["name"] == "eval:0")
    prep_seconds = sum(operation["elapsed_seconds"] for operation in operations[:first_eval])
    checked.require(prep_seconds <= 45, "Preparation cap exceeded before formal play", "budget")
    matches, differences, partials, lifecycle_files, all_raw_files = [], [], [], [], []
    last_finish_time = None
    for index, planned_block in enumerate(evaluation[:len(eval_names)]):
        block = read(reports / f"eval-{index}.json")
        marker = markers[f"eval:{index}"]
        complete = block["complete"]
        checked.require(type(complete) is bool and (marker["status"] == "completed") is complete
            and marker["implementation"] == implementation and block["schema"] == "stage3d-evaluation-block-v1"
            and block["block"] == index and block["config_sha256"] == campaign
            and block["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL and block["plan_sha256"] == plan_sha
            and block["implementation"] == implementation and block["planned"] == planned_block
            and block["boundaries"] == boundaries and block["states_before"] == states
            and 0 <= len(block["matches"]) <= 8, "Evaluation block identity differs", "evaluation")
        if complete:
            checked.require(len(block["matches"]) == 8 and block["states_after"] == states
                and block["termination_requested"] is False, "Completed block changed states or stopped", "evaluation")
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
                evidence = block["matches"][position]
                planned = {"match_index": pair[f"{arm}_match_index"], "seed": planned_block["wall_seed"],
                    "sampling_seed": pair["sampling_seed"], "learner_seat": pair["learner_seat"]}
                checked.require(evidence["arm"] == arm and evidence["learner_seat"] == pair["learner_seat"],
                                "Recovery arm/seat collection order differs", "evaluation")
                record_path = checked_artifact(evidence["json_path"], evidence["json_sha256"], data_root, checked)
                arrays = checked_artifact(evidence["npz_path"], evidence["npz_sha256"], data_root, checked)
                record = read(record_path)
                audit_campaign_record(record, campaign, BEHAVIOR_PROTOCOL, planned, checked)
                start_path = checked_artifact(evidence["lifecycle_start_path"], evidence["lifecycle_start_sha256"], data_root, checked)
                finish_path = checked_artifact(evidence["lifecycle_finish_path"], evidence["lifecycle_finish_sha256"], data_root, checked)
                start, finish = read(start_path), read(finish_path)
                audit_lifecycle(start, finish, campaign=campaign, plan_sha=plan_sha, block=index,
                    arm=arm, planned=planned, state=states[arm], record=record, evidence=evidence,
                    start_path=start_path, start_sha=evidence["lifecycle_start_sha256"], checked=checked)
                checked.require(marker["started_unix"] <= start["started_unix"] <= finish["finished_unix"]
                    <= marker["finished_unix"], "Match lifecycle lies outside operation", "lifecycle")
                checked.require(last_finish_time is None or start["started_unix"] >= last_finish_time,
                                "Match admitted before preceding artifact completion", "lifecycle")
                last_finish_time = finish["finished_unix"]
                lifecycle_files.extend((str(start_path), str(finish_path)))
                all_raw_files.extend((str(record_path), str(arrays)))
                if not record["completed"]:
                    checked.require(not complete and position == len(block["matches"]) - 1,
                                    "Incomplete match was followed or counted as completed", "lifecycle")
                    audit_encoded(arrays, record["learner_rows"], checked)
                    partials.append({"block": index, "match_index": planned["match_index"],
                        "interrupted": record["interrupted"], "failure": record["failure"],
                        "included_in_block_summary": False, "encoded_archive_checked": True,
                        "full_terminal_audit_applicable": False})
                    continue
                match = audit_match(evidence, planned, states[arm], opponent, BEHAVIOR_PROTOCOL,
                    data_root, checked, collection_role="evaluation", collection_batch=1000 + index)
                arms[arm] = match
                matches.append({"arm": arm, **match})
                block_matches.append(match)
                outcomes.append({"arm": arm, "learner_seat": pair["learner_seat"],
                    "utility": match["learner_utility"], "rank": match["rank"]})
            if len(arms) == 2:
                a, b = arms["candidate"], arms["baseline"]
                common = min(len(a["ordinal_wall_fingerprints"]), len(b["ordinal_wall_fingerprints"]))
                checked.require(common > 0 and a["ordinal_wall_fingerprints"][:common] == b["ordinal_wall_fingerprints"][:common],
                                "Paired physical wall prefixes differ", "evaluation")
                pairs.append({"learner_seat": pair["learner_seat"], "common_hands": common,
                    "common_walls_verified": True, "utility_difference": a["learner_utility"] - b["learner_utility"]})
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
                "Incomplete block summary is not a persisted prefix", "evaluation")
    experiment_complete = len(differences) == 32 and not failed
    mean = interval = None
    if experiment_complete:
        result = read(reports / "evaluation.json")
        checked.require(result["schema"] == "stage3d-evaluation-v1" and result["old_evaluation_pooled"] is False
            and result["optimizer_updates"] == 0, "Recovery endpoint pooled old data or trained", "evaluation")
        mean, interval = audit_fixed_endpoint(result, differences, matches, checked)
    else:
        checked.require(not (reports / "evaluation.json").exists(), "Incomplete campaign has a fixed endpoint", "evaluation")
    checked.require(len(all_raw_files) == len(set(all_raw_files)) and len(lifecycle_files) == len(set(lifecycle_files)),
                    "Duplicate raw artifacts or lifecycle markers", "identity")
    actual_raw = {str(path.resolve()) for path in (data_root / "evaluation").rglob("*") if path.is_file()}
    checked.require(actual_raw == set(all_raw_files), "Unreported or omitted raw artifacts", "scope")
    actual_lifecycle = {str(path.resolve()) for path in (data_root / "lifecycle").rglob("*") if path.is_file()}
    unexplained_lifecycle = actual_lifecycle - set(lifecycle_files)
    interrupted_start = []
    if unexplained_lifecycle:
        checked.require(not experiment_complete and len(unexplained_lifecycle) == 1,
                        "Unreported lifecycle evidence outside final interrupted attempt", "lifecycle")
        path = Path(next(iter(unexplained_lifecycle)))
        start = read(path)
        final_block = len(eval_names) - 1
        block = read(reports / f"eval-{final_block}.json")
        offset = len(block["matches"])
        pair, arm = evaluation[final_block]["pairs"][offset // 2], ("candidate", "baseline")[offset % 2]
        planned = {"match_index": pair[f"{arm}_match_index"], "seed": evaluation[final_block]["wall_seed"],
            "sampling_seed": pair["sampling_seed"], "learner_seat": pair["learner_seat"]}
        checked.require(offset < 8 and path.name == f"match-{planned['match_index']}-start.json"
            and start["schema"] == "stage3d-match-start-v1" and start["campaign_sha256"] == campaign
            and start["behavior_protocol_sha256"] == BEHAVIOR_PROTOCOL and start["plan_sha256"] == plan_sha
            and start["planned_match"] == planned and start["state"] == states[arm]
            and start["block"] == final_block and start["arm"] == arm,
            "Unfinished start is not the next fixed admitted match", "lifecycle")
        interrupted_start.append({"path": str(path), "sha256": digest(path), "match_index": planned["match_index"],
            "classification": "admitted but without durable trajectory completion; never counted or replaced"})
    checked.require(not list(data_root.rglob("*.pt")) and not list(data_root.rglob("*.pth")),
                    "Evaluation-only campaign wrote a model checkpoint", "scope")
    actual_bytes = sum(path.stat().st_size for directory in (data_root, reports, ROOT / "Books/stage3d")
                       for path in directory.rglob("*") if path.is_file())
    checked.require(actual_bytes < config["artifact_bytes"], "Current artifacts exceed cap", "budget")
    totals = {arm: {key: {kind: sum(match[key][kind] for match in matches if match["arm"] == arm)
        for kind in CALLS} for key in ("requested_calls", "declared_calls", "completed_calls", "robbed_kans")}
        for arm in ("candidate", "baseline")}
    input_paths = {config_path, plan_path, preflight_path, xml_path, history_manifest_path,
        reports / "previous-stage3c-snapshot.json", old_config_path,
        ROOT / "reports/stage3c/plan.json", ROOT / "reports/stage3c/batch-3.json", reference,
        Path(__file__).resolve(), ROOT / "scripts/audit_stage3c.py", ROOT / "scripts/stage3a_audit.py",
        ROOT / "tests/test_evaluation_recovery_audit.py"}
    input_paths.update(ROOT / path for path in implementation)
    input_paths.update(Path(boundary["path"]) for boundary in boundaries.values())
    input_paths.update(Path(path) for path in all_raw_files + lifecycle_files)
    input_paths.update(Path(path) for path in unexplained_lifecycle)
    input_paths.update(reports / f"eval-{index}.json" for index in range(len(eval_names)))
    for name in markers:
        if name == "prepare":
            input_paths.add(reports / "prepare-attempt.json")
        elif name.startswith("eval:"):
            input_paths.add(reports / f"eval-attempt-{name.split(':')[1]}.json")
        else:
            input_paths.add(reports / f"checks-{name.split(':')[1]}.json")
    if experiment_complete:
        input_paths.add(reports / "evaluation.json")
    raw_ledger = read(data_root / "budget-ledger.json")
    closed_ledger = [operation for operation in raw_ledger["operations"] if operation["status"] != "active"]
    active_ledger = [operation for operation in raw_ledger["operations"] if operation["status"] == "active"]
    return {"schema": "stage3d-independent-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True, "experiment_complete": experiment_complete, "engineering_evidence_pass": experiment_complete,
        "config_sha256": campaign, "behavior_protocol_sha256": BEHAVIOR_PROTOCOL, "plan_sha256": plan_sha,
        "implementation": implementation,
        "auditor_sha256": digest(Path(__file__)),
        "evidence_inputs": [{"path": str(path.resolve()), "sha256": digest(path)} for path in sorted(input_paths)],
        "trajectory_file_membership": sorted(actual_raw), "lifecycle_file_membership": sorted(actual_lifecycle),
        "closed_ledger_prefix": closed_ledger,
        "active_audit_operation": active_ledger[0] if active_ledger else None,
        "audit_helpers": {str(path.relative_to(ROOT)): digest(path) for path in
            (ROOT / "scripts/audit_stage3c.py", ROOT / "scripts/stage3a_audit.py")},
        "checks": dict(sorted(checked.counts.items())), "check_count": sum(checked.counts.values()),
        "budget": {**budget, "preparation_seconds": prep_seconds, "current_artifact_bytes": actual_bytes},
        "evaluation_blocks": len(differences), "evaluation_blocks_attempted": len(eval_names),
        "evaluation_matches": len(matches), "persisted_partial_matches": partials,
        "admitted_without_durable_completion": interrupted_start,
        "complete_block_differences": differences, "mean_utility_difference": mean,
        "confidence_interval": interval, "calls": totals,
        "evaluation_learner_rows": sum(match["learner_rows"] for match in matches),
        "states": states, "optimizer_updates": 0, "checkpoint_writes": 0, "candidate_promoted": False,
        "old_evaluation_pooled": False,
        "limitations": ["Saved evidence integrity, not competitive-strength certification.",
            "Endpoint file hashes and worker state manifests are checked; model forwards remain worker evidence.",
            "Physical wall commitments are compared; unseen physical walls are not reconstructed from hashes.",
            "Memory maxima are sampled process-tree RSS.",
            "If current_audit_lease is present, its own completion is absent from the closed-prefix budget."]}


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
    checked.require(result["schema"] == "stage3d-independent-audit-v1" and result["valid"] is True
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
    for directory, key in (("evaluation", "trajectory_file_membership"), ("lifecycle", "lifecycle_file_membership")):
        paths = sorted(str(path.resolve()) for path in (data_root / directory).rglob("*") if path.is_file())
        checked.require(paths == result[key], "Evidence membership changed after full audit", "closure")
    checked.require(not list(data_root.rglob("*.pt")) and not list(data_root.rglob("*.pth")),
                    "A new checkpoint appeared during audit closure", "closure")
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
    actual_bytes = sum(path.stat().st_size for directory in (data_root, reports, ROOT / "Books/stage3d")
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
    parser.add_argument("--config", default=str(ROOT / "configs/stage3d.json"))
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
