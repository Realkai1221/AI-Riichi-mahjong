"""Independent saved-evidence audit for the fixed Stage 3C experiment.

Uses the preserved Stage 3A stdlib NPY, probability and terminal-ledger auditors.
Does not import learner/model/engine code. Boundary file hashes and recorded
reload identities are checked; model forwards/gradients remain worker evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from stage3a_audit import (Checks, REFERENCE, RULE_PROFILE, audit_encoded, audit_row,
    audit_score_chain, canonical, checked_artifact, close, digest, finite,
    object_hash, read)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "stage3c-frozen-base-call-heads-v1"
SAMPLER = "python-random-float64-cdf-call-gate-v1"
CALLS = ("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN")
HEAD_CONFIG = {"hidden_width": 64, "activation": "ReLU", "layer_norm_affine": False,
    "initial_call_mass": .05, "context_interface": "frozen-context-output",
    "candidate_interface": "frozen-action-encoding-plus-efficiency"}


def audit_state(state, updates, protocol, checked, initial=None):
    checked.require(state["schema"] == SCHEMA and state["sampler"] == SAMPLER
        and state["protocol_sha256"] == protocol and state["reference_sha256"] == REFERENCE
        and state["device"] == "cpu", "State protocol/device/sampler mismatch", "identity")
    checked.require(state["updates"] == state["optimizer_steps_executed"] == updates
        and state["retired_batches"] == list(range(updates)), "Nonsequential update boundary", "update")
    keys = ("weights_sha256", "heads_sha256", "base_weights_sha256", "immutable_base_sha256",
            "optimizer_sha256", "rank_head_sha256")
    checked.require(all(re.fullmatch("[0-9a-f]{64}", state[key]) for key in keys),
                    "Malformed state identity", "identity")
    checked.require(state["policy_id"] == f"{SCHEMA}:update-{updates}:{state['weights_sha256']}",
                    "Policy ID does not bind model and update", "identity")
    checked.require(state["base_unchanged"] is True
        and state["base_weights_sha256"] == state["immutable_base_sha256"],
        "Frozen base changed", "frozen_base")
    checked.require(state["head_config"] == HEAD_CONFIG
        and state["model_config"]["feature_schema"] == "actor-features-v2",
        "Architecture/feature schema changed", "scope")
    checked.require(state["optimizer_config"] == {"name": "AdamW", "learning_rate": .0001,
        "weight_decay": 0., "betas": [.9, .999], "eps": 1e-8, "foreach": False,
        "fresh_optimizer": True, "parameter_scope": "new_heads_only"},
        "Heads-only optimizer recipe changed", "update")
    if initial is not None:
        for key in ("base_weights_sha256", "immutable_base_sha256", "rank_head_sha256",
                    "model_config", "head_config", "initialization_seed"):
            checked.require(state[key] == initial[key], f"Frozen state field changed: {key}", "frozen_base")


def audit_boundary(boundary, state, data_root, checked):
    checked.require({key: value for key, value in boundary.items() if key not in ("path", "sha256")} == state,
                    "Boundary/state identity mismatch", "checkpoint")
    checked_artifact(boundary["path"], boundary["sha256"], data_root, checked)


def call_events(events, seat):
    """Independent declaration versus completed-kan counting from event order."""
    declared, completed, robbed = Counter(), Counter(), Counter()
    pending = None
    for event in events:
        kind = event.get("type", "").upper()
        if kind in CALLS and event.get("actor") == seat:
            declared[kind] += 1
            if kind in ("CHI", "PON"):
                completed[kind] += 1
            else:
                pending = kind
        elif pending is not None and kind == "TSUMO" and event.get("actor") == seat:
            completed[pending] += 1
            pending = None
        elif pending is not None and kind == "HORA" and event.get("target") == seat:
            robbed[pending] += 1
            pending = None
        elif kind in ("START_KYOKU", "END_KYOKU", "END_GAME"):
            pending = None
    return ({kind: counter[kind] for kind in CALLS} for counter in (declared, completed, robbed))


def audit_match(evidence, planned, state, opponent, protocol, data_root, checked,
                *, collection_role="training", collection_batch=None):
    collection_batch = state["updates"] if collection_batch is None else collection_batch
    path = checked_artifact(evidence["json_path"], evidence["json_sha256"], data_root, checked)
    arrays = checked_artifact(evidence["npz_path"], evidence["npz_sha256"], data_root, checked)
    record = read(path)
    checked.require(record["schema"] == SCHEMA and record["protocol_sha256"] == protocol
        and record["batch_index"] == collection_batch and record["policy_id"] == state["policy_id"]
        and record["sampler"] == SAMPLER and record["collection_role"] == collection_role
        and record["training_eligible"] is (collection_role == "training"),
        "Trajectory behavior identity mismatch", "provenance")
    checked.require(all(record[key] == value for key, value in planned.items()),
                    "Trajectory differs from frozen seed/seat plan", "provenance")
    checked.require(record["rule_profile_sha256"] == RULE_PROFILE
        and record["view_hash_schema"] == "riichi-ai-player-view-v2"
        and record["max_steps"] == 100000, "Rules/actor contract/match ceiling differs", "scope")
    rows, requests, log = record["learner_rows"], record["requests"], record["auditor_log"]
    checked.require(record["encoded_artifact"] == {"path": str(arrays), "sha256": evidence["npz_sha256"],
        "rows": len(rows)}, "Encoded archive association mismatch", "identity")
    checked.require(record["auditor_log_sha256"] == object_hash(log)
        and record["hands"] == sum(event.get("type") == "start_kyoku" for event in log),
        "Full-log hash or explicit hand count mismatch", "terminal")
    focal = record["learner_seat"]
    expected_opponent = {**opponent, "role": "frozen_opponent", "policy_id": "frozen-argmax:" + REFERENCE}
    checked.require(len(record["policies"]) == 4 and focal in range(4), "Invalid seats", "provenance")
    for seat, policy in enumerate(record["policies"]):
        if seat != focal:
            checked.require(policy == expected_opponent, "Frozen opponent identity changed", "provenance")
        else:
            checked.require(policy["role"] == "learner" and policy["policy_id"] == state["policy_id"]
                and policy["weights_sha256"] == state["weights_sha256"]
                and policy["base_weights_sha256"] == state["immutable_base_sha256"]
                and policy["heads_sha256"] == state["heads_sha256"] and policy["update"] == state["updates"]
                and policy["reference_sha256"] == REFERENCE and policy["sampling_seed"] == planned["sampling_seed"]
                and policy["feature_schema"] == "actor-features-v2", "Learner manifest differs", "provenance")
    checked.require(record["completed"] is True and record["terminated"] is True
        and record["truncated"] is False and record["failure"] is None
        and record["stop_reason"] == "completed" and record["terminal_result_verified"] is True
        and record["conservation_verified"] is True, "Trajectory incomplete or failed", "terminal")
    scores, reconstruction = audit_score_chain(log, checked)
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    ranks = [order.index(seat) + 1 for seat in range(4)]
    utilities = [2.5 - rank for rank in ranks]
    checked.require(record["scores"] == record["reconstructed_scores"] == scores
        and record["terminal_score_reconstruction"] == reconstruction
        and record["ranks"] == record["reconstructed_ranks"] == ranks
        and record["utilities"] == utilities and record["learner_utility"] == utilities[focal],
        "Complete-match utility or terminal score chain mismatch", "terminal")
    checked.require(sum(scores) + 1000 * record["riichi_sticks"] == 100000,
                    "Terminal score/deposit conservation mismatch", "terminal")
    rng = random.Random(planned["sampling_seed"])
    checked.require(object_hash(rng.getstate()) == object_hash(record["sampling_rng_start"]),
                    "Initial sampling RNG differs", "behavior")
    for index, row in enumerate(rows):
        checked.require(row["row_index"] == index and row["match_index"] == planned["match_index"]
            and row["batch_index"] == collection_batch and row["seat"] == focal
            and row["policy_id"] == state["policy_id"], "Learner row provenance mismatch", "provenance")
        checked.require(type(row["hand_id"]) is int and 1 <= row["hand_id"] <= record["hands"]
            and (index == 0 or row["hand_id"] >= rows[index - 1]["hand_id"]),
            "Learner hand IDs nonsequential", "provenance")
        audit_row(row, checked)
        if not row["forced"]:
            checked.require(row["uniform_draw"] == rng.random(), "Behavior RNG stream differs", "behavior")
        kinds = [json.loads(key)[1] for key in row["semantic_keys"]]
        checked.require(kinds == row["action_kinds"] and row["selected_kind"] == kinds[row["selected_index"]],
                        "Action-family metadata differs", "calls")
        call_mass = math.fsum(probability for kind, probability in zip(kinds, row["probabilities"]) if kind in CALLS)
        call_count = sum(kind in CALLS for kind in kinds)
        has_both_groups = 0 < call_count < len(kinds)
        checked.require(row["call_candidates"] == sum(kind in CALLS for kind in kinds)
            and close(row["call_mass"], call_mass, 1e-12), "Call mass/candidate mismatch", "calls")
        checked.require(finite(row["gate_mass"]) and 0 <= row["gate_mass"] <= 1
            and close(call_mass, row["gate_mass"], 1e-12), "Gate and actual call mass differ", "calls")
        if state["updates"] == 0 and call_count:
            expected_mass = .05 if has_both_groups else 1.
            checked.require(close(call_mass, expected_mass, 1e-8), "Initial call mass changed", "calls")
            checked.require(all(close(probability, call_mass / call_count, 1e-12)
                for kind, probability in zip(kinds, row["probabilities"]) if kind in CALLS),
                "Initialized call head is not uniform within legal calls", "calls")
        win_mass = math.fsum(probability for kind, probability in zip(kinds, row["probabilities"])
                             if kind in ("RON", "TSUMO"))
        checked.require(close(row["win_mass_actual"], win_mass, 1e-12)
            and close(win_mass, row["win_mass_without_gate"] * (1 - row["gate_mass"]), 1e-12),
            "Win probability is inconsistent with gate rule", "calls")
        checked.require(close(row["win_mass_lost"], row["win_mass_without_gate"] - row["win_mass_actual"], 1e-12)
            and row["win_mass_lost"] >= -1e-12, "Lost win-probability mass mismatch", "calls")
        checked.require(row["post_call_state"] is (row["own_meld_count"] > 0)
            and row["open_hand_state"] is (row["own_open_meld_count"] > 0),
            "Post-call observation classification mismatch", "calls")
    checked.require(object_hash(rng.getstate()) == object_hash(record["sampling_rng_end"]),
                    "Final sampling RNG differs", "behavior")
    linked = []
    for step, request in enumerate(requests):
        identity = object_hash({"match_index": planned["match_index"], "step": step, "views": request["player_views"]})
        checked.require(request["step"] == step and request["pre_response_id"] == identity,
                        "Pre-response snapshot mismatch", "provenance")
        checked.require(set(request["choices"]) <= set(request["player_views"]),
                        "Submitted response lacks same-snapshot player view", "provenance")
        start, end = request["auditor_event_start"], request["auditor_event_end"]
        checked.require(type(start) is int and type(end) is int and 0 <= start <= end <= len(log)
            and (step == 0 or start == requests[step - 1]["auditor_event_end"]),
            "Atomic engine event ranges differ", "calls")
        declared_events = [event for event in log[start:end] if event.get("type", "").upper() in CALLS]
        checked.require(request["declared_call_events"] == declared_events,
                        "Per-response declared calls differ from full log", "calls")
        for seat_text, choice in request["choices"].items():
            seat = int(seat_text)
            checked.require(choice["policy_id"] == record["policies"][seat]["policy_id"]
                and choice["fallback"] is False and choice["overridden"] is False,
                "Action provenance/override mismatch", "provenance")
            if seat == focal:
                index = choice["row_index"]
                row = rows[index]
                checked.require(choice["eligible"] is True and row["step"] == step
                    and row["pre_response_id"] == identity
                    and row["player_view_sha256"] == request["player_views"][seat_text]
                    and choice["semantic_key"] == row["selected_semantic_key"]
                    and choice["physical_token"] == row["selected_physical_token"]
                    and choice["probability"] == row["probability"], "Learner action link mismatch", "provenance")
                outcomes = [outcome for outcome in record["call_resolution_evidence"]
                            if start <= outcome["event_index"] < end and outcome["actor"] == focal]
                expected_resolved = (any(outcome["resolved"] for outcome in outcomes
                    if outcome["kind"] == row["selected_kind"]) if row["selected_kind"] in CALLS else None)
                checked.require(row["requested_call"] is (row["selected_kind"] in CALLS)
                    and row["call_resolution_evidence"] == outcomes
                    and row["requested_call_resolved"] is expected_resolved,
                    "Requested/resolved row linkage differs", "calls")
                linked.append(index)
            else:
                checked.require(choice["eligible"] is False and choice["probability"] == 1.
                    and "row_index" not in choice, "Opponent action is learner data", "provenance")
    checked.require(linked == list(range(len(rows))) and len(requests) == record["steps"],
                    "Omitted/duplicate learner rows or simulator steps", "provenance")
    requested = {kind: sum(row["selected_kind"] == kind for row in rows) for kind in CALLS}
    declared, completed, robbed = call_events(log, focal)
    full_declarations = [index for index, event in enumerate(log) if event.get("type", "").upper() in CALLS]
    checked.require([outcome["event_index"] for outcome in record["call_resolution_evidence"]] == full_declarations,
                    "Omitted/extra call-resolution records", "calls")
    for outcome in record["call_resolution_evidence"]:
        event = log[outcome["event_index"]]
        checked.require(outcome["kind"] == event["type"].upper() and outcome["actor"] == event["actor"]
            and type(outcome["resolved"]) is bool, "Malformed completed-match call outcome", "calls")
        completion = outcome["completion_event_index"]
        if outcome["kind"] in ("CHI", "PON"):
            checked.require(outcome["resolved"] is True and completion == outcome["event_index"],
                            "Accepted chi/pon completion differs", "calls")
        else:
            expected = None
            for following in range(outcome["event_index"] + 1, len(log)):
                subsequent = log[following]
                if subsequent.get("type") in ("hora", "end_kyoku", "start_kyoku", "end_game"):
                    break
                if subsequent.get("type") == "tsumo" and subsequent.get("actor") == outcome["actor"]:
                    expected = following
                    break
            checked.require(completion == expected and outcome["resolved"] is (expected is not None),
                            "Kan completion does not match replacement draw before interruption", "calls")
    checked.require(record["requested_calls"] == requested, "Requested-call total mismatch", "calls")
    if "declared_calls" in record:
        checked.require(record["declared_calls"] == declared, "Declared-call total mismatch", "calls")
    checked.require(record["resolved_calls"] == completed, "Resolved-call total mismatch", "calls")
    checked.require(record["post_call_rows"] == sum(row["post_call_state"] for row in rows)
        and record["open_hand_rows"] == sum(row["open_hand_state"] for row in rows),
        "Post-call observation total mismatch", "calls")
    checked.require(record["win_competition_rows"] == sum(row["call_candidates"] > 0
        and row["win_mass_without_gate"] > 0 for row in rows)
        and close(record["total_win_probability_mass_lost"], math.fsum(row["win_mass_lost"] for row in rows)),
        "Win-probability competition summary differs", "calls")
    checked.require(record["learner_last_step"] == (rows[-1]["step"] if rows else None)
        and record["opponent_steps_after_last_learner"] == (record["steps"] - 1 - rows[-1]["step"] if rows else record["steps"]),
        "Reward linkage after final learner action differs", "terminal")
    checked.require(len(record["ordinal_wall_fingerprints"]) == record["hands"]
        and all(re.fullmatch("[0-9a-f]{64}", value) for value in record["ordinal_wall_fingerprints"]),
        "Ordinal physical-wall commitment count differs", "identity")
    audit_encoded(arrays, rows, checked)
    return {"match_index": record["match_index"], "completed": True, "learner_utility": utilities[focal],
        "rank": ranks[focal], "scores": scores, "learner_rows": len(rows),
        "optional_rows": sum(not row["forced"] for row in rows), "forced_rows": sum(row["forced"] for row in rows),
        "stored_log_probability_sum": math.fsum(row["log_probability"] for row in rows),
        "optional_entropy_sum": math.fsum(row["entropy"] for row in rows if not row["forced"]),
        "requested_calls": requested, "declared_calls": declared, "completed_calls": completed,
        "robbed_kans": robbed, "post_call_rows": record["post_call_rows"], "open_hand_rows": record["open_hand_rows"],
        "ordinal_wall_fingerprints": record["ordinal_wall_fingerprints"],
        "files": [str(path), str(arrays)]}


def audit_update(update, matches, before, planned, checked):
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
        checked.require(finite(update[field]) and update[field] > 0, "No-op or invalid learning diagnostic", "update")
    for component in ("gate", "call_head"):
        values = update["gradient_components"][component]
        checked.require(values["finite"] is True and values["l2_norm"] > 0
            and values["nonzero_elements"] > 0, f"No gradient in {component}", "update")
    checked.require(update["distribution_diagnostic_complete"] is True
        and update["distribution_diagnostic_rows"] == optional and update["numerical_no_op"] is False,
        "Post-update diagnostic is incomplete or no-op", "update")
    checked.require(update["after"]["heads_sha256"] != before["heads_sha256"],
                    "Head weights did not change", "update")
    return {"batch_index": before["updates"], "optional_rows": optional,
            "loss": loss, "heads_changed": True, "base_unchanged": True}


def audit_budget(config, protocol, reports, data_root, checked, allow_active):
    raw = read(data_root / "budget-ledger.json")
    saved = read(reports / "budget.json")
    checked.require(raw["schema"] == "package-a2-budget-v1" and raw["identity"] == protocol
        and raw["limit_seconds"] == 600, "Budget identity mismatch", "budget")
    operations = raw["operations"]
    active = [operation for operation in operations if operation["status"] == "active"]
    checked.require(not active or (allow_active and len(active) == 1 and active[0] == operations[-1]
        and active[0]["name"].startswith("check:") and "audit" in active[0]["name"]),
        "Unclosed numerical work outside the explicitly allowed current audit", "budget")
    closed = operations[:-1] if active else operations
    checked.require(saved["operations"] == closed and saved["identity"] == protocol,
                    "Saved budget does not equal closed ledger prefix", "budget")
    charged = 0.
    peaks, sizes, markers = [], [], {}
    for index, operation in enumerate(closed):
        name = operation["name"]
        checked.require(operation["id"] == index and operation["status"] in ("completed", "failed")
            and finite(operation["elapsed_seconds"]) and operation["elapsed_seconds"] >= 0,
            "Malformed numerical operation", "budget")
        if name == "prepare":
            marker_path, maximum = reports / "prepare-attempt.json", 30
        elif re.fullmatch(r"batch:[0-3]", name):
            marker_path, maximum = reports / f"attempt-{name.split(':')[1]}.json", 100
        elif re.fullmatch(r"eval:(?:[0-9]|[12][0-9]|3[01])", name):
            marker_path, maximum = reports / f"eval-attempt-{name.split(':')[1]}.json", 40
        elif re.fullmatch(r"check:[a-zA-Z0-9_-]+", name):
            marker_path, maximum = reports / f"checks-{name.split(':')[1]}.json", 90
        else:
            raise ValueError("Numerical operation outside fixed Stage 3C scope")
        checked.require(0 < operation["reserved_seconds"] <= maximum
            and operation["reserved_seconds"] <= 600 - charged - 45 + 1e-8,
            "Reservation exceeds operation cap or shutdown reserve", "budget")
        charged += operation["elapsed_seconds"]
        checked.require(operation["exceeded_reservation"] is False
            and operation["elapsed_seconds"] <= operation["reserved_seconds"],
            "Numerical reservation was exceeded", "budget")
        marker = read(marker_path)
        checked.require(marker["name"] == name and marker["config_sha256"] == protocol
            and marker["status"] == operation["status"] and marker["failure"] == operation["failure"],
            "Marker and durable operation differ", "budget")
        checked.require(marker["budget"]["operations"] == closed[:index + 1]
            and close(marker["budget"]["charged_seconds"], charged),
            "Operation budget snapshot differs from prefix", "budget")
        watchdog = marker["watchdog"]
        checked.require(watchdog["sampled_peak_tree_rss_bytes"] < config["memory_bytes"]
            and watchdog["artifact_bytes_peak"] < config["artifact_bytes"],
            "Recorded memory or artifact cap exceeded", "budget")
        if operation["status"] == "completed":
            checked.require(watchdog["exit_code"] == 0 and operation["failure"] is None,
                            "Completed operation has error evidence", "budget")
        else:
            checked.require(operation["failure"] is not None, "Failed operation hides reason", "budget")
        peaks.append(watchdog["sampled_peak_tree_rss_bytes"])
        sizes.append(watchdog["artifact_bytes_peak"])
        markers[name] = marker
    checked.require(charged <= 600 and close(saved["charged_seconds"], charged)
        and close(saved["remaining_seconds"], 600 - charged), "Cumulative budget mismatch", "budget")
    if active:
        checked.require(active[0]["reserved_seconds"] <= 600 - charged - 45,
                        "Current audit exceeds reserve", "budget")
    return {"limit_seconds": 600, "closed_charged_seconds": charged,
        "closed_operations": len(closed), "all_closed": not active,
        "current_audit_lease": active[0]["name"] if active else None,
        "current_audit_reservation_seconds": active[0]["reserved_seconds"] if active else 0,
        "failed_operations": sum(operation["status"] == "failed" for operation in closed),
        "sampled_peak_tree_rss_bytes": max(peaks, default=0), "artifact_bytes_peak": max(sizes, default=0)}, markers


def audit(config_path, expected_sha, *, allow_active=False):
    checked = Checks()
    config_path = Path(config_path).resolve()
    protocol = digest(config_path)
    checked.require(protocol == expected_sha, "Configuration differs from externally pinned SHA", "identity")
    config = read(config_path)
    exact = {"schema": "stage3c-config-v1", "limit_seconds": 600, "memory_bytes": 32 * 1024**3,
        "artifact_bytes": 1000000000, "reserve_seconds": 45, "per_batch_allowance_seconds": 100,
        "prepare_allowance_seconds": 30, "test_allowance_seconds": 90, "device": "cpu", "torch_threads": 1,
        "maximum_batches": 4, "matches_per_batch": 16, "maximum_matches": 320, "training_matches": 64,
        "evaluation_matches": 256, "evaluation_blocks": 32, "per_block_allowance_seconds": 40,
        "preparation_limit_seconds": 90, "initialization_seed": 310211, "maximum_updates": 4,
        "learning_rate": .0001, "weight_decay": 0., "gradient_clip": 1., "replay_atol": 0.,
        "replay_rtol": 0., "replay_log_atol": 0., "microbatch_size": 1, "max_steps": 100000,
        "reference_sha256": REFERENCE}
    checked.require(all(config.get(key) == value for key, value in exact.items()),
                    "Authorized recipe/cap changed", "scope")
    constraints = dict(book_training=False, test_split_loading=False, external_acquisition=False,
        paid_compute=False, fresh_strength_evaluation=True, opponent_actions_eligible=False,
        baseline="zero", entropy_bonus=0., reward_shaping=False, replay_reuse=False,
        rank_head_frozen=True, base_frozen=True)
    checked.require(config["constraints"] == constraints, "Scope changed", "scope")
    def seed(kind, index):
        return int.from_bytes(hashlib.sha256(f"riichi-ai-stage3c-20261021:{kind}:{index}".encode()).digest()[:8], "big")
    batches = [{"batch_index": batch, "matches": [{"match_index": index,
        "seed": seed("wall", index), "sampling_seed": seed("sampling", index), "learner_seat": index % 4}
        for index in range(batch * 16, batch * 16 + 16)]} for batch in range(4)]
    evaluation = [{"block": block, "wall_seed": seed("evaluation-wall", block), "pairs": [
        {"learner_seat": seat, "sampling_seed": seed("evaluation-sampling", block * 4 + seat),
         "candidate_match_index": 64 + block * 8 + seat * 2,
         "baseline_match_index": 65 + block * 8 + seat * 2} for seat in range(4)]} for block in range(32)]
    checked.require(config["batches"] == batches and config["evaluation"] == evaluation,
                    "Fixed seed/seat plan changed", "scope")
    training_walls = {match["seed"] for batch in batches for match in batch["matches"]}
    evaluation_walls = {block["wall_seed"] for block in evaluation}
    training_sampling = {match["sampling_seed"] for batch in batches for match in batch["matches"]}
    evaluation_sampling = {pair["sampling_seed"] for block in evaluation for pair in block["pairs"]}
    checked.require(len(training_walls) == 64 and len(evaluation_walls) == 32
        and training_walls.isdisjoint(evaluation_walls)
        and len(training_sampling) == 64 and len(evaluation_sampling) == 128
        and training_sampling.isdisjoint(evaluation_sampling), "Seed reuse or collision", "scope")
    data_root = (ROOT / config["data_root"]).resolve()
    reports = (ROOT / config["reports_root"]).resolve()
    reference = (ROOT / config["reference_checkpoint"]).resolve()
    checked.require(digest(reference) == REFERENCE, "Original checkpoint changed", "identity")
    plan_path = reports / "plan.json"
    plan, plan_sha = read(plan_path), digest(plan_path)
    checked.require(plan["schema"] == "stage3c-plan-v1" and plan["config_sha256"] == protocol
        and plan["batches"] == batches and plan["evaluation"] == evaluation
        and plan["constraints"] == constraints and plan["frozen_before_formal_collection"] is True
        and plan["reference_sha256"] == REFERENCE and Path(plan["reference_checkpoint"]).resolve() == reference,
        "Frozen plan differs", "identity")
    paths = sorted(path for path in (ROOT / "src/riichi_ai").glob("*.py") if not path.name.startswith("book_scenarios"))
    paths += sorted((ROOT / "tests").glob("test_stage3c*.py"))
    paths += [ROOT / "scripts/stage3c.py", ROOT / "scripts/package_a_stability.py",
              ROOT / "pyproject.toml", ROOT / "requirements-macos-arm64.lock"]
    implementation = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    checked.require(plan["implementation"] == implementation, "Frozen implementation differs", "identity")
    checked.require(plan["versions"] == {"torch": "2.14.1", "numpy": "2.5.4", "riichienv": "0.4.10",
        "mahjong": "2.0.0", "psutil": "7.2.2"}, "Recorded versions differ from pinned stack", "identity")
    budget, markers = audit_budget(config, protocol, reports, data_root, checked, allow_active)
    checked.require(markers["prepare"]["status"] == "completed"
        and markers["prepare"]["implementation"] == implementation, "Preparation did not pass", "scope")
    eval_names = [name for name in markers if name.startswith("eval:")]
    checked.require([name for name in markers if name.startswith("batch:")] == [f"batch:{i}" for i in range(4)]
        and 0 < len(eval_names) <= 32 and eval_names == [f"eval:{i}" for i in range(len(eval_names))],
        "Expected four updates and a fixed ordered evaluation prefix", "scope")
    failed_eval = [name for name in eval_names if markers[name]["status"] == "failed"]
    checked.require(not failed_eval or failed_eval == eval_names[-1:],
                    "Evaluation continued or retried after failed block", "scope")
    prep_seconds = sum(operation["elapsed_seconds"] for operation in read(data_root / "budget-ledger.json")["operations"]
                       if operation["status"] != "active" and operation["id"] < next(
                           operation["id"] for operation in read(data_root / "budget-ledger.json")["operations"]
                           if operation["name"] == "batch:0"))
    checked.require(prep_seconds <= 90, "Formal play began after preparation allowance exceeded", "budget")
    passed_postcall = []
    for name, marker in markers.items():
        if marker["status"] != "completed" or not name.startswith("check:"):
            continue
        if marker["finished_unix"] > markers["batch:0"]["started_unix"]:
            continue
        command = marker.get("command", [])
        xml_path = next((argument.split("=", 1)[1] for argument in command if argument.startswith("--junitxml=")), None)
        if xml_path is None:
            continue
        parsed = ET.parse(ROOT / xml_path).getroot()
        cases = [case for case in parsed.iter("testcase") if "test_stage3c_postcall" in case.get("classname", "")]
        if len(cases) == 8 and marker["implementation"] == implementation and all(not list(case.iter("failure")) and not list(case.iter("error"))
                                  and not list(case.iter("skipped")) for case in cases):
            passed_postcall.append({"operation": name, "xml_path": str((ROOT / xml_path).resolve()),
                                    "xml_sha256": digest(ROOT / xml_path), "cases": 8})
    checked.require(bool(passed_postcall), "No passing mandatory post-call gate before formal play", "scope")
    receipt = plan["preflight_receipt"]
    checked.require(receipt["receipt_sha256"] == digest(reports / "checks-final-preflight.json")
        and receipt["junit_sha256"] == digest(reports / "final-preflight.xml")
        and receipt["postcall_cases"] == 8 and receipt["passing_cases"] >= 43
        and receipt["tests"] == {str(path.relative_to(ROOT)): digest(path)
                                 for path in sorted((ROOT / "tests").glob("test_stage3c*.py"))},
        "Frozen mandatory preflight receipt differs", "scope")
    initial = plan["initial_state"]
    audit_state(initial, 0, protocol, checked)
    boundary = plan["initial_boundary"]
    audit_boundary(boundary, initial, data_root, checked)
    opponent = plan["opponent"]
    checked.require(opponent["checkpoint_sha256"] == REFERENCE
        and opponent["weights_sha256"] == initial["base_weights_sha256"]
        and opponent["decision_rule"] == "stable_semantic_argmax"
        and opponent["model_config"] == initial["model_config"], "Original opponent identity differs", "identity")
    preflight = plan["preflight"]
    checked.require(preflight["valid"] is True and preflight["base_logit_path_exact"] is True
        and preflight["probability_replay_max_absolute_error"] == 0
        and preflight["optimizer_steps"] == 0 and preflight["new_complete_matches"] == 0
        and preflight["state_unchanged"] is True and preflight["state"] == initial,
        "Initial mechanism/gradient preparation failed", "scope")
    state, training, update_summaries = initial, [], []
    for index in range(4):
        batch = read(reports / f"batch-{index}.json")
        marker = markers[f"batch:{index}"]
        checked.require(marker["status"] == "completed" and marker["implementation"] == implementation
            and batch["schema"] == "stage3c-batch-v1" and batch["complete"] is True
            and batch["config_sha256"] == protocol and batch["plan_sha256"] == plan_sha
            and batch["implementation"] == implementation and batch["planned_matches"] == batches[index]["matches"]
            and batch["input_state"] == state and batch["input_boundary"] == boundary
            and batch["opponent"] == opponent and len(batch["matches"]) == 16,
            "Training chain identity/completion mismatch", "update")
        matches = [audit_match(evidence, planned, state, opponent, protocol, data_root, checked)
                   for evidence, planned in zip(batch["matches"], batches[index]["matches"])]
        update_summaries.append(audit_update(batch["update"], matches, state, batches[index]["matches"], checked))
        checked.require(batch["optimizer_steps_executed_delta"] == batch["completed_update_count_delta"] == 1
            and batch["final_in_memory_state"] == batch["output_state"] == batch["update"]["after"]
            and batch["boundary_roundtrip_verified"] is True and batch["sources_unchanged"] is True,
            "Saved update boundary/reload mismatch", "checkpoint")
        state, boundary = batch["output_state"], batch["output_boundary"]
        audit_boundary(boundary, state, data_root, checked)
        training.extend(matches)
    states = {"candidate": state, "baseline": initial}
    boundaries = {"candidate": boundary, "baseline": plan["initial_boundary"]}
    evaluated, block_differences, partial_evaluation = [], [], []
    for index, planned_block in enumerate(evaluation[:len(eval_names)]):
        block = read(reports / f"eval-{index}.json")
        marker = markers[f"eval:{index}"]
        complete = block["complete"]
        checked.require(type(complete) is bool and (marker["status"] == "completed") is complete
            and marker["implementation"] == implementation
            and block["schema"] == "stage3c-evaluation-block-v1"
            and block["config_sha256"] == protocol and block["plan_sha256"] == plan_sha
            and block["implementation"] == implementation and block["planned"] == planned_block
            and block["boundaries"] == boundaries and block["states_before"] == states
            and 0 <= len(block["matches"]) <= 8, "Fixed evaluation arm or block identity changed", "evaluation")
        if complete:
            checked.require(block["states_after"] == states and len(block["matches"]) == 8,
                            "Completed evaluation changed state or omitted matches", "evaluation")
        else:
            checked.require(index == len(eval_names) - 1 and marker["failure"] is not None
                and block["mean_utility_difference"] is None,
                "Incomplete block reported endpoint or work continued", "evaluation")
        outcomes, pairs, differences = [], [], []
        for pair_index, pair in enumerate(planned_block["pairs"]):
            arms = {}
            for arm_index, arm in enumerate(("candidate", "baseline")):
                evidence_index = pair_index * 2 + arm_index
                if evidence_index >= len(block["matches"]):
                    continue
                evidence = block["matches"][evidence_index]
                checked.require(evidence["arm"] == arm and evidence["learner_seat"] == pair["learner_seat"],
                                "Evaluation arm/seat order changed", "evaluation")
                planned = {"match_index": pair[f"{arm}_match_index"], "seed": planned_block["wall_seed"],
                           "sampling_seed": pair["sampling_seed"], "learner_seat": pair["learner_seat"]}
                match = audit_match(evidence, planned, states[arm], opponent, protocol, data_root, checked,
                                    collection_role="evaluation", collection_batch=4 + index)
                arms[arm] = match
                evaluated.append({"arm": arm, **match})
                outcomes.append({"arm": arm, "learner_seat": pair["learner_seat"],
                                 "utility": match["learner_utility"], "rank": match["rank"]})
            if len(arms) != 2:
                continue
            candidate, baseline = arms["candidate"], arms["baseline"]
            common = min(len(candidate["ordinal_wall_fingerprints"]), len(baseline["ordinal_wall_fingerprints"]))
            checked.require(common > 0 and candidate["ordinal_wall_fingerprints"][:common]
                == baseline["ordinal_wall_fingerprints"][:common], "Actual paired physical walls differ", "evaluation")
            difference = candidate["learner_utility"] - baseline["learner_utility"]
            differences.append(difference)
            pairs.append({"learner_seat": pair["learner_seat"], "common_hands": common,
                          "common_walls_verified": True, "utility_difference": difference})
        if complete:
            mean = math.fsum(differences) / 4
            checked.require(block["outcomes"] == outcomes and block["pairs"] == pairs
                and block["mean_utility_difference"] == mean, "Paired block arithmetic differs", "evaluation")
            block_differences.append(mean)
        else:
            checked.require(block["outcomes"] == outcomes[:len(block["outcomes"])]
                and len(block["outcomes"]) <= len(outcomes)
                and block["pairs"] == pairs[:len(block["pairs"])] and len(block["pairs"]) <= len(pairs),
                "Incomplete worker report is not a valid persisted-outcome prefix", "evaluation")
            partial_evaluation.append({"block": index, "persisted_complete_matches": len(outcomes),
                "raw_completed_pairs": len(pairs), "recorded_outcomes": len(block["outcomes"]),
                "recorded_pairs": len(block["pairs"]), "included_in_block_summary": False,
                "failure": marker["failure"], "worker_final_state_present": "states_after" in block,
                "report_sha256": digest(reports / f"eval-{index}.json"),
                "failure_receipt_sha256": digest(reports / f"eval-attempt-{index}.json")})
    experiment_complete = len(block_differences) == 32 and not failed_eval
    mean = interval = None
    if experiment_complete:
        result = read(reports / "evaluation.json")
        mean = math.fsum(block_differences) / 32
        width = 6 * math.sqrt(math.log(40) / (2 * 32))
        interval = [max(-3., mean - width), min(3., mean + width)]
        checked.require(result["blocks"] == 32 and result["matches"] == 256
            and result["block_differences"] == block_differences and result["mean_utility_difference"] == mean
            and close(result["halfwidth"], width, 1e-14)
            and all(close(a, b, 1e-14) for a, b in zip(result["confidence_interval"], interval))
            and result["candidate_promoted"] is False, "Fixed sample inference arithmetic differs", "evaluation")
        for arm in ("candidate", "baseline"):
            mean_rank = math.fsum(match["rank"] for match in evaluated if match["arm"] == arm) / 128
            checked.require(result[f"{arm}_mean_rank"] == mean_rank, "Evaluation rank summary differs", "evaluation")
    else:
        checked.require(not (reports / "evaluation.json").exists(),
                        "Incomplete evaluation must not have the fixed-sample endpoint", "evaluation")
    all_matches = training + evaluated
    files = [file for match in all_matches for file in match["files"]]
    checked.require(len(files) == len(set(files)) == 2 * len(all_matches),
                    "Duplicate or omitted trajectory artifacts", "identity")
    actual = {str(path.resolve()) for directory in (data_root / "trajectories", data_root / "evaluation")
              for path in directory.rglob("*") if path.is_file()}
    checked.require(actual == set(files), "Extra/unreported trajectory artifacts", "scope")
    checked.require({path.name for path in (data_root / "boundaries").glob("*.pt")}
        == {"initial.pt", "batch-0.pt", "batch-1.pt", "batch-2.pt", "batch-3.pt"},
        "Unexpected checkpoint selection or extra update", "scope")
    actual_bytes = sum(path.stat().st_size for directory in (data_root, reports, ROOT / "Books/stage3c")
                       for path in directory.rglob("*") if path.is_file())
    checked.require(actual_bytes < config["artifact_bytes"], "Current artifact cap exceeded", "budget")
    totals = {key: {kind: sum(match[key][kind] for match in training) for kind in CALLS}
              for key in ("requested_calls", "declared_calls", "completed_calls", "robbed_kans")}
    return {"schema": "stage3c-independent-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True, "engineering_evidence_pass": experiment_complete,
        "training_engineering_evidence_pass": True, "experiment_complete": experiment_complete,
        "config_sha256": protocol, "plan_sha256": plan_sha,
        "implementation": implementation, "helper_sha256": digest(ROOT / "scripts/stage3a_audit.py"),
        "checks": dict(sorted(checked.counts.items())), "check_count": sum(checked.counts.values()),
        "budget": {**budget, "current_artifact_bytes": actual_bytes}, "mandatory_postcall_gate": passed_postcall,
        "training_matches": 64, "optimizer_updates": 4, "evaluation_matches": len(evaluated),
        "evaluation_blocks": len(block_differences), "evaluation_blocks_attempted": len(eval_names),
        "partial_evaluation": partial_evaluation,
        "completed_block_differences_descriptive": block_differences,
        "completed_block_mean_descriptive": (math.fsum(block_differences) / len(block_differences) if block_differences else None),
        "initial_state": initial, "final_state": state, "base_unchanged": True,
        "training_learner_rows": sum(match["learner_rows"] for match in training),
        "training_calls": totals, "training_post_call_rows": sum(match["post_call_rows"] for match in training),
        "updates": update_summaries, "mean_utility_difference": mean, "confidence_interval": interval,
        "candidate_promoted": False,
        "limitations": ["Complete saved evidence integrity and objective arithmetic; no competitive-strength certification.",
            "Boundary hashes and reload identities are verified; parameter forwards, gradients and replay remain worker evidence.",
            "Physical wall commitments are compared across arms; this auditor does not reconstruct unseen walls from hashes.",
            "Memory maxima are sampled process-tree RSS.",
            "If current_audit_lease is present, the final audit's own completion is not included in its closed-prefix budget."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/stage3c.json"))
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-active-audit", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve prior audit evidence; use a new output")
    result = audit(args.config, args.expected_config_sha256, allow_active=args.allow_active_audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(canonical({"valid": result["valid"], "checks": result["check_count"],
                     "engineering_evidence_pass": result["engineering_evidence_pass"]}))


if __name__ == "__main__":
    main()
