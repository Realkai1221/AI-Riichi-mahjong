"""Fixed saved-training diagnosis: no new matches, optimizer steps or evaluation inputs.

Preparation uses only the standard library. Run imports numerical dependencies only
inside run(), verifies the frozen plan and every consumed artifact, and uses the
historical actor encoding without passing the auditor log into a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "stage3f-saved-training-diagnosis-v1"
CALLS = ("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN")
WINS = ("RON", "TSUMO")
FEATURE_KEYS = ("tiles", "globals", "history", "history_mask", "action_tokens", "candidate_efficiency")
SOURCES = ("scripts/stage3f_diagnose.py", "tests/test_stage3f_diagnosis.py", "src/riichi_ai/stage3c.py",
           "src/riichi_ai/stage3a.py", "src/riichi_ai/model.py", "src/riichi_ai/features.py",
           "src/riichi_ai/efficiency.py", "src/riichi_ai/semantics.py", "src/riichi_ai/player.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def artifact(path, expected=None):
    path = Path(path).resolve()
    actual = digest(path)
    require(expected is None or expected == actual, f"Artifact hash mismatch: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def verify_artifact(item):
    require(artifact(item["path"], item["sha256"]) == item, "Artifact size changed")


def row_identity(item):
    return (item["batch_index"], item["match_index"], item["row_index"])


def strata(row):
    return {"win_vs_call": row["call_candidates"] > 0 and any(kind in WINS for kind in row["action_kinds"]),
            "selected_multiple_call": row["call_candidates"] > 1 and row["selected_kind"] in CALLS}


def select_rows(candidates, limit=256):
    require(type(limit) is int and 1 <= limit <= 256, "Selection limit must be 1 to 256")
    require(len({row_identity(item) for item in candidates}) == len(candidates), "Duplicate row identity")
    required = [item for item in candidates if any(item["strata"].values())]
    require(len(required) <= limit, "Required strata exceed fixed row ceiling")
    remaining = [item for item in candidates if not any(item["strata"].values())]
    remaining.sort(key=lambda item: hashlib.sha256(
        ("stage3f-call-row-v1:" + ":".join(map(str, row_identity(item)))).encode()).hexdigest())
    chosen = required + remaining[:limit - len(required)]
    return sorted(chosen, key=row_identity)


def validate_record(record, batch_index, policy_id):
    require(record.get("collection_role") == "training" and record.get("campaign_phase") == "training"
            and record.get("training_eligible") is True, "Evaluation or ineligible match rejected")
    require(record.get("completed") is True and record.get("terminated") is True
            and record.get("terminal_result_verified") is True and record.get("failure") is None
            and record.get("truncated") is False and record.get("interrupted") is False,
            "Incomplete or failed match rejected")
    require(record.get("batch_index") == batch_index and record.get("policy_id") == policy_id,
            "Behavior checkpoint association differs")
    require(record.get("learner_utility") in (-1.5, -.5, .5, 1.5), "Invalid complete-match utility")
    require(record.get("learner_seat") in range(4), "Invalid learner seat")
    for index, row in enumerate(record["learner_rows"]):
        require(row.get("row_index") == index and row.get("batch_index") == batch_index
                and row.get("match_index") == record["match_index"] and row.get("policy_id") == policy_id
                and row.get("seat") == record["learner_seat"] and row.get("training_eligible") is True
                and row.get("eligible") is True and row.get("fallback") is False
                and row.get("overridden") is False, "Wrong-seat, altered or ineligible learner row")


def prepare():
    candidates, matches, batches, inputs = [], {}, {}, {}
    for relative in SOURCES:
        item = artifact(ROOT / relative)
        inputs[item["path"]] = item
    # The old inventory is provenance only. No evaluation file is opened.
    item = artifact(ROOT / "reports/stage3e/manifest.json")
    inputs[item["path"]] = item
    for index in range(4, 32):
        path = ROOT / f"reports/stage3e/batch-{index}.json"
        item = artifact(path)
        inputs[item["path"]] = item
        batch = read(path)
        require(batch.get("complete") is True and batch.get("batch_index") == index,
                "First fixed training campaign is incomplete")
        require(len(batch["matches"]) == 16 and batch["input_state"]["updates"] == index,
                "Training batch horizon differs")
        for name in ("input_boundary", "output_boundary"):
            source = batch[name]
            item = artifact(source["path"], source["sha256"])
            inputs[item["path"]] = item
        batches[str(index)] = {"report": str(path.resolve()), "input_boundary": batch["input_boundary"]["path"],
                               "output_boundary": batch["output_boundary"]["path"],
                               "protocol_sha256": batch["behavior_protocol_sha256"],
                               "policy_id": batch["input_state"]["policy_id"]}
        for evidence in batch["matches"]:
            item = artifact(evidence["json_path"], evidence["json_sha256"])
            inputs[item["path"]] = item
            record = read(item["path"])
            validate_record(record, index, batches[str(index)]["policy_id"])
            match_id = str(record["match_index"])
            require(match_id not in matches, "Duplicate match")
            matches[match_id] = {"batch_index": index, "json_path": item["path"],
                                 "npz_path": evidence["npz_path"], "npz_sha256": evidence["npz_sha256"]}
            for row in record["learner_rows"]:
                if row["call_candidates"]:
                    candidates.append({"batch_index": index, "match_index": record["match_index"],
                        "row_index": row["row_index"], "encoded_sha256": row["encoded_sha256"],
                        "strata": strata(row)})
    selected = select_rows(candidates)
    full_matches = sorted(int(key) for key, value in matches.items() if value["batch_index"] == 4)
    selected_matches = sorted(set(full_matches) | {row["match_index"] for row in selected})
    for match_id in selected_matches:
        source = matches[str(match_id)]
        item = artifact(source["npz_path"], source["npz_sha256"])
        inputs[item["path"]] = item
    return {"schema": SCHEMA, "role": "training development diagnosis; not confirmation evidence",
        "selection_rule": "All win-versus-call and selected-multiple-call rows, then SHA256-ranked remainder",
        "row_limit": 256, "candidate_rows": len(candidates), "selected_rows": selected,
        "stratum_population_counts": {name: sum(item["strata"][name] for item in candidates)
                                      for name in ("win_vs_call", "selected_multiple_call")},
        "full_match_gradient_batch": 4, "full_match_ids": full_matches,
        "full_match_gradient_rule": "All eligible optional rows of all 16 matches; sum decisions, divide by 16",
        "selected_row_rule": "Selected rows describe output sensitivity only; not an unbiased gradient estimate",
        "matches": matches, "batches": batches, "inputs": sorted(inputs.values(), key=lambda item: item["path"]),
        "max_seconds": 240, "threads": 1, "device": "cpu", "optimizer_steps": 0,
        "new_matches": 0, "evaluation_rows": 0}


def validate_plan(plan):
    require(plan.get("schema") == SCHEMA and plan.get("row_limit") == 256
            and plan.get("full_match_gradient_batch") == 4 and plan.get("max_seconds") == 240
            and plan.get("threads") == 1 and plan.get("device") == "cpu"
            and plan.get("optimizer_steps") == plan.get("new_matches") == plan.get("evaluation_rows") == 0,
            "Fixed diagnostic scope changed")
    require(len(plan["selected_rows"]) <= 256
            and len({row_identity(row) for row in plan["selected_rows"]}) == len(plan["selected_rows"]),
            "Selected row ceiling or identity violated")
    require(len(plan["full_match_ids"]) == 16 and len(set(plan["full_match_ids"])) == 16,
            "Complete first batch required")
    require(all(plan["matches"][str(index)]["batch_index"] == 4 for index in plan["full_match_ids"]),
            "Full-match batch changed")


def visible_context(record, row, encoded):
    """Export only whitelisted public/own information, never full engine logs."""
    values, tiles = encoded["globals"], encoded["tiles"]
    seat = row["seat"]
    hand_starts = [event for event in record["auditor_log"] if event.get("type") == "start_kyoku"]
    require(1 <= row["hand_id"] <= len(hand_starts), "Missing explicit hand ordinal")
    start = hand_starts[row["hand_id"] - 1]
    scores = [None] * 4
    for relative in range(4):
        scores[(seat + relative) % 4] = round(float(values[relative]) * 50000)
    names = [f"{n}{suit}" for suit in "mps" for n in range(1, 10)] + ["E", "S", "W", "N", "P", "F", "C"]
    own = [{"tile": names[index], "count": round(float(tiles[index, 0]) * 4),
            "red_count": round(float(tiles[index, 1]))} for index in range(34) if tiles[index, 0] > 0]
    return {"hand_ordinal": row["hand_id"], "round_wind": start["bakaze"], "round_number": start["kyoku"],
        "honba": start["honba"], "dealer": start["oya"], "learner_seat": seat,
        "current_scores_absolute_seats": scores, "current_riichi_sticks": round(float(values[37]) * 10),
        "own_concealed_tiles": own, "own_meld_count": row["own_meld_count"],
        "own_open_meld_count": row["own_open_meld_count"],
        "legal_semantic_keys": row["semantic_keys"], "selected_semantic_key": row["selected_semantic_key"],
        "selected_kind": row["selected_kind"], "probabilities": row["probabilities"],
        "uniform_draw": row["uniform_draw"], "requested_call_resolved": row["requested_call_resolved"],
        "interpretation": "Visible-state review only. No counterfactual outcome or optimality claim."}


def run(plan, max_seconds):
    started = time.monotonic()
    deadline = started + max_seconds
    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError("Fixed saved-data diagnosis deadline reached")
    validate_plan(plan)
    require(0 < max_seconds <= plan["max_seconds"], "Run budget exceeds frozen limit")
    for item in plan["inputs"]:
        tick()
        verify_artifact(item)
    import numpy as np
    import torch
    from riichi_ai import stage3c as learner
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    def state(index, output=False):
        source = plan["batches"][str(index)]
        result = learner.load_boundary(source["output_boundary" if output else "input_boundary"],
            expected_protocol_sha256=source["protocol_sha256"], device="cpu")
        require(result.updates == index + int(output), "Boundary update differs")
        return result
    def load_match(match_id):
        entry = plan["matches"][str(match_id)]
        record = read(entry["json_path"])
        validate_record(record, entry["batch_index"], plan["batches"][str(entry["batch_index"])]["policy_id"])
        return record, np.load(entry["npz_path"], allow_pickle=False)
    def encoded_row(record, archive, index):
        encoded = {key: archive[f"row_{index:05d}__{key}"] for key in FEATURE_KEYS}
        require(learner.encoded_hash(encoded) == record["learner_rows"][index]["encoded_sha256"],
                "Encoded actor row changed")
        return encoded
    def replay(model, encoded, row):
        tick()
        output = model.distribution(encoded, "cpu")
        actual = output["probabilities"].detach().numpy()
        require(np.array_equal(actual, np.asarray(row["probabilities"], dtype=np.float64)),
                "Actual behavior probabilities did not replay exactly")
        require(math.log(float(actual[row["selected_index"]])) == row["log_probability"],
                "Actual behavior selected log probability differs")
        return output
    def vector(model, prefix):
        tensors = [parameter.grad.detach().double().flatten() if parameter.grad is not None
                   else torch.zeros(parameter.numel(), dtype=torch.float64)
                   for name, parameter in model.heads.named_parameters() if name.startswith(prefix)]
        return torch.cat(tensors)
    def norm(value):
        require(bool(torch.isfinite(value).all()), "Non-finite diagnostic vector")
        return float(torch.linalg.vector_norm(value))
    def cosine(left, right):
        denominator = norm(left) * norm(right)
        return float(torch.dot(left, right)) / denominator if denominator else None
    initial = state(4)
    initial_identity = learner.state_manifest(initial)
    initial.optimizer.zero_grad(set_to_none=True)
    match_vectors = {"gate": [], "call": []}
    match_metrics, total_loss, optional, forced, gradient_rows = [], 0., 0, 0, 0
    for match_id in plan["full_match_ids"]:
        tick()
        record, archive = load_match(match_id)
        before = {key: vector(initial.model, key) for key in match_vectors}
        count, local_loss = 0, 0.
        try:
            for index, row in enumerate(record["learner_rows"]):
                encoded = encoded_row(record, archive, index)
                output = replay(initial.model, encoded, row)
                if row["forced"]:
                    forced += 1
                    continue
                optional += 1
                count += 1
                loss = -output["log_probabilities"][row["selected_index"]] * record["learner_utility"] / 16
                require(bool(torch.isfinite(loss)), "Non-finite reconstructed loss")
                local_loss += float(loss.detach())
                if loss.requires_grad:
                    loss.backward()
                    gradient_rows += 1
            local = {key: vector(initial.model, key) - before[key] for key in match_vectors}
            for key in local:
                match_vectors[key].append(local[key])
            match_metrics.append({"match_index": match_id, "utility": record["learner_utility"],
                "optional_rows": count, "loss_contribution_divided_by_16": local_loss,
                "gate_gradient_l2": norm(local["gate"]), "conditional_call_gradient_l2": norm(local["call"])})
            total_loss += local_loss
        finally:
            archive.close()
    batch = read(plan["batches"]["4"]["report"])
    expected = batch["update"]
    require(optional == expected["optional_learner_rows"] and forced == expected["forced_learner_rows"]
            and gradient_rows == expected["gradient_eligible_rows"], "Full-match gradient population differs")
    require(math.isclose(total_loss, expected["differentiable_loss"], abs_tol=1e-10, rel_tol=1e-10),
            "Full-match reconstructed loss differs")
    components = {}
    for key in match_vectors:
        whole = vector(initial.model, key)
        parts = match_vectors[key]
        magnitude_sum = sum(norm(part) for part in parts)
        actual_norm = norm(whole)
        expected_norm = expected["gradient_components"]["gate" if key == "gate" else "call_head"]["l2_norm"]
        require(math.isclose(actual_norm, expected_norm, abs_tol=1e-7, rel_tol=1e-6),
                "Recorded gradient component did not reproduce")
        components[key] = {"gradient_l2": actual_norm, "recorded_gradient_l2": expected_norm,
            "sum_match_gradient_l2": magnitude_sum,
            "cancellation_ratio_norm_sum_over_sum_norms": actual_norm / magnitude_sum if magnitude_sum else None,
            "match_cosines_to_total": [cosine(part, whole) for part in parts],
            "pairwise_match_cosines": [cosine(left, right) for i, left in enumerate(parts) for right in parts[i + 1:]],
            "interpretation": "Descriptive within-batch cancellation; not an estimate of population gradient variance."}
    raw_norm = math.sqrt(sum(norm(vector(initial.model, key)) ** 2 for key in match_vectors))
    require(math.isclose(raw_norm, expected["gradient_norm_before_clip"], abs_tol=1e-7, rel_tol=1e-6),
            "Recorded gradient norm differs")
    applied_norm = float(torch.nn.utils.clip_grad_norm_(initial.model.heads.parameters(), 1., error_if_nonfinite=True))
    require(math.isclose(applied_norm, raw_norm, abs_tol=1e-7, rel_tol=1e-6), "Clip norm differs")
    after = state(4, output=True)
    optimizer_checks = []
    for (name, parameter), (after_name, parameter_after) in zip(initial.model.heads.named_parameters(), after.model.heads.named_parameters()):
        require(name == after_name, "Parameter order differs")
        before_moment = initial.optimizer.state[parameter]
        after_moment = after.optimizer.state[parameter_after]
        gradient = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        first = before_moment["exp_avg"] * .9 + gradient * .1
        second = before_moment["exp_avg_sq"] * .999 + gradient.square() * .001
        require(torch.allclose(first, after_moment["exp_avg"], atol=1e-7, rtol=1e-5)
                and torch.allclose(second, after_moment["exp_avg_sq"], atol=1e-9, rtol=1e-5),
                "Saved Adam moments differ from reconstructed clipped gradient")
        step = float(after_moment["step"])
        require(step == float(before_moment["step"]) + 1 == 5, "Saved Adam step differs")
        predicted_delta = -1e-4 * (after_moment["exp_avg"] / (1 - .9 ** step)) / (
            (after_moment["exp_avg_sq"] / (1 - .999 ** step)).sqrt() + 1e-8)
        actual_delta = parameter_after.detach() - parameter.detach()
        # fp32 application rounds tiny additions to existing parameter magnitudes.
        predicted_after = parameter.detach() + predicted_delta
        rounding_error = float((predicted_after - parameter_after.detach()).abs().max())
        require(rounding_error <= 6e-8, "Saved Adam parameter movement differs beyond fp32 rounding")
        optimizer_checks.append({"parameter": name, "actual_delta_l2": norm(actual_delta.double().flatten()),
            "actual_delta_max_abs": float(actual_delta.abs().max()), "formula_parameter_max_abs_error": rounding_error,
            "first_moment_l2": norm(after_moment["exp_avg"].double().flatten()),
            "second_moment_l2": norm(after_moment["exp_avg_sq"].double().flatten())})
    # A separate fixed stratified view checks actual historical behavior and update-4/32 common states.
    endpoint = state(31, output=True)
    endpoint_parameters = dict(endpoint.model.heads.named_parameters())
    parameter_movement = {}
    for prefix in ("gate", "call"):
        pairs = [(name, parameter) for name, parameter in initial.model.heads.named_parameters() if name.startswith(prefix)]
        initial_vector = torch.cat([parameter.detach().double().flatten() for _, parameter in pairs])
        final_vector = torch.cat([endpoint_parameters[name].detach().double().flatten() for name, _ in pairs])
        delta = final_vector - initial_vector
        parameter_movement[prefix] = {"initial_l2": norm(initial_vector), "final_l2": norm(final_vector),
            "delta_l2": norm(delta), "delta_max_abs": float(delta.abs().max()),
            "delta_cosine_to_first_batch_loss_gradient": cosine(delta, vector(initial.model, prefix))}
    selected_results, call_instead_win = [], []
    for batch_index in sorted({row["batch_index"] for row in plan["selected_rows"]}):
        behavior = initial if batch_index == 4 else state(batch_index)
        selected = [item for item in plan["selected_rows"] if item["batch_index"] == batch_index]
        for match_id in sorted({item["match_index"] for item in selected}):
            record, archive = load_match(match_id)
            try:
                for item in [item for item in selected if item["match_index"] == match_id]:
                    row = record["learner_rows"][item["row_index"]]
                    require(strata(row) == item["strata"] and row["call_candidates"] > 0,
                            "Selected stratum differs")
                    encoded = encoded_row(record, archive, item["row_index"])
                    require(row["encoded_sha256"] == item["encoded_sha256"], "Selection changed")
                    with torch.inference_mode():
                        replay(behavior.model, encoded, row)
                        left = initial.model.distribution(encoded, "cpu")
                        right = endpoint.model.distribution(encoded, "cpu")
                    delta = right["probabilities"] - left["probabilities"]
                    selected_results.append({**item, "selected_kind": row["selected_kind"],
                        "behavior_call_mass": row["call_mass"], "utility": record["learner_utility"],
                        "call_mass_update_4": float(left["call_mass"]), "call_mass_update_32": float(right["call_mass"]),
                        "call_mass_change": float(right["call_mass"] - left["call_mass"]),
                        "probability_max_abs_change": float(delta.abs().max()),
                        "probability_l1_change": float(delta.abs().sum()),
                        "conditional_call_max_abs_change": float((right["conditional_calls"] - left["conditional_calls"]).abs().max())})
                    if item["strata"]["win_vs_call"] and row["selected_kind"] in CALLS:
                        call_instead_win.append({**item, "visible_context": visible_context(record, row, encoded),
                            "eventual_match_utility_hindsight_only": record["learner_utility"]})
            finally:
                archive.close()
        if behavior is not initial:
            del behavior
    require(learner.state_manifest(initial) == initial_identity, "Diagnosis mutated model or optimizer state")
    for item in plan["inputs"]:
        tick()
        verify_artifact(item)
    scalar = lambda field: [row[field] for row in selected_results]
    movement = {field: {"mean": sum(scalar(field)) / len(selected_results), "min": min(scalar(field)), "max": max(scalar(field))}
                for field in ("call_mass_change", "probability_max_abs_change", "probability_l1_change", "conditional_call_max_abs_change")}
    updates = [read(plan["batches"][str(index)]["report"])["update"] for index in range(4, 32)]
    return {"schema": SCHEMA, "valid": True, "role": plan["role"], "elapsed_seconds": time.monotonic() - started,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "device": "cpu", "threads": 1, "optimizer_steps": 0, "new_matches": 0, "evaluation_rows": 0,
        "source_files_unchanged": True, "model_and_optimizer_unchanged": True,
        "full_batch": {"batch_index": 4, "matches": 16, "optional_rows": optional, "forced_rows": forced,
            "gradient_rows": gradient_rows, "loss": total_loss, "recorded_loss": expected["differentiable_loss"],
            "probability_replay_max_absolute_error": 0., "selected_log_replay_max_absolute_error": 0.,
            "gradient_norm": raw_norm, "clip_coefficient": min(1., 1. / (applied_norm + 1e-6)),
            "components": components, "per_match": match_metrics, "optimizer_checks": optimizer_checks},
        "recorded_training_updates": {"count": len(updates), "clipped_updates": sum(update["gradient_norm_before_clip"] > 1 for update in updates),
            "gradient_norm_min": min(update["gradient_norm_before_clip"] for update in updates),
            "gradient_norm_max": max(update["gradient_norm_before_clip"] for update in updates),
            "gradient_components": [{"batch_index": index, "gate_l2": update["gradient_components"]["gate"]["l2_norm"],
                "conditional_call_l2": update["gradient_components"]["call_head"]["l2_norm"]} for index, update in zip(range(4, 32), updates)]},
        "parameter_movement_update_4_to_32": parameter_movement,
        "selected_row_count": len(selected_results), "selected_rows": selected_results,
        "selected_common_state_movement": movement, "call_instead_of_win": call_instead_win,
        "limitations": ["One complete training batch reconstructs the mechanism, not long-run gradient variance or strength.",
            "The 256-row stratified sample is deliberately enriched for rare events and is not a representative policy-gradient estimate.",
            "Checkpoint differences on saved states do not estimate counterfactual match outcomes.",
            "No baseline, learning rate, model scale or target change is tested or chosen from evaluation results."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=240)
    args = parser.parse_args()
    require(not args.output.exists(), "Output exists; refusing overwrite or replacement")
    if args.prepare:
        result = prepare()
    else:
        require(args.plan is not None, "Run requires frozen plan")
        plan_hash = digest(args.plan)
        result = run(read(args.plan), args.max_seconds)
        require(digest(args.plan) == plan_hash, "Plan changed during run")
        result["plan_sha256"] = plan_hash
    write_new(args.output, result)
    print(json.dumps({"schema": SCHEMA, "mode": "prepare" if args.prepare else "run",
                      "output": str(args.output.resolve()), "sha256": digest(args.output),
                      "valid": result.get("valid"), "selected_rows": len(result.get("selected_rows", []))}))


if __name__ == "__main__":
    main()
