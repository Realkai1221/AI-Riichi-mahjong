"""Model-free integrity, accounting, safety replay and paired endpoint audit.

This reads JSON and hashes checkpoints/NPZ bytes; it never imports a numerical
library or deserializes a checkpoint or array. Validity and scientific acceptance
are separate: a preserved failed/stopped run is evidence, never a replacement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from package_a_diagnosis_audit import Audit, close, digest, index_hash, read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/package_a_continuation.json"
CONSTRAINTS = {"new_matches": 0, "action_test_predictions": 0,
               "test_split_loading": False, "external_acquisition": False,
               "reinforcement_learning": False}
ATTEMPTS = [(202610171, .0003), (202610171, .0001), (202610172, .0001), (202610172, .0003)]
DIAGNOSIS_MANIFEST_SHA256 = "f847120c42182001c740ca1c011ba537797f00fee0b4859fb1f65d1857960108"
DIAGNOSIS_PLAN_SHA256 = "5a753d8b4593854aee81b0857c10104105ba7347e1731cb5c52a93bbca22d858"
SAFETY_KEYS = ("configuration", "warmup_reference_samples", "references", "events", "tripped", "scale_alarm")


def plain_state_digest(value):
    """Independently reproduce recursive digest for JSON-only safety state."""
    h = hashlib.sha256()
    def add(item):
        if isinstance(item, dict):
            h.update(b"mapping{")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                add(key); add(item[key])
            h.update(b"}")
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode() + b"[")
            for child in item: add(child)
            h.update(b"]")
        else:
            h.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
    add(value)
    return h.hexdigest()


def replay_safety(config, history, diagnostics, initial=None):
    """Independent stdlib replay of fixed operational threshold formulas."""
    state = deepcopy(initial) if initial is not None else {
        "configuration": config, "warmup_reference_samples": {}, "references": {},
        "streaks": {}, "events": [], "tripped": False, "scale_alarm": False}
    if state["configuration"] != config:
        raise ValueError("Safety configuration changed")
    by_step = {d["step"]: d for d in diagnostics}
    if len(by_step) != len(diagnostics):
        raise ValueError("Duplicate safety diagnostic step")
    def condition(step, key, value, threshold, persistence):
        if not math.isfinite(value): raise ValueError("Nonfinite safety value")
        exceeded = value > threshold
        streak = state["streaks"].get(key, 0) + 1 if exceeded else 0
        state["streaks"][key] = streak
        if exceeded:
            abort = streak >= persistence
            state["events"].append(dict(step=step, metric=key, value=value, threshold=threshold,
                streak=streak, required_persistence=persistence, abort=abort))
            state["tripped"] |= abort
            state["scale_alarm"] |= abort and key.startswith("scale:")
    for row in history:
        step = row["step"]
        condition(step, "gradient_norm", row["gradient_norm"], config["gradient_threshold"], config["gradient_persistence"])
        condition(step, "policy_ce", row["policy_ce"], config["policy_ce_threshold"], config["policy_ce_persistence"])
        if step not in by_step: continue
        diagnostic = by_step[step]
        for population in ("sampled", "fixed"):
            for metric in ("context_rms", "legal_logit_abs", "centered_legal_logit_abs"):
                for statistic in ("mean", "maximum"):
                    key = f"scale:{population}:{metric}:{statistic}"
                    value = diagnostic[population][metric][statistic]
                    if not math.isfinite(value): raise ValueError("Nonfinite safety diagnostic")
                    if step <= config["warmup_steps"]:
                        state["warmup_reference_samples"].setdefault(key, []).append(value)
                        continue
                    if key not in state["references"]:
                        samples = state["warmup_reference_samples"].get(key, [])
                        if not samples: raise ValueError("Missing historical warmup reference")
                        state["references"][key] = max(config["scale_floor"], float(median(samples)))
                    threshold = config["scale_factor"] * state["references"][key]
                    if metric == "legal_logit_abs": threshold = max(config["raw_logit_absolute_floor"], threshold)
                    condition(step, key, value, threshold, config["persistence"])
    return state


def endpoint_table(evidence):
    table = {}
    for label, split, population in (("training_teacher", "training", "teacher"),
                                     ("validation_teacher", "validation", "teacher"),
                                     ("validation_student", "validation", "student")):
        fit = evidence["fit"][split]
        optional = fit["groups"][f"stratum:{population}:optional"]
        discard = fit["groups"][f"stratum:{population}:optional_teacher_discard"]
        table[label] = {"optional_rows": optional["rows"], "discard_rows": discard["rows"],
            "smooth_kl_f64": optional["metrics"]["smooth_kl_f64"]["mean"],
            "original_kl_f64": optional["metrics"]["original_kl_f64"]["mean"],
            "nonbest_mass_f32": optional["metrics"]["nonbest_mass_f32"]["mean"],
            "target_nonbest_mass": optional["metrics"]["target_nonbest_mass"]["mean"],
            "discard_agreement": discard["metrics"]["tie_agreement"]["mean"],
            "shanten_worsening_count": discard["severity"]["shanten_increments"]["count"],
            "shanten_worsening_rate": discard["severity"]["shanten_increments"]["rate"]}
    return table


def compare_pair(baseline, same, lower, *, eligible):
    """Componentwise preregistered criteria; no new composite quality score."""
    if same is None or lower is None:
        return {"eligible": False, "components": {}, "exploratory_passed": False,
                "reason": "Both complete endpoints are required"}
    components = {}
    for cohort in ("training_teacher", "validation_teacher"):
        components[f"{cohort}:smooth_kl_below_same"] = lower[cohort]["smooth_kl_f64"] < same[cohort]["smooth_kl_f64"]
    for cohort in ("validation_teacher", "validation_student"):
        for reference_name, reference in (("same", same), ("baseline", baseline)):
            components[f"{cohort}:agreement_at_least_{reference_name}"] = lower[cohort]["discard_agreement"] >= reference[cohort]["discard_agreement"]
            components[f"{cohort}:shanten_rate_at_most_{reference_name}"] = lower[cohort]["shanten_worsening_rate"] <= reference[cohort]["shanten_worsening_rate"]
    return {"eligible": bool(eligible), "components": components,
            "exploratory_passed": bool(eligible and all(components.values())),
            "baseline": baseline, "same": same, "lower": lower}


def audit(expected_config_sha256):
    checked = Audit()
    identity = digest(CONFIG)
    checked.check(len(expected_config_sha256) == 64 and identity == expected_config_sha256,
                  "Current config does not match explicitly supplied frozen SHA-256", "identity")
    config = read(CONFIG)
    reports, external = (ROOT / config["reports_root"]).resolve(), Path(config["data_root"])
    plan_path = reports / "plan.json"
    plan, plan_sha = read(plan_path), digest(plan_path)
    checked.check(config["constraints"] == plan["constraints"] == CONSTRAINTS, "Scope changed")
    checked.check((config["limit_seconds"], config["maximum_attempts"], config["starting_step"],
        config["additional_steps"], config["terminal_step"], config["memory_bytes"], config["artifact_bytes"])
        == (900, 4, 18500, 5000, 23500, 32 * 1024**3, 500_000_000), "Authorized limits changed")
    checked.check((config["reserve_seconds"], config["test_allowance_seconds"], config["prepare_allowance_seconds"],
        config["per_attempt_allowance_seconds"], config["epsilon"], config["batch_size"], config["device"])
        == (45, 120, 45, 180, .01, 256, "mps"), "Frozen operational settings changed")
    attempts = config["attempts"]
    checked.check([(a["seed"], a["lr"]) for a in attempts] == ATTEMPTS and plan["attempts"] == attempts,
                  "Four-arm order changed")
    checked.check(plan["config_sha256"] == identity and plan["frozen_before_formal_updates"] is True,
                  "Missing frozen pre-update identity", "identity")
    for path, sha in plan["implementation"].items():
        checked.check(digest(ROOT / path) == sha, f"Implementation changed: {path}", "identity")
    diagnosis = read(ROOT / config["diagnosis_plan"])
    checked.check(digest(ROOT / config["diagnosis_plan"]) == config["diagnosis_plan_sha256"]
        == plan["diagnosis_plan_sha256"] == DIAGNOSIS_PLAN_SHA256, "Prior frozen cohort plan changed", "identity")
    for key in ("train_indices", "validation_indices", "training_manifest", "validation_manifest"):
        checked.check(plan[key] == diagnosis[key], f"Prior cohort identity changed: {key}", "cohorts")
    checked.check(len(plan["train_indices"]) == 10000 and len(set(plan["train_indices"])) == 10000
        and plan["validation_indices"] == list(range(16024)), "Frozen cohort cardinality changed", "cohorts")
    historical_manifest_path = ROOT / "reports/package_a_diagnosis/manifest.json"
    checked.check(digest(historical_manifest_path) == DIAGNOSIS_MANIFEST_SHA256, "Prior diagnosis manifest changed", "identity")
    historical_files = {e["path"]: e for e in read(historical_manifest_path)["files"] if e["scope"] == "project"}
    baselines, baseline_shapes, source_reports = {}, {}, {}
    for seed in (202610171, 202610172):
        baseline_path = f"reports/package_a_diagnosis/model-smooth-{seed}-terminal.json"
        checked.check(digest(ROOT / baseline_path) == historical_files[baseline_path]["sha256"],
                      f"Prior endpoint summary changed: {seed}", "identity")
        baseline = read(ROOT / baseline_path)
        baselines[seed] = endpoint_table(baseline)
        baseline_shapes[seed] = {split: {key: group["rows"] for key, group in fit["groups"].items()}
                                 for split, fit in baseline["fit"].items()}
        source_reports[seed] = read(ROOT / f"reports/package_a/a2/run-smooth-{seed}.json")
    ledger_path = external / "budget-ledger.json"
    ledger = read(ledger_path)
    ledger_copy = reports / "budget-ledger.json"
    if ledger_copy.exists():
        checked.check(digest(ledger_copy) == digest(ledger_path), "Report ledger copy differs from final external ledger", "identity")
    operations = ledger["operations"]
    checked.check(ledger["identity"] == identity and ledger["limit_seconds"] == 900, "Ledger identity changed", "budget")
    checked.check([op["id"] for op in operations] == list(range(len(operations))) and len({op["name"] for op in operations}) == len(operations),
                  "Duplicate or unordered budget operations", "budget")
    checked.check(all(op["status"] in ("completed", "failed") and isinstance(op["elapsed_seconds"], (int, float))
        and math.isfinite(op["elapsed_seconds"]) and 0 <= op["elapsed_seconds"] <= op["reserved_seconds"]
        and not op["exceeded_reservation"] for op in operations), "Active/unreconciled or excessive lease", "budget")
    for op in operations:
        ceiling = (config["per_attempt_allowance_seconds"] if op["name"].startswith("continuation:") else
                   config["prepare_allowance_seconds"] if op["name"] == "prepare" else config["test_allowance_seconds"])
        checked.check(0 < op["reserved_seconds"] <= ceiling, f"Operation reservation exceeds allowance: {op['name']}", "budget")
    charged = sum(op["elapsed_seconds"] for op in operations)
    checked.check(charged <= 900, "Cumulative budget exceeded", "budget")
    for left, right in zip(operations, operations[1:]):
        checked.check(left["started_unix"] + left["elapsed_seconds"] <= right["started_unix"] + .1, "Overlapping leases", "budget")
    names = [f"continuation:{a['name']}" for a in attempts]
    checked.check([op["name"] for op in operations if op["name"].startswith("continuation:")] == names,
                  "Missing, extra or reordered formal attempts", "budget")
    markers = [read(reports / "prepare-attempt.json")] + [read(p) for p in sorted(reports.glob("checks-*.json"))]
    markers += [read(reports / f"attempt-{i}.json") for i in range(4)]
    checked.check({m["name"] for m in markers} == {op["name"] for op in operations}, "Marker/ledger coverage differs", "budget")
    checked.check({p.name for p in reports.glob("attempt-*.json")} == {f"attempt-{i}.json" for i in range(4)}, "Unexpected formal marker", "budget")
    by_name = {op["name"]: op for op in operations}
    for marker in markers:
        checked.check(marker["config_sha256"] == identity and marker["status"] == by_name[marker["name"]]["status"],
                      f"Marker identity/status mismatch: {marker['name']}", "identity")
        if marker["name"] == "prepare" or marker["name"].startswith("continuation:"):
            checked.check(marker["implementation"] == plan["implementation"], f"Frozen marker source changed: {marker['name']}", "identity")
        if marker.get("watchdog"):
            checked.check(marker["watchdog"]["sampled_peak_tree_rss_bytes"] < config["memory_bytes"], "Process memory exceeded", "budget")
    checked.check(markers[0]["status"] == "completed" and markers[0]["finished_unix"] <= read(reports / "attempt-0.json")["started_unix"],
                  "Formal attempts preceded preparation", "budget")
    endpoints, runtime, attempt_rows, arrays = {}, {}, [], set()
    endpoint_memory_peaks = []
    for index, a in enumerate(attempts):
        name, seed = a["name"], a["seed"]
        source = source_reports[seed]
        checked.check(digest(a["checkpoint"]) == a["checkpoint_sha256"] and
            digest(ROOT / a["source_report"]) == plan["inputs"][name]["source_report_sha256"], f"Starting source changed: {name}", "identity")
        marker, path = read(reports / f"attempt-{index}.json"), reports / f"run-{name}.json"
        row = {"name": name, "seed": seed, "lr": a["lr"], "status": marker["status"], "complete": False, "failure": marker["failure"]}
        if not path.exists():
            checked.check(marker["status"] == "failed", f"Missing successful run report: {name}", "completeness")
            attempt_rows.append(row); continue
        report = read(path)
        checked.check(report["attempt"] == a and report["config_sha256"] == identity and report["plan_sha256"] == plan_sha
            and report["implementation"] == plan["implementation"] and report["constraints"] == CONSTRAINTS,
            f"Run identity changed: {name}", "identity")
        row.update(report_sha256=digest(path), complete=report["complete"], failure=report.get("failure") or row["failure"])
        endpoint_peak = report["sampled_peak_endpoint_conservative_memory_bytes"]
        checked.check(type(endpoint_peak) is int and 0 <= endpoint_peak < config["memory_bytes"],
                      f"Endpoint memory ceiling exceeded: {name}", "budget")
        endpoint_memory_peaks.append(endpoint_peak)
        run = report.get("training")
        if run is not None:
            runtime[name] = run
            stop = run["completed_steps"]
            checked.check(run["starting_step"] == 18500 and run["requested_additional_steps"] == 5000
                and run["requested_steps"] == 23500 and run["additional_steps"] == stop - 18500
                and 18500 <= stop <= 23500, f"Step accounting mismatch: {name}", "completeness")
            checked.check(run["starting_checkpoint_sha256"] == a["checkpoint_sha256"] and run["source_report_sha256"] == plan["inputs"][name]["source_report_sha256"],
                          f"Restored input identity mismatch: {name}", "identity")
            settings = run["settings"]
            expected = {**source["settings"], "lr": a["lr"], "steps": 23500, "protocol_manifest": settings["protocol_manifest"]}
            checked.check(settings == expected and settings["protocol_manifest"] == {"config_sha256": identity, "plan_sha256": plan_sha, "attempt": a},
                          f"More than learning rate changed: {name}", "protocol")
            checked.check(run["data_manifest"] == plan["training_manifest"] and run["validation_manifest"] == plan["validation_manifest"],
                          f"Runtime cohort mismatch: {name}", "cohorts")
            checked.check([h["step"] for h in run["history"]] == list(range(18501, stop + 1)), f"Incomplete update history: {name}", "completeness")
            checked.check(all(all(math.isfinite(v) for v in h.values()) for h in run["history"]), f"Nonfinite update history: {name}", "runtime_assertions")
            original = replay_safety(source["safety"]["configuration"], source["history"], source["diagnostics"])
            checked.check(all(original[k] == source["safety"][k] for k in SAFETY_KEYS), f"Historical safety replay differs: {name}", "safety_replay")
            inherited = run["initial_safety_state"]
            checked.check(all(original[k] == inherited[k] for k in (*SAFETY_KEYS, "streaks"))
                and plain_state_digest(inherited) == run["initial_safety_state_sha256"], f"Inherited safety state differs: {name}", "safety_replay")
            resumed = replay_safety(inherited["configuration"], run["history"], run["diagnostics"], inherited)
            checked.check(all(resumed[k] == run["safety_state"][k] for k in (*SAFETY_KEYS, "streaks"))
                and all(resumed[k] == run["safety"][k] for k in SAFETY_KEYS), f"Continuation safety replay differs: {name}", "safety_replay")
            checked.check(run["initial_sampled_index_digest"] == source["sampled_index_digest"], f"Sampler chain was restarted: {name}", "runtime_assertions")
            checked.check(run["sampled_peak_conservative_memory_bytes"] < config["memory_bytes"], "Worker memory exceeded", "budget")
            row.update(completed_steps=stop, additional_steps=run["additional_steps"], best_step=run["best_step"],
                terminal_fixed_discard_agreement=run["terminal_fixed_discard_agreement"], durability_passed=run["durability_passed"],
                safety_events=len(run["safety"]["events"]), stop_reason=run["stop_reason"])
            if report["complete"]:
                checked.check(marker["status"] == "completed" and run["complete"] and not run["failure"] and not report.get("failure")
                    and stop == 23500 and run["stop_reason"] == "steps_completed" and run["sources_unchanged"]
                    and run["frozen_rank_head_unchanged"] and report["sources_unchanged"] and report["inputs_unchanged"],
                    f"Contradictory complete run: {name}", "runtime_assertions")
                checked.check(run["validation_schedule_complete"] and [v["step"] for v in run["validations"]] == list(range(19000, 23501, 500))
                    and all(v["complete"] for v in run["validations"]), f"Incomplete scheduled validation: {name}", "completeness")
                checked.check([d["step"] for d in run["diagnostics"]] == list(range(18600, 23501, 100))
                    and [d["step"] for d in run["sampled_prefixes"]] == list(range(18600, 23501, 100))
                    and run["sampled_prefixes"][-1]["sha256"] == run["sampled_index_digest"], f"Incomplete diagnostic/sample trace: {name}", "completeness")
                checked.check(all(run[k]["complete"] for k in ("terminal_fixed_validation", "terminal_full_validation", "terminal_probability_metrics")),
                              f"Incomplete terminal measurement: {name}", "completeness")
                durable = not run["safety"]["tripped"] and not run["safety"]["scale_alarm"] and run["terminal_fixed_discard_agreement"] >= .999
                checked.check(run["durability_passed"] == durable, f"Durability flag differs from fixed rule: {name}", "acceptance")
        for split, fit in report["fit"].items():
            indices = plan["train_indices"] if split == "training" else plan["validation_indices"]
            checked.check(fit["complete"] and fit["rows"] == fit["requested_rows"] == len(indices)
                and fit["indices_sha256"] == index_hash(indices) and fit["epsilon"] == .01 and fit["device"] == "mps", f"Endpoint cohort differs: {name}/{split}", "cohorts")
            checked.check({key: group["rows"] for key, group in fit["groups"].items()} == baseline_shapes[seed][split],
                          f"Endpoint group membership counts changed: {name}/{split}", "cohorts")
            for group_name, group in fit["groups"].items(): checked.group(group, f"{name}/{split}/{group_name}")
            checked.partition(fit["groups"], "all", ["optional", "forced"], f"{name}/{split}")
            checked.partition(fit["groups"], "optional", ["stratum:teacher:optional", "stratum:student:optional"], f"{name}/{split}")
        for split, metadata in report["arrays"].items():
            artifact = Path(metadata["path"])
            checked.check(artifact.parent == external / "arrays" and artifact.name == f"{name}-{split}.npz"
                and artifact.stat().st_size == metadata["bytes"] and digest(artifact) == metadata["sha256"], f"Endpoint array changed: {name}/{split}", "identity")
            arrays.add(artifact)
        for kind, metadata in report.get("checkpoints", {}).items():
            checked.check(metadata["path"] == run[f"{kind}_checkpoint"] and digest(metadata["path"]) == metadata["sha256"], f"Checkpoint bytes changed: {name}/{kind}", "identity")
        if report["complete"]:
            checked.check(set(report["fit"]) == set(report["arrays"]) == {"training", "validation"}
                and set(report["checkpoints"]) == {"selected", "terminal"}, f"Missing completed artifacts: {name}", "completeness")
            endpoints[name] = endpoint_table(report)
        attempt_rows.append(row)
    checked.check(arrays == set((external / "arrays").glob("*.npz")), "Unrecorded endpoint arrays", "identity")
    checked.check({p.name for p in reports.glob("run-*.json")} <= {f"run-{a['name']}.json" for a in attempts}, "Extra run report", "identity")
    pairs = []
    for seed in (202610171, 202610172):
        same_name, lower_name = f"same-{seed}", f"lower-{seed}"
        pair = compare_pair(baselines[seed], endpoints.get(same_name), endpoints.get(lower_name),
                            eligible=same_name in endpoints and lower_name in endpoints)
        pair["seed"] = seed
        if same_name in runtime and lower_name in runtime:
            same, lower = runtime[same_name], runtime[lower_name]
            for key in ("initial_state_sha256", "initial_optimizer_state_sha256", "initial_optimizer_without_lr_sha256",
                        "initial_rng_state_sha256", "initial_safety_state_sha256", "initial_sampled_index_digest"):
                checked.check(same[key] == lower[key], f"Paired starting state differs: {seed}/{key}", "paired_runtime_assertions")
            left = {p["step"]: p["sha256"] for p in same["sampled_prefixes"]}
            right = {p["step"]: p["sha256"] for p in lower["sampled_prefixes"]}
            common = sorted(set(left) & set(right))
            checked.check(all(left[step] == right[step] for step in common), f"Paired minibatch stream differs: {seed}", "paired_runtime_assertions")
            if pair["eligible"]:
                checked.check(len(common) == 50 and same["sampled_index_digest"] == lower["sampled_index_digest"],
                              f"Incomplete paired sampler trace: {seed}", "paired_runtime_assertions")
            pair["paired_sample_prefixes_verified"] = len(common)
        pairs.append(pair)
    artifact_bytes = sum(p.stat().st_size for root in (reports, external) for p in root.rglob("*") if p.is_file())
    checked.check(artifact_bytes < config["artifact_bytes"], "Artifact budget exceeded", "budget")
    all_complete = len(endpoints) == 4
    return {"schema": "package-a-continuation-audit-v1", "valid": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": identity, "plan_sha256": plan_sha,
        "checks_passed": sum(checked.counts.values()), "checks_by_category": dict(checked.counts),
        "attempts": attempt_rows, "all_four_complete": all_complete, "array_files_verified": len(arrays),
        "ledger_report_copy_verified": ledger_copy.exists(),
        "paired_comparisons": pairs, "exploratory_passed_both_seeds": all(p["exploratory_passed"] for p in pairs),
        "durability_passed_both_lower_arms": all(lower in endpoints and runtime[lower]["durability_passed"] for lower in ("lower-202610171", "lower-202610172")),
        "budget": {"charged_seconds": charged, "limit_seconds": 900, "remaining_seconds": 900 - charged,
            "operation_count": len(operations), "active_leases": 0, "formal_attempts": 4,
            "sampled_peak_tree_rss_bytes": max((m.get("watchdog", {}).get("sampled_peak_tree_rss_bytes", 0) for m in markers), default=0),
            "sampled_peak_training_conservative_bytes": max((r["sampled_peak_conservative_memory_bytes"] for r in runtime.values()), default=0),
            "sampled_peak_endpoint_conservative_bytes": max(endpoint_memory_peaks, default=0),
            "sampled_peak_conservative_worker_bytes": max([r["sampled_peak_conservative_memory_bytes"] for r in runtime.values()] + endpoint_memory_peaks, default=0),
            "artifact_bytes_before_audit_write": artifact_bytes},
        "limitations": ["This hashes checkpoint/NPZ bytes without deserializing them; recorded runtime state assertions are not independent reruns.",
            "Safety formulas and aggregate numerator/denominator arithmetic are independently replayed using stdlib operations on JSON.",
            "Primary decision criteria use full validation teacher/student strata; training quality is descriptive. KL uses optional rows and CPU-float64 normalization of saved float32 forward logits.",
            "Fixed-cohort component comparisons are descriptive; neither two seeds nor a durability pass establishes playing strength.",
            "Sampled memory can miss peaks; conservative process-plus-accelerator accounting may double count unified memory."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output", default=str(ROOT / "reports/package_a_continuation/audit.json"))
    args = parser.parse_args()
    try:
        result = audit(args.expected_config_sha256)
    except (OSError, KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        result = {"schema": "package-a-continuation-audit-v1", "valid": False,
                  "failure": {"type": type(error).__name__, "message": str(error)}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("paired_comparisons", "limitations")}, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
