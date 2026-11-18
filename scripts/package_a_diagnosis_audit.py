"""Audit frozen diagnosis records without importing numerical/model libraries.

Checks file identities and independently recalculates accounting and aggregate
arithmetic from saved summaries. Runtime assertions about inference/model state
are verified for consistency, not independently reproduced. NPZ files are hashed,
never deserialized. Run only after all leased work has finished.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/package_a_diagnosis.json"
CONFIG_SHA256 = "a61086f8258c5130b35e13f786b6809628a5715714e9a456b63c4450227775e3"
PLAN_SHA256 = "5a753d8b4593854aee81b0857c10104105ba7347e1731cb5c52a93bbca22d858"
CONSTRAINTS = {"optimizer_updates": 0, "new_matches": 0,
               "action_test_predictions": 0, "test_split_loading": False}
BRANCHES = {"cpu_float32", "mps_float32", "cpu_float64",
            "cpu_float32_softmax_float64", "mps_float32_softmax_float64"}
COMPARISONS = {
    "cpu_float32__vs__cpu_float32_softmax_float64",
    "cpu_float32__vs__cpu_float64", "cpu_float32__vs__mps_float32",
    "mps_float32__vs__cpu_float64", "mps_float32__vs__mps_float32_softmax_float64",
}
SUMMARY_METRICS = ("tie_agreement", "exact_agreement", "original_kl_f32", "smooth_kl_f32",
                   "nonbest_mass_f32", "target_nonbest_mass", "call_mass_f32",
                   "kan_mass_f32", "entropy_f32", "raw_logit_abs_mean",
                   "row_offset_abs", "centered_logit_abs_mean", "context_rms")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def index_hash(indices):
    value = hashlib.sha256()
    for index in indices:
        value.update(struct.pack("<q", index))
    return value.hexdigest()


def close(left, right):
    return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


class Audit:
    def __init__(self):
        self.counts = Counter()

    def check(self, condition, label, category="protocol"):
        if not condition:
            raise ValueError(label)
        self.counts[category] += 1

    def group(self, group, label):
        rows = group["rows"]
        self.check(type(rows) is int and rows >= 0, f"Invalid group rows: {label}", "aggregate_arithmetic")
        for name, metric in group["metrics"].items():
            n, total, mean = metric["denominator"], metric["sum"], metric["mean"]
            valid = type(n) is int and 0 <= n <= rows and math.isfinite(total)
            valid &= (n == rows or name == "best_minus_nonbest_max_gap")
            valid &= (mean is None and total == 0) if n == 0 else (
                mean is not None and close(mean, total / n)
                and metric["minimum"] <= mean + 1e-8 <= metric["maximum"] + 2e-8)
            self.check(valid, f"Metric numerator/denominator mismatch: {label}/{name}", "aggregate_arithmetic")
        severity = group["severity"]
        denominator = severity["optional_teacher_discard_rows"]
        self.check(0 <= denominator <= rows, f"Invalid severity denominator: {label}", "aggregate_arithmetic")
        for key in ("shanten_increments", "same_shanten_lost_ukeire"):
            value = severity[key]
            histogram = {int(k): v for k, v in value["histogram"].items()}
            count, total = sum(histogram.values()), sum(k * v for k, v in histogram.items())
            self.check(value["count"] == count and value["total"] == total
                and value["denominator"] == denominator
                and all(k > 0 and type(v) is int and v > 0 for k, v in histogram.items())
                and ((value["rate"] is None) if denominator == 0 else close(value["rate"], count / denominator))
                and ((value["mean_when_positive"] is None and value["maximum"] is None) if count == 0
                     else close(value["mean_when_positive"], total / count) and value["maximum"] == max(histogram)),
                f"Severity arithmetic mismatch: {label}/{key}", "aggregate_arithmetic")

    def partition(self, groups, parent, children, label):
        whole = groups[parent]
        pieces = [groups[key] for key in children]
        self.check(whole["rows"] == sum(g["rows"] for g in pieces), f"Partition row mismatch: {label}", "cohort_partitions")
        for name, metric in whole["metrics"].items():
            self.check(metric["denominator"] == sum(g["metrics"][name]["denominator"] for g in pieces)
                       and close(metric["sum"], sum(g["metrics"][name]["sum"] for g in pieces)),
                       f"Partition metric mismatch: {label}/{name}", "cohort_partitions")


def metric_mean(group, name):
    metric = group["metrics"][name]
    return metric["sum"] / metric["denominator"] if metric["denominator"] else None


def group_table(group):
    return {"rows": group["rows"], "metrics": {name: metric_mean(group, name) for name in SUMMARY_METRICS},
            "severity": group["severity"]}


def fit_table(fit):
    groups = fit["groups"]
    names = ("optional", "optional_teacher_discard", "stratum:teacher:optional",
             "stratum:student:optional", "stratum:teacher:optional_teacher_discard",
             "stratum:student:optional_teacher_discard", "optional:call_available:True",
             "optional:kan_available:True")
    table = {name: group_table(groups[name]) for name in names}
    equal = {}
    for population in ("optional", "optional_teacher_discard"):
        pieces = [groups[f"stratum:{stratum}:{population}"] for stratum in ("teacher", "student")]
        equal[population] = ({name: sum(metric_mean(g, name) for g in pieces) / 2
                             for name in SUMMARY_METRICS}
                            if all(g["rows"] for g in pieces) else None)
    return {"cohorts": table, "equal_teacher_student_strata": equal}


def audit():
    checked = Audit()
    checked.check(digest(CONFIG_PATH) == CONFIG_SHA256, "Diagnosis configuration changed", "identity")
    config = read(CONFIG_PATH)
    reports = (ROOT / config["reports_root"]).resolve()
    external = (ROOT / config["data_root"]).resolve()
    plan_path = reports / "plan.json"
    checked.check(digest(plan_path) == PLAN_SHA256, "Frozen row plan changed", "identity")
    plan = read(plan_path)
    checked.check(config["constraints"] == plan["constraints"] == CONSTRAINTS, "Inference-only scope changed")
    checked.check(config["limit_seconds"] == 600 and config["memory_bytes"] == 32 * 1024**3
        and config["artifact_bytes"] == 250_000_000 and config["maximum_model_attempts"] == 7,
        "Authorized resource ceiling changed")
    checked.check(plan["config_sha256"] == CONFIG_SHA256 and plan["frozen_before_model_outputs"] is True,
                  "Plan identity or precommit assertion missing", "identity")
    sources = sorted((ROOT / "src/riichi_ai").glob("*.py")) + [ROOT / "scripts/package_a_diagnosis.py",
        ROOT / "scripts/package_a_stability.py", ROOT / config["stage2b_config"]]
    implementation = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    checked.check(plan["implementation"] == implementation, "Frozen numerical implementation changed", "identity")
    checkpoints = config["checkpoints"]
    checked.check(len(checkpoints) == 7 and plan["checkpoints"] == checkpoints,
                  "Declared checkpoint set changed", "identity")
    rows = {"training": (plan["train_indices"], 10000, 490206),
            "validation": (plan["validation_indices"], 16024, 16024),
            "precision": (plan["precision_indices"], 64, 16024)}
    for split, (indices, count, limit) in rows.items():
        checked.check(len(indices) == len(set(indices)) == count
            and all(type(i) is int and 0 <= i < limit for i in indices)
            and indices == sorted(indices) and index_hash(indices) == plan["index_sha256"][split],
            f"Frozen indices invalid: {split}", "cohorts")
    checked.check(plan["validation_indices"] == list(range(16024)), "Validation is not the complete fixed split", "cohorts")
    historical = read(ROOT / "reports/package_a/a2/run-control-202610171.json")
    checked.check(plan["index_sha256"]["precision"] == historical["settings"]["diagnostic_indices_sha256"],
                  "Precision states differ from historical diagnostic states", "cohorts")
    checked.check(plan["training_manifest"]["rows"] == 490206
        and plan["training_manifest"]["stratum_rows"] == {"teacher": 490206, "student": 0}
        and plan["training_manifest"]["split"] == "train"
        and plan["validation_manifest"]["rows"] == 16024
        and plan["validation_manifest"]["stratum_rows"] == {"teacher": 8049, "student": 7975}
        and plan["validation_manifest"]["split"] == "validation", "Dataset population identity mismatch", "cohorts")
    stage = read(ROOT / config["stage2b_config"])
    for split, derived in (("training", "derived-stage2a"), ("validation", "derived-stage2b")):
        manifest = plan[f"{split}_manifest"]
        checked.check(digest(Path(manifest["path"]) / "manifest.json") == manifest["source_manifest_sha256"]
            and digest(Path(stage["data_root"]) / derived / "manifest.json") == manifest["derived_manifest_sha256"],
            f"Corpus manifest identity changed: {split}", "identity")
    ledger_path = external / "budget-ledger.json"
    ledger = read(ledger_path)
    operations = ledger["operations"]
    checked.check(ledger["identity"] == CONFIG_SHA256 and ledger["limit_seconds"] == 600,
                  "Budget identity or ceiling changed", "budget")
    checked.check(all(op["status"] == "completed" and not op["failure"] and not op["exceeded_reservation"]
        and 0 <= op["elapsed_seconds"] <= op["reserved_seconds"] for op in operations),
        "Active, failed, or over-reservation operation; diagnosis is incomplete", "budget")
    charged = sum(op["elapsed_seconds"] for op in operations)
    checked.check(charged <= 600 and [op["id"] for op in operations] == list(range(len(operations))),
                  "Cumulative budget or operation order invalid", "budget")
    for left, right in zip(operations, operations[1:]):
        checked.check(left["started_unix"] + left["elapsed_seconds"] <= right["started_unix"] + .1,
                      "Numerical operations overlap", "budget")
    expected_names = [f"checkpoint:{item['name']}" for item in checkpoints]
    checked.check([op["name"] for op in operations if op["name"].startswith("checkpoint:")] == expected_names,
                  "Checkpoint attempt count/order differs from the fixed seven", "budget")
    checked.check(len({op["name"] for op in operations}) == len(operations), "Repeated operation name", "budget")
    checked.check([p.name for p in sorted(reports.glob("attempt-*.json"))] == [f"attempt-{i}.json" for i in range(7)],
                  "Missing or extra model attempt markers", "budget")
    markers = [read(reports / "prepare-attempt.json")]
    markers += [read(p) for p in sorted(reports.glob("checks-*.json"))]
    markers += [read(reports / f"attempt-{i}.json") for i in range(7)]
    checked.check({m["name"] for m in markers} == {op["name"] for op in operations},
                  "Operation/marker coverage differs", "budget")
    marker_by_name = {m["name"]: m for m in markers}
    for marker in markers:
        checked.check(marker["status"] == "completed" and marker["failure"] is None
            and marker["config_sha256"] == CONFIG_SHA256 and marker["implementation"] == implementation,
            f"Incomplete or changed marker: {marker['name']}", "identity")
        checked.check(marker["watchdog"]["sampled_peak_tree_rss_bytes"] < config["memory_bytes"],
                      f"Sampled memory ceiling exceeded: {marker['name']}", "budget")
    checked.check(marker_by_name["prepare"]["finished_unix"] <= marker_by_name[expected_names[0]]["started_unix"],
                  "Model attempt began before preparation finished", "budget")
    checked.check(all(op["reserved_seconds"] <= config["per_model_allowance_seconds"]
        for op in operations if op["name"].startswith("checkpoint:")), "Model reservation exceeds declared allowance", "budget")
    models, recorded_arrays, precision_batch_hashes, group_shapes = [], set(), set(), {}
    for index, item in enumerate(checkpoints):
        label = item["name"]
        checked.check(digest(item["path"]) == item["sha256"], f"Checkpoint bytes changed: {label}", "identity")
        report_path = reports / f"model-{label}.json"
        checked.check(report_path.is_file(), f"Missing model report: {label}; diagnosis is incomplete", "completeness")
        report = read(report_path)
        checked.check(report["complete"] is True and "failure" not in report
            and report["checkpoint"] == item and report["config_sha256"] == CONFIG_SHA256
            and report["plan_sha256"] == PLAN_SHA256 and report["implementation"] == implementation
            and report["constraints"] == CONSTRAINTS, f"Incomplete or changed model report: {label}", "identity")
        checked.check(report["model_state_sha256_before"] == report["model_state_sha256_after"]
            and report["sources_unchanged"] is True and report["checkpoint_file_unchanged"] is True,
            f"Runtime model/source identity assertion failed: {label}", "runtime_assertions")
        checked.check(report["sampled_peak_conservative_memory_bytes"] < config["memory_bytes"],
                      f"Worker memory ceiling exceeded: {label}", "budget")
        checked.check(set(report["arrays"]) == {"training", "validation", "precision"},
                      f"Missing inference arrays: {label}", "completeness")
        for split, record in report["arrays"].items():
            path = Path(record["path"])
            checked.check(path == external / "arrays" / f"{label}-{split}.npz" and path.is_file()
                and path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                f"Array bytes/path changed: {label}/{split}", "identity")
            recorded_arrays.add(path)
        precision = report["precision"]
        precision_batch_hashes.add(precision["input_batch_sha256"])
        checked.check(precision["rows"] == 64 and precision["row_ids"] == plan["precision_indices"]
            and precision["mps_requested"] is True and precision["mps_available"] is True
            and set(precision["branches"]) == BRANCHES and set(precision["comparisons"]) == COMPARISONS
            and all(value is True for value in precision["checks"].values()),
            f"Precision protocol/runtime assertions invalid: {label}", "runtime_assertions")
        for name, branch in precision["branches"].items():
            checked.check(branch["rows"] == 64 and branch["finite"] is True
                and all(branch[k] == 0 for k in ("nonfinite_legal_logits", "nonfinite_probabilities",
                                                "nonfinite_context", "argmax_illegal_count"))
                and branch["padding_probability_maximum"] == 0,
                f"Invalid precision branch: {label}/{name}", "runtime_assertions")
        for name, comparison in precision["comparisons"].items():
            checked.check(comparison.get("valid") is not False and comparison["rows"] == 64
                and comparison["argmax_disagreement_rows"] == len(comparison["changed_rows"])
                and comparison["teacher_best_membership_changed_rows"] == sum(
                    row["left_teacher_best"] != row["right_teacher_best"] for row in comparison["changed_rows"]),
                f"Precision comparison invalid: {label}/{name}", "aggregate_arithmetic")
        for name in ("cpu_float32__vs__cpu_float32_softmax_float64", "mps_float32__vs__mps_float32_softmax_float64"):
            checked.check(precision["comparisons"][name]["argmax_disagreement_rows"] == 0,
                          f"Identical-logit normalization changed argmax: {label}", "runtime_assertions")
        for split, count in (("training", 10000), ("validation", 16024)):
            fit = report["fit"][split]
            checked.check(fit["complete"] is True and fit["rows"] == fit["requested_rows"] == count
                and fit["indices_sha256"] == plan["index_sha256"][split] and fit["epsilon"] == config["epsilon"]
                and fit["device"] == "mps" and fit["forward_dtype"] == "float32",
                f"Incomplete or changed fit cohort: {label}/{split}", "completeness")
            groups = fit["groups"]
            shape = {name: group["rows"] for name, group in groups.items()}
            if split not in group_shapes:
                group_shapes[split] = shape
            checked.check(shape == group_shapes[split] and groups["all"]["rows"] == count,
                          f"Cohort counts differ across checkpoints: {label}/{split}", "cohorts")
            checked.check(groups["stratum:teacher"]["rows"] == (10000 if split == "training" else 8049)
                and groups["stratum:student"]["rows"] == (0 if split == "training" else 7975),
                f"Teacher/student cohort mismatch: {label}/{split}", "cohorts")
            for name, group in groups.items():
                checked.group(group, f"{label}/{split}/{name}")
            partitions = [("all", ["optional", "forced"]), ("all", ["stratum:teacher", "stratum:student"]),
                          ("all", [name for name in groups if name.startswith("legal_count:")])]
            for population in ("optional", "forced", "optional_teacher_discard"):
                partitions.append((population, [f"stratum:{s}:{population}" for s in ("teacher", "student")]))
            for feature in ("call_available", "kan_available"):
                for population in ("all", "optional", "optional_teacher_discard"):
                    prefix = "" if population == "all" else population + ":"
                    partitions.append((population, [f"{prefix}{feature}:{v}" for v in (False, True)]))
            for parent, children in partitions:
                checked.partition(groups, parent, children, f"{label}/{split}/{parent}")
        models.append({"checkpoint": label, "checkpoint_sha256": item["sha256"], "report_sha256": digest(report_path),
            "fit": {split: fit_table(report["fit"][split]) for split in ("training", "validation")},
            "precision": {"rows": precision["rows"],
                "branches": {name: {key: branch[key] for key in ("legal_logit_abs", "legal_logit_row_offset_abs",
                    "centered_legal_logit_abs", "context_rms", "zero_probability_legal_candidates")}
                    for name, branch in precision["branches"].items()},
                "comparisons": {name: {key: comparison[key] for key in ("argmax_disagreement_rows",
                    "teacher_best_membership_changed_rows", "probability_row_l1_difference",
                    "teacher_best_probability_absolute_difference")}
                    for name, comparison in precision["comparisons"].items()}}})
    checked.check(len(recorded_arrays) == 21 and recorded_arrays == set((external / "arrays").glob("*.npz")),
                  "Expected exactly 21 recorded array files", "identity")
    checked.check(len(precision_batch_hashes) == 1, "Precision input batch differs across checkpoints", "cohorts")
    checked.check({p.name for p in reports.glob("model-*.json")} == {f"model-{item['name']}.json" for item in checkpoints},
                  "Missing or extra model reports", "completeness")
    artifact_bytes = sum(p.stat().st_size for root in (reports, external) for p in root.rglob("*") if p.is_file())
    checked.check(artifact_bytes < config["artifact_bytes"], "Diagnosis artifact ceiling exceeded", "budget")
    return {"schema": "package-a-diagnosis-audit-v1", "valid": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks_passed": sum(checked.counts.values()),
        "checks_by_category": dict(checked.counts), "config_sha256": CONFIG_SHA256, "plan_sha256": PLAN_SHA256,
        "ledger_sha256": digest(ledger_path), "models_completed": len(models), "array_files_verified": len(recorded_arrays),
        "budget": {"charged_seconds": charged, "limit_seconds": 600, "remaining_seconds": 600 - charged,
            "operation_count": len(operations), "active_leases": 0, "model_attempts": 7,
            "sampled_peak_tree_rss_bytes": max(m["watchdog"]["sampled_peak_tree_rss_bytes"] for m in markers),
            "sampled_peak_conservative_worker_bytes": max(read(reports / f"model-{item['name']}.json")["sampled_peak_conservative_memory_bytes"] for item in checkpoints),
            "artifact_bytes_before_audit_write": artifact_bytes, "artifact_limit_bytes": config["artifact_bytes"]},
        "cohort_counts": group_shapes, "models": models,
        "limitations": [
            "This audit hashes NPZ files without reading arrays, and recalculates arithmetic only from stored group numerators/denominators.",
            "Model-state preservation, no-update execution, and numerical inference assertions are recorded runtime evidence, not independently repeated inference.",
            "The row plan and source/configuration hashes are pinned. Full corpus byte verification remains with the historical snapshot verifiers.",
            "Memory peaks are sampled. The conservative worker measure can double-count shared memory. Artifact count is the pre-write snapshot.",
            "Valid execution does not establish playing strength, future training stability, or a causal collapse mechanism."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports/package_a_diagnosis/audit.json"))
    args = parser.parse_args()
    try:
        result = audit()
    except (OSError, KeyError, ValueError, TypeError) as error:
        result = {"schema": "package-a-diagnosis-audit-v1", "valid": False,
                  "failure": {"type": type(error).__name__, "message": str(error)}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({key: result[key] for key in ("valid", "checks_passed", "checks_by_category", "models_completed",
                                                "array_files_verified", "budget", "failure") if key in result}, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
