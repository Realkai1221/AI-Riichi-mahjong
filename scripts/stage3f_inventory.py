"""Write-once Stage 3F integrity inventory, including stopped development runs.

Only stdlib file, source, lifecycle and budget bookkeeping is performed here.
No arrays, checkpoints, engines or model computations are loaded. Numerical
findings remain separately hashed evidence; integrity does not establish skill.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from stage3d_inventory import require, read, digest, safe_path, files_under, entries_for

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/stage3f"
MANIFEST = REPORTS / "manifest.json"
VERIFICATION = REPORTS / "inventory-verification.json"
SCHEMA = "stage3f-inventory-v1"
HISTORICAL_COMMIT = "ef0b3612379986469d82ce46415180eb14c3885d"
HISTORICAL_MANIFEST = "053cd27818b93f4a621251709cca48fa5e54a327c0f15adcf6f316d2d75db7f7"
FORMAL_ORDER = ("parity-serial", "parity-parallel", "parity-check", "workers-1", "workers-4", "workers-8", "workers-14", "admission", "pilot")
AUXILIARY_PROJECT_PATHS = (
    "scripts/stage3f_inventory.py", "scripts/stage3d_inventory.py", "scripts/verify_stage3e_snapshot.py",
    "scripts/stage3f_diagnose.py", "scripts/audit_stage3f_corpus.py",
    "tests/test_stage3f_diagnosis.py", "configs/stage3f.json",
    "docs/STAGE3F_PROGRAM.md", "docs/STAGE3F_PROGRAM_PROPOSAL.md", "docs/STAGE3F_DIAGNOSIS_REVIEW.md",
    "docs/STAGE3F_REVIEW.md", "reports/stage3f/REPORT.md",
    "AGENTS.md", ".gitignore", "README.md", "docs/STATUS.md", "docs/IMPLEMENTATION_PLAN.md", "docs/DECISIONS.md",
    "pyproject.toml", "requirements-macos-arm64.lock",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def selected_files():
    config, plan = read(ROOT / "configs/stage3f.json"), read(REPORTS / "plan.json")
    external = Path(config["data_root"])
    require(external.is_absolute() and external == external.resolve() and not external.is_relative_to(ROOT),
            "External output root must be canonical and outside the project")
    require((ROOT / config["reports_root"]).resolve() == REPORTS.resolve(), "Report root differs")
    roots = {"project": ROOT, "external": external}
    require(isinstance(plan.get("sources"), dict) and bool(plan["sources"]), "Frozen plan sources missing")
    project = {safe_path(ROOT, path) for path in (*plan["sources"], *AUXILIARY_PROJECT_PATHS)}
    # New post-freeze auditors and packaging helpers are included explicitly by
    # Stage 3F filename membership without changing the frozen actor sources.
    for directory, pattern in (("scripts", "*stage3f*.py"), ("src/riichi_ai", "stage3f*.py"),
                               ("tests", "*stage3f*.py"), ("docs", "STAGE3F*.md")):
        project.update((ROOT / directory).glob(pattern))
    project.update(files_under(REPORTS))
    project.difference_update((MANIFEST, VERIFICATION))
    selected = {"project": project, "external": files_under(external, ignore_locks=True)}
    seen = set()
    for scope, paths in selected.items():
        require(bool(paths), f"Empty scope: {scope}")
        for path in paths:
            safe_path(roots[scope], path.relative_to(roots[scope]).as_posix())
            require(path.is_file() and path.resolve() not in seen, f"Missing or duplicate selected file: {path}")
            seen.add(path.resolve())
    return config, plan, roots, selected


def _junit(path):
    document = ET.parse(path).getroot()
    suites = [document] if document.tag == "testsuite" else list(document.findall("testsuite"))
    require(bool(suites), "JUnit report contains no suites")
    totals = {key: sum(int(suite.get(key, "0")) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), **totals}


def budget_state(config, roots, plan):
    config_sha = digest(ROOT / "configs/stage3f.json")
    ledger, saved = read(roots["external"] / "budget-ledger.json"), read(REPORTS / "budget.json")
    require(ledger.get("schema") == "package-a2-budget-v1" and ledger.get("identity") == config_sha
            and ledger.get("limit_seconds") == config["limit_seconds"] == 7200
            and all(saved.get(key) == value for key, value in ledger.items()), "Live and saved ledger differ")
    operations = ledger["operations"]
    require([op["id"] for op in operations] == list(range(len(operations)))
            and len({op["name"] for op in operations}) == len(operations), "Duplicate/nonsequential numerical operation")
    charged = diagnosis = 0.
    failed, receipts, tests = [], [], []
    formal_failed = False
    completed_formal = []
    archives = {digest(path) for path in (REPORTS / "source-archive").glob("*") if path.is_file()}
    for op in operations:
        elapsed, reserved = op.get("elapsed_seconds"), op.get("reserved_seconds")
        require(op.get("status") in ("completed", "failed") and type(elapsed) in (int, float)
                and type(reserved) in (int, float) and math.isfinite(elapsed) and math.isfinite(reserved)
                and 0 <= elapsed <= reserved and reserved > 0 and op.get("exceeded_reservation") is False,
                "Unclosed, invalid or exceeded numerical lease")
        require(reserved + config["reserve_seconds"] <= 7200 - charged + 1e-8, "Lease consumed protected shutdown reserve")
        if op["name"].startswith("diagnosis-"):
            require(diagnosis + reserved <= config["diagnosis_limit_seconds"] == 300, "Diagnosis reservation exceeds cumulative cap")
            diagnosis += elapsed
        charged += elapsed
        path = REPORTS / f"operation-{op['name']}.json"
        receipt = read(path)
        require(receipt.get("schema") == "stage3f-operation-v1" and receipt.get("name") == op["name"]
                and receipt.get("status") == op["status"] and receipt.get("ledger_operation") == op
                and receipt.get("config_sha256") == config_sha and receipt.get("reserved_seconds") == reserved,
                "Closed operation receipt differs from ledger")
        require(receipt.get("sampled_peak_tree_rss_bytes", 0) < config["memory_bytes"]
                and receipt.get("artifact_bytes_peak", 0) < config["artifact_bytes"], "Recorded resource ceiling reached")
        for name, expected in receipt["sources"].items():
            source = safe_path(ROOT, name)
            require(digest(source) == expected or expected in archives,
                    f"Operation source version is neither current nor preserved: {name}")
        command = receipt.get("command", [])
        for argument in command:
            if argument.startswith("--junitxml="):
                test = _junit(safe_path(ROOT, argument.split("=", 1)[1]))
                require(op["status"] != "completed" or test["failures"] + test["errors"] == 0,
                        "Successful test operation contains failed cases")
                tests.append({"operation": op["name"], **test})
        if op["status"] == "failed":
            require(receipt.get("failure") and op.get("failure"), "Failed operation lost its failure evidence")
            failed.append(op["name"])
        if op["name"] in FORMAL_ORDER:
            require(not formal_failed, "Formal collection continued after a failed formal phase")
            expected = FORMAL_ORDER[len(completed_formal)]
            require(op["name"] == expected, "Formal phases are not an ordered prefix")
            formal_failed = op["status"] == "failed"
            if not formal_failed:
                completed_formal.append(op["name"])
        receipts.append({"operation": op["name"], "path": str(path.relative_to(ROOT)), "sha256": digest(path)})
    require(charged <= 7200 and diagnosis <= 300 and saved["charged_seconds"] == charged
            and saved["remaining_seconds"] == 7200 - charged, "Budget totals or diagnosis sub-budget differ")
    known_receipts = {REPORTS / f"operation-{op['name']}.json" for op in operations}
    require(set(REPORTS.glob("operation-*.json")) == known_receipts, "Operation receipt/ledger membership differs")
    release = read(REPORTS / "operation-release-preflight.json")
    require(release["status"] == "completed" and all(release["sources"].get(k) == v for k, v in plan["sources"].items()),
            "Successful source-bound release preflight is missing")
    # Preserve the observed development failure rather than silently replacing it.
    require("scheduling-tests" in failed, "Original failed scheduling preflight was removed")
    failed_test = next((test for test in tests if test["operation"] == "scheduling-tests"), None)
    require(failed_test and failed_test["failures"] == 5 and failed_test["errors"] == 0,
            "Original five scheduling-test failures were not retained")
    return {"limit_seconds": 7200, "charged_seconds": charged, "remaining_seconds": 7200 - charged,
            "diagnosis_limit_seconds": 300, "diagnosis_charged_seconds": diagnosis, "all_closed": True,
            "operations": len(operations), "failed_operations": failed,
            "completed_formal_phases": completed_formal, "formal_phase_failed": formal_failed,
            "receipts": receipts, "test_reports": tests,
            "ledger_sha256": digest(roots["external"] / "budget-ledger.json"), "budget_sha256": digest(REPORTS / "budget.json")}


def collection_state(config, plan, roots, budget):
    external = roots["external"]
    expected = {"parity-serial": plan["parity"], "parity-parallel": plan["parity"],
                **{f"workers-{item['workers']}": item["matches"] for item in plan["worker_probes"]}, "pilot": plan["pilot"]}
    require(len(plan["parity"]) == 16 and len(plan["pilot"]) == 1000
            and [item["workers"] for item in plan["worker_probes"]] == [1, 4, 8, 14]
            and all(len(item["matches"]) == 32 for item in plan["worker_probes"])
            and config["max_match_executions"] == plan["max_match_executions"] == 1200,
            "Fixed match counts or worker grid changed")
    starts = sorted(external.glob("**/lifecycle/*.started.json"))
    finishes = set(external.glob("**/lifecycle/*.finished.json"))
    require(len(starts) <= 1200, "Full-match execution ceiling exceeded")
    recognized_finishes, complete, partial, unclosed = set(), [], [], []
    for path in starts:
        phase_root = path.parent.parent
        start = read(path)
        planned = start["plan"]
        name = planned["match_id"]
        require(path.name == name + ".started.json" and planned["role"] == "development"
                and start["reference_sha256"] == config["reference_sha256"]
                and start["call_checkpoint_sha256"] == config["call_sha256"], "Started match identity or checkpoint changed")
        finish = path.with_name(name + ".finished.json")
        if not finish.exists():
            unclosed.append(str(path.relative_to(external)))
            continue
        recognized_finishes.add(finish)
        marker, report = read(finish), read(phase_root / "reports" / (name + ".json"))
        require(marker["report_sha256"] == canonical_hash(report) and marker["payload_sha256"] == report["payload_sha256"]
                and marker["completed"] == report["completed"] and all(report.get(key) == value for key, value in planned.items()),
                "Durable match report commitment differs")
        deterministic = {key: value for key, value in report.items()
                         if key not in ("payload_sha256", "elapsed_seconds", "peak_worker_rss_bytes", "timing_seconds")}
        require(canonical_hash(deterministic) == report["payload_sha256"] and report["source_role"] == "development-only"
                and report["training_eligible"] is False and report["ppo_eligible"] is False,
                "Match payload or development-only role changed")
        for relative, metadata in report["artifacts"].items():
            artifact = safe_path(phase_root, relative)
            require(artifact.stat().st_size == metadata["bytes"] and digest(artifact) == metadata["sha256"],
                    "Saved match artifact differs from its report")
        item = {"directory": str(phase_root.relative_to(external)), "match_id": name, "rows": report["rows"]}
        if report["completed"]:
            require(report["failure"] is None and report["terminal_result_verified"] is True, "Complete match lacks terminal evidence")
            complete.append(item)
        else:
            require(report["failure"] is not None, "Incomplete match hides failure evidence")
            partial.append(item)
    require(recognized_finishes == finishes, "Finish marker has no matching start")
    phases = {}
    for phase, assignments in expected.items():
        path = external / phase / "manifest.json"
        phase_starts = [p for p in starts if p.parent.parent == external / phase]
        started_ids = [read(p)["plan"]["match_id"] for p in phase_starts]
        by_id = {item["match_id"]: item for item in assignments}
        require(set(started_ids) <= set(by_id), "Phase started an unplanned match")
        for started in phase_starts:
            require(read(started)["plan"] == by_id[read(started)["plan"]["match_id"]], "Phase plan assignment changed")
        summary = {"planned_matches": len(assignments), "started_matches": len(started_ids),
                   "completed_matches": sum(item["directory"] == phase for item in complete),
                   "partial_matches": sum(item["directory"] == phase for item in partial),
                   "unclosed_starts": sum(Path(p).parts[0] == phase for p in unclosed), "state": "not_started"}
        if path.exists():
            manifest = read(path)
            require(manifest["schema"] == "riichi-ai-stage3f-corpus-v1" and manifest["phase"] == phase
                    and manifest["planned"] == assignments and manifest["config_sha256"] == digest(ROOT / "configs/stage3f.json")
                    and manifest["plan_sha256"] == digest(REPORTS / "plan.json")
                    and manifest["source_role"] == "development-only" and manifest["optimizer_steps"] == 0
                    and manifest["training_eligible"] is False and manifest["ppo_eligible"] is False,
                    "Corpus manifest scope or identity differs")
            reports = [read(p) for p in sorted((external / phase / "reports").glob("match-*.json"))]
            require(manifest["matches"] == reports, "Corpus manifest report membership changed")
            require(manifest["splits"] == {"development": [r["match_id"] for r in reports]}, "Development split changed")
            if manifest["complete"]:
                require(manifest["failure"] is None and len(reports) == len(assignments)
                        and all(r["completed"] for r in reports) and summary["unclosed_starts"] == 0,
                        "Complete corpus lacks complete planned matches")
                summary["state"] = "complete"
            else:
                summary["state"] = "incomplete"
            summary["manifest_sha256"] = digest(path)
        elif phase_starts:
            summary["state"] = "interrupted_without_manifest"
        if phase in budget["completed_formal_phases"]:
            require(summary["state"] == "complete", "Successful collection operation lacks its completed corpus")
        phases[phase] = summary
    require(not list(external.rglob("*.pt")) and not list(external.rglob("*.pth")), "Engineering pilot wrote a persistent checkpoint")
    fixture_cases = {
        "test_truncated_attempt_is_reproducible_durable_and_never_eligible": (2, 2),
        "test_all_views_built_before_any_policy_and_actor_input_is_immutable": (1, 1),
        "test_expired_deadline_saves_failure_without_fake_terminal_return": (1, 1),
        "test_resolve_failure_retains_attempted_choices_without_claiming_submission": (1, 1),
        "test_checkpoint_mismatch_is_persisted_before_any_engine_creation": (1, 0),
    }
    fixture_starts = fixture_engine_starts = 0
    fixture_evidence = []
    for test_report in budget["test_reports"]:
        xml_path = safe_path(ROOT, test_report["path"])
        for case in ET.parse(xml_path).getroot().iter("testcase"):
            name = case.get("name", "").split("[", 1)[0]
            if name not in fixture_cases or "stage3f_collect" not in case.get("classname", ""):
                continue
            require(case.find("failure") is None and case.find("error") is None and case.find("skipped") is None,
                    "Partial-fixture accounting needs its successful recorded case")
            admitted, initialized = fixture_cases[name]
            fixture_starts += admitted
            fixture_engine_starts += initialized
            fixture_evidence.append({"operation": test_report["operation"], "test": name,
                                     "lifecycle_starts": admitted, "engine_initializations": initialized})
    require(len(starts) + fixture_starts <= 1200, "Durable plus known fixture starts exceed execution ceiling")
    return {"phases": phases, "match_starts": len(starts),
            "known_partial_fixture_lifecycle_starts": fixture_starts,
            "known_partial_fixture_engine_initializations": fixture_engine_starts,
            "counted_starts_including_partial_fixtures": len(starts) + fixture_starts,
            "partial_fixture_accounting": fixture_evidence,
            "fixture_evidence_limitation": "Known partial starts are counted from successful preserved JUnit cases and frozen fixture definitions. Pytest temporary artifacts are not separately inventoried; these fixtures complete no match. Unregistered executions outside this campaign cannot be excluded by a file inventory.", "completed_match_executions": len(complete),
            "persisted_partial_executions": len(partial), "unclosed_match_starts": unclosed,
            "replay_or_other_match_starts": sum(p.parent.parent.name not in expected for p in starts),
            "collection_complete": all(item["state"] == "complete" for item in phases.values()),
            "pilot_complete": phases["pilot"]["state"] == "complete", "optimizer_updates": 0,
            "checkpoint_writes": 0, "candidate_promoted": False, "strength_evaluation_performed": False}


def numerical_evidence(config, plan, roots, collection):
    evidence = []
    pilot_hash = collection["phases"]["pilot"].get("manifest_sha256")
    pilot_audit = replay_pass = benchmark_pass = False
    expected_replays = [plan["pilot"][index]["match_id"] for index in plan["replay_indices"]]
    for path in sorted(REPORTS.glob("*.json")):
        if path in (MANIFEST, VERIFICATION):
            continue
        data = read(path)
        schema = data.get("schema", "")
        if schema == "riichi-ai-stage3f-corpus-audit-v1":
            valid = data.get("valid") is True
            pilot_audit |= valid and data.get("manifest_sha256") == pilot_hash
            evidence.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "schema": schema, "valid": valid})
        elif schema == "riichi-ai-stage3f-replay-audit-v1":
            valid = data.get("valid") is True
            replay_pass |= (valid and len(expected_replays) == 10
                and [item.get("match_id") for item in data.get("matches", [])] == expected_replays
                and all(item.get("matched") is True for item in data["matches"]))
            evidence.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "schema": schema, "valid": valid})
        elif schema == "riichi-ai-stage3f-stream-benchmark-v1":
            require(data.get("complete") is True and all(item.get("optimizer_steps", 0) == 0 for item in data["devices"]),
                    "Value computation probe unexpectedly optimized a model")
            require(all(item.get("model_weights_unchanged", True) is True for item in data["devices"]), "Value probe mutated its weights")
            valid = data.get("numerical_checks_passed") is True
            benchmark_pass |= valid and data.get("manifest_sha256") == pilot_hash
            evidence.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "schema": schema, "valid": valid})
    diagnosis = read(REPORTS / "diagnosis.json")
    require(diagnosis.get("valid") is True and diagnosis.get("optimizer_steps") == diagnosis.get("new_matches") == diagnosis.get("evaluation_rows") == 0
            and diagnosis.get("source_files_unchanged") is True and diagnosis.get("model_and_optimizer_unchanged") is True
            and diagnosis.get("plan_sha256") == digest(REPORTS / "diagnosis-plan.json"), "Saved-data diagnosis identity or scope changed")
    return {"recorded_numerical_evidence": evidence, "diagnosis_sha256": digest(REPORTS / "diagnosis.json"),
            "pilot_bulk_audit_pass": pilot_audit, "ten_match_same_engine_replay_pass": replay_pass,
            "pilot_backend_probe_pass": benchmark_pass,
            "engineering_evidence_complete": collection["collection_complete"] and pilot_audit and replay_pass and benchmark_pass,
            "numeric_arithmetic_repeated_by_inventory": False, "independent_rule_certification": False}


def packaging_state(config, plan, roots):
    require(config["schema"] == "stage3f-engineering-config-v1" and config["training_updates"] == 0
            and config["policy_promotion"] is False and config["corpus_training_eligible"] is False
            and config["round1_corpus_started"] is False and config["role"] == "development",
            "Engineering campaign scope changed")
    require(plan["config_sha256"] == digest(ROOT / "configs/stage3f.json") and plan["training_updates"] == 0
            and plan["training_eligible"] is False and plan["ppo_eligible"] is False, "Frozen plan identity or eligibility differs")
    for name, expected in plan["sources"].items():
        require(digest(safe_path(ROOT, name)) == expected, f"Frozen source changed: {name}")
    for stem in ("reference", "call"):
        require(digest(config[stem + "_checkpoint"]) == config[stem + "_sha256"], "Historical input checkpoint changed")
    require(digest(ROOT / "docs/STAGE3F_PROGRAM_PROPOSAL.md") == config["proposal_sha256"], "Original proposal copy changed")
    previous_path = REPORTS / "previous-stage3e-snapshot.json"
    previous = read(previous_path)
    require(previous.get("valid") is True and previous.get("verified_files") == 3121
            and previous.get("snapshot_commit") == config["historical_stage3e_commit"] == HISTORICAL_COMMIT
            and previous.get("manifest_sha256") == HISTORICAL_MANIFEST, "Historical Stage 3E snapshot receipt differs")
    budget = budget_state(config, roots, plan)
    collection = collection_state(config, plan, roots, budget)
    numerical = numerical_evidence(config, plan, roots, collection)
    # Exclude this self-referential pair from the stable recorded byte total.
    artifact_bytes = sum(path.stat().st_size for root in (roots["external"], REPORTS)
                         for path in root.rglob("*") if path.is_file() and path not in (MANIFEST, VERIFICATION))
    require(artifact_bytes < config["artifact_bytes"], "Final artifact cap exceeded")
    return {"config_sha256": digest(ROOT / "configs/stage3f.json"), "plan_sha256": digest(REPORTS / "plan.json"),
            "historical_stage3e_snapshot_receipt_sha256": digest(previous_path),
            "historical_snapshot_files_previously_verified": 3121,
            "artifact_bytes_excluding_inventory": artifact_bytes, "artifact_limit_bytes": config["artifact_bytes"],
            "budget": budget, "collection": collection, "numerical_evidence": numerical,
            "policy_promotion": False, "training_updates": 0, "round1_started": False}


def create():
    require(not MANIFEST.exists(), "Inventory already exists; verify it instead")
    config, plan, roots, selected = selected_files()
    state = packaging_state(config, plan, roots)
    entries = entries_for(roots, selected)
    result = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "working_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "roots": {scope: str(path) for scope, path in roots.items()}, "packaging_state": state,
        "explicit_auxiliary_project_members": list(AUXILIARY_PROJECT_PATHS),
        "file_count": len(entries), "total_bytes": sum(entry["bytes"] for entry in entries),
        "scope_counts": dict(Counter(entry["scope"] for entry in entries)), "files": entries,
        "exclusions": ["This manifest and inventory-verification.json.", "External runtime lock files.",
            "Historical inputs remain bound to their separate Stage 3E snapshot and current input hashes."],
        "limitations": ["Integrity-valid may describe an incomplete or stopped engineering pilot.",
            "No policy learning, promotion, playing-strength inference or independent rule certification.",
            "Historical snapshot receipt is retained; its complete numerical audits are not repeated.",
            "Current HEAD may precede the result commit; explicit hashes bind final bytes."]}
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    require(state["artifact_bytes_excluding_inventory"] + len(payload.encode()) + 1_000_000 < config["artifact_bytes"],
            "Final inventory/verification reserve exceeds artifact cap")
    with MANIFEST.open("x") as handle:
        handle.write(payload)
    return {"created": True, "file_count": len(entries), "total_bytes": result["total_bytes"],
            "scope_counts": result["scope_counts"], "manifest_sha256": digest(MANIFEST),
            "pilot_complete": state["collection"]["pilot_complete"],
            "engineering_evidence_complete": state["numerical_evidence"]["engineering_evidence_complete"]}


def verify():
    inventory = read(MANIFEST)
    config, plan, roots, selected = selected_files()
    require(inventory["schema"] == SCHEMA and inventory["roots"] == {scope: str(path) for scope, path in roots.items()}
            and inventory["explicit_auxiliary_project_members"] == list(AUXILIARY_PROJECT_PATHS), "Inventory schema or roots changed")
    require(inventory["packaging_state"] == packaging_state(config, plan, roots), "Final packaging state changed")
    known, unique, errors = {scope: set() for scope in roots}, set(), []
    for entry in inventory["files"]:
        scope = entry["scope"]
        require(scope in roots, "Unknown inventory scope")
        path = safe_path(roots[scope], entry["path"])
        require(path not in known[scope] and path.resolve() not in unique and type(entry["bytes"]) is int and entry["bytes"] >= 0,
                "Duplicate artifact or invalid byte count")
        known[scope].add(path)
        unique.add(path.resolve())
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            errors.append({"scope": scope, "path": entry["path"], "reason": "missing or changed"})
    extra = {scope: sorted(str(p) for p in selected[scope] - known[scope]) for scope in roots}
    missing = {scope: sorted(str(p) for p in known[scope] - selected[scope]) for scope in roots}
    counts = dict(Counter(entry["scope"] for entry in inventory["files"]))
    totals = inventory["file_count"] == len(inventory["files"]) and inventory["scope_counts"] == counts and inventory["total_bytes"] == sum(e["bytes"] for e in inventory["files"])
    actual_artifact_bytes = sum(path.stat().st_size for root in (roots["external"], REPORTS) for path in root.rglob("*") if path.is_file())
    require(actual_artifact_bytes < config["artifact_bytes"], "Final bundle including inventory exceeds artifact cap")
    return {"valid": not errors and not any(extra.values()) and not any(missing.values()) and totals,
            "files_checked": len(inventory["files"]), "total_bytes": inventory["total_bytes"], "scope_counts": counts,
            "counts_match": totals, "errors": errors, "extra_scope_files": extra, "missing_scope_files": missing,
            "manifest_sha256": digest(MANIFEST), "scope": "Live Stage 3F bundle; separate historical Stage 3E receipt retained."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    require(not args.output or (args.verify and Path(args.output).resolve() == VERIFICATION),
            "Only the standard excluded verification output is permitted")
    result = create() if args.create else verify()
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("created") or result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
