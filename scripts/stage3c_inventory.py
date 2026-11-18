"""Write-once inventory of the complete or stopped Stage 3C experiment and book evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/stage3c"
MANIFEST = REPORTS / "manifest.json"
VERIFICATION = REPORTS / "inventory-verification.json"
SCHEMA = "stage3c-inventory-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "Inventory paths must be nonempty portable relative paths")
    parts = PurePosixPath(relative)
    require(parts.parts and not parts.is_absolute() and ".." not in parts.parts
            and parts.as_posix() == relative, "Unsafe or noncanonical inventory path")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Inventory path escapes its root")
    current = root
    for part in parts.parts:
        current /= part
        require(not current.is_symlink(), "Linked files are not accepted inventory artifacts")
    return path


def files_under(root, *, ignore_locks=False):
    require(root.is_dir() and not root.is_symlink(), f"Missing or linked scope root: {root}")
    paths = set()
    for path in root.rglob("*"):
        if ignore_locks and path.name.endswith(".lock"):
            continue
        require(not path.is_symlink(), f"Unexpected linked artifact: {path}")
        if path.is_file():
            paths.add(path)
    return paths


def selected_files():
    config, plan = read(ROOT / "configs/stage3c.json"), read(REPORTS / "plan.json")
    external = Path(config["data_root"])
    require(external.is_absolute() and external == external.resolve()
            and not external.is_relative_to(ROOT), "External output root must be canonical and outside project")
    require((ROOT / config["reports_root"]).resolve() == REPORTS.resolve(), "Report root differs")
    roots = {"project": ROOT, "external": external, "books": ROOT / "Books/stage3c"}
    project = {safe_path(ROOT, path) for path in plan["implementation"]}
    project.update((ROOT / "scripts").glob("stage3c*.py"))
    project.update(ROOT / path for path in (
        "scripts/verify_stage3b_snapshot.py", "scripts/audit_stage3c.py",
        "scripts/analyze_stage3c.py", "scripts/stage3a_audit.py",
        "tests/test_call_behavior_analysis.py"))
    project.update((ROOT / "tests").glob("test_stage3c*.py"))
    project.update((ROOT / "docs").glob("STAGE3C*.md"))
    project.update(ROOT / path for path in (
        "configs/stage3c.json", "AGENTS.md", ".gitignore", "README.md",
        "docs/STATUS.md", "docs/IMPLEMENTATION_PLAN.md", "docs/DECISIONS.md"))
    project.update(files_under(REPORTS))
    project.difference_update((MANIFEST, VERIFICATION))
    selected = {"project": project, "external": files_under(external, ignore_locks=True),
                "books": files_under(roots["books"])}
    unique = set()
    for scope, paths in selected.items():
        require(bool(paths), f"Empty scope: {scope}")
        for path in paths:
            require(path.is_file(), f"Missing selected file: {path}")
            safe_path(roots[scope], path.relative_to(roots[scope]).as_posix())
            require(path.resolve() not in unique, "An artifact appears in multiple scopes")
            unique.add(path.resolve())
    return config, plan, roots, selected


def experiment_state(config, plan, audit):
    """Require the fixed training chain and a complete or explicitly stopped eval."""
    for index in range(4):
        marker = read(REPORTS / f"attempt-{index}.json")
        batch = read(REPORTS / f"batch-{index}.json")
        require(marker["status"] == "completed" and batch["complete"] is True
                and len(batch["matches"]) == 16, "All four fixed training updates must be complete")
    require(len(list(REPORTS.glob("attempt-*.json"))) == 4, "Unexpected training attempts")
    attempts = sorted(REPORTS.glob("eval-attempt-*.json"), key=lambda path: int(path.stem.rsplit("-", 1)[1]))
    require([int(path.stem.rsplit("-", 1)[1]) for path in attempts] == list(range(len(attempts)))
            and 1 <= len(attempts) <= 32, "Evaluation attempts are not a bounded contiguous prefix")
    result_ids = sorted(int(path.stem[5:]) for path in REPORTS.glob("eval-*.json")
                        if path.stem[5:].isdigit())
    require(result_ids == list(range(len(attempts))), "Evaluation results lack matching durable attempts")
    complete, matches, failed = 0, 0, []
    for index, path in enumerate(attempts):
        marker, block = read(path), read(REPORTS / f"eval-{index}.json")
        require(marker["name"] == f"eval:{index}" and block["block"] == index,
                "Evaluation receipt identity differs")
        matches += len(block["matches"])
        if marker["status"] == "completed":
            require(not failed and block["complete"] is True and len(block["matches"]) == 8,
                    "Completed evaluation block is incomplete or follows a failure")
            complete += 1
        else:
            require(marker["status"] == "failed" and index == len(attempts) - 1
                    and marker["failure"] is not None and block["complete"] is False,
                    "Only a final, explicitly failed evaluation attempt may stop the prefix")
            failed.append(index)
    is_complete = complete == 32 and not failed
    require(audit.get("experiment_complete") is is_complete
            and audit.get("training_engineering_evidence_pass") is True
            and audit.get("engineering_evidence_pass") is is_complete
            and audit.get("training_matches") == 64 and audit.get("optimizer_updates") == 4
            and audit.get("evaluation_blocks") == complete
            and audit.get("evaluation_blocks_attempted") == len(attempts)
            and audit.get("evaluation_matches") == matches,
            "Audit experiment status or saved-game counts differ")
    stop_hash = None
    if not is_complete:
        require(len(failed) == 1, "Incomplete evaluation requires its explicit failure receipt")
        stop = read(REPORTS / "evaluation-stop.json")
        require(stop.get("failed_operation") == f"eval:{failed[0]}"
                and stop.get("formal_resume_performed") is False
                and stop.get("no_retry_rule_retained") is True
                and stop.get("fixed_32_block_inference_available") is False
                and stop.get("original_runtime_and_test_bytes_unchanged") is True,
                "Stopped evaluation lacks its preserved no-retry/source-integrity record")
        require(not (REPORTS / "evaluation.json").exists()
                and audit.get("mean_utility_difference") is None
                and audit.get("confidence_interval") is None,
                "Stopped evaluation cannot publish the fixed 32-block inference")
        stop_hash = digest(REPORTS / "evaluation-stop.json")
    require(audit.get("candidate_promoted") is False, "This package cannot promote a policy")
    return dict(experiment_complete=is_complete, training_matches=64, optimizer_updates=4,
                completed_evaluation_blocks=complete, attempted_evaluation_blocks=len(attempts),
                saved_evaluation_matches=matches, failed_evaluation_blocks=failed,
                evaluation_stop_sha256=stop_hash, candidate_promoted=False)


def packaging_state(config, plan, roots):
    config_hash = digest(ROOT / "configs/stage3c.json")
    require(plan["config_sha256"] == config_hash and plan["frozen_before_formal_collection"] is True,
            "Plan/configuration identity differs")
    require(digest(config["reference_checkpoint"]) == config["reference_sha256"]
            == "5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f",
            "Original reference checkpoint changed")
    audit = read(REPORTS / "audit.json")
    require(audit.get("schema") == "stage3c-independent-audit-v1" and audit.get("valid") is True
            and audit.get("config_sha256") == config_hash
            and audit.get("plan_sha256") == digest(REPORTS / "plan.json"),
            "Final valid independent audit must bind the current plan/configuration")
    require(audit["implementation"] == plan["implementation"], "Frozen implementation identity changed")
    for path, expected in plan["implementation"].items():
        require(digest(safe_path(ROOT, path)) == expected, f"Frozen numerical source changed: {path}")
    require(audit.get("helper_sha256") == digest(ROOT / "scripts/stage3a_audit.py"),
            "Independent audit helper changed")
    ledger = read(roots["external"] / "budget-ledger.json")
    saved = read(REPORTS / "budget.json")
    require(ledger["identity"] == config_hash and ledger["limit_seconds"] == config["limit_seconds"] == 600
            and all(saved.get(key) == value for key, value in ledger.items()),
            "Live and saved budget identities or contents differ")
    operations = ledger["operations"]
    require([op["id"] for op in operations] == list(range(len(operations)))
            and len({op["name"] for op in operations}) == len(operations),
            "Duplicate or nonsequential operations")
    for operation in operations:
        elapsed, reserved = operation.get("elapsed_seconds"), operation.get("reserved_seconds")
        require(operation["status"] in ("completed", "failed")
                and type(elapsed) in (int, float) and type(reserved) in (int, float)
                and math.isfinite(elapsed) and math.isfinite(reserved)
                and 0 <= elapsed <= reserved and reserved > 0
                and operation.get("exceeded_reservation") is False,
                "Unclosed, invalid or excessive numerical lease prevents packaging")
    charged = sum(op["elapsed_seconds"] for op in operations)
    require(0 <= charged <= 600 and saved["charged_seconds"] == charged
            and saved["remaining_seconds"] == 600 - charged, "Saved budget totals differ")
    budget = dict(limit_seconds=600, charged_seconds=charged, remaining_seconds=600 - charged,
                  operations=len(operations), all_closed=True,
                  completed_operations=sum(op["status"] == "completed" for op in operations),
                  failed_operations=sum(op["status"] == "failed" for op in operations))
    audited_budget = audit["budget"]
    expected_budget = dict(limit_seconds=600, closed_charged_seconds=charged,
        closed_operations=len(operations), all_closed=True, current_audit_lease=None,
        current_audit_reservation_seconds=0, failed_operations=budget["failed_operations"])
    require(all(audited_budget.get(key) == value for key, value in expected_budget.items()),
            "Audit is stale relative to the final closed ledger")
    experiment = experiment_state(config, plan, audit)
    previous = read(REPORTS / "previous-stage3b-snapshot.json")
    require(previous.get("valid") is True and previous.get("verified_files") == 86
            and previous.get("snapshot_commit") == "cc10de1c4134f58703e180f2a9d3ee67df375662"
            and previous.get("manifest_sha256") == "8ed799b62f0167b985571262fe5a1854ace8e9bf37e9b64ebd30e8f9284315c4",
            "Historical Stage 3B verification differs")
    artifact_bytes = sum(path.stat().st_size for root in (roots["external"], REPORTS, roots["books"])
                         for path in root.rglob("*") if path.is_file())
    require(artifact_bytes < config["artifact_bytes"], "Current artifact cap exceeded")
    return dict(config_sha256=config_hash, plan_sha256=digest(REPORTS / "plan.json"),
                audit_sha256=digest(REPORTS / "audit.json"),
                ledger_sha256=digest(roots["external"] / "budget-ledger.json"),
                saved_budget_sha256=digest(REPORTS / "budget.json"),
                previous_stage3b_verification_sha256=digest(REPORTS / "previous-stage3b-snapshot.json"),
                experiment=experiment, budget=budget)


def entries_for(roots, selected):
    return [dict(scope=scope, path=path.relative_to(roots[scope]).as_posix(),
                 bytes=path.stat().st_size, sha256=digest(path))
            for scope in sorted(selected) for path in sorted(selected[scope])]


def create():
    require(not MANIFEST.exists(), "Inventory already exists; verify it instead")
    config, plan, roots, selected = selected_files()
    gate = packaging_state(config, plan, roots)
    entries = entries_for(roots, selected)
    result = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        working_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        roots={scope: str(path) for scope, path in roots.items()}, packaging_state=gate,
        file_count=len(entries), total_bytes=sum(e["bytes"] for e in entries),
        scope_counts=dict(Counter(e["scope"] for e in entries)), files=entries,
        exclusions=["This manifest and reports/stage3c/inventory-verification.json.",
                    "External runtime lock files.",
                    "Original supplied book PDFs and historical phase artifacts remain separately pinned inputs."],
        limitations=["File integrity does not establish playing strength, full rule certification or strategic label correctness.",
                     "A stopped experiment can be integrity-valid while experiment_complete remains false; no replacement games or policy promotion.",
                     "Book observation and answer exports remain local development material.",
                     "Working HEAD may precede the results commit; explicit file hashes bind final bytes."])
    with MANIFEST.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return dict(created=True, file_count=len(entries), total_bytes=result["total_bytes"],
                scope_counts=result["scope_counts"], manifest_sha256=digest(MANIFEST))


def verify():
    inventory = read(MANIFEST)
    require(inventory.get("schema") == SCHEMA, "Unknown inventory schema")
    config, plan, roots, selected = selected_files()
    require(inventory["roots"] == {scope: str(path) for scope, path in roots.items()},
            "Inventory roots changed")
    require(inventory["packaging_state"] == packaging_state(config, plan, roots),
            "Final packaging state changed")
    known, unique, errors = {scope: set() for scope in roots}, set(), []
    for entry in inventory["files"]:
        scope = entry["scope"]
        require(scope in roots, "Unknown inventory scope")
        path = safe_path(roots[scope], entry["path"])
        require(path not in known[scope] and path.resolve() not in unique, "Duplicate artifact")
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0, "Invalid byte count")
        known[scope].add(path)
        unique.add(path.resolve())
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            errors.append({"scope": scope, "path": entry["path"], "reason": "missing or changed"})
    extra = {scope: sorted(str(p) for p in selected[scope] - known[scope]) for scope in roots}
    missing = {scope: sorted(str(p) for p in known[scope] - selected[scope]) for scope in roots}
    counts = dict(Counter(entry["scope"] for entry in inventory["files"]))
    counts_match = (inventory["file_count"] == len(inventory["files"])
        and inventory["total_bytes"] == sum(e["bytes"] for e in inventory["files"])
        and inventory["scope_counts"] == counts)
    return dict(valid=not errors and not any(extra.values()) and not any(missing.values()) and counts_match,
        files_checked=len(inventory["files"]), total_bytes=inventory["total_bytes"], scope_counts=counts,
        counts_match=counts_match, errors=errors, extra_scope_files=extra, missing_scope_files=missing,
        manifest_sha256=digest(MANIFEST),
        scope="Live final Stage 3C bundle; historical Stage 3B bytes verified separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--create", action="store_true")
    command.add_argument("--verify", action="store_true")
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
