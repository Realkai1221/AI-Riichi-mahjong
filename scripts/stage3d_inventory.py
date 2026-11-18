"""Write-once inventory of complete or stopped Stage 3D recovery and local book evidence.

Only explicit project members and artifact directories are packaged; unrelated
source/test additions cannot silently enlarge the declared project file set.
No engine, model, checkpoint deserialization or numerical experiment is run.
"""

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
REPORTS = ROOT / "reports/stage3d"
MANIFEST = REPORTS / "manifest.json"
VERIFICATION = REPORTS / "inventory-verification.json"
SCHEMA = "stage3d-inventory-v1"


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


AUXILIARY_PROJECT_PATHS = (
    "scripts/stage3d_inventory.py",
    "scripts/verify_stage3c_snapshot.py",
    "scripts/audit_stage3d.py",
    "scripts/audit_stage3c.py",
    "scripts/stage3a_audit.py",
    "scripts/analyze_stage3d.py",
    "scripts/analyze_stage3c.py",
    "scripts/render_stage3d_book_pages.py",
    "tests/test_evaluation_recovery_audit.py",
    "tests/test_recovery_behavior_analysis.py",
    "tests/test_call_behavior_analysis.py",
    "tests/test_stage3a_audit.py",
    "configs/stage3d.json",
    "docs/STAGE3D_PLAN.md",
    "docs/STAGE3D_REVIEW.md",
    "docs/STAGE3D_BOOK_REVIEW.md",
    "reports/stage3d/REPORT.md",
    "reports/stage3d/audit-prefix.json",
    "AGENTS.md",
    ".gitignore",
    "README.md",
    "docs/STATUS.md",
    "docs/IMPLEMENTATION_PLAN.md",
    "docs/DECISIONS.md",
)
EXPECTED_BOUNDARIES = {
    "candidate": ("4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967", 4),
    "baseline": ("3baca74ddcb011ea7346396621c709e7499835c2cf4b9e6e201688aa52e75bf2", 0),
}
HISTORICAL_COMMIT = "fbba75bc5e2301b67f1addd68844d3b4589cbfdf"
HISTORICAL_MANIFEST = "c4f7090dbce4067bb93bb21ad5751dbc830c84e2708e0bee8503e8cb0ac9c8ce"


def selected_files():
    config, plan = read(ROOT / "configs/stage3d.json"), read(REPORTS / "plan.json")
    external = Path(config["data_root"])
    require(external.is_absolute() and external == external.resolve()
            and not external.is_relative_to(ROOT), "External output root must be canonical and outside project")
    require((ROOT / config["reports_root"]).resolve() == REPORTS.resolve(), "Report root differs")
    roots = {"project": ROOT, "external": external, "books": ROOT / "Books/stage3d"}
    members = config["source_paths"]
    require(isinstance(members, list) and members and len(members) == len(set(members))
            and set(members) == set(plan["implementation"]),
            "Frozen implementation membership differs from the explicit configuration list")
    project = {safe_path(ROOT, path) for path in members}
    require(len(AUXILIARY_PROJECT_PATHS) == len(set(AUXILIARY_PROJECT_PATHS)),
            "Auxiliary project membership contains duplicates")
    project.update(safe_path(ROOT, path) for path in AUXILIARY_PROJECT_PATHS)
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


def experiment_state(config, plan, audit, roots):
    """Derive completed, partial and admitted counts without running any policy."""
    require(config["evaluation_blocks"] == 32 and config["maximum_matches"] == 256
            and len(plan["evaluation"]) == 32 and plan["evaluation"] == config["evaluation"],
            "Expected the fixed 32-block, 256-match evaluation-only campaign")
    require(read(REPORTS / "prepare-attempt.json")["status"] == "completed",
            "Frozen preparation did not complete")
    attempts = sorted(REPORTS.glob("eval-attempt-*.json"), key=lambda path: int(path.stem.rsplit("-", 1)[1]))
    require([int(path.stem.rsplit("-", 1)[1]) for path in attempts] == list(range(len(attempts)))
            and 1 <= len(attempts) <= 32, "Evaluation attempts are not a bounded contiguous prefix")
    result_ids = sorted(int(path.stem[5:]) for path in REPORTS.glob("eval-*.json")
                        if path.stem[5:].isdigit())
    require(result_ids == list(range(len(attempts))), "Evaluation results lack matching durable attempts")
    completed_blocks, completed_matches, saved_records = 0, 0, 0
    failed, partials, seen_indices = [], [], set()
    for index, path in enumerate(attempts):
        marker, block = read(path), read(REPORTS / f"eval-{index}.json")
        require(marker["name"] == f"eval:{index}" and block["block"] == index
                and block["planned"] == plan["evaluation"][index], "Evaluation receipt identity differs")
        require(block["states_before"] == plan["states"] and 0 <= len(block["matches"]) <= 8,
                "Fixed endpoint states or persisted match count differ")
        this_completed = 0
        for position, evidence in enumerate(block["matches"]):
            pair = plan["evaluation"][index]["pairs"][position // 2]
            arm = ("candidate", "baseline")[position % 2]
            expected_index = pair[f"{arm}_match_index"]
            absolute = Path(evidence["json_path"])
            require(absolute.is_absolute() and absolute.is_relative_to(roots["external"]),
                    "Saved trajectory path escapes campaign output")
            record_path = safe_path(roots["external"], absolute.relative_to(roots["external"]).as_posix())
            require(digest(record_path) == evidence["json_sha256"], "Saved trajectory identity changed")
            record = read(record_path)
            require(record["match_index"] == expected_index and expected_index not in seen_indices
                    and evidence["arm"] == arm and record["campaign_block"] == index,
                    "Saved match index/arm differs or was duplicated")
            seen_indices.add(expected_index)
            saved_records += 1
            require(type(record["completed"]) is bool, "Completed match flag must be Boolean")
            if record["completed"]:
                require(record["terminated"] is True and record["terminal_result_verified"] is True
                        and record["failure"] is None and record["truncated"] is False
                        and record["interrupted"] is False, "Completed record lacks a valid terminal result")
                this_completed += 1
                completed_matches += 1
            else:
                require(position == len(block["matches"]) - 1 and block["complete"] is False,
                        "Incomplete trajectory was followed by another persisted match")
                partials.append(dict(block=index, match_index=expected_index,
                                     interrupted=record["interrupted"], failure=record["failure"]))
        if marker["status"] == "completed":
            require(not failed and block["complete"] is True and this_completed == 8
                    and block["states_after"] == plan["states"]
                    and block["termination_requested"] is False,
                    "Completed block is incomplete, changed model state or follows a failure")
            completed_blocks += 1
        else:
            require(marker["status"] == "failed" and index == len(attempts) - 1
                    and marker["failure"] is not None and block["complete"] is False
                    and block["mean_utility_difference"] is None,
                    "Only a final explicitly failed attempt may interrupt collection")
            failed.append(index)
    lifecycle = roots["external"] / "lifecycle"
    unclosed_starts = sorted(path for path in lifecycle.glob("match-*-start.json")
                            if not path.with_name(path.name.replace("-start.json", "-finish.json")).exists())
    admitted = audit.get("admitted_without_durable_completion")
    require(isinstance(admitted, list) and len(admitted) == len(unclosed_starts)
            and len(unclosed_starts) <= 1, "Audit admitted-without-completion count differs")
    for path, item in zip(unclosed_starts, sorted(admitted, key=lambda item: item["path"])):
        require(str(path) == item["path"] and digest(path) == item["sha256"]
                and read(path)["planned_match"]["match_index"] == item["match_index"]
                and item["match_index"] not in seen_indices,
                "Unfinished lifecycle identity differs or was also counted as completed")
    audited_partials = audit.get("persisted_partial_matches")
    require(isinstance(audited_partials, list) and len(audited_partials) == len(partials)
            and all(all(saved.get(key) == value for key, value in actual.items())
                    for saved, actual in zip(audited_partials, partials)),
            "Audit persisted incomplete trajectory count or identity differs")
    is_complete = completed_blocks == 32 and not failed
    require(audit.get("experiment_complete") is is_complete
            and audit.get("engineering_evidence_pass") is is_complete
            and audit.get("evaluation_blocks") == completed_blocks
            and audit.get("evaluation_blocks_attempted") == len(attempts)
            and audit.get("evaluation_matches") == completed_matches,
            "Audit experiment status or completed-game counts differ")
    require(audit.get("optimizer_updates") == 0 and audit.get("checkpoint_writes") == 0
            and audit.get("candidate_promoted") is False and audit.get("old_evaluation_pooled") is False,
            "Recovery must remain evaluation-only with no promotion or historical pooling")
    require(not list(roots["external"].rglob("*.pt")) and not list(roots["external"].rglob("*.pth")),
            "Evaluation-only campaign wrote checkpoint artifacts")
    if is_complete:
        result = read(REPORTS / "evaluation.json")
        require(completed_matches == 256 and not partials and not admitted
                and result.get("blocks") == 32 and result.get("matches") == 256
                and result.get("old_evaluation_pooled") is False
                and result.get("optimizer_updates") == 0 and result.get("candidate_promoted") is False
                and result.get("mean_utility_difference") == audit.get("mean_utility_difference")
                and result.get("confidence_interval") == audit.get("confidence_interval"),
                "Complete fixed endpoint differs from final audit")
    else:
        require(not (REPORTS / "evaluation.json").exists()
                and audit.get("mean_utility_difference") is None
                and audit.get("confidence_interval") is None,
                "Incomplete campaign cannot publish the fixed 32-block inference")
    return dict(experiment_complete=is_complete, training_matches=0, optimizer_updates=0,
                checkpoint_writes=0, completed_evaluation_blocks=completed_blocks,
                attempted_evaluation_blocks=len(attempts), completed_evaluation_matches=completed_matches,
                persisted_evaluation_records=saved_records, persisted_incomplete_matches=len(partials),
                admitted_without_durable_completion=len(admitted), failed_evaluation_blocks=failed,
                old_evaluation_pooled=False, candidate_promoted=False)


def packaging_state(config, plan, roots):
    config_hash = digest(ROOT / "configs/stage3d.json")
    require(plan["config_sha256"] == config_hash and plan["frozen_before_formal_collection"] is True,
            "Plan/configuration identity differs")
    require(digest(config["reference_checkpoint"]) == config["reference_sha256"]
            == "5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f",
            "Original reference checkpoint changed")
    require(plan["boundaries"] == config["boundaries"]
            and set(config["boundaries"]) == set(EXPECTED_BOUNDARIES), "Fixed endpoint identities differ")
    boundary_hashes = {}
    for arm, (expected_hash, expected_updates) in EXPECTED_BOUNDARIES.items():
        boundary = config["boundaries"][arm]
        path = Path(boundary["path"])
        require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
                and boundary["sha256"] == digest(path) == expected_hash
                and boundary["updates"] == plan["states"][arm]["updates"] == expected_updates,
                f"Frozen {arm} checkpoint changed")
        boundary_hashes[arm] = expected_hash
    audit = read(REPORTS / "audit.json")
    require(audit.get("schema") == "stage3d-independent-audit-v1" and audit.get("valid") is True
            and audit.get("config_sha256") == config_hash
            and audit.get("plan_sha256") == digest(REPORTS / "plan.json")
            and audit.get("behavior_protocol_sha256") == plan["behavior_protocol_sha256"]
            == config["behavior_protocol_sha256"],
            "Final valid independent audit must bind current plan/configuration/behavior")
    require(audit["implementation"] == plan["implementation"]
            and audit["states"] == plan["states"], "Frozen implementation or endpoint states changed")
    for path, expected in plan["implementation"].items():
        require(digest(safe_path(ROOT, path)) == expected, f"Frozen numerical source changed: {path}")
    require(audit.get("auditor_sha256") == digest(ROOT / "scripts/audit_stage3d.py")
            and audit.get("audit_helpers") == {path: digest(ROOT / path) for path in
                ("scripts/audit_stage3c.py", "scripts/stage3a_audit.py")},
            "Independent auditor or helper source changed")
    prefix_path = REPORTS / "audit-prefix.json"
    prefix = read(prefix_path)
    require(audit.get("prefix_audit") == {"path": str(prefix_path), "sha256": digest(prefix_path)}
            and audit.get("closure", {}).get("valid") is True
            and audit["closure"].get("full_match_numeric_arithmetic_repeated") is False
            and prefix.get("budget", {}).get("all_closed") is False
            and prefix.get("active_audit_operation") is not None,
            "Final audit does not bind the preserved leased full-audit prefix")
    retained_keys = (
        "schema", "valid", "config_sha256", "behavior_protocol_sha256", "plan_sha256",
        "implementation", "auditor_sha256", "audit_helpers", "states",
        "experiment_complete", "engineering_evidence_pass", "evaluation_blocks",
        "evaluation_blocks_attempted", "evaluation_matches", "persisted_partial_matches",
        "admitted_without_durable_completion", "complete_block_differences",
        "mean_utility_difference", "confidence_interval", "calls", "evaluation_learner_rows",
        "optimizer_updates", "checkpoint_writes", "candidate_promoted", "old_evaluation_pooled",
    )
    require(all(key in prefix and audit.get(key) == prefix[key] for key in retained_keys),
            "Final closure changed full-audit evidence or numerical findings")
    ledger = read(roots["external"] / "budget-ledger.json")
    saved = read(REPORTS / "budget.json")
    require(ledger["schema"] == "package-a2-budget-v1" and ledger["identity"] == config_hash
            and ledger["limit_seconds"] == config["limit_seconds"] == 400
            and all(saved.get(key) == value for key, value in ledger.items()),
            "Live and saved budget identities or contents differ")
    operations = ledger["operations"]
    require(audit.get("closed_ledger_prefix") == operations
            and audit.get("active_audit_operation") is None,
            "Final audit closure does not bind the complete closed ledger")
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
    require(0 <= charged <= 400 and saved["charged_seconds"] == charged
            and saved["remaining_seconds"] == 400 - charged, "Saved budget totals differ")
    budget = dict(limit_seconds=400, charged_seconds=charged, remaining_seconds=400 - charged,
                  operations=len(operations), all_closed=True,
                  completed_operations=sum(op["status"] == "completed" for op in operations),
                  failed_operations=sum(op["status"] == "failed" for op in operations))
    audited_budget = audit["budget"]
    expected_budget = dict(limit_seconds=400, closed_operations=len(operations), all_closed=True,
        current_audit_lease=None, current_audit_reservation_seconds=0,
        failed_operations=budget["failed_operations"])
    require(all(audited_budget.get(key) == value for key, value in expected_budget.items())
            and math.isclose(audited_budget["closed_charged_seconds"], charged, rel_tol=0, abs_tol=1e-8),
            "Final audit is stale relative to the final closed ledger")
    first_eval = next((op["id"] for op in operations if op["name"] == "eval:0"), None)
    require(first_eval is not None, "No formal evaluation attempt exists")
    preparation = sum(op["elapsed_seconds"] for op in operations[:first_eval])
    require(preparation <= config["preparation_limit_seconds"] == 45
            and math.isclose(audited_budget["preparation_seconds"], preparation, rel_tol=0, abs_tol=1e-8),
            "Preparation allowance or audited preparation duration differs")
    experiment = experiment_state(config, plan, audit, roots)
    previous = read(REPORTS / "previous-stage3c-snapshot.json")
    require(previous.get("valid") is True and previous.get("verified_files") == 383
            and previous.get("snapshot_commit") == HISTORICAL_COMMIT
            and previous.get("manifest_sha256") == HISTORICAL_MANIFEST,
            "Historical Stage 3C verification differs")
    analysis_hash = None
    if experiment["experiment_complete"]:
        analysis = read(REPORTS / "behavior.json")
        require(analysis.get("schema") == "stage3d-saved-data-descriptive-behavior-v1"
                and analysis.get("valid") is True and analysis.get("complete_experiment") is True
                and analysis.get("input_match_count") == 256
                and analysis.get("config_sha256") == config_hash
                and analysis.get("plan_sha256") == digest(REPORTS / "plan.json")
                and analysis.get("analysis_source_sha256") == digest(ROOT / "scripts/analyze_stage3d.py")
                and analysis.get("analysis_helper_sha256") == digest(ROOT / "scripts/analyze_stage3c.py")
                and analysis.get("fixed_endpoint_reference", {}).get("sha256") == digest(REPORTS / "evaluation.json")
                and analysis.get("old_evaluation_pooled") is False
                and analysis.get("new_games") == 0 and analysis.get("new_optimizer_updates") == 0,
                "Final descriptive analysis identity or evaluation-only scope differs")
        analysis_hash = digest(REPORTS / "behavior.json")
    else:
        require(not (REPORTS / "behavior.json").exists(),
                "Incomplete campaign cannot have the complete-endpoint descriptive analysis")
    artifact_bytes = sum(path.stat().st_size for root in (roots["external"], REPORTS, roots["books"])
                         for path in root.rglob("*") if path.is_file())
    require(artifact_bytes < config["artifact_bytes"], "Current artifact cap exceeded")
    return dict(config_sha256=config_hash, plan_sha256=digest(REPORTS / "plan.json"),
                audit_sha256=digest(REPORTS / "audit.json"), analysis_sha256=analysis_hash,
                prefix_audit_sha256=digest(prefix_path),
                ledger_sha256=digest(roots["external"] / "budget-ledger.json"),
                saved_budget_sha256=digest(REPORTS / "budget.json"),
                previous_stage3c_verification_sha256=digest(REPORTS / "previous-stage3c-snapshot.json"),
                frozen_checkpoint_sha256=boundary_hashes, experiment=experiment, budget=budget)


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
        explicit_auxiliary_project_members=list(AUXILIARY_PROJECT_PATHS),
        exclusions=["This manifest and reports/stage3d/inventory-verification.json.",
                    "External runtime lock files.",
                    "Original supplied book PDFs and historical phase artifacts and input checkpoints remain separately pinned inputs."],
        limitations=["File integrity does not establish playing strength, full rule certification or strategic label correctness.",
                     "A stopped experiment can be integrity-valid while experiment_complete remains false; no replacement games or policy promotion.",
                     "Book observation and answer exports remain local development material.",
                     "Working HEAD may precede the results commit; explicit file hashes bind final bytes.",
                     "Unrelated project sources are not implicitly selected; reports, external and local book artifact directories are exhaustive."])
    with MANIFEST.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return dict(created=True, file_count=len(entries), total_bytes=result["total_bytes"],
                scope_counts=result["scope_counts"], manifest_sha256=digest(MANIFEST))


def verify():
    inventory = read(MANIFEST)
    require(inventory.get("schema") == SCHEMA, "Unknown inventory schema")
    config, plan, roots, selected = selected_files()
    require(inventory.get("explicit_auxiliary_project_members") == list(AUXILIARY_PROJECT_PATHS),
            "Explicit auxiliary project membership changed")
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
        scope="Live final Stage 3D bundle; historical Stage 3C bytes verified separately.")


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
