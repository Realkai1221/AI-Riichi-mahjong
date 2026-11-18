"""Write-once integrity inventory of the fixed Stage 3E continuation.

This is stdlib file/ledger bookkeeping only. It never loads a checkpoint, runs
an engine or repeats the independent audit's match/probability arithmetic.
Incomplete campaigns remain distinguishable from completed fixed endpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

from stage3d_inventory import require, read, digest, safe_path, files_under, entries_for

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/stage3e"
MANIFEST = REPORTS / "manifest.json"
VERIFICATION = REPORTS / "inventory-verification.json"
SCHEMA = "stage3e-inventory-v1"
INITIAL_SHA = "4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967"
REFERENCE_SHA = "5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f"
HISTORICAL_COMMIT = "7d3380f2d9c595b4da6d99871ede8ccd92380608"
HISTORICAL_MANIFEST = "30234190420f681c5f7ec857288da04df102e9dc030749aa6c950cd8b4f89058"
AUXILIARY_PROJECT_PATHS = (
    "scripts/stage3e_inventory.py", "scripts/stage3d_inventory.py",
    "scripts/verify_stage3d_snapshot.py", "scripts/verify_stage3c_snapshot.py",
    "scripts/audit_stage3e.py", "scripts/audit_stage3d.py", "scripts/audit_stage3c.py", "scripts/stage3a_audit.py",
    "scripts/analyze_stage3e.py", "scripts/analyze_stage3d.py", "scripts/analyze_stage3c.py",
    "scripts/prepare_stage3e_book_calls.py", "src/riichi_ai/book_calls.py", "tests/test_book_calls.py",
    "tests/test_continuation_horizon_audit.py", "tests/test_fixed_horizon_analysis.py",
    "tests/test_evaluation_recovery_audit.py", "tests/test_recovery_behavior_analysis.py",
    "tests/test_call_behavior_analysis.py", "tests/test_stage3a_audit.py",
    "configs/stage3e.json", "docs/STAGE3E_PLAN.md", "docs/STAGE3E_REVIEW.md", "docs/STAGE3E_BOOK_REVIEW.md",
    "reports/stage3e/REPORT.md", "reports/stage3e/audit-prefix.json",
    "AGENTS.md", ".gitignore", "README.md", "docs/STATUS.md", "docs/IMPLEMENTATION_PLAN.md", "docs/DECISIONS.md",
)


def selected_files():
    config, plan = read(ROOT / "configs/stage3e.json"), read(REPORTS / "plan.json")
    external = Path(config["data_root"])
    require(external.is_absolute() and external == external.resolve() and not external.is_relative_to(ROOT),
            "External output root must be canonical and outside project")
    require((ROOT / config["reports_root"]).resolve() == REPORTS.resolve(), "Report root differs")
    roots = {"project": ROOT, "external": external, "books": ROOT / "Books/stage3e"}
    members = config["source_paths"]
    require(isinstance(members, list) and members and len(members) == len(set(members))
            and set(members) == set(plan["implementation"]), "Explicit frozen implementation membership differs")
    require(len(AUXILIARY_PROJECT_PATHS) == len(set(AUXILIARY_PROJECT_PATHS)), "Duplicate auxiliary member")
    project = {safe_path(ROOT, path) for path in [*members, *AUXILIARY_PROJECT_PATHS]}
    project.update(files_under(REPORTS))
    project.difference_update((MANIFEST, VERIFICATION))
    selected = {"project": project, "external": files_under(external, ignore_locks=True),
                "books": files_under(roots["books"])}
    seen = set()
    for scope, paths in selected.items():
        require(bool(paths), f"Empty scope: {scope}")
        for path in paths:
            require(path.is_file(), f"Missing selected file: {path}")
            safe_path(roots[scope], path.relative_to(roots[scope]).as_posix())
            require(path.resolve() not in seen, "Duplicate artifact across scopes")
            seen.add(path.resolve())
    return config, plan, roots, selected


def _boundary(boundary, external=None):
    path = Path(boundary["path"])
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink()
            and digest(path) == boundary["sha256"], "Checkpoint file hash or path differs")
    if external is not None:
        safe_path(external, path.relative_to(external).as_posix())
        require(path.parent == external / "boundaries", "Saved boundary escapes its directory")
    return {"path": str(path), "sha256": boundary["sha256"], "updates": boundary["updates"]}


def _attempts(prefix, start, maximum):
    paths = sorted((p for p in REPORTS.glob(f"{prefix}*.json")
                    if p.stem[len(prefix):].isdigit()), key=lambda p: int(p.stem[len(prefix):]))
    indices = [int(p.stem[len(prefix):]) for p in paths]
    require(len(paths) <= maximum and indices == list(range(start, start + len(paths))),
            f"{prefix} attempts are not a contiguous bounded prefix")
    markers = [read(path) for path in paths]
    require(all(m["status"] in ("completed", "failed") for m in markers), "Attempt is not closed")
    failures = [i for i, m in zip(indices, markers) if m["status"] == "failed"]
    require(not failures or failures == [indices[-1]], "Work continued after a failed formal attempt")
    return list(zip(indices, markers)), failures


def _saved_match(evidence, expected, role, campaign, plan_sha, external, seen):
    path = Path(evidence["json_path"])
    require(path.is_absolute() and path.is_relative_to(external), "Saved match escapes campaign")
    safe_path(external, path.relative_to(external).as_posix())
    require(digest(path) == evidence["json_sha256"], "Saved match identity changed")
    record = read(path)
    require(all(record.get(key) == value for key, value in expected.items())
            and record["match_index"] not in seen, "Unexpected or repeated match identity")
    seen.add(record["match_index"])
    require(record["campaign_sha256"] == campaign and record["campaign_plan_sha256"] == plan_sha
            and record["collection_role"] == role and record["campaign_phase"] == role,
            "Match campaign or role differs")
    require(type(record["completed"]) is bool, "Completed flag must be Boolean")
    if record["completed"]:
        require(record["terminated"] is True and record["terminal_result_verified"] is True
                and record["failure"] is None and record["truncated"] is False
                and record["interrupted"] is False, "Complete trajectory lacks terminal evidence")
    return record


def experiment_state(config, plan, audit, roots):
    require(config["maximum_batches"] == 28 and config["matches_per_batch"] == 16
            and config["training_matches"] == 448 and config["evaluation_blocks"] == 32
            and config["evaluation_matches"] == 256 and config["maximum_matches"] == 704
            and plan["batches"] == config["batches"] and len(plan["batches"]) == 28
            and plan["evaluation"] == config["evaluation"] and len(plan["evaluation"]) == 32,
            "Fixed 448-training/256-evaluation plan differs")
    require(read(REPORTS / "prepare-attempt.json")["status"] == "completed", "Preparation incomplete")
    campaign, plan_sha = digest(ROOT / "configs/stage3e.json"), digest(REPORTS / "plan.json")
    train_attempts, failed_training = _attempts("attempt-", 4, 28)
    eval_attempts, failed_evaluation = _attempts("eval-attempt-", 0, 32)
    before, boundary = plan["initial_state"], plan["initial_boundary"]
    seen, partials, boundaries = set(), [], []
    training_matches = evaluation_matches = training_records = evaluation_records = 0
    completed_batches = completed_blocks = optimizer_steps = completed_updates = 0
    for index, marker in train_attempts:
        path = REPORTS / f"batch-{index}.json"
        require(marker["name"] == f"batch:{index}", "Training receipt name differs")
        if not path.exists():
            require(marker["status"] == "failed", "Successful training lacks result")
            continue
        batch = read(path)
        planned = plan["batches"][index - 4]["matches"]
        require(batch["batch_index"] == index and batch["planned_matches"] == planned
                and batch["input_boundary"] == boundary and batch["input_state"] == before
                and len(batch["matches"]) <= 16, "Training continuation chain differs")
        for position, evidence in enumerate(batch["matches"]):
            record = _saved_match(evidence, planned[position], "training", campaign, plan_sha, roots["external"], seen)
            training_records += 1
            require(record["campaign_batch_index"] == index and record["campaign_arm"] == "learner",
                    "Training batch provenance differs")
            if record["completed"]:
                training_matches += 1
            else:
                require(position == len(batch["matches"]) - 1 and batch["complete"] is False,
                        "Work continued after incomplete training match")
                partials.append({"phase": "training", "batch_index": index, "match_index": record["match_index"]})
        step_delta, update_delta = batch["optimizer_steps_executed_delta"], batch["completed_update_count_delta"]
        require(type(step_delta) is int and type(update_delta) is int and 0 <= update_delta <= step_delta <= 1,
                "Unexpected update count within one batch")
        optimizer_steps += step_delta
        completed_updates += update_delta
        if batch["output_boundary"] is not None:
            saved = _boundary(batch["output_boundary"], roots["external"])
            require(saved["updates"] == index + 1, "Saved boundary update index differs")
            boundaries.append(saved)
        if marker["status"] == "completed":
            require(batch["complete"] is True and len(batch["matches"]) == 16 and step_delta == update_delta == 1
                    and batch["output_boundary"] is not None and batch["boundary_roundtrip_verified"] is True
                    and batch["sources_unchanged"] is True and batch["termination_requested"] is False,
                    "Successful batch lacks exact completed update boundary")
            before, boundary = batch["output_state"], batch["output_boundary"]
            require(before["updates"] == before["optimizer_steps_executed"] == index + 1
                    and before["retired_batches"] == list(range(index + 1)) and before["base_unchanged"] is True,
                    "Absolute update chain differs")
            completed_batches += 1
        else:
            require(batch["complete"] is False and marker["failure"] is not None,
                    "Failed batch hides incomplete status")
    require(not eval_attempts or (completed_batches == 28 and not failed_training),
            "Evaluation started without the complete fixed training horizon")
    fixed_states = {"candidate": before, "baseline": plan["initial_state"]}
    fixed_boundaries = {"candidate": boundary, "baseline": plan["initial_boundary"]}
    for index, marker in eval_attempts:
        path = REPORTS / f"eval-{index}.json"
        require(marker["name"] == f"eval:{index}", "Evaluation receipt name differs")
        if not path.exists():
            require(marker["status"] == "failed", "Successful evaluation lacks result")
            continue
        block = read(path)
        require(block["block"] == index and block["planned"] == plan["evaluation"][index]
                and block["states_before"] == fixed_states and block["boundaries"] == fixed_boundaries
                and len(block["matches"]) <= 8, "Evaluation endpoints or fixed block differs")
        this_completed = 0
        for position, evidence in enumerate(block["matches"]):
            pair = plan["evaluation"][index]["pairs"][position // 2]
            arm = ("candidate", "baseline")[position % 2]
            expected = {"match_index": pair[f"{arm}_match_index"], "seed": block["planned"]["wall_seed"],
                        "sampling_seed": pair["sampling_seed"], "learner_seat": pair["learner_seat"]}
            record = _saved_match(evidence, expected, "evaluation", campaign, plan_sha, roots["external"], seen)
            require(evidence["arm"] == record["campaign_arm"] == arm and record["campaign_block"] == index
                    and record["training_eligible"] is False, "Evaluation role or arm differs")
            evaluation_records += 1
            if record["completed"]:
                evaluation_matches += 1
                this_completed += 1
            else:
                require(position == len(block["matches"]) - 1 and block["complete"] is False,
                        "Work continued after incomplete evaluation match")
                partials.append({"phase": "evaluation", "block": index, "match_index": record["match_index"]})
        if marker["status"] == "completed":
            require(block["complete"] is True and this_completed == 8 and block["states_after"] == fixed_states
                    and block["termination_requested"] is False, "Successful evaluation block lacks fixed endpoint")
            completed_blocks += 1
        else:
            require(block["complete"] is False and marker["failure"] is not None
                    and block["mean_utility_difference"] is None, "Failed block has a completed endpoint")
    lifecycle = roots["external"] / "lifecycle"
    unclosed = sorted(path for path in lifecycle.glob("match-*-start.json")
                      if not path.with_name(path.name.replace("-start.json", "-finish.json")).exists())
    admitted = audit["admitted_without_durable_completion"]
    require(len(unclosed) == len(admitted) <= 1, "Unclosed match admission count differs")
    for path, item in zip(unclosed, sorted(admitted, key=lambda item: item["path"])):
        require(str(path) == item["path"] and digest(path) == item["sha256"]
                and read(path)["planned_match"]["match_index"] == item["match_index"]
                and item["match_index"] not in seen, "Unclosed admission identity differs")
    require({p["match_index"] for p in partials} == {p["match_index"] for p in audit["persisted_partial_matches"]},
            "Persisted partial trajectory count differs from audit")
    complete = completed_batches == 28 and completed_blocks == 32 and not failed_training and not failed_evaluation
    require(audit["experiment_complete"] is complete and audit["engineering_evidence_pass"] is complete
            and audit["evaluation_blocks"] == completed_blocks and audit["evaluation_blocks_attempted"] == len(eval_attempts)
            and audit["evaluation_matches"] == evaluation_matches, "Audited experiment state differs")
    require(audit["training_complete"] is (completed_batches == 28 and not failed_training)
            and audit["training_batches_attempted"] == len(train_attempts)
            and audit["training_matches"] == training_matches and audit["starting_updates"] == 4
            and audit["candidate_updates"] == before["updates"] and audit["optimizer_updates"] == completed_batches
            and [row["batch_index"] for row in audit["updates"]] == list(range(4, 4 + completed_batches))
            and audit["evaluation_optimizer_updates"] == 0 and audit["candidate_promoted"] is False
            and audit["old_evaluation_pooled"] is False and audit["states"] == fixed_states,
            "Audited training horizon, update chain or frozen endpoint differs")
    if complete:
        result = read(REPORTS / "evaluation.json")
        require(training_matches == 448 and evaluation_matches == 256 and not partials and not admitted
                and result["blocks"] == 32 and result["matches"] == 256 and result["training_updates"] == 28
                and result["evaluation_optimizer_updates"] == 0 and result["candidate_promoted"] is False
                and result["old_evaluation_pooled"] is False and result["mean_utility_difference"] == audit["mean_utility_difference"]
                and result["confidence_interval"] == audit["confidence_interval"], "Final endpoint and audit differ")
    else:
        require(not (REPORTS / "evaluation.json").exists() and audit["mean_utility_difference"] is None
                and audit["confidence_interval"] is None, "Incomplete campaign published fixed-endpoint inference")
    return {"experiment_complete": complete, "attempted_training_batches": len(train_attempts),
            "completed_training_batches": completed_batches, "completed_training_matches": training_matches,
            "persisted_training_records": training_records, "optimizer_steps_executed": optimizer_steps,
            "completed_updates": completed_updates, "initial_absolute_update": 4,
            "last_successful_absolute_update": before["updates"], "saved_boundaries": boundaries,
            "attempted_evaluation_blocks": len(eval_attempts), "completed_evaluation_blocks": completed_blocks,
            "completed_evaluation_matches": evaluation_matches, "persisted_evaluation_records": evaluation_records,
            "persisted_incomplete_matches": len(partials), "admitted_without_durable_completion": len(admitted),
            "failed_training_batches": failed_training, "failed_evaluation_blocks": failed_evaluation,
            "old_evaluation_pooled": False, "candidate_promoted": False}


def packaging_state(config, plan, roots):
    campaign, plan_sha = digest(ROOT / "configs/stage3e.json"), digest(REPORTS / "plan.json")
    require(plan["config_sha256"] == campaign and plan["frozen_before_formal_collection"] is True,
            "Plan/configuration identity differs")
    require(digest(config["reference_checkpoint"]) == config["reference_sha256"] == REFERENCE_SHA,
            "Original reference checkpoint changed")
    initial = _boundary(config["initial_boundary"])
    require(initial["sha256"] == INITIAL_SHA and initial["updates"] == 4
            and config["initial_boundary"] == plan["initial_boundary"]
            and plan["initial_state"]["updates"] == 4, "Initial continuation boundary changed")
    audit = read(REPORTS / "audit.json")
    require(audit["schema"] == "stage3e-independent-audit-v1" and audit["valid"] is True
            and audit["config_sha256"] == campaign and audit["plan_sha256"] == plan_sha
            and audit["behavior_protocol_sha256"] == plan["behavior_protocol_sha256"] == config["behavior_protocol_sha256"]
            and audit["implementation"] == plan["implementation"], "Final independent audit identity differs")
    for path, expected in plan["implementation"].items():
        require(digest(safe_path(ROOT, path)) == expected, f"Frozen source changed: {path}")
    require(audit["auditor_sha256"] == digest(ROOT / "scripts/audit_stage3e.py"), "Auditor source changed")
    for path, expected in audit["audit_helpers"].items():
        require(digest(safe_path(ROOT, path)) == expected, f"Audit helper source changed: {path}")
    prefix_path = REPORTS / "audit-prefix.json"
    prefix = read(prefix_path)
    require(audit.get("prefix_audit") == {"path": str(prefix_path), "sha256": digest(prefix_path)}
            and audit.get("closure", {}).get("valid") is True
            and audit["closure"].get("full_match_numeric_arithmetic_repeated") is False
            and prefix.get("budget", {}).get("all_closed") is False
            and prefix.get("active_audit_operation") is not None, "Audit closure lacks its preserved leased prefix")
    mutable_closure_keys = {"created_utc", "full_audit_created_utc", "budget", "closed_ledger_prefix",
                            "active_audit_operation", "prefix_audit", "closure"}
    require(all(audit.get(key) == value for key, value in prefix.items() if key not in mutable_closure_keys),
            "Audit closure changed full-audit findings")
    for item in audit["evidence_inputs"]:
        require(digest(item["path"]) == item["sha256"], "Full-audit evidence changed before packaging")
    trajectory_paths = sorted(str(path.resolve()) for name in ("trajectories", "evaluation")
                              for path in (roots["external"] / name).rglob("*") if path.is_file())
    lifecycle_paths = sorted(str(path.resolve()) for path in (roots["external"] / "lifecycle").rglob("*") if path.is_file())
    checkpoint_paths = sorted(str(path.resolve()) for path in roots["external"].rglob("*")
                              if path.is_file() and path.suffix in (".pt", ".pth"))
    require(trajectory_paths == audit["trajectory_file_membership"]
            and lifecycle_paths == audit["lifecycle_file_membership"]
            and checkpoint_paths == audit["checkpoint_file_membership"]
            and len(checkpoint_paths) == audit["checkpoint_writes"], "Audited artifact membership changed")
    ledger, saved = read(roots["external"] / "budget-ledger.json"), read(REPORTS / "budget.json")
    require(ledger["schema"] == "package-a2-budget-v1" and ledger["identity"] == campaign
            and ledger["limit_seconds"] == config["limit_seconds"] == 1200
            and all(saved.get(key) == value for key, value in ledger.items()), "Saved/live ledger differs")
    operations = ledger["operations"]
    require(audit["closed_ledger_prefix"] == operations and audit["active_audit_operation"] is None
            and [op["id"] for op in operations] == list(range(len(operations)))
            and len({op["name"] for op in operations}) == len(operations), "Final ledger is unclosed or repeated")
    charged = 0.
    for op in operations:
        elapsed, reserved = op.get("elapsed_seconds"), op.get("reserved_seconds")
        require(op["status"] in ("completed", "failed") and type(elapsed) in (int, float)
                and type(reserved) in (int, float) and math.isfinite(elapsed) and math.isfinite(reserved)
                and 0 <= elapsed <= reserved and reserved > 0 and op["exceeded_reservation"] is False,
                "Invalid or excessive numerical lease")
        reserve = 500 if op["name"].startswith("batch:") else 100 if op["name"].startswith("eval:") else 60
        require(reserved <= 1200 - charged - reserve + 1e-8, "Admission consumed protected reserve")
        charged += elapsed
    require(charged <= 1200 and saved["charged_seconds"] == charged and saved["remaining_seconds"] == 1200 - charged,
            "Budget totals differ")
    audited = audit["budget"]
    require(audited["all_closed"] is True and audited["limit_seconds"] == 1200
            and audited["closed_operations"] == len(operations) and audited["current_audit_lease"] is None
            and math.isclose(audited["closed_charged_seconds"], charged, rel_tol=0, abs_tol=1e-8),
            "Audit is stale relative to final ledger")
    first_train = next((op["id"] for op in operations if op["name"] == "batch:4"), len(operations))
    preparation = sum(op["elapsed_seconds"] for op in operations[:first_train])
    require(preparation <= config["preparation_limit_seconds"] == 60
            and math.isclose(audited["preparation_seconds"], preparation, rel_tol=0, abs_tol=1e-8),
            "Preparation budget differs")
    experiment = experiment_state(config, plan, audit, roots)
    previous_path = REPORTS / "previous-stage3d-snapshot.json"
    previous = read(previous_path)
    require(previous["valid"] is True and previous["verified_files"] == 1216
            and previous["snapshot_commit"] == HISTORICAL_COMMIT and previous["manifest_sha256"] == HISTORICAL_MANIFEST,
            "Historical Stage 3D snapshot differs")
    analysis_path = REPORTS / "behavior.json"
    analysis_hash = None
    if experiment["experiment_complete"]:
        analysis = read(analysis_path)
        require(analysis.get("schema") == "stage3e-saved-data-fixed-horizon-descriptive-v1"
                and analysis.get("valid") is True and analysis.get("config_sha256") == campaign
                and analysis.get("plan_sha256") == plan_sha and analysis.get("complete_experiment") is True
                and analysis.get("input_match_count") == 704
                and analysis.get("analysis_source_sha256") == digest(ROOT / "scripts/analyze_stage3e.py")
                and analysis.get("analysis_helpers") == {name: digest(ROOT / "scripts" / name)
                    for name in ("analyze_stage3c.py", "analyze_stage3d.py")}
                and analysis.get("fixed_endpoint_reference", {}).get("sha256") == digest(REPORTS / "evaluation.json")
                and analysis.get("models_loaded") == 0 and analysis.get("new_games") == 0
                and analysis.get("new_optimizer_updates") == 0, "Descriptive analysis identity differs")
        analysis_hash = digest(analysis_path)
    else:
        require(not analysis_path.exists(), "Incomplete campaign has complete-endpoint descriptive analysis")
    books = read(roots["books"] / "validation.json")
    require(books.get("valid") is True and books.get("records") == 2
            and books.get("training_eligible") is False and books.get("evaluation_eligible") is False
            and books.get("full_policy_ready") is False and books.get("prior_exports_semantically_unchanged") is True,
            "Partial book validation or quarantine differs")
    for reference in [*books["input_references"].values(), *books["artifacts"]]:
        path = safe_path(ROOT, reference["path"])
        require(path.stat().st_size == reference["bytes"] and digest(path) == reference["sha256"],
                "Book input or output changed after preparation")
    artifact_bytes = sum(path.stat().st_size for root in (roots["external"], REPORTS, roots["books"])
                         for path in root.rglob("*") if path.is_file())
    require(artifact_bytes < config["artifact_bytes"], "Artifact cap exceeded")
    return {"config_sha256": campaign, "plan_sha256": plan_sha, "audit_sha256": digest(REPORTS / "audit.json"),
            "prefix_audit_sha256": digest(prefix_path), "analysis_sha256": analysis_hash,
            "ledger_sha256": digest(roots["external"] / "budget-ledger.json"), "saved_budget_sha256": digest(REPORTS / "budget.json"),
            "previous_stage3d_verification_sha256": digest(previous_path), "initial_checkpoint": initial,
            "experiment": experiment, "budget": {"limit_seconds": 1200, "charged_seconds": charged,
                "remaining_seconds": 1200 - charged, "all_closed": True, "operations": len(operations),
                "completed_operations": sum(op["status"] == "completed" for op in operations),
                "failed_operations": sum(op["status"] == "failed" for op in operations)}}


def create():
    require(not MANIFEST.exists(), "Inventory already exists; verify it instead")
    config, plan, roots, selected = selected_files()
    gate = packaging_state(config, plan, roots)
    entries = entries_for(roots, selected)
    result = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "working_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "roots": {scope: str(path) for scope, path in roots.items()}, "packaging_state": gate,
        "file_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
        "scope_counts": dict(Counter(e["scope"] for e in entries)), "files": entries,
        "explicit_auxiliary_project_members": list(AUXILIARY_PROJECT_PATHS),
        "exclusions": ["This manifest and reports/stage3e/inventory-verification.json.", "External runtime lock files.",
            "Original books and earlier campaign artifacts/input checkpoints remain separately pinned inputs."],
        "limitations": ["Integrity is not playing strength, full rule certification or strategic label validation.",
            "A stopped experiment can be integrity-valid while experiment_complete is false; no replacements or promotion.",
            "Book records remain quarantined local preparation data.",
            "Working HEAD can precede the results commit; explicit hashes bind final bytes.",
            "Project source membership is explicit; report/external/book artifact directories are exhaustive."]}
    with MANIFEST.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"created": True, "file_count": len(entries), "total_bytes": result["total_bytes"],
            "scope_counts": result["scope_counts"], "manifest_sha256": digest(MANIFEST)}


def verify():
    inventory = read(MANIFEST)
    config, plan, roots, selected = selected_files()
    require(inventory["schema"] == SCHEMA and inventory["explicit_auxiliary_project_members"] == list(AUXILIARY_PROJECT_PATHS)
            and inventory["roots"] == {scope: str(path) for scope, path in roots.items()}, "Inventory schema/membership/roots changed")
    require(inventory["packaging_state"] == packaging_state(config, plan, roots), "Final packaging state changed")
    known, unique, errors = {scope: set() for scope in roots}, set(), []
    for entry in inventory["files"]:
        scope = entry["scope"]
        require(scope in roots, "Unknown inventory scope")
        path = safe_path(roots[scope], entry["path"])
        require(path not in known[scope] and path.resolve() not in unique and type(entry["bytes"]) is int
                and entry["bytes"] >= 0, "Duplicate artifact or invalid byte count")
        known[scope].add(path)
        unique.add(path.resolve())
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            errors.append({"scope": scope, "path": entry["path"], "reason": "missing or changed"})
    extra = {scope: sorted(str(p) for p in selected[scope] - known[scope]) for scope in roots}
    missing = {scope: sorted(str(p) for p in known[scope] - selected[scope]) for scope in roots}
    counts = dict(Counter(entry["scope"] for entry in inventory["files"]))
    totals = (inventory["file_count"] == len(inventory["files"])
              and inventory["total_bytes"] == sum(e["bytes"] for e in inventory["files"])
              and inventory["scope_counts"] == counts)
    return {"valid": not errors and not any(extra.values()) and not any(missing.values()) and totals,
            "files_checked": len(inventory["files"]), "total_bytes": inventory["total_bytes"], "scope_counts": counts,
            "counts_match": totals, "errors": errors, "extra_scope_files": extra, "missing_scope_files": missing,
            "manifest_sha256": digest(MANIFEST), "scope": "Live Stage 3E bundle; historical Stage 3D snapshot verified separately."}


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
