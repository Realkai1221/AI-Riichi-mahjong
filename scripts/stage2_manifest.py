"""Create or verify the completed Stage 2A evidence inventory; never train."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT / "reports/stage2/manifest.json"
IMPLEMENTATION = "8b14e0d842d55d4c0230fad8bb6f7bd5ce91d1c1"


def record(path: Path, root: Path) -> dict:
    content = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and "__pycache__" not in path.parts
                  and path.name != ".DS_Store")


def create() -> None:
    if DESTINATION.exists():
        raise FileExistsError("Historical manifest exists; use verify, do not overwrite")
    plan = json.loads((PROJECT / "configs/stage2_pilot.json").read_text())
    external = Path(plan["data_root"])
    audit = json.loads((PROJECT / "reports/stage2/resource-and-provenance-audit.json").read_text())
    if not audit["valid"]:
        raise ValueError("Resource/provenance audit did not pass")
    for item in audit["evidence_files"]:
        if record(PROJECT / item["path"], PROJECT)["sha256"] != item["sha256"]:
            raise ValueError(f"Audited evidence changed: {item['path']}")
    for path in sorted((PROJECT / "src/riichi_ai").glob("*.py")):
        committed = subprocess.run(
            ["git", "show", f"{IMPLEMENTATION}:{path.relative_to(PROJECT)}"],
            cwd=PROJECT, check=True, capture_output=True).stdout
        if path.read_bytes() != committed:
            raise ValueError(f"Implementation changed: {path}")

    project_paths = {PROJECT / name for name in (
        "AGENTS.md", "README.md", "pyproject.toml", "requirements-macos-arm64.lock", ".gitignore")}
    for directory in ("src", "tests", "scripts", "configs", "docs", "reports/stage2"):
        project_paths.update(files(PROJECT / directory))
    project_paths.add(PROJECT / "reports/stage1/manifest.json")
    project_paths.discard(DESTINATION)
    project_records = [record(path, PROJECT) for path in sorted(project_paths)]
    external_records = [record(path, external) for path in files(external)]
    budget = json.loads((PROJECT / "reports/stage2/extended-training-budget.json").read_text())
    resource = audit["training_resources"]
    content = {
        "schema": "riichi-ai-stage2-evidence-v1",
        "review_date": "2026-10-17",
        "implementation_commit": IMPLEMENTATION,
        "stage1_commit": "1a91128",
        "package_bytes_match_implementation": True,
        "project_root": str(PROJECT), "external_root": str(external),
        "project_files": project_records, "external_files": external_records,
        "scope": "Inventory excludes itself, Git metadata, caches and OS metadata. "
                 "External artifact bytes are not committed to Git. Hashes establish identity, "
                 "not independent rules correctness or competitive strength. "
                 "Earlier evaluation dirty flags are preserved; package bytes match the implementation commit.",
        "outcome": {
            "engineering_pilot": "complete_for_documented_scope",
            "optional_discard_validation_agreement": 0.8246937537901758,
            "teacher_retention": "not_established; development futility; no confirmation run",
            "stronger_than_panel": "not_established",
            "extended_action_test_evaluated": False,
            "stage2b": "proposed_for_review; not_executed",
            "reinforcement_learning_started": False,
        },
        "tests": {"tests": 183, "subtests": 165, "failures": 0, "skips": 0,
                  "evidence": "reports/stage2/tests.xml"},
        "resources": {
            "training_budget_seconds": budget["cumulative_seconds"],
            "training_limit_seconds": budget["limit_seconds"],
            "peak_sampled_conservative_training_memory_bytes": resource["peak_sampled_conservative_memory_bytes"],
            "memory_target_bytes": plan["limits"]["memory_gib"] * 1024**3,
            "external_logical_file_bytes": sum(item["bytes"] for item in external_records),
            "inventoried_project_file_bytes_excluding_manifest": sum(item["bytes"] for item in project_records),
            "artifact_limit_bytes": plan["limits"]["new_artifact_bytes"],
            "notes": "Budget includes learning checks/loading/training/validation, not total session time. "
                     "Memory sums sampled RSS and MPS driver memory conservatively. "
                     "Storage uses logical lengths; project inventory also includes preserved earlier files.",
        },
    }
    encoded = (json.dumps(content, indent=2, sort_keys=True) + "\n").encode()
    conservative_disk = (sum(item["bytes"] for item in project_records)
                         + sum(item["bytes"] for item in external_records) + len(encoded))
    if (not budget["within_budget"] or conservative_disk > plan["limits"]["new_artifact_bytes"]
            or resource["peak_sampled_conservative_memory_bytes"] > plan["limits"]["memory_gib"] * 1024**3):
        raise ValueError("Recorded campaign exceeds a declared resource limit")
    DESTINATION.write_bytes(encoded)
    verify()


def verify() -> None:
    manifest = json.loads(DESTINATION.read_text())
    count = 0
    for root, key in ((PROJECT, "project_files"), (Path(manifest["external_root"]), "external_files")):
        for expected in manifest[key]:
            actual = record(root / expected["path"], root)
            if actual != expected:
                raise ValueError(f"Evidence differs: {root / expected['path']}")
            count += 1
    print(json.dumps({"valid": True, "verified_files": count, "manifest": str(DESTINATION),
                      "manifest_sha256": hashlib.sha256(DESTINATION.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    args = parser.parse_args()
    {"create": create, "verify": verify}[args.command]()
