"""Create or verify the final Package A inventory without loading model code.

Create only after reports and documentation are final. The inventory is written
once, excludes itself, and preserves all existing evidence bytes. Verification
does not rewrite it. Historical inventories and corpora are referenced through
their pinned snapshot verification reports rather than copied into this one.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports/package_a/manifest.json"
PROJECT_DIRECTORIES = ("src", "tests", "scripts", "configs", "docs")
PROJECT_TOP_LEVEL = ("AGENTS.md", "README.md", "pyproject.toml", "requirements-macos-arm64.lock")
RUNTIME_CACHE_DIRECTORIES = ("__pycache__", ".pytest_cache")
RUNTIME_CACHE_SUFFIXES = (".pyc", ".pyo")
RUNTIME_CACHE_FILENAMES = (".DS_Store",)
SNAPSHOTS = {
    "stage2a": {
        "snapshot_commit": "e19a6cb959a72bdfbf3bb62d7b9c327dd01337a7",
        "manifest_sha256": "a027a087c66f22d4b9819498a3f45c1c4bbd967cde7ea65251e285e98b833db9",
        "verification_report": "reports/package_a/stage2a-integrity.json",
    },
    "stage2b": {
        "snapshot_commit": "60ceff6b7fd8d4d40f96d5ceec54a2cac720ba3c",
        "manifest_sha256": "7743a8582b39343ceef16ac3b3ff3373bbf8d5e437b6e46718517307b11fc87b",
        "verification_report": "reports/package_a/stage2b-integrity.json",
    },
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def record(scope, path, root):
    before = path.stat()
    sha256 = digest(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while being inventoried: {path}")
    return {"scope": scope, "path": path.relative_to(root).as_posix(),
            "bytes": after.st_size, "sha256": sha256}


def excluded_report(path, reports_root):
    """Avoid self-reference, including redirected verification command output."""
    if path == reports_root / "manifest.json":
        return True
    name = path.name.lower().replace("_", "-")
    return "inventory-verification" in name or "manifest-verification" in name


def files_under(root, exclude_runtime_caches=False):
    paths = {path for path in root.rglob("*") if path.is_file()}
    if exclude_runtime_caches:
        paths = {path for path in paths if not (
            any(part in RUNTIME_CACHE_DIRECTORIES for part in path.relative_to(root).parts)
            or path.suffix in RUNTIME_CACHE_SUFFIXES or path.name in RUNTIME_CACHE_FILENAMES)}
    return paths


def project_files(project_root, reports_root):
    paths = set()
    for folder in PROJECT_DIRECTORIES:
        paths.update(files_under(project_root / folder, exclude_runtime_caches=True))
    paths.update(project_root / name for name in PROJECT_TOP_LEVEL if (project_root / name).is_file())
    paths.update(p for p in files_under(reports_root) if not excluded_report(p, reports_root))
    return paths


def create():
    if MANIFEST.exists():
        raise FileExistsError("Package A inventory already exists; verify it instead of overwriting frozen evidence")
    config = json.loads((ROOT / "configs/package_a.json").read_text())
    external_root = (ROOT / config["stability"]["data_root"]).resolve()
    reports_root = ROOT / "reports/package_a"
    if not external_root.is_dir() or not reports_root.is_dir():
        raise FileNotFoundError("Both Package A report and external artifact roots must exist")
    historical = {}
    for name, expected in SNAPSHOTS.items():
        path = ROOT / expected["verification_report"]
        result = json.loads(path.read_text())
        if not result["valid"] or any(result[k] != expected[k] for k in ("snapshot_commit", "manifest_sha256")):
            raise ValueError(f"Historical snapshot verification does not match its pinned identity: {name}")
        historical[name] = {**expected, "verification_report_sha256": digest(path),
                            "verified_files": result["verified_files"]}
    entries = [record("project", p, ROOT) for p in sorted(project_files(ROOT, reports_root))]
    entries += [record("external", p, external_root) for p in sorted(files_under(external_root))]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    inventory = {
        "schema": "package-a-final-inventory-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {"project": str(ROOT), "reports": str(reports_root), "external": str(external_root)},
        "working_head": head,
        "working_tree_dirty": dirty,
        "config_sha256": digest(ROOT / "configs/package_a.json"),
        "historical_snapshots": historical,
        "project_directories": list(PROJECT_DIRECTORIES),
        "optional_top_level_files": list(PROJECT_TOP_LEVEL),
        "excluded_report_files": ["reports/package_a/manifest.json", "filenames containing inventory-verification or manifest-verification (case-insensitive; underscores treated as hyphens)"],
        "excluded_runtime_caches": {
            "scope": list(PROJECT_DIRECTORIES),
            "directory_names": list(RUNTIME_CACHE_DIRECTORIES),
            "file_suffixes": list(RUNTIME_CACHE_SUFFIXES),
            "file_names": list(RUNTIME_CACHE_FILENAMES),
            "reason": "Generated runtime and filesystem caches are ignored; actual source, configuration, documentation and evidence are preserved.",
        },
        "file_count": len(entries),
        "total_bytes_excluding_inventory": sum(e["bytes"] for e in entries),
        "files": entries,
        "limitations": [
            "The manifest itself is excluded; its SHA-256 is returned separately by create and verify.",
            "This is an inventory of current bytes excluding the explicitly listed runtime caches under project source directories. A Git commit alone does not identify an uncommitted working tree.",
            "Historical Stage 2A/2B inventories and old bulk corpora are not included or regenerated; pinned snapshot verification reports are referenced.",
            "Verification checks recorded bytes, not scientific validity, playing strength, permissions or an independent signature.",
        ],
    }
    # Exclusive creation prevents accidental replacement of a frozen inventory.
    with MANIFEST.open("x") as stream:
        json.dump(inventory, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"created": True, "manifest": str(MANIFEST), "file_count": len(entries),
            "project_files": sum(e["scope"] == "project" for e in entries),
            "external_files": sum(e["scope"] == "external" for e in entries),
            "total_bytes_excluding_inventory": inventory["total_bytes_excluding_inventory"],
            "manifest_bytes": MANIFEST.stat().st_size, "manifest_sha256": digest(MANIFEST)}


def safe_recorded_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid relative inventory path: {relative}")
    return root.joinpath(*path.parts)


def verify(check_extra=False):
    inventory = json.loads(MANIFEST.read_text())
    if inventory["schema"] != "package-a-final-inventory-v1":
        raise ValueError("Unexpected Package A inventory schema")
    roots = {k: Path(v) for k, v in inventory["roots"].items()}
    entries = inventory["files"]
    missing, changed, duplicates = [], [], []
    recorded = {"project": set(), "external": set()}
    for entry in entries:
        scope = entry["scope"]
        if scope not in recorded:
            raise ValueError(f"Unknown inventory scope: {scope}")
        path = safe_recorded_path(roots[scope], entry["path"])
        if path in recorded[scope]:
            duplicates.append(str(path))
        recorded[scope].add(path)
        if not path.is_file():
            missing.append(str(path))
        elif path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            changed.append(str(path))
    extras = []
    if check_extra:
        reports_files = {p for p in files_under(roots["reports"]) if not excluded_report(p, roots["reports"])}
        extras = sorted(str(p) for p in (reports_files - recorded["project"]) | (files_under(roots["external"]) - recorded["external"]))
    count_matches = inventory["file_count"] == len(entries)
    bytes_match = inventory["total_bytes_excluding_inventory"] == sum(e["bytes"] for e in entries)
    return {"valid": not (missing or changed or duplicates or extras) and count_matches and bytes_match,
            "manifest": str(MANIFEST), "manifest_sha256": digest(MANIFEST),
            "verified_files": len(entries) - len(missing) - len(changed),
            "recorded_files": len(entries), "file_count_matches": count_matches,
            "recorded_byte_total_matches": bytes_match,
            "missing": missing, "changed": changed, "duplicate_paths": duplicates,
            "checked_extra_files": check_extra, "extra_files": extras,
            "extra_file_scope": "Package A reports and external A2 artifacts; declared inventory/verification outputs excluded",
            "scope": "Recorded file bytes only. Historical snapshot evidence is referenced; the inventory itself is excluded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create")
    check = commands.add_parser("verify")
    check.add_argument("--check-extra", action="store_true", help="Also reject new files under Package A reports/external roots")
    args = parser.parse_args()
    try:
        result = create() if args.command == "create" else verify(args.check_extra)
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        result = {"valid": False, "failure": {"type": type(error).__name__, "message": str(error)}}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("created") or result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
