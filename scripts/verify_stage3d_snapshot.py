"""Verify Stage 3D without requiring later working-tree documents to stay frozen.

Project bytes are read from its completed Git snapshot. External training files
and local book evidence remain checked on disk against the pinned manifest.
This stdlib-only command does not load models or run numerical diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "7d3380f2d9c595b4da6d99871ede8ccd92380608"
MANIFEST_PATH = "reports/stage3d/manifest.json"
MANIFEST_SHA256 = "30234190420f681c5f7ec857288da04df102e9dc030749aa6c950cd8b4f89058"
EXPECTED_COUNTS = {"project": 179, "external": 1025, "books": 12}
EXPECTED_BYTES = 267783998


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def committed(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{COMMIT}:{path}"], cwd=ROOT,
        check=True, capture_output=True,
    ).stdout


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value: str) -> PurePosixPath:
    require(isinstance(value, str) and bool(value) and "\\" not in value,
            "Inventory paths must be nonempty portable relative paths")
    path = PurePosixPath(value)
    require(not path.is_absolute() and bool(path.parts) and ".." not in path.parts
            and path.as_posix() == value, f"Invalid historical inventory path: {value}")
    return path


def artifact_at(root: Path, relative: PurePosixPath) -> Path:
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(),
            f"Historical artifact root is missing or linked: {root}")
    artifact = root.joinpath(*relative.parts)
    require(artifact.resolve().is_relative_to(root.resolve()), "Artifact escapes its root")
    current = root
    for component in relative.parts:
        current /= component
        require(not current.is_symlink(), f"Linked historical artifact: {current}")
    return artifact


def verify() -> dict:
    raw = committed(MANIFEST_PATH)
    manifest_hash = hashlib.sha256(raw).hexdigest()
    require(manifest_hash == MANIFEST_SHA256, "Historical Stage 3D manifest hash changed")
    manifest = json.loads(raw)
    require(manifest["schema"] == "stage3d-inventory-v1", "Unexpected inventory schema")
    entries = manifest["files"]
    require(manifest["file_count"] == len(entries) == sum(EXPECTED_COUNTS.values()),
            "Historical Stage 3D file count mismatch")
    require(manifest["scope_counts"] == EXPECTED_COUNTS, "Historical scope counts changed")
    require(manifest["total_bytes"] == sum(entry["bytes"] for entry in entries) == EXPECTED_BYTES,
            "Historical Stage 3D byte total mismatch")
    expected_roots = {
        "project": ROOT,
        "external": Path.home() / "Library/Application Support/RiichiAI/stage3d",
        "books": ROOT / "Books/stage3d",
    }
    require(manifest["roots"] == {key: str(path) for key, path in expected_roots.items()},
            "Historical Stage 3D evidence roots differ")
    counts = dict.fromkeys(EXPECTED_COUNTS, 0)
    seen = set()
    for entry in entries:
        scope, relative = entry["scope"], relative_path(entry["path"])
        require(scope in counts, f"Unexpected evidence scope: {scope}")
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0, "Invalid byte count")
        require(isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in entry["sha256"]),
                "Invalid artifact SHA-256")
        key = (scope, relative.as_posix())
        require(key not in seen, f"Duplicate historical evidence: {key}")
        seen.add(key)
        if scope == "project":
            require(relative.as_posix() not in (
                MANIFEST_PATH, "reports/stage3d/inventory-verification.json"),
                "Historical inventory must exclude its own manifest/verification")
            content = committed(relative.as_posix())
            require(len(content) == entry["bytes"]
                    and hashlib.sha256(content).hexdigest() == entry["sha256"],
                    f"Historical project bytes changed: {relative}")
        else:
            artifact = artifact_at(expected_roots[scope], relative)
            require(artifact.is_file() and artifact.stat().st_size == entry["bytes"]
                    and digest(artifact) == entry["sha256"],
                    f"Historical local evidence missing or changed: {artifact}")
        counts[scope] += 1
    require(counts == EXPECTED_COUNTS, f"Historical Stage 3D scope totals differ: {counts}")
    return {
        "valid": True,
        "snapshot_commit": COMMIT,
        "manifest_sha256": manifest_hash,
        "verified_files": sum(counts.values()),
        "verified_project_files": counts["project"],
        "verified_external_files": counts["external"],
        "verified_book_files": counts["books"],
        "verified_bytes_excluding_inventory": EXPECTED_BYTES,
        "scope": (
            "Project bytes checked in the completed Stage 3D Git snapshot; "
            "unchanged external training and local book artifacts checked on disk. "
            "Current project files may evolve. This verifies recorded identities, "
            "not playing strength or strategic label correctness."
        ),
    }


def main() -> int:
    try:
        result = verify()
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        result = {"valid": False, "failure": {
            "type": type(error).__name__, "message": str(error)}}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
