"""Verify completed Package A evidence against its immutable Git snapshot.

Project sources and documentation may evolve during later investigations. Their
historical bytes are checked in Git; the original external artifacts are checked
on disk. This command is read-only and does not regenerate an inventory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c7ae395babcbc56d056a37995ad23952b9d9d4ba"
MANIFEST_PATH = "reports/package_a/manifest.json"
MANIFEST_SHA256 = "78891fb2a16653329500d21f4f2e4249ec1e00c9c3c4074b5b6e580ac4ae971b"
EXPECTED_COUNTS = {"project": 146, "external": 22}


def committed(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{COMMIT}:{path}"], cwd=ROOT,
        check=True, capture_output=True,
    ).stdout


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid historical inventory path: {value}")
    return path


def verify() -> dict:
    raw = committed(MANIFEST_PATH)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != MANIFEST_SHA256:
        raise ValueError("Historical Package A manifest does not match its pinned SHA-256")
    manifest = json.loads(raw)
    if manifest["schema"] != "package-a-final-inventory-v1":
        raise ValueError("Unexpected historical Package A inventory schema")
    entries = manifest["files"]
    if manifest["file_count"] != len(entries):
        raise ValueError("Historical Package A inventory file count mismatch")
    if manifest["total_bytes_excluding_inventory"] != sum(entry["bytes"] for entry in entries):
        raise ValueError("Historical Package A inventory byte total mismatch")
    external_root = Path(manifest["roots"]["external"])
    if not external_root.is_absolute():
        raise ValueError("Historical Package A external root must be absolute")
    counts = {"project": 0, "external": 0}
    seen = set()
    for entry in entries:
        scope = entry["scope"]
        path = relative_path(entry["path"])
        if scope not in counts:
            raise ValueError(f"Unknown Package A evidence scope: {scope}")
        key = (scope, path.as_posix())
        if key in seen:
            raise ValueError(f"Duplicate historical inventory entry: {scope}:{path}")
        seen.add(key)
        if scope == "project":
            if path.as_posix() == MANIFEST_PATH:
                raise ValueError("Historical Package A inventory must exclude itself")
            content = committed(path.as_posix())
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Historical Git evidence mismatch: {path}")
        else:
            artifact = external_root.joinpath(*path.parts)
            if not artifact.is_file() or artifact.stat().st_size != entry["bytes"] or digest(artifact) != entry["sha256"]:
                raise ValueError(f"External Package A artifact missing or changed: {artifact}")
        counts[scope] += 1
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Historical Package A scope counts mismatch: {counts}")
    return {
        "valid": True,
        "snapshot_commit": COMMIT,
        "verified_files": sum(counts.values()),
        "verified_project_files": counts["project"],
        "verified_external_files": counts["external"],
        "manifest_sha256": manifest_sha256,
        "verified_bytes_excluding_inventory": manifest["total_bytes_excluding_inventory"],
        "scope": (
            "Project files checked in the completed Package A Git snapshot; "
            "unchanged external artifacts checked on disk. The pinned inventory "
            "excludes itself. Current project files may evolve. This verifies "
            "recorded file identities, not scientific acceptance or playing strength."
        ),
    }


def main() -> int:
    try:
        result = verify()
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        result = {"valid": False, "failure": {"type": type(error).__name__, "message": str(error)}}
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
