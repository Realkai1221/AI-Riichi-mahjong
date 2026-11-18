"""Verify completed Stage 2B evidence against its immutable Git snapshot.

Project sources and documentation may evolve after Stage 2B. Their historical
bytes are checked in Git; the original bulk artifacts remain checked on disk.
This command is read-only and never regenerates the historical inventory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "60ceff6b7fd8d4d40f96d5ceec54a2cac720ba3c"
MANIFEST_PATH = "reports/stage2b/manifest.json"
MANIFEST_SHA256 = "7743a8582b39343ceef16ac3b3ff3373bbf8d5e437b6e46718517307b11fc87b"


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


def verify() -> dict:
    raw = committed(MANIFEST_PATH)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != MANIFEST_SHA256:
        raise ValueError("Historical Stage 2B manifest does not match its pinned SHA-256")
    manifest = json.loads(raw)
    entries = manifest["files"]
    if manifest["file_count"] != len(entries):
        raise ValueError("Historical Stage 2B inventory file count mismatch")
    counts = {"project": 0, "bulk": 0}
    for entry in entries:
        if entry["scope"] == "project":
            content = committed(entry["path"])
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Historical Git evidence mismatch: {entry['path']}")
        elif entry["scope"] == "bulk":
            path = Path(entry["path"])
            if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError(f"External Stage 2B artifact missing or changed: {path}")
        else:
            raise ValueError(f"Unknown Stage 2B evidence scope: {entry['scope']}")
        counts[entry["scope"]] += 1
    return {
        "valid": True,
        "snapshot_commit": COMMIT,
        "verified_files": sum(counts.values()),
        "verified_project_files": counts["project"],
        "verified_bulk_files": counts["bulk"],
        "manifest_sha256": manifest_sha256,
        "scope": (
            "Project files checked in the completed Stage 2B Git snapshot; "
            "unchanged external bulk artifacts checked on disk. "
            "Current project sources and documentation may evolve."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
