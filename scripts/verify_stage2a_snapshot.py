"""Verify Stage 2A's original inventory against its immutable Git snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e19a6cb959a72bdfbf3bb62d7b9c327dd01337a7"


def committed(path: str) -> bytes:
    return subprocess.run(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT,
                          check=True, capture_output=True).stdout


def verify() -> dict:
    raw = committed("reports/stage2/manifest.json")
    manifest = json.loads(raw)
    count = 0
    for entry in manifest["project_files"]:
        content = committed(entry["path"])
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Historical Git evidence mismatch: {entry['path']}")
        count += 1
    external = Path(manifest["external_root"])
    for entry in manifest["external_files"]:
        content = (external / entry["path"]).read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"External Stage 2A artifact mismatch: {entry['path']}")
        count += 1
    return {"valid": True, "snapshot_commit": COMMIT, "verified_files": count,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "Project files checked in the completed Stage 2A Git snapshot; unchanged external bulk artifacts checked on disk. Current Stage 2B sources intentionally evolve."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
