"""Small, model-free checks for the immutable Stage 2B evidence verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_stage2b_snapshot.py"
SPEC = importlib.util.spec_from_file_location("verify_stage2b_snapshot", SCRIPT)
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class Stage2BSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bulk = self.root / "artifact.bin"
        self.bulk.write_bytes(b"original bulk")
        self.project_path = "docs/STATUS.md"
        self.project_content = b"completed historical stage"
        self.entries = [
            self.entry("project", self.project_path, self.project_content),
            self.entry("bulk", str(self.bulk), self.bulk.read_bytes()),
        ]
        self.manifest = {"file_count": len(self.entries), "files": self.entries}
        self.raw = json.dumps(self.manifest).encode()
        self.blobs = {
            snapshot.MANIFEST_PATH: self.raw,
            self.project_path: self.project_content,
        }
        self.addCleanup(patch.stopall)
        patch.object(snapshot, "ROOT", self.root).start()
        patch.object(snapshot, "committed", side_effect=lambda path: self.blobs[path]).start()
        patch.object(snapshot, "MANIFEST_SHA256", hashlib.sha256(self.raw).hexdigest()).start()

    @staticmethod
    def entry(scope, path, content):
        return {"scope": scope, "path": path, "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()}

    def test_current_project_edits_do_not_replace_historical_evidence(self):
        current = self.root / self.project_path
        current.parent.mkdir()
        current.write_text("new stage, different bytes")
        result = snapshot.verify()
        self.assertTrue(result["valid"])
        self.assertEqual(result["verified_files"], 2)
        self.assertEqual(result["verified_project_files"], 1)
        self.assertEqual(result["verified_bulk_files"], 1)

    def test_changed_manifest_is_rejected_before_entries_are_trusted(self):
        self.blobs[snapshot.MANIFEST_PATH] = self.raw + b"\n"
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            snapshot.verify()

    def test_same_size_changed_historical_project_blob_is_rejected(self):
        self.blobs[self.project_path] = b"x" * len(self.project_content)
        with self.assertRaisesRegex(ValueError, "Historical Git evidence mismatch"):
            snapshot.verify()

    def test_same_size_changed_bulk_artifact_is_rejected(self):
        self.bulk.write_bytes(b"x" * self.bulk.stat().st_size)
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            snapshot.verify()

    def test_missing_bulk_artifact_is_rejected(self):
        self.bulk.unlink()
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            snapshot.verify()


if __name__ == "__main__":
    unittest.main()
