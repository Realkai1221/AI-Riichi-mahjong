"""Model-free checks for immutable Package A diagnosis evidence verification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_package_a_diagnosis_snapshot.py"
SPEC = importlib.util.spec_from_file_location("verify_package_a_diagnosis_snapshot", SCRIPT)
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class PackageADiagnosisSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.external = self.root / "external"
        self.external.mkdir()
        self.artifact = self.external / "artifact.bin"
        self.artifact.write_bytes(b"original external bytes")
        self.project_path = "docs/STATUS.md"
        self.project_content = b"completed historical stage"
        self.entries = [
            self.entry("project", self.project_path, self.project_content),
            self.entry("external", "artifact.bin", self.artifact.read_bytes()),
        ]
        self.manifest = {
            "schema": "package-a-diagnosis-inventory-v1",
            "file_count": len(self.entries),
            "total_bytes": sum(entry["bytes"] for entry in self.entries),
            "roots": {"external": str(self.external)},
            "files": self.entries,
        }
        self.blobs = {self.project_path: self.project_content}
        self.addCleanup(patch.stopall)
        patch.object(snapshot, "ROOT", self.root).start()
        patch.object(snapshot, "committed", side_effect=lambda path: self.blobs[path]).start()
        patch.object(snapshot, "EXPECTED_COUNTS", {"project": 1, "external": 1}).start()
        self.pin_fixture_manifest()

    @staticmethod
    def entry(scope, path, content):
        return {"scope": scope, "path": path, "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()}

    def pin_fixture_manifest(self):
        raw = json.dumps(self.manifest).encode()
        self.blobs[snapshot.MANIFEST_PATH] = raw
        patch.object(snapshot, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest()).start()

    def test_current_project_edits_do_not_replace_historical_evidence(self):
        current = self.root / self.project_path
        current.parent.mkdir()
        current.write_text("new stage, different bytes")
        result = snapshot.verify()
        self.assertTrue(result["valid"])
        self.assertEqual(result["verified_files"], 2)
        self.assertEqual(result["verified_project_files"], 1)
        self.assertEqual(result["verified_external_files"], 1)

    def test_changed_manifest_is_rejected_before_entries_are_trusted(self):
        self.blobs[snapshot.MANIFEST_PATH] += b"\n"
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            snapshot.verify()

    def test_same_size_changed_historical_project_blob_is_rejected(self):
        self.blobs[self.project_path] = b"x" * len(self.project_content)
        with self.assertRaisesRegex(ValueError, "Historical Git evidence mismatch"):
            snapshot.verify()

    def test_same_size_changed_external_artifact_is_rejected(self):
        self.artifact.write_bytes(b"x" * self.artifact.stat().st_size)
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            snapshot.verify()

    def test_missing_external_artifact_is_rejected(self):
        self.artifact.unlink()
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            snapshot.verify()

    def test_inventory_count_or_byte_total_mismatch_is_rejected(self):
        for field in ("file_count", "total_bytes"):
            with self.subTest(field=field):
                original = self.manifest[field]
                self.manifest[field] = original + 1
                self.pin_fixture_manifest()
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    snapshot.verify()
                self.manifest[field] = original

    def test_duplicate_or_unsafe_inventory_paths_are_rejected(self):
        for path, expected in ((self.project_path, "Duplicate"), ("../outside", "Invalid")):
            with self.subTest(path=path):
                self.entries[1] = self.entry("project", path, self.project_content)
                self.manifest["total_bytes"] = sum(entry["bytes"] for entry in self.entries)
                self.pin_fixture_manifest()
                with self.assertRaisesRegex(ValueError, expected):
                    snapshot.verify()

    def test_unknown_scope_and_manifest_self_reference_are_rejected(self):
        for scope, path, expected in (
            ("other", "artifact.bin", "Unknown"),
            ("project", snapshot.MANIFEST_PATH, "exclude itself"),
        ):
            with self.subTest(scope=scope):
                self.entries[1] = self.entry(scope, path, b"original external bytes")
                self.pin_fixture_manifest()
                with self.assertRaisesRegex(ValueError, expected):
                    snapshot.verify()

    def test_scope_counts_are_pinned(self):
        with patch.object(snapshot, "EXPECTED_COUNTS", {"project": 2, "external": 0}):
            with self.assertRaisesRegex(ValueError, "scope counts mismatch"):
                snapshot.verify()


if __name__ == "__main__":
    unittest.main()
