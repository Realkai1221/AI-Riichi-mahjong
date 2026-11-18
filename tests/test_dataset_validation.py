"""Real small replays and coherently rehashed corruption audit fixtures."""

import hashlib
import io
import json
import shutil

import numpy as np
import pytest

from riichi_ai.dataset import _pack_arrays, generate_dataset
from riichi_ai.dataset_validation import verify_dataset


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    path = tmp_path_factory.mktemp("verification-pilot")
    manifest = generate_dataset(path, matches=3, root_seed=2026101794)
    assert manifest["valid"], manifest["failure"]
    return path, manifest


def test_every_row_validated_and_first_middle_last_matches_replayed(pilot):
    path, manifest = pilot
    report = verify_dataset(path)
    assert report["valid"], report["errors"]
    assert report["checked_matches"] == report["replayed_matches"] == 3
    assert report["checked_artifacts"] == 9
    assert report["checked_rows"] == report["replayed_rows"] == manifest["decisions"]
    assert report["replayed_match_ids"] == [match["match_id"] for match in manifest["matches"]]


def _rewrite_array(path, area, mutate):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    relative = f"{area}/match-000000.npz"
    with np.load(io.BytesIO((path / relative).read_bytes()), allow_pickle=False) as shard:
        arrays = {key: shard[key] for key in shard.files}
    mutate(arrays)
    raw = _pack_arrays(arrays)
    (path / relative).write_bytes(raw)
    manifest["matches"][0]["artifacts"][relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))


def test_replay_detects_rehashed_finite_actor_change(pilot, tmp_path):
    shutil.copytree(pilot[0], tmp_path / "changed")
    path = tmp_path / "changed"
    _rewrite_array(path, "actor", lambda arrays: arrays["globals"].__setitem__((0, 0), arrays["globals"][0, 0] + 1))
    report = verify_dataset(path, replay_matches=1)
    assert not report["valid"]
    assert any("Actor replay differs at row 0: globals" in error["message"] for error in report["errors"])


def test_all_rows_check_rejects_rehashed_wrong_terminal_target(pilot, tmp_path):
    shutil.copytree(pilot[0], tmp_path / "changed")
    path = tmp_path / "changed"
    _rewrite_array(path, "targets", lambda arrays: arrays["final_rank"].__setitem__(0, arrays["final_rank"][0] % 4 + 1))
    report = verify_dataset(path, replay_matches=0)
    assert not report["valid"]
    assert any("Terminal rank labels" in error["message"] for error in report["errors"])


def test_corrupted_auditor_is_not_hidden_by_actor_only_integrity(pilot, tmp_path):
    shutil.copytree(pilot[0], tmp_path / "changed")
    path = tmp_path / "changed"
    audit = path / "auditor/match-000000.json.gz"
    audit.write_bytes(audit.read_bytes() + b"changed")
    report = verify_dataset(path, replay_matches=0)
    assert not report["valid"]
    assert any("Artifact hash/length mismatch" in error["message"] for error in report["errors"])


def test_verifier_rejects_manifest_split_reassignment(pilot, tmp_path):
    shutil.copytree(pilot[0], tmp_path / "changed")
    path = tmp_path / "changed"
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["validation"] = [manifest["splits"]["train"][0]]
    manifest_path.write_text(json.dumps(manifest))
    report = verify_dataset(path)
    assert not report["valid"] and report["checked_artifacts"] == 0
    assert "split membership" in report["errors"][0]["message"]
