"""Bounded streaming, strict shards, causal prefixes and exact restart fixtures."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from riichi_ai.dataset import _pack_arrays
from riichi_ai.features import ACTION_FIELDS, GLOBAL_FEATURES, HISTORY_FIELDS, TILE_CHANNELS
from riichi_ai.semantics import ACTION_KINDS
from riichi_ai.stage3f_stream import ACTOR_SCHEMA, CORPUS_SCHEMA, TARGET_SCHEMA, StreamDataset, validate_arrays


def arrays():
    rows = 8
    seat = np.tile(np.arange(4, dtype=np.int8), 2)
    prefix = np.repeat([1, 3], 4).astype(np.int32)
    tokens = np.zeros((rows * 2, ACTION_FIELDS), dtype=np.int16)
    tokens[::2, 0] = ACTION_KINDS.index("DISCARD") + 1
    tokens[1::2, 0] = ACTION_KINDS.index("PASS") + 1
    tokens[::2, 1] = 1
    efficiency = np.zeros((rows * 2, 4), dtype=np.float32)
    efficiency[::2, 3] = 1
    history = np.zeros((12, HISTORY_FIELDS), dtype=np.int16)
    history[:, 0] = 1
    globals_ = np.zeros((rows, GLOBAL_FEATURES), dtype=np.float32)
    globals_[:, 24:28] = np.eye(4, dtype=np.float32)[seat]
    globals_[:, 41] = prefix / 128
    actor = {"tiles": np.zeros((rows, 34, TILE_CHANNELS), dtype=np.float32), "globals": globals_,
        "action_tokens": tokens, "candidate_efficiency": efficiency,
        "candidate_offsets": np.arange(0, rows * 2 + 1, 2, dtype=np.int64),
        "history_tokens": history, "history_start": (seat * 3).astype(np.int64),
        "history_length": prefix.astype(np.int16), "history_segment_start": (seat * 3).astype(np.int64),
        "history_segment_length": np.full(rows, 3, dtype=np.int32), "history_prefix_length": prefix}
    chosen = np.arange(rows, dtype=np.int16) % 2
    targets = {"chosen_index": chosen, "seat": seat, "step": np.arange(rows, dtype=np.int32),
        "hand_id": np.zeros(rows, dtype=np.int32), "final_rank": seat + 1,
        "final_utility": np.asarray([1.5, .5, -.5, -1.5], dtype=np.float32)[seat],
        "behavior_probability": np.asarray([.25, .75], dtype=np.float64)[chosen],
        "behavior_id": seat.copy(), "behavior_probabilities": np.tile([.25, .75], rows).astype(np.float64)}
    return actor, targets


def write_corpus(directory: Path, matches=3):
    directory.mkdir(exist_ok=True)
    reports = []
    for index in range(matches):
        name = f"match-{index:04d}"
        actor, targets = arrays()
        artifacts = {}
        for area, group in (("actor", actor), ("targets", targets)):
            relative = f"{area}/{name}.npz"
            payload = _pack_arrays(group)
            path = directory / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
            artifacts[relative] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        # This deliberately nonexistent path must never be opened by the loader.
        artifacts[f"auditor/{name}.json.gz"] = {"sha256": "0" * 64, "bytes": 99}
        reports.append({"match_id": name, "completed": True, "rows": 8, "artifacts": artifacts,
            "source_role": "development-only", "training_eligible": False, "ppo_eligible": False,
            "actor_schema": ACTOR_SCHEMA, "target_schema": TARGET_SCHEMA,
            "uncompressed_actor_bytes": sum(a.nbytes for a in actor.values()),
            "uncompressed_target_bytes": sum(a.nbytes for a in targets.values())})
    manifest = {"schema": CORPUS_SCHEMA, "complete": True, "source_role": "development-only",
        "matches": reports, "splits": {"development": [r["match_id"] for r in reports]}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def identities(batches):
    return [(b.match_id, row) for b in batches for row in b.row_indices.tolist()]


def test_complete_shards_are_streamed_once_with_actor_target_separation(tmp_path):
    write_corpus(tmp_path)
    dataset = StreamDataset(tmp_path)
    assert dataset.shards_opened == 0
    batches = list(dataset.iter_batches(batch_rows=3, seed=71))
    assert len(identities(batches)) == len(set(identities(batches))) == 24
    assert dataset.shards_opened == 3
    assert dataset.loaded_match is None
    for batch in batches:
        assert batch.rows <= 3
        assert not set(batch.actor) & {"seat", "final_utility", "behavior_probability", "hand_id"}
        assert np.all(batch.actor["candidate_mask"])
        assert np.allclose(batch.targets["behavior_probabilities"].sum(1), 1)
        assert np.all(batch.actor["history"][~batch.actor["history_mask"]] == 0)
    assert not (tmp_path / "auditor").exists()


def test_order_is_reproducible_and_changes_between_epochs(tmp_path):
    write_corpus(tmp_path)
    one = identities(StreamDataset(tmp_path).iter_batches(seed=17, epoch=1))
    assert one == identities(StreamDataset(tmp_path).iter_batches(seed=17, epoch=1))
    assert one != identities(StreamDataset(tmp_path).iter_batches(seed=17, epoch=2))


@pytest.mark.parametrize("stop_after", [1, 3, 5, 9])
def test_exact_cursor_resume_at_midmatch_match_and_terminal_boundaries(tmp_path, stop_after):
    write_corpus(tmp_path)
    expected = list(StreamDataset(tmp_path).iter_batches(seed=33, epoch=4, batch_rows=3))
    prefix = expected[:stop_after]
    resumed = list(StreamDataset(tmp_path).iter_batches(seed=33, epoch=4, batch_rows=3, cursor=prefix[-1].cursor))
    assert identities(prefix + resumed) == identities(expected)
    for actual, wanted in zip(resumed, expected[stop_after:]):
        assert actual.cursor == wanted.cursor
        for group in ("actor", "targets"):
            for key in getattr(actual, group):
                np.testing.assert_array_equal(getattr(actual, group)[key], getattr(wanted, group)[key])


@pytest.mark.parametrize("setting,value", [("seed", 9), ("epoch", 5), ("batch_rows", 4),
                                          ("manifest_sha256", "0" * 64), ("row_position", 1)])
def test_cursor_rejects_changed_identity_or_nonboundary(tmp_path, setting, value):
    write_corpus(tmp_path)
    first = next(StreamDataset(tmp_path).iter_batches(seed=33, epoch=4, batch_rows=3))
    cursor = {**first.cursor, setting: value}
    with pytest.raises(ValueError, match="cursor|Resume"):
        list(StreamDataset(tmp_path).iter_batches(seed=33, epoch=4, batch_rows=3, cursor=cursor))


@pytest.mark.parametrize("mutation", ["incomplete", "wrong_role", "duplicate", "split_overlap", "unknown_split_member", "missing_artifact"])
def test_manifest_rejection(tmp_path, mutation):
    manifest = write_corpus(tmp_path)
    if mutation == "incomplete":
        manifest["matches"][0]["completed"] = False
    elif mutation == "wrong_role":
        manifest["source_role"] = "sealed-confirmation"
    elif mutation == "duplicate":
        manifest["matches"].append(manifest["matches"][0])
    elif mutation == "split_overlap":
        manifest["splits"]["other"] = [manifest["matches"][0]["match_id"]]
    elif mutation == "unknown_split_member":
        manifest["splits"]["development"].append("unknown")
    else:
        manifest["matches"][0]["artifacts"].pop("actor/match-0000.npz")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        StreamDataset(tmp_path)


def test_unmanifested_shard_rejected(tmp_path):
    write_corpus(tmp_path)
    (tmp_path / "actor/unknown.npz").write_bytes(b"unmanifested")
    with pytest.raises(ValueError, match="Unmanifested"):
        StreamDataset(tmp_path)


def test_corrupt_numeric_hash_rejected_before_decode(tmp_path):
    write_corpus(tmp_path, matches=1)
    (tmp_path / "actor/match-0000.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        next(StreamDataset(tmp_path).iter_batches())


def test_shard_decompression_bound_rejected_before_allocation(tmp_path):
    write_corpus(tmp_path, matches=1)
    with pytest.raises(MemoryError, match="decompression"):
        next(StreamDataset(tmp_path, max_shard_bytes=1000, memory_limit_bytes=3000).iter_batches())


def test_batch_memory_bound_checked(tmp_path):
    manifest = write_corpus(tmp_path, matches=1)
    shard_bytes = manifest["matches"][0]["uncompressed_actor_bytes"] + manifest["matches"][0]["uncompressed_target_bytes"]
    with pytest.raises(MemoryError, match="batch"):
        next(StreamDataset(tmp_path, max_shard_bytes=shard_bytes + 5000,
            memory_limit_bytes=2 * (shard_bytes + 5000) + 1).iter_batches(batch_rows=8))


@pytest.mark.parametrize("mutation", ["future_prefix", "cross_segment", "alias_segment", "backward_prefix", "bad_chosen", "bad_probability", "bad_utility", "bad_seat", "nonfinite", "extra_actor", "negative_token", "visible_prefix_mismatch"])
def test_row_validation_rejects_causal_or_target_corruption(mutation):
    actor, targets = arrays()
    if mutation == "visible_prefix_mismatch":
        actor["globals"][0, 41] = 3 / 128
    elif mutation == "future_prefix":
        actor["history_prefix_length"][0] = 4
    elif mutation == "cross_segment":
        actor["history_start"][0] += 1
    elif mutation == "alias_segment":
        actor["history_segment_start"][1] = 0
        actor["history_start"][1] = 0
    elif mutation == "backward_prefix":
        actor["history_prefix_length"][4] = actor["history_length"][4] = 0
        actor["globals"][4, 41] = 0
    elif mutation == "bad_chosen":
        targets["chosen_index"][0] = 2
    elif mutation == "bad_probability":
        targets["behavior_probability"][0] = .8
    elif mutation == "bad_utility":
        targets["final_utility"][0] = 0
    elif mutation == "bad_seat":
        targets["seat"][0] = 4
    elif mutation == "nonfinite":
        actor["tiles"][0, 0, 0] = np.nan
    elif mutation == "extra_actor":
        actor["final_scores"] = np.zeros(4)
    else:
        actor["action_tokens"][0, 0] = -1
    with pytest.raises(ValueError):
        validate_arrays(actor, targets, 8)


def test_valid_fixture_and_actor_numpy_to_torch_staging(tmp_path):
    import torch
    write_corpus(tmp_path, matches=1)
    batch = next(StreamDataset(tmp_path).iter_batches(batch_rows=3))
    tensors = batch.torch_actor()
    assert tensors["history"].dtype == torch.long
    assert tensors["history_mask"].dtype == torch.bool
    for key, value in tensors.items():
        np.testing.assert_array_equal(value.numpy(), batch.actor[key])
        assert value.is_contiguous()
    tensors["tiles"][0, 0, 0] = 9
    assert batch.actor["tiles"][0, 0, 0] == 0


def test_value_probe_backward_is_read_only_and_candidate_masked(tmp_path):
    import torch
    from riichi_ai.model import PolicyNetwork, _state_hash, efficiency_model_config
    write_corpus(tmp_path, matches=1)
    batch = next(StreamDataset(tmp_path).iter_batches(batch_rows=3))
    module_path = Path(__file__).resolve().parents[1] / "scripts/stage3f_stream_benchmark.py"
    spec = importlib.util.spec_from_file_location("stage3f_stream_benchmark_fixture", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = efficiency_model_config({"suit_width": 16, "history_width": 8, "hidden_width": 32,
                                      "context_width": 16, "candidate_width": 8})
    model = PolicyNetwork(config)
    for parameter in model.rank_head.parameters():
        parameter.requires_grad_(False)
    before = _state_hash(model.state_dict())
    result = module.numerical_snapshot(model, batch, "cpu")
    assert result["finite"] and result["input_exact"] and result["gradient_l2"] > 0
    assert _state_hash(model.state_dict()) == before
    assert all(p.grad is None for p in model.rank_head.parameters())
