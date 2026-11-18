"""Small real-match tests for data isolation, alignment and reproducibility."""

import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from riichi_ai.dataset import generate_dataset, load_split, plan_matches, validate_manifest
from riichi_ai.features import encode_view
from riichi_ai.player import PlayerAdapter
from riichi_ai.semantics import semantic_actions
from riichi_ai.simulator import create_env


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic-pilot")
    manifest = generate_dataset(path, matches=3, root_seed=2026101791)
    assert manifest["valid"], manifest["failure"]
    return path, manifest


def test_plan_splits_independent_matches_before_seat_extraction():
    first = plan_matches(1000, 2026101701)
    assert first == plan_matches(1000, 2026101701)
    assert [sum(r["split"] == split for r in first) for split in ("train", "validation", "test")] == [800, 100, 100]
    assert len({r["seed"] for r in first}) == 1000
    assert len({r["independent_group"] for r in first}) == 1000
    assert set(r["seed"] for r in first).isdisjoint(r["seed"] for r in plan_matches(1000, 2026101702))
    with pytest.raises(ValueError, match="Stage 1"):
        plan_matches(10, 2026101601)


def test_plan_rejects_derived_development_seed_collision(monkeypatch):
    from riichi_ai import dataset
    seed = plan_matches(2, 99)[0]["seed"]
    monkeypatch.setattr(dataset, "_stage1_seeds", lambda: {seed})
    with pytest.raises(ValueError, match="collides"):
        plan_matches(2, 99)


def test_completed_rows_have_only_true_terminal_rank_targets(pilot):
    path, manifest = pilot
    data = load_split(path, "train")
    by_id = {m["match_id"]: m for m in manifest["matches"]}
    assert len(data) == sum(by_id[name]["decisions"] for name in data.match_ids)
    for sample in data:
        assert sample["rank_target"] == by_id[sample["match_id"]]["final_ranks"][sample["seat"]] - 1
        assert np.isclose(sample["policy_target"].sum(), 1)
        assert sample["best_mask"][sample["chosen_index"]]
        assert np.array_equal(sample["policy_target"] > 0, sample["best_mask"])
        assert sample["optional"] == (len(sample["policy_target"]) > 1)
        assert set(sample["encoded"]) == {"tiles", "globals", "history", "history_mask", "action_tokens"}
    assert manifest["semantic_forced"] >= manifest["physical_forced"]
    assert manifest["split_leakage"]["exact_duplicate_game_hashes"] == []
    assert manifest["closed_hand_limitation"]


def test_actor_loader_does_not_read_or_depend_on_auditor_artifacts(pilot, tmp_path):
    original, _ = pilot
    path = tmp_path / "copied"
    shutil.copytree(original, path)
    expected = load_split(path, "train", max_matches=1)[0]
    shutil.rmtree(path / "auditor")
    actual = load_split(path, "train", max_matches=1)[0]
    for key in expected["encoded"]:
        assert np.array_equal(expected["encoded"][key], actual["encoded"][key])
    assert np.array_equal(expected["policy_target"], actual["policy_target"])
    with pytest.raises(NotImplementedError):
        load_split(path, "train", include_beliefs=True)


def test_deduplicated_history_and_candidates_replay_to_identical_actor_inputs(pilot):
    path, manifest = pilot
    first = manifest["matches"][0]
    audit = json.loads(gzip.decompress((path / f"auditor/{first['match_id']}.json.gz").read_bytes()))
    data = load_split(path, "train", max_matches=1)
    env, observations = create_env(first["seed"])
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    row = 0
    for batch in audit["policy_requests"]:
        views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
        chosen = {}
        for request in batch["requests"]:
            seat = request["seat"]
            view = views[seat]
            actions = semantic_actions(view)
            expected = encode_view(view, actions)
            sample = data[row]
            assert request["row"] == row
            assert (sample["seat"], sample["step"], sample["hand_id"]) == (seat, batch["step"], view.hand_id)
            for key in expected:
                assert np.array_equal(expected[key], sample["encoded"][key]), (row, key)
            assert np.array_equal(sample["best_mask"], request["best_mask"])
            selected = actions[request["chosen_index"]].representative
            assert selected.token == request["submitted_action"]
            assert request["behavior_probability"] == 1
            chosen[seat] = adapters[seat].resolve(selected, observations[seat])
            row += 1
        observations = env.step(chosen)
    assert env.done() and row == len(data)
    assert env.mjai_log == audit["canonical_events"]
    with np.load(path / f"actor/{first['match_id']}.npz", allow_pickle=False) as arrays:
        # Prefix references reconstruct histories without saving every prefix.
        assert len(arrays["history_tokens"]) < int(arrays["history_length"].sum())


def test_training_prefixes_and_optional_filter_preserve_value_rows(pilot):
    path, _ = pilot
    first = load_split(path, "train", max_matches=1)
    both = load_split(path, "train", max_matches=2)
    optional = load_split(path, "train", max_matches=2, optional_only=True)
    assert first.match_ids == both.match_ids[:1]
    assert len(optional) == sum(sample["optional"] for sample in both)
    assert len(optional) < len(both)
    assert len(load_split(path, "validation")) == 0  # Tiny pilots can have empty holdouts.
    with pytest.raises(ValueError, match="only"):
        load_split(path, "train", max_matches=3)


def test_preloaded_and_bounded_cache_samples_agree(pilot):
    path, _ = pilot
    eager = load_split(path, "train", preload=True)
    bounded = load_split(path, "train", preload=False, cache_matches=1)
    assert len(eager._cache) == 2 and len(bounded._cache) == 1
    for index in (0, len(eager) - 1, 3, len(eager) // 2):
        one, two = eager[index], bounded[index]
        assert one["match_id"] == two["match_id"]
        for key in one["encoded"]:
            assert np.array_equal(one["encoded"][key], two["encoded"][key])
    with pytest.raises(MemoryError):
        load_split(path, "train", memory_limit_bytes=1)


def test_truncated_match_has_a_failure_record_and_no_terminal_training_shards(tmp_path):
    manifest = generate_dataset(tmp_path, matches=2, root_seed=2026101792, max_steps=1)
    assert not manifest["valid"]
    assert manifest["completed_matches"] == 0
    assert manifest["failure"]["type"] == "TruncatedMatch"
    assert manifest["rejected_matches"] == ["match-000000"]
    assert manifest["not_written_matches"] == ["match-000001"]
    assert not (tmp_path / "actor").exists()
    audit = json.loads(gzip.decompress((tmp_path / "auditor/match-000000.json.gz").read_bytes()))
    assert audit["final_ranks"] is None
    with pytest.raises(ValueError, match="valid"):
        load_split(tmp_path, "train")


def test_rare_path_counts_separate_unknown_from_zero(pilot):
    _, manifest = pilot
    assert manifest["observed_rare_path_counts"]["ankan_declarations"] == 0
    for match in manifest["matches"]:
        for record in match["rare_paths"]["uninstrumented"].values():
            assert record == {"status": "not_instrumented", "count": None}
    assert "riichi_refused_for_wall_length" in manifest["uninstrumented_rare_paths"]


def test_parallel_and_serial_outputs_match_exactly(pilot, tmp_path):
    _, serial = pilot
    parallel = generate_dataset(tmp_path, matches=3, root_seed=2026101791, workers=2)
    assert parallel["valid"], parallel["failure"]
    assert serial["splits"] == parallel["splits"]
    for first, second in zip(serial["matches"], parallel["matches"]):
        assert first["seed"] == second["seed"]
        assert first["game_hash"] == second["game_hash"]
        assert first["artifacts"] == second["artifacts"]
        assert first["chosen_action_counts"] == second["chosen_action_counts"]


def test_corrupt_numeric_artifact_is_rejected(pilot, tmp_path):
    original, _ = pilot
    path = tmp_path / "copied"
    shutil.copytree(original, path)
    actor = path / "actor/match-000000.npz"
    actor.write_bytes(actor.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Corrupted"):
        load_split(path, "train")


@pytest.mark.parametrize("corruption", ["shared_split", "moved_split", "reordered_split", "wrong_seed",
                                         "wrong_group", "duplicate_report", "duplicate_game", "wrong_plan"])
def test_manifest_edit_cannot_silently_change_match_isolation(pilot, tmp_path, corruption):
    original, _ = pilot
    path = tmp_path / "copied"
    shutil.copytree(original, path)
    manifest = json.loads((path / "manifest.json").read_text())
    if corruption == "shared_split":
        manifest["splits"]["validation"].append(manifest["splits"]["train"][0])
    elif corruption == "moved_split":
        manifest["splits"]["validation"].append(manifest["splits"]["train"].pop(0))
    elif corruption == "reordered_split":
        manifest["splits"]["train"].reverse()
    elif corruption == "wrong_seed":
        manifest["matches"][0]["seed"] = manifest["matches"][1]["seed"]
    elif corruption == "wrong_group":
        manifest["matches"][0]["independent_group"] = manifest["matches"][1]["independent_group"]
    elif corruption == "duplicate_report":
        manifest["matches"][1] = manifest["matches"][0]
    elif corruption == "duplicate_game":
        manifest["matches"][1]["game_hash"] = manifest["matches"][0]["game_hash"]
    else:
        plan_path = path / "plan.json"
        plan_path.write_text(plan_path.read_text() + " ")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_manifest(path)
    with pytest.raises(ValueError):
        load_split(path, "train")


def test_bounded_submission_is_ordered_and_does_not_require_python314_map():
    from riichi_ai.dataset import _bounded_results
    submitted = []

    class Future:
        def __init__(self, job):
            self.job = job

        def result(self):
            return self.job

    class Executor:
        def submit(self, function, job):
            submitted.append(job)
            return Future(job)

    jobs = [{"index": index} for index in range(7)]
    iterator = _bounded_results(Executor(), jobs, workers=2)
    assert next(iterator) == jobs[0]
    assert len(submitted) == 2
    assert next(iterator) == jobs[1]
    assert len(submitted) == 3
    assert list(iterator) == jobs[2:]
    assert submitted == jobs


def test_disk_budget_stops_before_writing_large_artifact_and_preserves_manifest(tmp_path):
    manifest = generate_dataset(tmp_path, matches=2, root_seed=2026101793, disk_limit_bytes=100000)
    assert not manifest["valid"] and manifest["failure"]["type"] == "DiskBudgetExceeded"
    assert (tmp_path / "manifest.json").exists()
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= 100000


def test_existing_dataset_is_never_overwritten(pilot):
    path, _ = pilot
    with pytest.raises(FileExistsError):
        generate_dataset(path, matches=2)
