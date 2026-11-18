"""Observer commitments and small spawned-process evaluation regressions."""

from dataclasses import replace
import copy
import json
import os
import random

import pytest

from riichi_ai import evaluation
from riichi_ai.auditing import PlayerViewHasher, VIEW_HASH_SCHEMA, reference_view_digest
from riichi_ai.player import EventView, PlayerAdapter, action_spec
from riichi_ai.rules import audit_physical_tiles, wall_for_hands
from riichi_ai.simulator import create_env


def initial(wall):
    env, _ = create_env(0)
    obs = env.reset(wall=wall, seed=0, oya=0, round_wind=0, honba=0,
                    kyotaku=0, scores=[25_000] * 4)
    audit_physical_tiles(env)
    return env, obs


def pick(observation, kind, tile=None):
    return next(action for action in observation.legal_actions()
                if action_spec(action).kind == kind and (tile is None or action.tile == tile))


def test_view_hash_preserves_one_observer_under_complete_hidden_world_changes():
    # Limited legal-world family: opponent initial allocations and future live
    # wall differ; own deal and all draws through the dealer's second turn agree.
    wall = list(range(136))
    random.Random(17).shuffle(wall)
    changed = wall.copy()
    slots = lambda seat: [block * 16 + seat * 4 + offset
                          for block in range(3) for offset in range(4)] + [48 + seat]
    for first, second in zip(slots(1), slots(2)):
        changed[first], changed[second] = changed[second], changed[first]
    changed[57:122] = reversed(changed[57:122])
    commitments, opponent_commitments = [], []
    for complete in (wall, changed):
        env, obs = initial(complete)
        adapter, hasher = PlayerAdapter(0), PlayerViewHasher()
        views = [adapter.build(obs[0])]
        opponent_commitments.append(reference_view_digest(PlayerAdapter(1).build(env.get_observation(1))))
        for _ in range(30):
            moves = {}
            for seat, observation in obs.items():
                passing = [a for a in observation.legal_actions() if action_spec(a).kind == "PASS"]
                moves[seat] = passing[0] if passing else pick(observation, "DISCARD", env.drawn_tile)
            obs = env.step(moves)
            audit_physical_tiles(env)
            if 0 in obs:
                views.append(adapter.build(obs[0]))
                if any(a.kind == "DISCARD" for a in views[-1].legal_actions):
                    break
        assert any(event.kind == "dahai" for event in views[-1].history)
        digests = [hasher.digest(view) for view in views]
        assert digests == [reference_view_digest(view) for view in views]
        commitments.append(digests)
    assert commitments[0] == commitments[1]
    # The changed hand is visible to its owner. A combined four-seat digest
    # is deliberately not invariant to this opponent-private change.
    assert opponent_commitments[0] != opponent_commitments[1]


def test_inactive_private_draw_changes_do_not_change_observer_commitment():
    wall = list(range(136))
    random.Random(17).shuffle(wall)
    changed = wall.copy()
    changed[0], changed[52] = changed[52], changed[0]
    results = []
    for complete in (wall, changed):
        env, obs = initial(complete)
        adapter, hasher = PlayerAdapter(1), PlayerViewHasher()
        views = [adapter.build(env.get_observation(1))]
        assert views[0].drawn_tile is None
        # Same held discard and public tsumogiri=False in both legal worlds.
        obs = env.step({0: pick(obs[0], "DISCARD", wall[1])})
        for _ in range(10):
            audit_physical_tiles(env)
            if 1 in obs:
                views.append(adapter.build(obs[1]))
                if any(action.kind == "DISCARD" for action in views[-1].legal_actions):
                    break
            obs = env.step({seat: pick(observation, "PASS") for seat, observation in obs.items()})
        assert any(event.kind == "dahai" for event in views[-1].history)
        results.append([hasher.digest(view) for view in views])
    assert results[0] == results[1]


def test_cached_hash_binds_actual_history_and_hand_id_after_prefix_changes():
    _, obs = create_env(73)
    base = PlayerAdapter(0).build(obs[0])
    public = EventView(kind="dahai", actor=1, tile="1m", tile_type=0, tsumogiri=False)
    extended = replace(base, history=base.history + (public,))
    rewritten = replace(extended, history=extended.history[:-1] + (replace(public, tsumogiri=True),))
    reset = replace(base, hand_id=base.hand_id + 1)
    changed_score = replace(base, scores=(26_000, 24_000, 25_000, 25_000))
    changed_legal = replace(base, legal_actions=base.legal_actions[:-1])
    hasher = PlayerViewHasher()
    # Append, equal-length mutation, truncation, empty/reset prefixes, and
    # repeated snapshots all use the actual view rather than cached assumptions.
    sequence = (base, extended, extended, rewritten, base, replace(base, history=()),
                extended, reset, changed_score, changed_legal, base)
    for view in sequence:
        assert hasher.digest(view) == reference_view_digest(view)
    digest = reference_view_digest(base)
    for view in (extended, rewritten, reset, changed_score, changed_legal):
        assert reference_view_digest(view) != digest


def test_actual_abortive_renchan_resets_cached_history_without_reusing_hand_id():
    env, obs = initial(wall_for_hands({0: "19m19p19s1234567z"}, ["2m"]))
    adapter, hasher = PlayerAdapter(0), PlayerViewHasher()
    before = adapter.build(obs[0])
    first_digest = hasher.digest(before)
    obs = env.step({0: pick(obs[0], "KYUSHU_KYUHAI")})
    after = adapter.build(obs[0])
    assert after.oya == before.oya and after.hand_id == before.hand_id + 1
    assert after.history[0].kind == "start_kyoku"
    assert hasher.digest(after) == reference_view_digest(after) != first_digest


@pytest.fixture(scope="module")
def serial_and_parallel():
    kwargs = dict(candidate="shanten_v2", baseline="random", blocks=2, seed=987_651)
    serial = evaluation.evaluate_paired(**kwargs, workers=1)
    parallel = evaluation.evaluate_paired(**kwargs, workers=2)
    return serial, parallel


def test_spawned_blocks_match_serial_digests_and_statistics(serial_and_parallel):
    serial, parallel = serial_and_parallel
    assert serial["valid"], {key: serial[key] for key in ("sources_unchanged", "policies_unchanged", "failed_games")}
    assert parallel["valid"], {key: parallel[key] for key in ("sources_unchanged", "policies_unchanged", "failed_games")}
    assert serial["completed_games"] == parallel["completed_games"] == 16
    assert serial["blocks"] == parallel["blocks"]
    assert serial["statistics"] == parallel["statistics"]
    assert serial["arm_summaries"] == parallel["arm_summaries"]
    assert [(m["block"], m["focal_seat"], m["arm"], m["digest"]) for m in serial["matches"]] == [
        (m["block"], m["focal_seat"], m["arm"], m["digest"]) for m in parallel["matches"]]


def test_digest_excludes_timing_and_report_pairing_annotations(serial_and_parallel):
    serial, _ = serial_and_parallel
    excluded = {
        "digest", "elapsed_seconds", "timing_seconds", "policy_latency_by_seat", "policy_inference_by_seat",
        "policy_inference_timing_by_seat",
        "setup_seconds", "finalization_seconds", "unallocated_loop_seconds",
        "block", "arm", "focal_seat", "paired_common_walls_match", "paired_common_hands",
        "policies_match_frozen_panel",
    }
    for match in serial["matches"]:
        assert match["schema"] == "riichi-ai-evaluation-v2"
        assert match["player_view_hash_schema"] == VIEW_HASH_SCHEMA
        payload = {key: value for key, value in match.items() if key not in excluded}
        assert evaluation._hash(payload) == match["digest"]
        assert all(value >= 0 for value in match["timing_seconds"].values())
        assert sum(match["timing_seconds"].values()) <= match["elapsed_seconds"]
        assert {"view_hash", "wall_hash"} <= match["timing_seconds"].keys()
        assert match["policy_inference_by_seat"] == [None] * 4


def test_match_reports_per_wrapper_neural_inference_counters(tmp_path):
    import torch
    from riichi_ai.model import PolicyNetwork

    path = tmp_path / "counter-weights.pt"
    model = PolicyNetwork({"suit_width": 8, "history_width": 4, "hidden_width": 16,
                           "context_width": 8, "candidate_width": 4})
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, path)
    evaluation._load_neural.cache_clear()
    try:
        policy_name = f"checkpoint:{path}"
        # One initial optional discard invokes a network only for the dealer.
        # The same cached weights must still have fresh per-match wrappers.
        records = [evaluation.run_match(51, [policy_name] * 4, max_steps=1) for _ in range(2)]
        assert records[0]["digest"] == records[1]["digest"]
        for record in records:
            assert record["failure"] is None
            counters = record["policy_inference_by_seat"]
            assert counters[0] == {"decisions": 1, "forced_decisions": 0, "network_calls": 1}
            assert counters[1:] == [{"decisions": 0, "forced_decisions": 0, "network_calls": 0}] * 3
            assert all("inference_stats" not in manifest for manifest in record["policies"])
    finally:
        evaluation._load_neural.cache_clear()


def test_parallel_truncations_preserve_records_and_suppress_inference(tmp_path):
    report = evaluation.evaluate_paired("random", "random", blocks=2, workers=2,
                                        seed=987_652, max_steps=1, failure_dir=tmp_path)
    assert not report["valid"] and report["statistics"] is None
    assert report["completed_games"] == 0 and report["truncated_games"] == 16
    assert report["hanchans_per_hour"] == 0
    assert len(report["matches"]) == 16
    for match in report["matches"]:
        assert match["ranks"] is None and match["utilities"] is None
        record = json.loads(open(match["failure_record"]).read())
        assert record["record"]["digest"] == match["digest"]
        assert len(record["submitted_actions"]) == 1


@pytest.mark.parametrize("workers", [0, 17, True, 1.5])
def test_invalid_worker_counts_fail_before_game_execution(monkeypatch, workers):
    monkeypatch.setattr(evaluation, "run_match", lambda *args, **kwargs: pytest.fail("Unexpected game"))
    with pytest.raises(ValueError, match="workers"):
        evaluation.evaluate_paired("random", "random", blocks=1, workers=workers)


def test_checkpoint_cache_detects_changed_bytes_with_unchanged_size_and_mtime(tmp_path):
    import torch
    from riichi_ai.model import PolicyNetwork

    path = tmp_path / "weights.pt"
    model = PolicyNetwork({"suit_width": 8, "history_width": 4, "hidden_width": 16,
                           "context_width": 8, "candidate_width": 4})
    payload = {"model_config": model.config, "model_state_dict": model.state_dict()}
    torch.save(payload, path)
    original = path.stat()
    evaluation._load_neural.cache_clear()
    try:
        first = evaluation._evaluation_policy(f"checkpoint:{path}", 0)
        fresh_wrapper = evaluation._evaluation_policy(f"checkpoint:{path}", 1)
        assert fresh_wrapper is not first and fresh_wrapper.model is first.model
        _, obs = create_env(51)
        view = PlayerAdapter(0).build(obs[0])
        first.choose(replace(view, legal_actions=(view.legal_actions[0],)))
        assert first.inference_stats()["decisions"] == 1
        assert fresh_wrapper.inference_stats()["decisions"] == 0
        with torch.no_grad():
            next(model.parameters()).add_(1)
        torch.save(payload, path)
        assert path.stat().st_size == original.st_size
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert path.stat().st_mtime_ns == original.st_mtime_ns
        changed = evaluation._evaluation_policy(f"checkpoint:{path}", 0)
        assert changed.checkpoint_hash != first.checkpoint_hash
        assert changed.weights_hash != first.weights_hash
        assert changed.model is not first.model
    finally:
        evaluation._load_neural.cache_clear()


def test_transient_wrong_policy_invalidates_block_despite_unchanged_final_panel(monkeypatch, serial_and_parallel):
    serial, _ = serial_and_parallel
    matches = copy.deepcopy(serial["matches"][:8])
    block = copy.deepcopy(serial["blocks"][0])
    # Simulate one transiently wrong checkpoint/version, with the initial
    # and final panel unchanged. Actual per-game provenance must still reject it.
    matches[0]["policies"][2]["version"] = "unexpected-transient-version"
    monkeypatch.setattr(evaluation, "_evaluate_block", lambda job: (matches, block))
    report = evaluation.evaluate_paired("shanten_v2", "random", blocks=1, seed=987_651)
    assert report["policies_unchanged"]
    assert not report["match_policies_match_frozen_panel"]
    assert not report["matches"][0]["policies_match_frozen_panel"]
    assert not report["blocks"][0]["valid"]
    assert report["completed_games"] == 8
    assert not report["valid"] and report["statistics"] is None
