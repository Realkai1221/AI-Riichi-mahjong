"""Bounded numerical/engine fixtures, not a playing-strength experiment."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import gzip
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from riichi_ai import stage3a
from riichi_ai.auditing import reference_view_digest
from riichi_ai.model import PolicyNetwork, NeuralPolicy, efficiency_model_config, collate_encoded
from riichi_ai.features import encode_view, efficiency_from_encoded
from riichi_ai.semantics import semantic_actions
from riichi_ai.player import PlayerAdapter, action_spec
from riichi_ai.rules import wall_for_hands
from riichi_ai.simulator import create_env


SMALL = {"suit_width": 8, "history_width": 4, "hidden_width": 16,
         "context_width": 8, "candidate_width": 8}
PROTOCOL = "0" * 64


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def reference(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(992)
        model = PolicyNetwork(efficiency_model_config(SMALL))
    path = tmp_path / "reference.pt"
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(reference, device="cpu"):
    return stage3a.initialize(*reference, protocol_sha256=PROTOCOL, device=device)


def initial_view(seed=311):
    _, observations = create_env(seed)
    return PlayerAdapter(0).build(observations[0])


def synthetic_batch(state, batch_index=0, matches=2, counts=(1, 3)):
    """Complete labels are synthetic: only objective/provenance math is tested."""
    result, plans = [], []
    for index in range(matches):
        plan = {"match_index": index + batch_index * 10, "seed": 311 + index,
                "learner_seat": 0, "sampling_seed": 817 + index}
        view = initial_view(plan["seed"])
        rng = random.Random(plan["sampling_seed"])
        rows, encoded = [], []
        for step in range(counts[index % len(counts)]):
            _, row, arrays = stage3a.actor_decision(
                state.model, view, rng, device=state.device,
                policy_id=state.policy_id, view_hash=reference_view_digest(view))
            row.update({"step": step, "pre_response_id": "synthetic", "row_index": step,
                        "match_index": plan["match_index"], "batch_index": batch_index})
            rows.append(row)
            encoded.append(arrays)
        scores = [40000, 25000, 20000, 15000] if index % 2 == 0 else [15000, 20000, 25000, 40000]
        ranks = [1, 2, 3, 4] if index % 2 == 0 else [4, 3, 2, 1]
        log = [{"type": "start_kyoku", "scores": scores, "kyotaku": 0}, {"type": "end_game"}]
        record = {**plan, "batch_index": batch_index, "protocol_sha256": PROTOCOL,
                  "policy_id": state.policy_id, "completed": True, "terminated": True,
                  "truncated": False, "failure": None, "scores": scores,
                  "ranks": ranks, "terminal_result_verified": True,
                  "learner_utility": stage3a.RANK_UTILITY[ranks[0] - 1],
                  "auditor_log": log, "learner_rows": rows}
        result.append(stage3a.MatchTrajectory(record, encoded))
        plans.append(plan)
    return result, plans


def test_fresh_optimizer_rank_freeze_and_reference_hash(reference):
    state = initialize(reference)
    assert state.optimizer.state_dict()["state"] == {}
    assert all(not p.requires_grad for p in state.model.rank_head.parameters())
    rank_ids = {id(p) for p in state.model.rank_head.parameters()}
    assert not rank_ids.intersection(id(p) for group in state.optimizer.param_groups for p in group["params"])
    assert state.updates == 0 and state.retired_batches == []
    with pytest.raises(ValueError, match="hash mismatch"):
        stage3a.initialize(reference[0], "f" * 64, protocol_sha256=PROTOCOL)


def test_categorical_sampler_uses_actual_probabilities_and_no_forced_rng_draw():
    logits = torch.tensor([0., -100., -1000.])
    probabilities = stage3a.normalized_probabilities(logits)
    assert probabilities[1] > 0  # Float64 avoids the smaller float32 range.
    assert probabilities[2] == 0  # Explicit numerical support, never sampled.
    rng = random.Random(81)
    before = rng.getstate()
    assert stage3a.sample_categorical(np.ones(1), rng) == (0, None)
    assert rng.getstate() == before
    class Zero:
        def random(self):
            return 0.0
    assert stage3a.sample_categorical(np.array([0., .2, .8]), Zero())[0] == 1
    with pytest.raises(ValueError, match="finite"):
        stage3a.normalized_probabilities(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="Invalid"):
        stage3a.sample_categorical(np.array([.1, .1]), rng)


def test_actual_semantic_sampling_records_resolvable_action_and_encoding(reference):
    state = initialize(reference)
    _, observations = create_env(311)
    adapter = PlayerAdapter(0)
    view = adapter.build(observations[0])
    selected, row, arrays = stage3a.actor_decision(
        state.model, view, random.Random(9), device="cpu", policy_id=state.policy_id,
        view_hash=reference_view_digest(view))
    assert action_spec(adapter.resolve(selected, observations[0])) == selected
    assert row["selected_physical_token"] == selected.token
    assert row["probability"] == row["probabilities"][row["selected_index"]]
    assert row["log_probability"] == math.log(row["probability"])
    assert row["encoded_sha256"] == stage3a.encoded_hash(arrays)
    assert set(arrays) == {"tiles", "globals", "history", "history_mask", "action_tokens", "candidate_efficiency"}
    assert not row["fallback"] and not row["overridden"]


def test_terminal_ties_and_truncation_do_not_receive_fake_rewards():
    scores = [25000] * 4
    log = [{"type": "start_kyoku", "scores": scores, "kyotaku": 0}, {"type": "end_game"}]
    result = stage3a.terminal_result(scores, [1, 2, 3, 4], log, completed=True)
    assert result["utilities"] == [1.5, .5, -.5, -1.5]
    assert stage3a.terminal_result(scores, [], [], completed=False)["utilities"] is None
    with pytest.raises(ValueError, match="ranks disagree"):
        stage3a.terminal_result(scores, [4, 3, 2, 1], log, completed=True)
    with pytest.raises(ValueError, match="end_game marker"):
        stage3a.terminal_result(scores, [1, 2, 3, 4], [], completed=True)


def test_terminal_score_chain_handles_riichi_multiple_hands_and_final_deposits():
    log = [{"type": "start_kyoku", "scores": [25000] * 4, "kyotaku": 0},
           {"type": "reach_accepted", "actor": 1},
           {"type": "hora", "deltas": [-8000, 9000, 0, 0]},
           {"type": "end_kyoku"},
           {"type": "start_kyoku", "scores": [17000, 33000, 25000, 25000], "kyotaku": 0},
           {"type": "reach_accepted", "actor": 2},
           {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
           {"type": "end_kyoku"}, {"type": "end_game"}]
    scores, evidence = stage3a.reconstruct_log_scores(log)
    assert scores == [17000, 34000, 24000, 25000]
    assert evidence["terminal_deposit_award"] == 1000
    assert evidence["terminal_deposit_recipient"] == 1
    assert evidence["hands"] == 2 and evidence["accepted_riichi"] == 2
    broken = deepcopy(log)
    broken[4]["scores"][0] += 1000
    with pytest.raises(ValueError, match="Adjacent hand"):
        stage3a.reconstruct_log_scores(broken)
    with pytest.raises(ValueError, match="confirm final scores"):
        stage3a.terminal_result([18000, 34000, 23000, 25000], [4, 1, 3, 2], log, completed=True)


def test_terminal_score_chain_matches_archived_complete_engine_logs():
    root = Path.home() / "Library/Application Support/RiichiAI/stage2/checks-data/auditor"
    paths = sorted(root.glob("*.json.gz"))
    if not paths:
        pytest.skip("Local archived engine fixtures unavailable")
    for path in paths:
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record["completed"]:
            result = stage3a.terminal_result(record["final_scores"], record["final_ranks"],
                                             record["canonical_events"], completed=True)
            assert result["reconstructed_scores"] == record["final_scores"]


def test_actual_final_draw_deposit_award_matches_reconstructed_log():
    cases = json.loads((Path(__file__).parent / "fixtures/stage1/cases.json").read_text())["cases"]
    case = cases["exhaustive_dealer_noten"]
    dealer = 3
    wall = wall_for_hands({(int(seat) + dealer) % 4: hand for seat, hand in case["hands"].items()},
                          case["draws"], dealer=dealer)
    env, _ = create_env(0)
    observations = env.reset(wall=wall, seed=0, oya=dealer, round_wind=1, honba=0, kyotaku=1,
                             scores=[29000, 23000, 23000, 24000])
    for _ in range(300):
        if env.done():
            break
        choices = {}
        for seat, observation in observations.items():
            legal = observation.legal_actions()
            passes = [action for action in legal if action_spec(action).kind == "PASS"]
            choices[seat] = passes[0] if passes else next(
                action for action in legal if action_spec(action).kind == "DISCARD" and action.tile == env.drawn_tile)
        observations = env.step(choices)
    assert env.done() and env.riichi_sticks == 0
    assert list(env.scores()) == [33000, 22000, 22000, 23000]
    result = stage3a.terminal_result(list(env.scores()), list(env.ranks()), env.mjai_log, completed=True)
    assert result["reconstructed_scores"] == list(env.scores())
    assert result["terminal_score_reconstruction"]["terminal_deposit_award"] == 1000


def test_actual_four_tenpai_draw_does_not_double_charge_riichi_log_delta():
    hands = {seat: f"123456m123p123s{seat + 1}z" for seat in range(4)}
    wall = wall_for_hands(hands, ["5z", "6z"], dealer=3)
    env, _ = create_env(0)
    observations = env.reset(wall=wall, seed=0, oya=3, round_wind=1, honba=0, kyotaku=0,
                             scores=[20000, 20000, 20000, 40000])
    declared = False
    for _ in range(400):
        if env.done():
            break
        choices = {}
        for seat, observation in observations.items():
            legal = observation.legal_actions()
            riichi = [action for action in legal if action_spec(action).kind == "RIICHI"]
            passing = [action for action in legal if action_spec(action).kind == "PASS"]
            if seat == 0 and not declared and riichi:
                choices[seat], declared = riichi[0], True
            elif passing:
                choices[seat] = passing[0]
            else:
                choices[seat] = next(action for action in legal
                                     if action_spec(action).kind == "DISCARD" and action.tile == env.drawn_tile)
        observations = env.step(choices)
    assert env.done() and declared and env.riichi_sticks == 0
    assert list(env.scores()) == [19000, 20000, 20000, 41000]
    draw = next(event for event in env.mjai_log if event.get("type") == "ryukyoku")
    assert draw["reason"] == "exhaustive_draw" and draw["deltas"] == [-1000, 0, 0, 0]
    result = stage3a.terminal_result(list(env.scores()), list(env.ranks()), env.mjai_log, completed=True)
    correction = result["terminal_score_reconstruction"]["upstream_log_delta_corrections"]
    assert len(correction) == 1 and correction[0]["effective_deltas"] == [0] * 4


def test_nagashi_log_removes_only_known_riichi_component_and_rejects_unexplained_draw():
    log = [{"type": "start_kyoku", "scores": [25000] * 4, "kyotaku": 0},
           {"type": "reach_accepted", "actor": 1},
           {"type": "ryukyoku", "reason": "nagashimangan", "deltas": [12000, -5000, -4000, -4000]},
           {"type": "end_kyoku"}, {"type": "end_game"}]
    scores, evidence = stage3a.reconstruct_log_scores(log)
    assert scores == [38000, 20000, 21000, 21000]
    assert evidence["upstream_log_delta_corrections"][0]["effective_deltas"] == [12000, -4000, -4000, -4000]
    log[2] = {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [-1000, 0, 0, 0]}
    with pytest.raises(ValueError, match="Unexplained"):
        stage3a.reconstruct_log_scores(log)


def test_objective_is_match_mean_not_decision_mean_and_rank_head_stays_frozen(reference):
    state = initialize(reference)
    trajectories, plans = synthetic_batch(state, counts=(1, 3))
    expected = -sum(t.record["learner_utility"] * sum(r["log_probability"] for r in t.record["learner_rows"])
                    for t in trajectories) / 2
    result = stage3a.update_batch(state, trajectories, batch_index=0, planned_matches=plans, microbatch_size=2)
    assert result["loss_denominator_matches"] == 2
    assert result["optional_learner_rows"] == 4 and result["opponent_rows_in_loss"] == 0
    assert result["behavior_reconstructed_loss"] == pytest.approx(expected, abs=1e-6)
    assert result["differentiable_loss"] == pytest.approx(expected, abs=1e-6)
    assert result["optimizer_steps"] == 1 and result["rank_head_unchanged"]
    assert result["parameter_max_absolute_change"] > 0
    assert result["distribution_max_absolute_change"] > 0
    assert result["gradients_finite"] and result["parameters_finite"]
    assert state.retired_batches == [0]
    with pytest.raises(ValueError, match="out of order"):
        stage3a.update_batch(state, trajectories, batch_index=0, planned_matches=plans)


@pytest.mark.parametrize("mutation", ["partial", "truncated", "fallback", "override", "opponent", "version", "reward", "encoding", "probability", "seed"])
def test_invalid_batch_never_changes_optimizer_or_weights(reference, mutation):
    state = initialize(reference)
    trajectories, plans = synthetic_batch(state)
    before = stage3a.state_manifest(state)
    if mutation == "partial":
        trajectories.pop()
    elif mutation == "truncated":
        trajectories[0].record.update(completed=False, terminated=False, truncated=True)
    elif mutation == "fallback":
        trajectories[0].record["learner_rows"][0]["fallback"] = True
    elif mutation == "override":
        trajectories[0].record["learner_rows"][0]["overridden"] = True
    elif mutation == "opponent":
        trajectories[0].record["learner_rows"][0]["seat"] = 2
    elif mutation == "version":
        trajectories[0].record["policy_id"] = "stale"
    elif mutation == "reward":
        trajectories[0].record["learner_utility"] = 99
    elif mutation == "encoding":
        trajectories[0].encoded[0]["globals"][0] += 1
    elif mutation == "probability":
        row = trajectories[0].record["learner_rows"][0]
        probabilities = np.full(len(row["probabilities"]), 1 / len(row["probabilities"]))
        row.update(probabilities=probabilities.tolist(), probability=float(probabilities[row["selected_index"]]),
                   log_probability=math.log(float(probabilities[row["selected_index"]])))
    elif mutation == "seed":
        trajectories[0].record["seed"] += 1
    with pytest.raises(ValueError):
        stage3a.update_batch(state, trajectories, batch_index=0, planned_matches=plans)
    assert stage3a.state_manifest(state) == before


def test_exact_cpu_boundary_resume_weights_optimizer_and_next_sampling(reference, tmp_path):
    full = initialize(reference)
    first, plan = synthetic_batch(full)
    stage3a.update_batch(full, first, batch_index=0, planned_matches=plan, microbatch_size=2)
    path = tmp_path / "boundary.pt"
    boundary = stage3a.save_boundary(full, path)
    assert len(boundary["sha256"]) == 64
    resumed = stage3a.load_boundary(path, expected_protocol_sha256=PROTOCOL)
    assert stage3a.state_manifest(full) == stage3a.state_manifest(resumed)
    full_batch, full_plan = synthetic_batch(full, batch_index=1)
    resumed_batch, resumed_plan = synthetic_batch(resumed, batch_index=1)
    assert [t.record for t in full_batch] == [t.record for t in resumed_batch]
    stage3a.update_batch(full, full_batch, batch_index=1, planned_matches=full_plan, microbatch_size=2)
    stage3a.update_batch(resumed, resumed_batch, batch_index=1, planned_matches=resumed_plan, microbatch_size=2)
    assert stage3a.state_manifest(full) == stage3a.state_manifest(resumed)
    with pytest.raises(FileExistsError):
        stage3a.save_boundary(full, path)
    with pytest.raises(ValueError, match="protocol mismatch"):
        stage3a.load_boundary(path, expected_protocol_sha256="f" * 64)


def test_persisted_actor_arrays_are_exact_and_exclusive(reference, tmp_path):
    state = initialize(reference)
    trajectories, _ = synthetic_batch(state)
    result = stage3a.persist_match(trajectories[0], tmp_path / "matches")
    record = json.loads(open(result["json_path"]).read())
    with np.load(result["npz_path"], allow_pickle=False) as archive:
        encoded = {key.split("__", 1)[1]: archive[key] for key in archive.files}
        assert stage3a.encoded_hash(encoded) == record["learner_rows"][0]["encoded_sha256"]
    with pytest.raises(FileExistsError):
        stage3a.persist_match(trajectories[0], tmp_path / "matches")


def test_hidden_world_changes_do_not_change_learner_features_probabilities_or_draw(reference):
    state = initialize(reference)
    wall = list(range(136))
    random.Random(17).shuffle(wall)
    changed = wall.copy()
    slots = lambda seat: [block * 16 + seat * 4 + offset for block in range(3) for offset in range(4)] + [48 + seat]
    for first, second in zip(slots(1), slots(2)):
        changed[first], changed[second] = changed[second], changed[first]
    changed[57:122] = reversed(changed[57:122])
    results = []
    for physical in (wall, changed):
        env, _ = create_env(0)
        observations = env.reset(wall=physical, seed=0, oya=0, round_wind=0, honba=0, kyotaku=0, scores=[25000] * 4)
        view = PlayerAdapter(0).build(observations[0])
        _, row, encoded = stage3a.actor_decision(state.model, view, random.Random(731), device="cpu",
                                                policy_id=state.policy_id, view_hash=reference_view_digest(view))
        results.append((row, stage3a.encoded_hash(encoded)))
    assert results[0] == results[1]


def test_collector_preserves_truncation_without_update(reference):
    state = initialize(reference)
    opponent = NeuralPolicy.from_checkpoint(reference[0])
    before = stage3a.state_manifest(state)
    trajectory = stage3a.collect_match(state, opponent, batch_index=0, match_index=0,
                                      seed=311, learner_seat=0, sampling_seed=71, max_steps=1)
    record = trajectory.record
    assert not record["completed"] and record["truncated"] and not record["terminated"]
    assert record["learner_utility"] is None and record["utilities"] is None
    assert len(record["learner_rows"]) == 1 and record["steps"] == 1
    assert stage3a.state_manifest(state) == before


def test_simultaneous_requests_all_build_before_any_choice(reference, monkeypatch):
    env, _ = create_env(0)
    wall = wall_for_hands({1: "12m456m789p456s23z", 2: "33m234p678234s23z",
                           3: "12m123p123789s11z"}, draws=["3m"])
    observations = env.reset(wall=wall, seed=0, oya=0, round_wind=0, honba=0, kyotaku=0, scores=[25000] * 4)
    discard = next(a for a in observations[0].legal_actions() if action_spec(a).kind == "DISCARD" and a.tile == env.drawn_tile)
    observations = env.step({0: discard})
    assert len(observations) > 1
    events = []
    original_build, original_actor = PlayerAdapter.build, stage3a.actor_decision
    def build(adapter, observation):
        events.append(("build", adapter.seat))
        return original_build(adapter, observation)
    def actor(*args, **kwargs):
        events.append(("choose", args[1].seat))
        return original_actor(*args, **kwargs)
    monkeypatch.setattr(PlayerAdapter, "build", build)
    monkeypatch.setattr(stage3a, "actor_decision", actor)
    monkeypatch.setattr(stage3a, "create_env", lambda seed: (env, observations))
    state = initialize(reference)
    opponent = NeuralPolicy.from_checkpoint(reference[0])
    original_choose = opponent.choose
    def choose(view):
        events.append(("choose", view.seat))
        return original_choose(view)
    monkeypatch.setattr(opponent, "choose", choose)
    trajectory = stage3a.collect_match(state, opponent, batch_index=0, match_index=0,
                                      seed=0, learner_seat=1, sampling_seed=33, max_steps=1)
    assert trajectory.record["failure"] is None
    count = len(observations)
    assert all(event[0] == "build" for event in events[:count])
    request = trajectory.record["requests"][0]
    assert len(request["choices"]) > 1
    assert all(not choice["eligible"] for seat, choice in request["choices"].items() if seat != "1")
    assert trajectory.record["learner_rows"][0]["pre_response_id"] == request["pre_response_id"]


def test_opponent_terminal_move_delivers_learner_reward_without_new_learner_action(reference, monkeypatch):
    # A deterministic South-4 nonlearner dealer heavenly-hand fixture. The
    # policy-selection stub isolates collector termination/reward behavior.
    env, _ = create_env(0)
    wall = wall_for_hands({3: "111222333m444p5s"}, draws=["5s"], dealer=3)
    observations = env.reset(wall=wall, seed=0, oya=3, round_wind=1, honba=0, kyotaku=0,
                             scores=[20000, 20000, 20000, 40000])
    monkeypatch.setattr(stage3a, "create_env", lambda seed: (env, observations))
    state = initialize(reference)
    opponent = NeuralPolicy.from_checkpoint(reference[0])
    monkeypatch.setattr(opponent, "choose", lambda view: next(a for a in view.legal_actions if a.kind == "TSUMO"))
    trajectory = stage3a.collect_match(state, opponent, batch_index=0, match_index=0,
                                      seed=0, learner_seat=0, sampling_seed=11, max_steps=1)
    record = trajectory.record
    assert record["completed"], record["failure"]
    assert record["learner_rows"] == [] and record["learner_last_step"] is None
    assert record["opponent_steps_after_last_learner"] == 1
    assert record["learner_utility"] == stage3a.RANK_UTILITY[record["ranks"][0] - 1]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_update_and_exact_boundary_weights_roundtrip(reference, tmp_path):
    state = initialize(reference, device="mps")
    trajectories, plans = synthetic_batch(state, matches=1)
    result = stage3a.update_batch(state, trajectories, batch_index=0, planned_matches=plans, microbatch_size=1)
    assert result["parameters_finite"] and result["rank_head_unchanged"]
    path = tmp_path / "mps-boundary.pt"
    stage3a.save_boundary(state, path)
    resumed = stage3a.load_boundary(path, expected_protocol_sha256=PROTOCOL, device="mps")
    assert stage3a.state_manifest(state) == stage3a.state_manifest(resumed)


def test_actual_reference_encoder_equivalence_and_fixed_microbatch_replay(record_property):
    path = Path.home() / "Library/Application Support/RiichiAI/stage2b/models/selected.pt"
    if not path.is_file():
        pytest.skip("Local original reference is unavailable")
    expected = "5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f"
    policy = NeuralPolicy.from_checkpoint(path)
    assert policy.checkpoint_hash == expected
    rows, singles = [], []
    for seed in range(720, 752):
        view = initial_view(seed)
        actions = semantic_actions(view)
        direct = encode_view(view, actions, feature_schema=policy.model.config["feature_schema"])
        historical = encode_view(view, actions)
        historical["candidate_efficiency"] = efficiency_from_encoded(historical["tiles"], historical["action_tokens"])
        assert stage3a.encoded_hash(direct) == stage3a.encoded_hash(historical)
        rows.append(direct)
        with torch.inference_mode():
            logits, _ = policy.model(collate_encoded([direct]))
            singles.append(stage3a.normalized_probabilities(logits[0]))
    with torch.inference_mode():
        logits, _ = policy.model(collate_encoded(rows))
    maximum = 0.0
    for index, row in enumerate(rows):
        batched = stage3a.normalized_probabilities(logits[index, :len(row["action_tokens"])])
        maximum = max(maximum, float(np.max(np.abs(batched - singles[index]))))
        with torch.inference_mode():
            replay_logits, _ = policy.model(collate_encoded([row]))
            replay = stage3a.normalized_probabilities(replay_logits[0])
        assert np.array_equal(replay, singles[index])
    record_property("actual_reference_batch_replay_max_absolute_error", maximum)
    record_property("formal_single_row_replay_max_absolute_error", 0.0)
