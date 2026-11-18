"""Local mechanism tests; synthetic outcomes are not strength evidence."""
from copy import deepcopy
import hashlib
import math
import random

import numpy as np
import pytest
import torch

from riichi_ai import stage3c
from riichi_ai.auditing import reference_view_digest
from riichi_ai.features import encode_view
from riichi_ai.model import NeuralPolicy, PolicyNetwork, collate_encoded, efficiency_model_config
from riichi_ai.player import PlayerAdapter, action_spec
from riichi_ai.rules import wall_for_hands
from riichi_ai.simulator import create_env

SMALL = {"suit_width": 8, "history_width": 4, "hidden_width": 16,
         "context_width": 8, "candidate_width": 8}
PROTOCOL = "3" * 64


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


def initialize(reference):
    return stage3c.initialize(*reference, protocol_sha256=PROTOCOL, initialization_seed=310211)


def call_view():
    env, _ = create_env(0)
    observations = env.reset(wall=wall_for_hands({1: "1240m123p456s123z"}, draws=["3m"]),
        seed=0, oya=0, round_wind=0, honba=0, kyotaku=0, scores=[25000] * 4)
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    for seat, observation in observations.items():
        adapters[seat].build(observation)
    action = next(a for a in observations[0].legal_actions()
                  if action_spec(a).kind == "DISCARD" and a.tile == env.drawn_tile)
    observations = env.step({0: action})
    return adapters[1].build(observations[1])


def synthetic_batch(state):
    """Use real legal features, but deliberately synthetic complete labels."""
    trajectories, plans = [], []
    view = call_view()
    for index, count in enumerate((1, 3)):
        plan = {"match_index": index, "seed": index + 1, "learner_seat": 1, "sampling_seed": 100 + index}
        rows, arrays = [], []
        for step in range(count):
            # Known CDF draw chooses a supported call; no game is simulated here.
            class FixedDraw:
                def random(self):
                    return .01
            _, row, encoded = stage3c.actor_decision(state.model, view, FixedDraw(), device="cpu",
                policy_id=state.policy_id, view_hash=reference_view_digest(view))
            row.update({"step": step, "row_index": step, "batch_index": 0, "match_index": index,
                        "pre_response_id": "synthetic-fixture-only"})
            rows.append(row)
            arrays.append(encoded)
        scores = [15000, 40000, 25000, 20000] if index == 0 else [25000, 15000, 20000, 40000]
        ranks = [4, 1, 2, 3] if index == 0 else [2, 4, 3, 1]
        record = {**plan, "schema": stage3c.SCHEMA, "sampler": stage3c.SAMPLER,
            "batch_index": 0, "protocol_sha256": PROTOCOL, "policy_id": state.policy_id,
            "collection_role": "training", "training_eligible": True,
            "completed": True, "terminated": True, "truncated": False, "failure": None,
            "scores": scores, "ranks": ranks, "terminal_result_verified": True,
            "learner_utility": (1.5, .5, -.5, -1.5)[ranks[1] - 1],
            "auditor_log": [{"type": "start_kyoku", "scores": scores, "kyotaku": 0}, {"type": "end_game"}],
            "learner_rows": rows}
        trajectories.append(stage3c.MatchTrajectory(record, arrays))
        plans.append(plan)
    return trajectories, plans


def group(base, gate, calls, kinds):
    return stage3c.grouped_distribution(torch.tensor(base, dtype=torch.float64),
        torch.tensor(gate, dtype=torch.float64, requires_grad=True),
        torch.tensor(calls, dtype=torch.float64, requires_grad=True),
        torch.tensor([stage3c.ACTION_KINDS.index(kind) + 1 for kind in kinds]))


def test_extreme_base_calls_get_trainable_positive_mass():
    result = group([0., -100000., -100001.], math.log(.05/.95), [0., 0., 0.], ["PASS", "CHI", "PON"])
    np.testing.assert_allclose(result["probabilities"].detach(), [.95, .025, .025], atol=1e-15, rtol=0)
    assert torch.isfinite(result["log_probabilities"]).all()
    np.testing.assert_allclose(result["log_probabilities"].detach(), np.log([.95, .025, .025]), atol=1e-14)


def test_analytic_gate_and_multiple_call_gradients():
    base = torch.tensor([0., -100000., -100001.], dtype=torch.float64)
    gate = torch.tensor(math.log(.05/.95), dtype=torch.float64, requires_grad=True)
    calls = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    kinds = torch.tensor([stage3c.ACTION_KINDS.index(kind) + 1 for kind in ("PASS", "CHI", "PON")])
    result = stage3c.grouped_distribution(base, gate, calls, kinds)
    result["log_probabilities"][1].backward()
    assert gate.grad.item() == pytest.approx(.95, abs=1e-14)
    np.testing.assert_allclose(calls.grad, [0., .5, -.5], atol=1e-14)


def test_single_call_has_gate_but_no_conditional_gradient():
    gate = torch.tensor(-2., dtype=torch.float64, requires_grad=True)
    calls = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    result = stage3c.grouped_distribution(torch.tensor([0., -1e6], dtype=torch.float64), gate, calls,
        torch.tensor([11, 2]))
    result["log_probabilities"][1].backward()
    assert gate.grad.item() > 0
    assert torch.count_nonzero(calls.grad) == 0


def test_noncall_conditional_softmax_ignores_rounded_full_distribution():
    result = group([-100000., -100001., 0.], -2., [0., 0., 0.], ["DISCARD", "PASS", "CHI"])
    ratio = result["probabilities"][0] / result["probabilities"][1]
    assert ratio.item() == pytest.approx(math.e)
    assert torch.isfinite(result["log_probabilities"]).all()


@pytest.mark.parametrize("kinds, expected", [(["CHI", "PON"], [.5, .5]), (["PASS"], [1.]),
                                              (["PASS", "DISCARD"], [.5, .5])])
def test_single_available_group_gets_all_probability(kinds, expected):
    result = group([0.] * len(kinds), -1e6, [0.] * len(kinds), kinds)
    np.testing.assert_allclose(result["probabilities"].detach(), expected)


def test_winning_action_mass_loss_is_reported_without_override():
    result = group([0., 0., -1e6], math.log(.05/.95), [0.] * 3, ["RON", "PASS", "PON"])
    assert result["win_mass_without_gate"].item() == pytest.approx(.5)
    assert result["win_mass_actual"].item() == pytest.approx(.475)
    assert result["win_mass_lost"].item() == pytest.approx(.025)


def test_staged_cpu_transfer_keeps_autograd():
    value = torch.tensor([1., 2.], requires_grad=True)
    transformed = stage3c.cpu_double(value)
    assert transformed.dtype == torch.float64 and transformed.device.type == "cpu"
    transformed.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_frozen_base_architecture_and_seed_reproduce(reference):
    first, second = initialize(reference), initialize(reference)
    assert stage3c.state_manifest(first) == stage3c.state_manifest(second)
    assert all(not p.requires_grad for p in first.model.base.parameters())
    optimized = {id(p) for group in first.optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in first.model.heads.parameters()}
    assert first.optimizer.state_dict()["state"] == {}
    with pytest.raises(ValueError, match="hash mismatch"):
        stage3c.initialize(reference[0], "f" * 64, protocol_sha256=PROTOCOL)
    with pytest.raises(ValueError, match="learning rate"):
        stage3c.initialize(*reference, protocol_sha256=PROTOCOL, learning_rate=1e-3)


def test_real_legal_row_base_logits_sampling_and_exact_replay(reference):
    state = initialize(reference)
    view = call_view()
    selected, row, encoded = stage3c.actor_decision(state.model, view, random.Random(31), device="cpu",
        policy_id=state.policy_id, view_hash=reference_view_digest(view))
    batch = collate_encoded([encoded], "cpu")
    with torch.inference_mode():
        old_logits, _ = state.model.base(batch)
        exposed, _, _ = state.model.frozen_representations(batch)
        assert torch.equal(old_logits, exposed)
        actual = state.model.distribution(encoded)["probabilities"].numpy()
    assert np.array_equal(actual, row["probabilities"])
    assert row["log_probability"] == math.log(actual[row["selected_index"]])
    assert selected.token == row["selected_physical_token"]
    assert row["call_candidates"] >= 2
    assert row["call_mass"] == pytest.approx(.05, abs=1e-8)
    with pytest.raises(ValueError, match="single-row"):
        state.model.frozen_representations(collate_encoded([encoded, encoded], "cpu"))


def test_prepare_direct_gradients_preserves_every_state_identity(reference):
    state = initialize(reference)
    before = stage3c.state_manifest(state)
    result = stage3c.prepare(state)
    assert result["valid"] and result["optimizer_steps"] == 0
    assert result["new_complete_matches"] == 0
    for component in result["gradient_components"].values():
        assert component["l2_norm"] > 0 and component["finite"]
        assert component["hidden_layer_l2_norm"] == 0
    assert stage3c.state_manifest(state) == before


def test_complete_match_objective_updates_only_heads(reference, tmp_path):
    state = initialize(reference)
    trajectories, plans = synthetic_batch(state)
    expected = -sum(t.record["learner_utility"] * sum(row["log_probability"] for row in t.record["learner_rows"])
                    for t in trajectories) / 2
    result = stage3c.update_batch(state, trajectories, batch_index=0, planned_matches=plans)
    assert result["loss_denominator_matches"] == 2 and result["opponent_rows_in_loss"] == 0
    assert result["differentiable_loss"] == pytest.approx(expected, abs=1e-12)
    assert result["stored_behavior_loss"] == pytest.approx(expected, abs=1e-12)
    assert result["base_unchanged"] and result["rank_head_unchanged"]
    assert result["probability_replay_max_absolute_error"] == 0
    assert result["selected_log_probability_replay_max_error"] == 0
    assert result["parameter_max_absolute_change"] > 0
    assert result["distribution_max_absolute_change"] > 0
    assert result["gradient_components"]["gate"]["l2_norm"] > 0
    assert result["gradient_components"]["call_head"]["l2_norm"] > 0
    path = tmp_path / "boundary.pt"
    stage3c.save_boundary(state, path)
    resumed = stage3c.load_boundary(path, expected_protocol_sha256=PROTOCOL)
    assert stage3c.state_manifest(resumed) == stage3c.state_manifest(state)
    assert resumed.retired_batches == [0]
    with pytest.raises(ValueError, match="out of order"):
        stage3c.update_batch(state, trajectories, batch_index=0, planned_matches=plans)


@pytest.mark.parametrize("mutation", ["partial", "truncated", "opponent", "reward", "encoding", "probability", "seed"])
def test_rejected_batch_preserves_parameters_and_optimizer(reference, mutation):
    state = initialize(reference)
    trajectories, plans = synthetic_batch(state)
    before = stage3c.state_manifest(state)
    row = trajectories[0].record["learner_rows"][0]
    if mutation == "partial":
        trajectories.pop()
    elif mutation == "truncated":
        trajectories[0].record["truncated"] = True
    elif mutation == "opponent":
        row["seat"] = 0
    elif mutation == "reward":
        trajectories[0].record["learner_utility"] = -.5
    elif mutation == "encoding":
        trajectories[0].encoded[0]["globals"][0] += 1
    elif mutation == "probability":
        row["probabilities"][0] += .01
    elif mutation == "seed":
        plans[0]["seed"] += 1
    with pytest.raises(ValueError):
        stage3c.update_batch(state, trajectories, batch_index=0, planned_matches=plans)
    assert stage3c.state_manifest(state) == before


def test_boundary_hash_guard_and_exclusive_write(reference, tmp_path):
    state = initialize(reference)
    path = tmp_path / "initial.pt"
    stage3c.save_boundary(state, path)
    with pytest.raises(FileExistsError):
        stage3c.save_boundary(state, path)
    with pytest.raises(ValueError, match="protocol"):
        stage3c.load_boundary(path, expected_protocol_sha256="x" * 64)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model_state_dict"]["base.suit_encoder.0.weight"][0, 0] += 1
    broken = tmp_path / "broken.pt"
    torch.save(checkpoint, broken)
    with pytest.raises(ValueError, match="Frozen base"):
        stage3c.load_boundary(broken, expected_protocol_sha256=PROTOCOL)


def test_partial_collector_has_no_terminal_reward_and_persists_rows(reference, tmp_path):
    state = initialize(reference)
    opponent = NeuralPolicy.from_checkpoint(reference[0], device="cpu")
    trajectory = stage3c.collect_match(state, opponent, batch_index=0, match_index=0,
        seed=22, learner_seat=0, sampling_seed=55, max_steps=2)
    record = trajectory.record
    assert record["truncated"] and not record["completed"]
    assert record["learner_utility"] is None and record["utilities"] is None
    assert len(record["ordinal_wall_fingerprints"]) == 1
    assert record["learner_rows"]
    for row in record["learner_rows"]:
        assert row["seat"] == 0 and row["eligible"]
    saved = stage3c.persist_match(trajectory, tmp_path)
    assert saved["json_sha256"] and saved["npz_sha256"]


def test_kan_resolution_distinguishes_declared_completed_robbed_and_pending():
    log = [{"type": "pon", "actor": 1}, {"type": "kakan", "actor": 1},
           {"type": "hora", "actor": 2, "target": 1}, {"type": "end_kyoku"},
           {"type": "start_kyoku"}, {"type": "ankan", "actor": 0}, {"type": "dora"},
           {"type": "tsumo", "actor": 0}, {"type": "daiminkan", "actor": 3}]
    resolved = stage3c.call_resolutions(log)
    assert [item["resolved"] for item in resolved] == [True, False, True, None]
    assert resolved[2]["completion_event_index"] == 7


def test_evaluation_role_allows_block_index_but_never_training(reference):
    state = initialize(reference)
    opponent = NeuralPolicy.from_checkpoint(reference[0], device="cpu")
    trajectory = stage3c.collect_match(state, opponent, batch_index=12, match_index=88,
        seed=22, learner_seat=0, sampling_seed=55, max_steps=1, collection_role="evaluation")
    assert trajectory.record["collection_role"] == "evaluation"
    assert trajectory.record["training_eligible"] is False
    assert trajectory.record["batch_index"] == 12
    assert state.updates == 0
    trajectories, plans = synthetic_batch(state)
    trajectories[0].record["collection_role"] = "evaluation"
    trajectories[0].record["training_eligible"] = False
    before = stage3c.state_manifest(state)
    with pytest.raises(ValueError, match="ineligible matches"):
        stage3c.update_batch(state, trajectories, batch_index=0, planned_matches=plans)
    assert stage3c.state_manifest(state) == before


def test_collection_retains_failure_when_trace_not_yet_appended(reference, monkeypatch):
    state = initialize(reference)
    opponent = NeuralPolicy.from_checkpoint(reference[0], device="cpu")
    original = PlayerAdapter.resolve
    def fail_resolve(self, action, observation):
        raise ValueError("Injected pre-step resolution failure")
    monkeypatch.setattr(PlayerAdapter, "resolve", fail_resolve)
    trajectory = stage3c.collect_match(state, opponent, batch_index=0, match_index=0,
        seed=22, learner_seat=0, sampling_seed=55, max_steps=2)
    assert not trajectory.record["completed"]
    assert trajectory.record["failure"]["message"] == "Injected pre-step resolution failure"
    assert trajectory.record["requests"] == []
    assert trajectory.record["learner_rows"]
    assert trajectory.record["learner_utility"] is None
