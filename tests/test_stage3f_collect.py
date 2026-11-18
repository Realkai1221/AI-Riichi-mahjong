"""Collector boundary/probability checks; short engine attempts are not strength evidence."""
from copy import deepcopy
from dataclasses import is_dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from riichi_ai import stage3c, stage3f_collect as collector
from riichi_ai.model import PolicyNetwork, efficiency_model_config
from riichi_ai.player import PlayerView


def actions(*kinds):
    return [SimpleNamespace(kind=kind) for kind in kinds]


class Draws:
    def __init__(self, *values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


@pytest.mark.parametrize("component_draw,action_draw,component,index", [
    (0., 0., "semantic_uniform", 0), (.049, .99, "semantic_uniform", 2),
    (.05, 0., "gated_base", 1), (.9, .99, "gated_base", 2),
])
def test_explorer_retains_marginal_and_latent_component_probabilities(component_draw, action_draw, component, index):
    result = collector.explore(actions("DISCARD", "PASS", "PON"), [0., .8, .2], Draws(component_draw, action_draw))
    expected = .95 * np.array([0., .8, .2]) + .05 / 3
    np.testing.assert_allclose(result["probabilities"], expected, rtol=0, atol=1e-16)
    assert result["selected_index"] == index and result["selected_component"] == component
    assert result["component_draw"] == component_draw and result["uniform_draw"] == action_draw
    assert result["probability"] == result["probabilities"][index]
    weight, conditional = (.05, 1 / 3) if component == "semantic_uniform" else (.95, [.0, .8, .2][index])
    assert result["component_probability"] == weight
    assert result["conditional_probability"] == conditional
    assert result["joint_probability"] == weight * conditional


@pytest.mark.parametrize("kinds,expected", [(("PASS", "RON", "PON"), 1), (("TSUMO", "ANKAN"), 0), (("RON", "TSUMO"), 0)])
def test_win_rule_is_deterministic_and_consumes_no_rng(kinds, expected):
    rng = random.Random(42)
    before = rng.getstate()
    result = collector.explore(actions(*kinds), np.full(len(kinds), 1 / len(kinds)), rng)
    assert result["selected_component"] == "win_heuristic"
    assert result["selected_index"] == expected
    assert result["probabilities"][expected] == result["probability"] == 1.
    assert sum(result["probabilities"]) == 1.
    assert result["component_draw"] is result["uniform_draw"] is None
    assert rng.getstate() == before


def test_exploration_uses_semantic_not_physical_candidate_count():
    # A semantic CHI may have several interchangeable physical representatives.
    choices = actions("CHI", "PASS")
    choices[0].physical_tokens = ("a", "b", "c")
    choices[1].physical_tokens = ("d",)
    result = collector.explore(choices, [.05, .95], Draws(0., .6))
    assert result["uniform_probabilities"] == [.5, .5]
    assert result["selected_index"] == 1


@pytest.mark.parametrize("values", [[], [.4, .4], [-.1, 1.1], [float("nan"), 1.], [float("inf"), 0.]])
def test_explorer_rejects_invalid_probabilities(values):
    with pytest.raises(ValueError, match="normalized semantic"):
        collector.explore(actions("PASS", "CHI"), values, random.Random(1))


def test_per_seat_plans_and_randomness_are_schedule_independent():
    before = random.getstate()
    plans = [collector.derive_plan(91, index) for index in range(10)]
    assert random.getstate() == before
    assert [collector.derive_plan(91, index) for index in reversed(range(10))][::-1] == plans
    assert len({plan["seed"] for plan in plans}) == 10
    for plan in plans:
        collector.validate_plan(plan)
        assert sorted(plan["seat_behaviors"]) == sorted(collector.BEHAVIORS)
        assert plan["role"] == "development" and len(set(plan["sampling_seeds"])) == 4
    rng1, rng2 = random.Random(plans[0]["sampling_seeds"][0]), random.Random(plans[0]["sampling_seeds"][0])
    choices = actions("CHI", "PASS")
    assert [collector.explore(choices, [.05, .95], rng1) for _ in range(5)] == [collector.explore(choices, [.05, .95], rng2) for _ in range(5)]


@pytest.mark.parametrize("key,value", [
    ("match_id", "../../escape"), ("role", "test"), ("role", "training"),
    ("seed", True), ("index", -1), ("sampling_seeds", [1, 1, 2, 3]),
    ("seat_behaviors", ["reference"] * 4),
])
def test_invalid_plan_cannot_create_artifacts(tmp_path, key, value):
    plan = collector.derive_plan(91, 0)
    plan[key] = value
    with pytest.raises(ValueError):
        collector.run_one(plan, "missing", "a" * 64, "missing", tmp_path / "new", expected_call_sha256="b" * 64)
    assert not (tmp_path / "new").exists()


def test_compact_history_has_explicit_prefix_and_never_borrows_future_or_other_hand():
    def feature(value):
        return {"tiles": np.full((34, 23), value, dtype=np.float32), "globals": np.zeros(45, dtype=np.float32),
                "action_tokens": np.zeros((2, 8), dtype=np.int16), "candidate_efficiency": np.zeros((2, 4), dtype=np.float32)}
    first = np.arange(1400, dtype=np.int16).reshape(140, 10)
    other = np.full((5, 10), 30000, dtype=np.int16)
    actor = collector.pack_actor([feature(0), feature(1), feature(2)],
        {(0, 1): [first[:2], first[2:]], (0, 2): [other]}, [((0, 1), 2), ((0, 1), 140), ((0, 2), 5)])
    assert actor["history_start"].tolist() == [0, 12, 140]
    assert actor["history_length"].tolist() == [2, 128, 5]
    assert actor["history_segment_start"].tolist() == [0, 0, 140]
    assert actor["history_segment_length"].tolist() == [140, 140, 5]
    assert actor["history_prefix_length"].tolist() == [2, 140, 5]
    np.testing.assert_array_equal(actor["history_tokens"][:2], first[:2])
    np.testing.assert_array_equal(actor["history_tokens"][12:140], first[-128:])
    np.testing.assert_array_equal(actor["history_tokens"][140:], other)
    assert actor["candidate_offsets"].tolist() == [0, 2, 4, 6]
    assert not ({"final_utility", "seat", "seed", "behavior_probabilities", "canonical_events"} & actor.keys())
    assert collector._pack_arrays(actor) == collector._pack_arrays(deepcopy(actor))


@pytest.fixture
def checkpoints(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = efficiency_model_config({"suit_width": 8, "history_width": 4, "hidden_width": 16,
                                     "context_width": 8, "candidate_width": 8})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(981)
        model = PolicyNetwork(config)
    reference_path = tmp_path / "reference.pt"
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, reference_path)
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    state = stage3c.initialize(reference_path, reference_sha, protocol_sha256="f" * 64, initialization_seed=310211)
    call_path = tmp_path / "call.pt"
    saved = stage3c.save_boundary(state, call_path)
    yield reference_path, reference_sha, call_path, saved["sha256"]
    collector._policies.cache_clear()
    torch.set_num_threads(old_threads)


def collect(tmp_path, checkpoints, *, name="run", **kwargs):
    reference_path, reference_sha, call_path, call_sha = checkpoints
    return collector.run_one(collector.derive_plan(101, 0), reference_path, reference_sha, call_path,
                             tmp_path / name, expected_call_sha256=call_sha, **kwargs)


def load_provenance(tmp_path, name="run"):
    return json.loads(gzip.decompress((tmp_path / name / "provenance/match-000000.json.gz").read_bytes()))


def test_truncated_attempt_is_reproducible_durable_and_never_eligible(tmp_path, checkpoints):
    first = collect(tmp_path, checkpoints, name="first", max_steps=3)
    second = collect(tmp_path, checkpoints, name="second", max_steps=3)
    assert first["payload_sha256"] == second["payload_sha256"]
    assert first["artifacts"] == second["artifacts"]
    assert first["steps"] == 3 and first["truncated"] and not first["completed"]
    assert not first["training_eligible"] and not first["ppo_eligible"]
    assert first["utilities"] is first["ranks"] is None
    assert not (tmp_path / "first/actor").exists()
    assert not (tmp_path / "first/targets/match-000000.npz").exists()
    assert (tmp_path / "first/lifecycle/match-000000.started.json").exists()
    assert (tmp_path / "first/lifecycle/match-000000.finished.json").exists()
    provenance = load_provenance(tmp_path, "first")
    assert provenance["rows"]
    for row in provenance["rows"]:
        assert row["training_eligible"] is row["ppo_eligible"] is False
        assert row["probability"] == row["probabilities"][row["selected_index"]]
    with pytest.raises(FileExistsError):
        collect(tmp_path, checkpoints, name="first", max_steps=3)


def test_all_views_built_before_any_policy_and_actor_input_is_immutable(tmp_path, checkpoints, monkeypatch):
    phase, observed = {"deciding": False}, []
    real_build, real_decision, real_create = collector.PlayerAdapter.build, collector._decision, collector.create_env
    def build(self, observation):
        assert not phase["deciding"], "A player view was built after an actor already chose"
        observed.append(("build", self.seat))
        return real_build(self, observation)
    def decision(name, view, reference, call, rng):
        phase["deciding"] = True
        assert isinstance(view, PlayerView) and is_dataclass(view) and view.__dataclass_params__.frozen
        assert not hasattr(view, "wall") and not hasattr(view, "seed")
        observed.append(("decision", view.seat))
        return real_decision(name, view, reference, call, rng)
    class Proxy:
        def __init__(self, env):
            self.env = env
        def __getattr__(self, key):
            return getattr(self.env, key)
        def step(self, choices):
            observed.append(("step", sorted(choices)))
            result = self.env.step(choices)
            phase["deciding"] = False
            return result
    def create(seed):
        env, observations = real_create(seed)
        return Proxy(env), observations
    monkeypatch.setattr(collector.PlayerAdapter, "build", build)
    monkeypatch.setattr(collector, "_decision", decision)
    monkeypatch.setattr(collector, "create_env", create)
    report = collect(tmp_path, checkpoints, max_steps=3)
    assert report["steps"] == 3 and report["truncated"]
    assert sum(kind == "step" for kind, _ in observed) == 3
    assert any(kind == "decision" for kind, _ in observed)


def test_expired_deadline_saves_failure_without_fake_terminal_return(tmp_path, checkpoints):
    report = collect(tmp_path, checkpoints, deadline=0.)
    assert report["failure"]["type"] == "TimeoutError" and report["steps"] == 0
    assert report["rows"] == 0 and report["utilities"] is None
    assert report["completed"] is False and report["terminal_result_verified"] is False


def test_resolve_failure_retains_attempted_choices_without_claiming_submission(tmp_path, checkpoints, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("injected resolution error")
    monkeypatch.setattr(collector.PlayerAdapter, "resolve", fail)
    report = collect(tmp_path, checkpoints, max_steps=3)
    assert report["failure"]["message"] == "injected resolution error" and report["steps"] == 0
    provenance = load_provenance(tmp_path)
    assert provenance["rows"] and len(provenance["requests"]) == 1
    assert provenance["requests"][0]["submitted"] is False
    assert not any(name.endswith(".npz") for name in report["artifacts"])


def test_checkpoint_mismatch_is_persisted_before_any_engine_creation(tmp_path, checkpoints, monkeypatch):
    def unexpected(*args):
        pytest.fail("Checkpoint mismatch reached engine")
    monkeypatch.setattr(collector, "create_env", unexpected)
    reference_path, reference_sha, call_path, _ = checkpoints
    report = collector.run_one(collector.derive_plan(91, 0), reference_path, reference_sha, call_path,
                              tmp_path / "mismatch", expected_call_sha256="0" * 64)
    assert report["failure"]["message"] == "Frozen checkpoint identity mismatch"
    assert report["steps"] == report["rows"] == 0 and not report["completed"]
