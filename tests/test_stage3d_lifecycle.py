"""Short real-engine lifecycle checks; no complete matches or strength claims."""
from copy import deepcopy
from dataclasses import is_dataclass
import hashlib
import json
from pathlib import Path
import random
import stat
import threading

import numpy as np
import pytest
import torch

from riichi_ai import stage3c, stage3d
from riichi_ai.model import NeuralPolicy, PolicyNetwork, efficiency_model_config
from riichi_ai.player import PlayerAdapter, PlayerView

PROTOCOL = "d" * 64
PLAN = dict(batch_index=3, match_index=17, seed=22, learner_seat=0, sampling_seed=55)
SMALL = {"suit_width": 8, "history_width": 4, "hidden_width": 16,
         "context_width": 8, "candidate_width": 8}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def policies(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(992)
        model = PolicyNetwork(efficiency_model_config(SMALL))
    path = tmp_path / "reference.pt"
    torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state = stage3c.initialize(path, digest, protocol_sha256=PROTOCOL, initialization_seed=310211)
    return state, NeuralPolicy.from_checkpoint(path, device="cpu")


def _common_behavior(record):
    result = deepcopy(record)
    for key in ("collector_schema", "elapsed_seconds", "interrupted", "eligibility_scope"):
        result.pop(key, None)
    for row in result["learner_rows"]:
        row.pop("training_eligible", None)
    return result


def _same_arrays(first, second):
    assert len(first.encoded) == len(second.encoded)
    for left, right in zip(first.encoded, second.encoded):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


def test_bounded_evaluation_behavior_identical_to_original(policies):
    state, opponent = policies
    before = stage3c.state_manifest(state)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    old = stage3c.collect_match(state, opponent, **PLAN, max_steps=3, collection_role="evaluation")
    new = stage3d.collect_evaluation_match(state, opponent, **PLAN, max_steps=3)
    assert _common_behavior(old.record) == _common_behavior(new.record)
    _same_arrays(old, new)
    assert new.record["steps"] == 3 and new.record["truncated"]
    assert new.record["collector_schema"] == stage3d.COLLECTOR_SCHEMA
    assert new.record["training_eligible"] is False
    # Historical "eligible" identifies a learner behavior choice. The match
    # and explicit row training flags prohibit optimization in this collector.
    assert all(row["eligible"] is True for row in new.record["learner_rows"])
    assert all(row["training_eligible"] is False for row in new.record["learner_rows"])
    assert stage3c.state_manifest(state) == before
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_cooperative_cancel_retains_exact_prior_two_steps(policies):
    state, opponent = policies
    checks = []
    def stop():
        checks.append(True)
        return len(checks) == 3
    old = stage3c.collect_match(state, opponent, **PLAN, max_steps=2, collection_role="evaluation")
    new = stage3d.collect_evaluation_match(state, opponent, **PLAN, max_steps=20, stop_requested=stop)
    record = new.record
    assert len(checks) == 3 and record["steps"] == 2
    assert record["interrupted"] and record["stop_reason"] == "cancelled"
    assert record["failure"] == {"type": "CollectionCancelled",
                                  "message": "Evaluation collection cancellation requested", "step": 2}
    assert not record["completed"] and not record["terminated"]
    assert record["learner_utility"] is None and record["utilities"] is None
    old_behavior, new_behavior = _common_behavior(old.record), _common_behavior(record)
    for key in ("max_steps", "failure", "stop_reason", "truncated", "conservation_verified"):
        old_behavior.pop(key)
        new_behavior.pop(key)
    assert old_behavior == new_behavior
    _same_arrays(old, new)


def test_cancel_before_first_response_keeps_initial_auditor_log(policies, monkeypatch):
    state, opponent = policies
    def unexpected_actor(*args, **kwargs):
        pytest.fail("Cancellation was not checked before the first actor request")
    monkeypatch.setattr(stage3d, "actor_decision", unexpected_actor)
    trajectory = stage3d.collect_evaluation_match(state, opponent, **PLAN, stop_requested=lambda: True)
    assert trajectory.record["steps"] == 0
    assert trajectory.record["requests"] == trajectory.record["learner_rows"] == trajectory.encoded == []
    assert trajectory.record["auditor_log"]
    assert trajectory.record["learner_utility"] is None
    assert trajectory.record["failure"]["type"] == "CollectionCancelled"


def test_cancellation_during_actor_finishes_response_before_stopping(policies, monkeypatch):
    state, opponent = policies
    flag = threading.Event()
    actual_actor = stage3d.actor_decision
    seen = []
    def actor(model, view, rng, **kwargs):
        assert isinstance(view, PlayerView) and is_dataclass(view)
        assert view.__dataclass_params__.frozen
        assert not hasattr(view, "wall") and not hasattr(view, "seed")
        seen.append(view)
        result = actual_actor(model, view, rng, **kwargs)
        flag.set()
        return result
    monkeypatch.setattr(stage3d, "actor_decision", actor)
    new = stage3d.collect_evaluation_match(state, opponent, **PLAN, max_steps=20, stop_requested=flag.is_set)
    old = stage3c.collect_match(state, opponent, **PLAN, max_steps=1, collection_role="evaluation")
    assert seen and new.record["steps"] == 1
    assert new.record["requests"] == old.record["requests"]
    assert new.record["auditor_log"] == old.record["auditor_log"]
    assert new.record["sampling_rng_end"] == old.record["sampling_rng_end"]
    _same_arrays(new, old)
    assert new.record["stop_reason"] == "cancelled"


def test_callback_exception_preserves_prior_trace(policies):
    state, opponent = policies
    checks = []
    def stop():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError("Injected callback failure")
        return False
    trajectory = stage3d.collect_evaluation_match(state, opponent, **PLAN, stop_requested=stop)
    record = trajectory.record
    assert record["steps"] == 1 and len(record["requests"]) == 1
    assert record["failure"] == {"type": "RuntimeError", "message": "Injected callback failure", "step": 1}
    assert record["stop_reason"] == "failure" and not record["interrupted"]
    assert record["learner_rows"] and trajectory.encoded
    assert record["learner_utility"] is None


def test_exception_before_resolved_request_retains_row_and_arrays(policies, monkeypatch, tmp_path):
    state, opponent = policies
    def fail_resolve(self, action, observation):
        raise ValueError("Injected pre-step resolution failure")
    monkeypatch.setattr(PlayerAdapter, "resolve", fail_resolve)
    trajectory = stage3d.collect_evaluation_match(state, opponent, **PLAN, max_steps=2)
    record = trajectory.record
    assert record["failure"] == {"type": "ValueError", "message": "Injected pre-step resolution failure", "step": 0}
    assert record["requests"] == [] and record["learner_rows"] and trajectory.encoded
    assert record["learner_utility"] is None
    artifacts = stage3d.persist_evaluation_match(trajectory, tmp_path / "evidence")
    saved = json.loads(Path(artifacts["json_path"]).read_text())
    assert saved["failure"] == record["failure"] and saved["training_eligible"] is False
    assert saved["encoded_artifact"]["sha256"] == artifacts["npz_sha256"]
    with np.load(artifacts["npz_path"], allow_pickle=False) as arrays:
        assert len(arrays.files) == sum(len(row) for row in trajectory.encoded)


def test_cancelled_persistence_is_durable_and_exclusive(policies, monkeypatch, tmp_path):
    state, opponent = policies
    trajectory = stage3d.collect_evaluation_match(state, opponent, **PLAN, stop_requested=lambda: True)
    synced = []
    actual_sync = stage3d.os.fsync
    def sync(descriptor):
        synced.append(stage3d.os.fstat(descriptor).st_mode)
        actual_sync(descriptor)
    monkeypatch.setattr(stage3d.os, "fsync", sync)
    destination = tmp_path / "cancelled"
    artifacts = stage3d.persist_evaluation_match(trajectory, destination)
    assert len(synced) == 4
    assert all(stat.S_ISREG(mode) for mode in synced[:2])
    assert all(stat.S_ISDIR(mode) for mode in synced[2:])
    hashes = {key: hashlib.sha256(Path(artifacts[key]).read_bytes()).hexdigest()
              for key in ("json_path", "npz_path")}
    assert hashes["json_path"] == artifacts["json_sha256"]
    assert hashes["npz_path"] == artifacts["npz_sha256"]
    with pytest.raises(FileExistsError):
        stage3d.persist_evaluation_match(trajectory, destination)
    assert {key: hashlib.sha256(Path(artifacts[key]).read_bytes()).hexdigest() for key in hashes} == hashes


def test_persistence_rejects_non_evaluation_before_artifact_creation(policies, tmp_path):
    state, opponent = policies
    trajectory = stage3d.collect_evaluation_match(state, opponent, **PLAN, stop_requested=lambda: True)
    trajectory.record["training_eligible"] = True
    destination = tmp_path / "ineligible"
    with pytest.raises(ValueError, match="evaluation-only"):
        stage3d.persist_evaluation_match(trajectory, destination)
    assert not destination.exists()


def test_evaluation_rows_rejected_by_training_guard(policies):
    state, opponent = policies
    before = stage3c.state_manifest(state)
    trajectory = stage3d.collect_evaluation_match(state, opponent, **PLAN, max_steps=1)
    with pytest.raises(ValueError, match="ineligible matches"):
        stage3c.update_batch(state, [trajectory], batch_index=0, planned_matches=[PLAN])
    assert stage3c.state_manifest(state) == before


def test_deadline_retains_original_failure_semantics(policies):
    state, opponent = policies
    trajectory = stage3d.collect_evaluation_match(state, opponent, **PLAN, deadline=0)
    assert trajectory.record["failure"]["type"] == "TimeoutError"
    assert trajectory.record["stop_reason"] == "failure" and not trajectory.record["interrupted"]
    assert trajectory.record["steps"] == 0 and trajectory.record["learner_utility"] is None
