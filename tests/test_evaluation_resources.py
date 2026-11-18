"""Resource-stop evidence and bounded scheduling without expensive campaigns."""

from pathlib import Path

import pytest

from riichi_ai import evaluation


@pytest.fixture
def block_stub(monkeypatch):
    """A complete eight-game panel isolates scheduling from simulator rules."""
    returned = []

    def block(job):
        candidate, baseline, opponents, index, seed, *_ = job
        records = []
        for focal in range(4):
            for arm, policy in (("candidate", candidate), ("baseline", baseline)):
                names = [None] * 4
                names[focal] = policy
                for offset, opponent in enumerate(opponents, 1):
                    names[(focal + offset) % 4] = opponent
                records.append({
                    "block": index, "focal_seat": focal, "arm": arm,
                    "seed": evaluation._seed(seed, "block", index),
                    "policies": [evaluation._evaluation_manifest(name, evaluation._seed(seed, "policy", index, focal, seat))
                                 for seat, name in enumerate(names)],
                    "completed": True, "decisions": 1, "ranks": [1, 2, 3, 4],
                    "scores": [40000, 30000, 20000, 10000], "failure": None, "stop_reason": "completed",
                    "timing_seconds": {key: 0 for key in ("encode", "policy", "resolve", "step", "audit", "view_hash", "wall_hash")},
                    "setup_seconds": 0, "finalization_seconds": 0, "unallocated_loop_seconds": 0,
                })
        returned.append(index)
        return records, {"block": index, "wall_seed": evaluation._seed(seed, "block", index),
                         "valid": True, "seat_utility_differences": [0] * 4, "mean_utility_difference": 0}

    monkeypatch.setattr(evaluation, "_evaluate_block", block)
    monkeypatch.setattr(evaluation, "source_manifest", lambda: {"frozen_test_sources": True})
    return returned


def test_preflight_real_disk_cap_stops_before_any_block(block_stub, tmp_path):
    (tmp_path / "existing.bin").write_bytes(b"1234567890")
    result = evaluation.evaluate_paired("random", "random", blocks=3, workers=2,
                                        artifact_roots=[tmp_path], disk_limit_bytes=9)
    assert block_stub == []
    assert not result["valid"] and result["statistics"] is None
    assert result["completed_blocks"] == result["completed_games"] == 0
    assert result["resource_failure"]["type"] == "DiskBudgetExceeded"
    assert result["resource_failure"]["measured_bytes"] == 10


def test_serial_stop_preserves_completed_block_but_cannot_promote(block_stub, monkeypatch):
    monkeypatch.setattr(evaluation, "_evaluation_resources", lambda _: {
        "process_tree_rss_bytes": 11 if block_stub else 1, "artifact_bytes": 0})
    result = evaluation.evaluate_paired("random", "random", blocks=3, memory_limit_bytes=10)
    assert block_stub == [0]
    assert result["completed_blocks"] == 1 and result["completed_games"] == 8
    assert result["blocks"][0]["valid"]  # Complete evidence remains interpretable.
    assert result["match_policies_match_frozen_panel"]
    assert not result["valid"] and result["statistics"] is None
    assert result["resource_failure"]["type"] == "MemoryBudgetExceeded"


class ImmediatePool:
    """Executor stand-in tracks submitted rather than executed outstanding work."""

    instances = []

    def __init__(self, *, max_workers, **_):
        self.max_workers, self.outstanding, self.maximum = max_workers, 0, 0
        self.submitted, self.terminated, self.shutdown_called = [], False, False
        self.instances.append(self)

    def submit(self, function, job):
        self.submitted.append(job[3])
        self.outstanding += 1
        self.maximum = max(self.maximum, self.outstanding)
        pool = self

        class Future:
            def result(self, timeout=None):
                pool.outstanding -= 1
                return function(job)

        return Future()

    def terminate_workers(self):
        self.terminated = True

    def shutdown(self, *, wait):
        self.shutdown_called = True


def test_parallel_stop_terminates_pending_work_and_keeps_order(block_stub, monkeypatch):
    monkeypatch.setattr(evaluation, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(evaluation, "_evaluation_resources", lambda _: {
        "process_tree_rss_bytes": 11 if block_stub else 1, "artifact_bytes": 0})
    result = evaluation.evaluate_paired("random", "random", blocks=7, workers=2, memory_limit_bytes=10)
    pool = ImmediatePool.instances[-1]
    assert pool.submitted == [0, 1] and pool.maximum == 2
    assert pool.terminated and not pool.shutdown_called
    assert block_stub == [0]
    assert [block["block"] for block in result["blocks"]] == [0]
    assert not result["valid"] and result["statistics"] is None


def test_unlimited_defaults_remain_complete_and_queue_is_bounded(block_stub, monkeypatch):
    monkeypatch.setattr(evaluation, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(evaluation, "_evaluation_resources", lambda _: pytest.fail("No resource sampling requested"))
    result = evaluation.evaluate_paired("random", "random", blocks=7, workers=2)
    pool = ImmediatePool.instances[-1]
    assert pool.maximum == 2 and pool.submitted == list(range(7))
    assert pool.shutdown_called and not pool.terminated
    assert result["valid"] and result["statistics"] is not None
    assert result["resource_failure"] is None and result["completed_blocks"] == 7
    assert block_stub == list(range(7))


@pytest.mark.parametrize("arguments", [
    {"memory_limit_bytes": 0}, {"disk_limit_bytes": -1},
    {"memory_limit_bytes": True}, {"disk_limit_bytes": 10},
])
def test_invalid_limit_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        evaluation.evaluate_paired("random", "random", **arguments)


def test_artifact_roots_do_not_double_count_nested_files(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    (child / "payload").write_bytes(b"123")
    assert evaluation._evaluation_resources([tmp_path, child])["artifact_bytes"] == 3
