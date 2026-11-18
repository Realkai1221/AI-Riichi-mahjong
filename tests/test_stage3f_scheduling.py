"""Fixed collection counts, failed-phase gates, and deadline admission."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_script("stage3f_launcher_scheduling_test", "stage3f.py")
collector = load_script("stage3f_collection_scheduling_test", "stage3f_collect.py")


def scope(tmp_path):
    reports, data = tmp_path / "reports", tmp_path / "data"
    reports.mkdir()
    data.mkdir()
    (reports / "plan.json").write_text("{}")
    return {"root_seed": 2026102201, "worker_grid": [1, 4, 8, 14], "parity_matches": 16,
            "matches_per_worker_probe": 32, "pilot_matches": 1000,
            "max_match_executions": 1200, "data_root": str(data), "reports_root": str(reports),
            "reserve_seconds": 120, "limit_seconds": 7200, "diagnosis_limit_seconds": 300}


@pytest.mark.parametrize("failed_name", ["parity-serial", "parity-parallel", "workers-1", "workers-14", "pilot"])
def test_failed_formal_phase_blocks_following_numerical_phase(failed_name):
    snapshot = {"remaining_seconds": 7000,
                "operations": [{"name": failed_name, "status": "failed", "elapsed_seconds": 10}]}
    with pytest.raises(ValueError):
        launcher.admission(snapshot, "workers-8", 60, {"reserve_seconds": 120, "diagnosis_limit_seconds": 300})


def test_expired_serial_deadline_starts_no_match(tmp_path, monkeypatch):
    config = scope(tmp_path)
    calls = []
    monkeypatch.setattr(collector, "worker", lambda job: calls.append(job))
    with pytest.raises((RuntimeError, TimeoutError)):
        collector.collect(config, {"config_sha256": "0" * 64}, "expired",
                          [{"match_id": "match-000001"}], 1, deadline=-1)
    assert calls == []
    assert not list((Path(config["data_root"]) / "expired").glob("lifecycle/*.started.json"))


def test_expired_parallel_deadline_submits_no_match(tmp_path, monkeypatch):
    config = scope(tmp_path)
    submitted = []

    class Pool:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, job):
            submitted.append(job)
            raise AssertionError("Submission after expired deadline")

    monkeypatch.setattr(collector, "ProcessPoolExecutor", Pool)
    with pytest.raises((RuntimeError, TimeoutError)):
        collector.collect(config, {"config_sha256": "0" * 64}, "expired",
                          [{"match_id": "match-000001"}], 4, deadline=-1)
    assert submitted == []


@pytest.mark.parametrize("key,value", [("parity_matches", 17), ("pilot_matches", 1001),
                                        ("matches_per_worker_probe", 33), ("worker_grid", [1, 8]),
                                        ("max_match_executions", 1159)])
def test_prepare_rejects_changed_fixed_counts_or_insufficient_match_cap(tmp_path, monkeypatch, key, value):
    config = scope(tmp_path)
    fake_root = tmp_path / "project"
    (fake_root / "configs").mkdir(parents=True)
    (fake_root / "configs/stage3f.json").write_text("{}")
    monkeypatch.setattr(collector, "ROOT", fake_root)
    monkeypatch.setattr(collector, "frozen_sources", lambda: {})
    config[key] = value
    with pytest.raises(ValueError):
        collector.prepare(config)


def test_fixed_plan_accounts_for_duplicate_parity_executions(tmp_path, monkeypatch):
    config = scope(tmp_path)
    fake_root = tmp_path / "project"
    (fake_root / "configs").mkdir(parents=True)
    (fake_root / "configs/stage3f.json").write_text("{}")
    monkeypatch.setattr(collector, "ROOT", fake_root)
    monkeypatch.setattr(collector, "frozen_sources", lambda: {})
    plan = collector.prepare(config)
    assert len(plan["parity"]) == 16
    assert [p["workers"] for p in plan["worker_probes"]] == [1, 4, 8, 14]
    assert all(len(p["matches"]) == 32 for p in plan["worker_probes"])
    assert len(plan["pilot"]) == 1000
    assert plan["parity_executions"] + plan["worker_probe_executions"] + plan["pilot_executions"] == 1160
    ids = [p["match_id"] for p in plan["parity"]]
    ids += [p["match_id"] for group in plan["worker_probes"] for p in group["matches"]]
    ids += [p["match_id"] for p in plan["pilot"]]
    assert len(ids) == len(set(ids)) == 1144
    assert plan["training_eligible"] is False and plan["ppo_eligible"] is False


def test_total_execution_cap_counts_preexisting_failed_starts(tmp_path, monkeypatch):
    config = scope(tmp_path)
    directory = Path(config["data_root"]) / "old" / "lifecycle"
    directory.mkdir(parents=True)
    (directory / "failed.started.json").write_text("{}")
    calls = []
    monkeypatch.setattr(collector, "worker", lambda job: calls.append(job))
    oversized = [{"match_id": f"match-{index:06d}"} for index in range(1200)]
    with pytest.raises(ValueError, match="ceiling"):
        collector.collect(config, {"config_sha256": "0" * 64}, "pilot", oversized, 1, deadline=float("inf"))
    assert calls == []
