"""Stage 3G resource admission, source pinning, and process-tree shutdown."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('stage3g_controller', SCRIPTS / 'stage3g.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def config():
    return dict(reserve_seconds=120, phase_limits_seconds=dict(m.PHASE_LIMITS))


def snap(remaining=7200, ops=()):
    return dict(remaining_seconds=remaining, operations=list(ops))


def op(name, seconds, status='completed'):
    return dict(name=name, elapsed_seconds=seconds, status=status)


def test_shutdown_and_future_reserves_are_both_protected():
    m.admission(snap(480), 'preflight', 60, config(), future_reserve_seconds=300)
    with pytest.raises(ValueError, match='protected reserves'):
        m.admission(snap(479), 'preflight', 60, config(), future_reserve_seconds=300)


@pytest.mark.parametrize('name', ['', '../escape', 'x/y', 'a b', 'Upper', 'unmapped'])
def test_invalid_name(name):
    with pytest.raises(ValueError):
        m.admission(snap(), name, 60, config())


@pytest.mark.parametrize('seconds', [float('nan'), float('inf'), 0, 14, -1])
def test_invalid_allowance(seconds):
    with pytest.raises(ValueError):
        m.admission(snap(), 'preflight', seconds, config())


@pytest.mark.parametrize('reserve', [float('nan'), float('inf'), -1])
def test_invalid_future_reserve(reserve):
    with pytest.raises(ValueError):
        m.admission(snap(), 'preflight', 60, config(), reserve)


def test_active_and_repeat_rejected():
    with pytest.raises(ValueError, match='Repeated operation'):
        m.admission(snap(ops=[op('preflight-a', None, 'active')]), 'audit', 60, config())
    with pytest.raises(ValueError, match='Repeated operation'):
        m.admission(snap(ops=[op('preflight-a', 1)]), 'preflight-a', 60, config())


def test_phase_cumulative_counts_failure_and_separates_phases():
    history = [op('preflight-old', 565, 'failed'), op('audit-old', 270)]
    m.admission(snap(ops=history), 'preflight-final', 35, config())
    with pytest.raises(ValueError, match='preflight cap'):
        m.admission(snap(ops=history), 'preflight-final', 36, config())
    with pytest.raises(ValueError, match='audit cap'):
        m.admission(snap(ops=history), 'audit-final', 15, config())
    m.admission(snap(ops=history), 'collect', 900, config())


def test_learning_cap_is_shared_across_arms():
    history = [op('train-mc_a0', 1000), op('train-mc_a1', 1000), op('train-delta_a0', 1000)]
    m.admission(snap(ops=history), 'train-delta_a1', 1000, config())
    with pytest.raises(ValueError, match='learning cap'):
        m.admission(snap(ops=history), 'train-delta_a1', 1001, config())


@pytest.mark.parametrize('failed', sorted(m.FORMAL_NAMES))
def test_failed_formal_stops_new_formal_but_allows_audit(failed):
    history = [op(failed, 10, 'failed')]
    next_formal = 'development' if failed != 'development' else 'train-mc_a0'
    with pytest.raises(ValueError, match='failed formal'):
        m.admission(snap(ops=history), next_formal, 60, config())
    m.admission(snap(ops=history), 'audit-failure', 60, config())


def test_calibration_gate_can_fail_as_completed_operation():
    history = [op('grp-fit', 200)]
    # Gate decisions are saved data, not subprocess failures; MC remains admissible.
    m.admission(snap(ops=history), 'train-mc_a0', 1000, config())


def test_test_failure_does_not_disable_later_formal_after_fix():
    history = [op('preflight-old', 10, 'failed')]
    m.admission(snap(ops=history), 'collect', 900, config())


def test_controller_exclusive_lock(tmp_path):
    with m.controller_lock(tmp_path):
        with pytest.raises(RuntimeError, match='controller is active'):
            with m.controller_lock(tmp_path):
                pass
    with m.controller_lock(tmp_path):
        pass


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    for path in ['scripts/stage3g.py', 'scripts/package_a_stability.py', 'configs/stage3g.json']:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path)
    source = tmp_path / 'worker.py'
    source.write_text('old source')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'sources': {'worker.py': m.digest(source)}}))
    return manifest, source


def test_explicit_source_membership_ignores_new_unlisted_file(tmp_path, monkeypatch):
    manifest, source = source_fixture(tmp_path, monkeypatch)
    before = m.sources(manifest)
    (tmp_path / 'new.py').write_text('new unrelated source')
    assert m.sources(manifest) == before
    source.write_text('changed source')
    with pytest.raises(RuntimeError, match='Frozen source changed'):
        m.sources(manifest)


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'a/../b', 'a//b', 'a\\b', ''])
def test_source_manifest_rejects_unsafe_membership(tmp_path, monkeypatch, name):
    manifest, _ = source_fixture(tmp_path, monkeypatch)
    manifest.write_text(json.dumps({'sources': {name: '0' * 64}}))
    with pytest.raises(ValueError, match='source membership'):
        m.sources(manifest)


def test_fixed_launcher_pin_cannot_be_overwritten_with_stale_value(tmp_path, monkeypatch):
    manifest, _ = source_fixture(tmp_path, monkeypatch)
    manifest.write_text(json.dumps({'sources': {'scripts/stage3g.py': '0' * 64}}))
    with pytest.raises(RuntimeError, match='Frozen source changed'):
        m.sources(manifest)


def test_exit_still_terminates_group_after_worker_already_exited(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'budget-ledger.json').write_text(json.dumps({'operations': [
        {'name': 'preflight-exit', 'status': 'active', 'token': 'token'}]}))
    process = SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(m.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(m, 'artifacts_bytes', lambda c: 0)
    monkeypatch.setattr(m, 'verify_sources', lambda s: None)
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: SimpleNamespace(free=1000))
    calls = []
    def killpg(pid, sig):
        calls.append((pid, sig))
        if len(calls) == 3:
            raise ProcessLookupError
    monkeypatch.setattr(m.os, 'killpg', killpg)
    receipt = {'name': 'preflight-exit'}
    m.execute(['not-executed'], 20, dict(data_root=str(data), memory_bytes=10**15,
              artifact_bytes=1000, free_disk_reserve_bytes=10), {}, receipt)
    assert calls == [(12345, 0), (12345, signal.SIGTERM), (12345, 0)]
    assert receipt['termination_requested'] and receipt['exit_code'] == 0
