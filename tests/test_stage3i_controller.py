"""Stage 3I reservation, immutable source, and subprocess cleanup controls."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('stage3i_controller', SCRIPTS / 'stage3i.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def config():
    return dict(reserve_seconds=150, phase_limits_seconds=dict(m.PHASE_LIMITS), max_match_executions=10000)


def snap(remaining=5400, ops=()):
    return dict(remaining_seconds=remaining, operations=list(ops))


def op(name, seconds, status='completed'):
    return dict(name=name, elapsed_seconds=seconds, status=status)


def test_shutdown_and_later_phase_reservations_are_both_protected():
    m.admission(snap(2250), 'training-formal', 1800, config(), future_reserve_seconds=300)
    with pytest.raises(ValueError, match='protected reserves'):
        m.admission(snap(2249), 'training-formal', 1800, config(), future_reserve_seconds=300)


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


def test_cumulative_failures_count_against_each_phase():
    history = [op('preflight-old', 400, 'failed'), op('audit-old', 290)]
    m.admission(snap(ops=history), 'preflight-fixed', 50, config())
    with pytest.raises(ValueError, match='preflight cap'):
        m.admission(snap(ops=history), 'preflight-fixed', 51, config())
    with pytest.raises(ValueError, match='audit cap'):
        m.admission(snap(ops=history), 'audit-final', 15, config())
    m.admission(snap(ops=history), 'features-formal', 600, config())


@pytest.mark.parametrize('failed', sorted(m.FORMAL_NAMES))
def test_failed_formal_stops_other_formal_but_retains_audit(failed):
    history = [op(failed, 10, 'failed')]
    following = 'evaluation-formal' if failed != 'evaluation-formal' else 'training-formal'
    with pytest.raises(ValueError, match='failed formal'):
        m.admission(snap(ops=history), following, 60, config())
    m.admission(snap(ops=history), 'audit-failure', 60, config())


@pytest.mark.parametrize('alias', ['features-retry', 'training-retry', 'calibration-retry', 'evaluation-retry'])
def test_formal_alias_cannot_evade_failed_formal_stop(alias):
    with pytest.raises(ValueError):
        m.admission(snap(ops=[op('training-formal', 10, 'failed')]), alias, 60, config())


def test_completed_features_and_training_allow_later_phases():
    history = [op('features-formal', 200), op('training-formal', 1000)]
    m.admission(snap(ops=history), 'calibration-formal', 300, config())
    m.admission(snap(ops=history), 'evaluation-formal', 1800, config())


def test_preflight_failure_allows_a_corrected_formal_attempt():
    m.admission(snap(ops=[op('preflight-old', 10, 'failed')]), 'features-formal', 600, config())


def match_receipt(name, attempts, status='completed'):
    return dict(schema='stage3i-operation-v1', name=name, reserved_match_attempts=attempts, status=status)


def test_attempt_limit_counts_failed_and_prepared_reservations():
    history = [match_receipt('preflight-games', 40, 'failed'),
               match_receipt('evaluation-formal', 9900, 'prepared')]
    assert m.match_admission(history, 'audit-replay', 60, config()) == 9940
    with pytest.raises(ValueError, match='campaign ceiling'):
        m.match_admission(history, 'audit-replay', 61, config())


@pytest.mark.parametrize('name', ['evaluation-formal', 'preflight-games', 'preflight-games-fixed', 'audit-replay'])
def test_game_operations_require_attempt_reservation(name):
    with pytest.raises(ValueError, match='positive match-attempt'):
        m.match_admission([], name, 0, config())
    assert m.match_admission([], name, 20, config()) == 0


@pytest.mark.parametrize('value', [-1, 1.5, True, float('inf')])
def test_invalid_match_reservation(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        m.match_admission([], 'evaluation-formal', value, config())


def test_nongame_operations_cannot_disguise_matches():
    for name in ('training-formal', 'features-formal', 'calibration-formal', 'preflight-training'):
        with pytest.raises(ValueError, match='cannot reserve'):
            m.match_admission([], name, 1, config())


def test_match_history_must_be_complete_and_unambiguous():
    with pytest.raises(ValueError, match='Malformed'):
        m.match_admission([{'schema':'stage3i-operation-v1','name':'preflight-old'}], 'evaluation-formal', 1, config())
    record = match_receipt('preflight-games', 20)
    with pytest.raises(ValueError, match='duplicate'):
        m.match_admission([record, record], 'evaluation-formal', 1, config())
    with pytest.raises(ValueError, match='Repeated'):
        m.match_admission([record], 'preflight-games', 1, config())


def test_controller_lock_serializes_the_admission_gap(tmp_path):
    with m.controller_lock(tmp_path):
        with pytest.raises(RuntimeError, match='controller is active'):
            with m.controller_lock(tmp_path):
                pass
    with m.controller_lock(tmp_path):
        pass


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    for path in ['scripts/stage3i.py', 'scripts/package_a_stability.py', 'configs/stage3i.json']:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path)
    source = tmp_path / 'worker.py'
    source.write_text('old source')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'sources': {'worker.py': m.digest(source)}}))
    return manifest, source


def test_explicit_source_membership_pins_only_selected_files(tmp_path, monkeypatch):
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


def test_stale_manifest_cannot_unpin_controller(tmp_path, monkeypatch):
    manifest, _ = source_fixture(tmp_path, monkeypatch)
    manifest.write_text(json.dumps({'sources': {'scripts/stage3i.py': '0' * 64}}))
    with pytest.raises(RuntimeError, match='Frozen source changed'):
        m.sources(manifest)


def test_exit_terminates_descendant_group_after_worker_exits(tmp_path, monkeypatch):
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
    receipt = {'name': 'preflight-exit', 'reserved_match_attempts': 0}
    m.execute(['not-executed'], 20, dict(data_root=str(data), memory_bytes=10**15,
              artifact_bytes=1000, free_disk_reserve_bytes=10), {}, receipt)
    assert calls == [(12345, 0), (12345, signal.SIGTERM), (12345, 0)]
    assert receipt['termination_requested'] and receipt['exit_code'] == 0


def saved_config(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    monkeypatch.setattr(m, 'ROOT', project)
    path = project / 'stage3i.json'
    monkeypatch.setattr(m, 'CONFIG', path)
    reference = tmp_path / 'reference.pt'
    reference.write_bytes(b'fixed historical checkpoint fixture')
    cfg = dict(schema='stage3i-config-v1', limit_seconds=5400, reserve_seconds=150,
               memory_bytes=32*1024**3, artifact_bytes=8_000_000_000,
               free_disk_reserve_bytes=10_000_000_000, new_training_matches=0,
               max_match_executions=10000, policy_promotion=False,
               phase_limits_seconds=dict(m.PHASE_LIMITS), reports_root='reports/stage3i',
               data_root=str(tmp_path/'external'), reference_checkpoint=str(reference),
               reference_sha256=m.digest(reference), call_checkpoint=str(reference), call_sha256=m.digest(reference))
    path.write_text(json.dumps(cfg))
    return path, cfg


def test_configuration_fixes_scope_and_historical_identities(tmp_path, monkeypatch):
    path, _ = saved_config(tmp_path, monkeypatch)
    loaded, identity = m.read_config()
    assert identity == m.digest(path)
    assert loaded['new_training_matches'] == 0
    assert loaded['max_match_executions'] == 10000


@pytest.mark.parametrize('field,value', [('new_training_matches',1), ('max_match_executions',10001),
    ('policy_promotion',True), ('limit_seconds',5401), ('reserve_seconds',149),
    ('artifact_bytes',8_000_000_001), ('memory_bytes',64*1024**3),
    ('free_disk_reserve_bytes',0), ('schema','stage3i-unbounded-v1')])
def test_scope_expansion_rejected(tmp_path, monkeypatch, field, value):
    path, cfg = saved_config(tmp_path, monkeypatch)
    cfg[field] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='scope changed'):
        m.read_config()


def test_historical_checkpoint_mutation_rejected(tmp_path, monkeypatch):
    _, cfg = saved_config(tmp_path, monkeypatch)
    Path(cfg['reference_checkpoint']).write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='Historical checkpoint changed'):
        m.read_config()
