"""Resource/failure boundaries for the separate Stage 3J campaign."""
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('stage3j_controller', SCRIPTS / 'stage3j.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def config():
    return dict(reserve_seconds=150, phase_limits_seconds=dict(m.PHASE_LIMITS), max_match_executions=9000)

def snapshot(remaining=5400, operations=()):
    return dict(remaining_seconds=remaining, operations=list(operations))

def test_future_audit_and_shutdown_reserves_survive_admission():
    assert m.admission(snapshot(4250), 'evaluation-formal', 3500, config(), 600) == 'evaluation'
    with pytest.raises(ValueError, match='protected reserves'):
        m.admission(snapshot(4249), 'evaluation-formal', 3500, config(), 600)

def test_phase_cap_includes_failed_work():
    old = dict(name='preflight-failed', elapsed_seconds=550, status='failed')
    m.admission(snapshot(operations=[old]), 'preflight-repaired', 50, config())
    with pytest.raises(ValueError, match='preflight cap'):
        m.admission(snapshot(operations=[old]), 'preflight-repaired', 51, config())

@pytest.mark.parametrize('name', ['evaluation-retry', 'training-formal', 'calibration-formal', 'features-formal'])
def test_no_training_or_disguised_formal_retries(name):
    with pytest.raises(ValueError):
        m.admission(snapshot(), name, 60, config())

def test_failed_formal_cannot_repeat_but_can_be_audited():
    old = dict(name='evaluation-formal', elapsed_seconds=20, status='failed')
    with pytest.raises(ValueError):
        m.admission(snapshot(operations=[old]), 'evaluation-formal', 60, config())
    m.admission(snapshot(operations=[old]), 'audit-failure', 60, config())

@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0, 14, -1])
def test_invalid_reservation(value):
    with pytest.raises(ValueError):
        m.admission(snapshot(), 'preflight-tests', value, config())

def test_conservative_attempts_and_final_replay_reservation():
    history = [dict(schema='stage3j-operation-v1', name='preflight-games', reserved_match_attempts=64, status='completed'),
               dict(schema='stage3j-operation-v1', name='evaluation-formal', reserved_match_attempts=8192, status='failed')]
    assert m.match_admission(history, 'audit-replay', 32, config()) == 8256
    with pytest.raises(ValueError, match='campaign ceiling'):
        m.match_admission(history, 'audit-replay', 745, config())
    with pytest.raises(ValueError):
        m.match_admission([], 'evaluation-formal', 0, config())
    with pytest.raises(ValueError):
        m.match_admission([], 'preflight-tests', 1, config())

def test_lock_serializes_even_admission_gap(tmp_path):
    with m.controller_lock(tmp_path):
        with pytest.raises(RuntimeError, match='controller is active'):
            with m.controller_lock(tmp_path):
                pass
