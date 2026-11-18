"""Budget/identity invariants without running a timed benchmark campaign."""
from pathlib import Path
import json
import time

import pytest

from riichi_ai import stage2b_runtime as runtime


def test_trial_refuses_deadline_without_starting_workers(tmp_path):
    with pytest.raises(RuntimeError, match='Insufficient'):
        runtime.benchmark_trial(checkpoint='unused', workers=1, duration=1200, root_seed=1,
                                namespace='sustained', output=tmp_path / 'result.json',
                                deadline=time.monotonic() + 1, memory_limit=1000,
                                artifact_roots=[], disk_limit=1000)
    assert not list(tmp_path.iterdir())


def test_campaign_freezes_best_measured_workers_and_stops_escalation(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'frozen.pt'
    checkpoint.write_bytes(b'identity')
    calls = []
    speed = {1: 100, 4: 200, 8: 180, 14: 400}
    def trial(**kwargs):
        calls.append(kwargs)
        return {'valid': True, 'workers': kwargs['workers'], 'decisions_per_second': speed[kwargs['workers']],
                'completed_matches': 10, 'peak_process_tree_rss': 123}
    monkeypatch.setattr(runtime, 'benchmark_trial', trial)
    monkeypatch.setattr(runtime, 'corpus_and_historical_wall_seeds', lambda plan: set())
    plan = {'limits': {'benchmark_total_seconds': 1800, 'memory_gib': 32, 'new_artifact_bytes': 5000,
                       'confirmation_max_blocks': 256}, 'evaluation': {'pilot_blocks': 30},
            'benchmark': {'worker_trials': [1, 4, 8, 14], 'short_trial_seconds': 35, 'sustained_seconds': 1200},
            'data_root': str(tmp_path / 'data'), 'seed_namespaces': {'benchmark': 1720, 'development': 1718, 'confirmation': 1719}}
    selection = {'checkpoint': str(checkpoint), 'checkpoint_sha256': runtime.file_hash(checkpoint)}
    result = runtime.benchmark_campaign(plan, selection, tmp_path / 'reports')
    assert [call['workers'] for call in calls] == [1, 4, 8, 4]
    assert len({call['deadline'] for call in calls}) == 1
    assert calls[-1]['duration'] == 1200
    assert result['valid'] and result['within_budget']
    chosen = json.loads((tmp_path / 'reports/benchmark-worker-selection.json').read_text())
    assert chosen['workers'] == 4
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        runtime.benchmark_campaign(plan, selection, tmp_path / 'reports')


def test_completed_evidence_is_not_overwritten(tmp_path):
    output = tmp_path / 'result.json'
    output.write_text('{}')
    with pytest.raises(FileExistsError):
        runtime.benchmark_trial(checkpoint='unused', workers=1, duration=1, root_seed=1,
                                namespace='sustained', output=output,
                                deadline=time.monotonic() + 60, memory_limit=1000,
                                artifact_roots=[], disk_limit=1000)
    assert output.read_text() == '{}'


def test_trial_does_not_call_shortened_duration_complete(tmp_path):
    with pytest.raises(RuntimeError, match='Insufficient'):
        runtime.benchmark_trial(checkpoint='unused', workers=1, duration=1200, root_seed=1,
                                namespace='sustained', output=tmp_path / 'result.json',
                                deadline=time.monotonic() + 1100, memory_limit=1000,
                                artifact_roots=[], disk_limit=1000)


def test_benchmark_loaded_identity_is_checked_per_seat(monkeypatch):
    monkeypatch.setattr(runtime, 'run_match', lambda *args, **kwargs: {
        'seed': 1, 'completed': True, 'policies': [dict(checkpoint_sha256='changed')] * 4,
        'digest': 'original-game-digest'})
    result = runtime._benchmark_match(1, 'unused', 'frozen', 'unused')
    assert not result['completed']
    assert result['stop_reason'] == 'policy_identity_mismatch'
    assert len(result['loaded_policies']) == 4
    assert result['digest'] == 'original-game-digest'
