"""Frozen Stage 3C resource, seed and no-replacement contracts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('stage3c_controller', ROOT / 'scripts/stage3c.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_disjoint_fixed_training_evaluation_and_sampling_plan():
    _, config, identity = controller.settings(ROOT / 'configs/stage3c.json')
    training = [m for b in config['batches'] for m in b['matches']]
    evaluation = config['evaluation']
    tw = {m['seed'] for m in training}
    ts = {m['sampling_seed'] for m in training}
    ew = {b['wall_seed'] for b in evaluation}
    es = {p['sampling_seed'] for b in evaluation for p in b['pairs']}
    assert [len(s) for s in (tw, ts, ew, es)] == [64, 64, 32, 128]
    assert len(tw | ts | ew | es) == 288
    ids = [m['match_index'] for m in training]
    ids += [p[f'{a}_match_index'] for b in evaluation for p in b['pairs'] for a in ('candidate', 'baseline')]
    assert ids == list(range(320))
    assert [m['learner_seat'] for m in training] == list(range(4)) * 16
    assert all([p['learner_seat'] for p in b['pairs']] == list(range(4)) for b in evaluation)
    assert config['limit_seconds'] == 600
    assert config['learning_rate'] == 1e-4
    assert config['constraints']['base_frozen']
    assert not config['constraints']['opponent_actions_eligible']
    assert len(identity) == 64


@pytest.mark.parametrize('key,value', [('limit_seconds',601),('device','mps'),
    ('learning_rate',.001),('maximum_matches',321),('replay_atol',1e-6),('initialization_seed',1),
    ('preparation_limit_seconds',91),('evaluation_blocks',33)])
def test_fixed_protocol_mutation_rejected(tmp_path, key, value):
    config = json.loads((ROOT / 'configs/stage3c.json').read_text())
    config[key] = value
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        controller.settings(path)


def _training_complete(root):
    (root / 'prepare-attempt.json').write_text(json.dumps({'status':'completed'}))
    (root / 'plan.json').write_text('{}')
    for index in range(4):
        (root / f'attempt-{index}.json').write_text(json.dumps({'status':'completed'}))
        (root / f'batch-{index}.json').write_text(json.dumps({'complete':True}))


def test_failed_eval_blocks_never_replaced(tmp_path):
    _training_complete(tmp_path)
    controller.check_evaluation_chain(tmp_path, 0)
    (tmp_path / 'eval-attempt-0.json').write_text(json.dumps({'status':'failed'}))
    (tmp_path / 'eval-0.json').write_text(json.dumps({'complete':False}))
    for invalid in (-1, 0, 1, 32):
        with pytest.raises(ValueError):
            controller.check_evaluation_chain(tmp_path, invalid)


def test_incomplete_training_never_evaluated(tmp_path):
    _training_complete(tmp_path)
    (tmp_path / 'batch-3.json').write_text(json.dumps({'complete':False}))
    with pytest.raises(ValueError):
        controller.check_evaluation_chain(tmp_path, 0)


def test_failed_training_never_replaced(tmp_path):
    (tmp_path / 'prepare-attempt.json').write_text(json.dumps({'status':'completed'}))
    (tmp_path / 'plan.json').write_text('{}')
    controller.check_chain(tmp_path, 0)
    (tmp_path / 'attempt-0.json').write_text(json.dumps({'status':'failed'}))
    (tmp_path / 'batch-0.json').write_text(json.dumps({'complete':False}))
    for invalid in (0, 1, 4):
        with pytest.raises(ValueError):
            controller.check_chain(tmp_path, invalid)


def test_preflight_rejects_missing_postcall_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'implementation', lambda: {'source':'fixed'})
    (tmp_path / 'checks-final-preflight.json').write_text(json.dumps({'status':'completed','implementation':{'source':'fixed'}}))
    (tmp_path / 'final-preflight.xml').write_text('<testsuites><testsuite>' + '<testcase classname="tests.test_stage3c"/>'*43 + '</testsuite></testsuites>')
    with pytest.raises(ValueError, match='Missing required'):
        controller.verify_preflight(tmp_path)


def test_preflight_rejects_changed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, 'implementation', lambda: {'source':'new'})
    (tmp_path / 'checks-final-preflight.json').write_text(json.dumps({'status':'completed','implementation':{'source':'old'}}))
    with pytest.raises(ValueError, match='current-implementation'):
        controller.verify_preflight(tmp_path)
