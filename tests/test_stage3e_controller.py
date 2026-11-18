"""Fixed horizon, no retry, seed separation and protected endpoint capacity."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('horizon_controller',ROOT/'scripts/stage3e.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_exact_horizon_and_fresh_nonoverlapping_seeds():
    _,config,_=c.settings(ROOT/'configs/stage3e.json')
    assert c.verify_fresh_seeds(config)==dict(fresh_seed_count=1056,stage3c_stage3d_disjoint=True)
    assert [b['batch_index'] for b in config['batches']]==list(range(4,32))
    assert [m['match_index'] for b in config['batches'] for m in b['matches']]==list(range(576,1024))
    assert [p[f'{a}_match_index'] for b in config['evaluation'] for p in b['pairs'] for a in ('candidate','baseline')]==list(range(1024,1280))
    assert config['initial_boundary']['updates']==4 and config['maximum_updates']==32


def test_protected_evaluation_audit_and_shutdown_capacity():
    config=c.fixed_settings()
    assert c.operation_reserve(config,'train')==500
    assert c.operation_reserve(config,'evaluate')==100
    assert c.operation_reserve(config,'budgeted')==60
    assert config['per_batch_allowance_seconds']==45


@pytest.mark.parametrize('key,value',[('limit_seconds',1201),('learning_rate',.001),('maximum_updates',33),('training_matches',464),('protected_evaluation_seconds',399),('protected_audit_seconds',39),('reserve_seconds',59),('device','mps')])
def test_recipe_or_budget_mutation_is_rejected(tmp_path,key,value):
    config=json.loads((ROOT/'configs/stage3e.json').read_text());config[key]=value
    path=tmp_path/'mutated.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError):c.settings(path)


def test_training_absolute_sequence_and_failure_are_not_replaceable(tmp_path):
    (tmp_path/'prepare-attempt.json').write_text(json.dumps({'status':'completed'}))
    c.check_chain(tmp_path,'train',4)
    with pytest.raises(ValueError):c.check_chain(tmp_path,'train',5)
    (tmp_path/'attempt-4.json').write_text(json.dumps({'status':'failed'}))
    (tmp_path/'batch-4.json').write_text(json.dumps({'complete':False}))
    for phase,index in [('train',4),('train',5),('evaluate',0)]:
        with pytest.raises(ValueError):c.check_chain(tmp_path,phase,index)


def test_evaluation_requires_entire_final_horizon_and_no_retries(tmp_path):
    (tmp_path/'prepare-attempt.json').write_text(json.dumps({'status':'completed'}))
    for i in range(4,32):
        (tmp_path/f'attempt-{i}.json').write_text(json.dumps({'status':'completed'}))
        (tmp_path/f'batch-{i}.json').write_text(json.dumps({'complete':True}))
    c.check_chain(tmp_path,'evaluate',0)
    (tmp_path/'eval-attempt-0.json').write_text(json.dumps({'status':'failed'}))
    (tmp_path/'eval-0.json').write_text(json.dumps({'complete':False}))
    for i in (0,1,32):
        with pytest.raises(ValueError):c.check_chain(tmp_path,'evaluate',i)


def test_runtime_membership_is_explicit_and_unrelated_additions_ignored(tmp_path):
    assert len(c.SOURCE_PATHS)==len(set(c.SOURCE_PATHS))
    assert tuple(sorted(c.SOURCE_PATHS))==c.SOURCE_PATHS
    (tmp_path/'runtime.py').write_text('original')
    pins=c.hash_sources(tmp_path,['runtime.py'])
    (tmp_path/'book_calls.py').write_text('independent')
    c.verify_source_identity(pins,tmp_path)
    (tmp_path/'runtime.py').write_text('modified')
    with pytest.raises(RuntimeError):c.verify_source_identity(pins,tmp_path)
