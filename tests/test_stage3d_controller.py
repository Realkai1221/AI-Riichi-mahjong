"""Recovery workflow contracts: fixed membership, fresh plans, durable attempts."""
import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('recovery_controller',ROOT/'scripts/stage3d.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_explicit_sources_ignore_unrelated_addition(tmp_path):
    (tmp_path/'runtime.py').write_text('original')
    pins=c.hash_sources(tmp_path,('runtime.py',))
    (tmp_path/'new_analysis.py').write_text('independent')
    c.verify_source_identity(pins,tmp_path)
    assert tuple(pins)==('runtime.py',)


def test_explicit_sources_reject_modified_or_missing(tmp_path):
    path=tmp_path/'runtime.py';path.write_text('original')
    pins=c.hash_sources(tmp_path,('runtime.py',))
    path.write_text('changed')
    with pytest.raises(RuntimeError):c.verify_source_identity(pins,tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError):c.verify_source_identity(pins,tmp_path)


@pytest.mark.parametrize('members',[(),('x','x'),('../escape',),('/absolute',)])
def test_invalid_source_membership_rejected(tmp_path,members):
    with pytest.raises((ValueError,FileNotFoundError)):c.hash_sources(tmp_path,members)


def test_fresh_seed_plan_and_fixed_endpoints():
    _,config,_=c.settings(ROOT/'configs/stage3d.json')
    old=json.loads((ROOT/'configs/stage3c.json').read_text())
    old_seeds={m[k] for b in old['batches'] for m in b['matches'] for k in ('seed','sampling_seed')}
    old_seeds|={b['wall_seed'] for b in old['evaluation']}
    old_seeds|={p['sampling_seed'] for b in old['evaluation'] for p in b['pairs']}
    fresh=[b['wall_seed'] for b in config['evaluation']]+[p['sampling_seed'] for b in config['evaluation'] for p in b['pairs']]
    assert len(set(fresh))==160 and not old_seeds&set(fresh)
    ids=[p[f'{arm}_match_index'] for b in config['evaluation'] for p in b['pairs'] for arm in ('candidate','baseline')]
    assert ids==list(range(320,576))
    assert config['constraints']['optimizer_updates']==config['constraints']['checkpoint_writes']==0
    assert config['boundaries']['candidate']['updates']==4 and config['boundaries']['baseline']['updates']==0


@pytest.mark.parametrize('key,value',[('limit_seconds',401),('maximum_matches',257),('device','mps'),('preparation_limit_seconds',46),('reserve_seconds',29)])
def test_protocol_mutations_rejected(tmp_path,key,value):
    config=json.loads((ROOT/'configs/stage3d.json').read_text());config[key]=value
    path=tmp_path/'bad.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError):c.settings(path)


def test_failed_block_prevents_retry_and_continuation(tmp_path):
    (tmp_path/'prepare-attempt.json').write_text(json.dumps({'status':'completed'}))
    c.check_chain(tmp_path,0)
    (tmp_path/'eval-attempt-0.json').write_text(json.dumps({'status':'failed'}))
    (tmp_path/'eval-0.json').write_text(json.dumps({'complete':False}))
    for index in (-1,0,1,32):
        with pytest.raises(ValueError):c.check_chain(tmp_path,index)


def test_lifecycle_marker_is_durable_and_never_overwritten(tmp_path,monkeypatch):
    modes=[];real=c.os.fsync
    def sync(fd):modes.append(c.os.fstat(fd).st_mode);real(fd)
    monkeypatch.setattr(c.os,'fsync',sync)
    path=tmp_path/'lifecycle'/'start.json'
    c.write_once_json(path,{'match_index':320,'phase':'started'})
    assert json.loads(path.read_text())['match_index']==320
    assert stat.S_ISREG(modes[0]) and all(stat.S_ISDIR(mode) for mode in modes[1:])
    original=path.read_bytes()
    with pytest.raises(FileExistsError):c.write_once_json(path,{'phase':'replaced'})
    assert path.read_bytes()==original
