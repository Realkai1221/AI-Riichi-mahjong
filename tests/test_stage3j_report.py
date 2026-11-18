"""Analysis uses shared blocks, never seats or training seeds as extra N."""
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('stage3j_report',SCRIPTS/'stage3j_report.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def panel():
    seeds=[20261024011,20261024012,20261024013]
    names=['reference_wrapped','hash_tie',*[f'tie_seed_{s}' for s in seeds],*[f'small_seed_{s}' for s in seeds]]
    jobs=[];records=[]
    for block in range(2):
        for focal in range(4):
            for arm in names:
                job=dict(match_id=f'{block}-{focal}-{arm}',block=block,focal_seat=focal,arm=arm)
                rank=(1 if block==0 else 3) if arm.startswith(('tie','small')) else 2
                order=[s for s in range(4) if s!=focal];ranks=[0]*4;ranks[focal]=rank
                for seat,r in zip(order,[r for r in range(1,5) if r!=rank]):ranks[seat]=r
                jobs.append(job);records.append(dict(**job,completed=True,failure=None,training_eligible=False,
                    ppo_eligible=False,ranks=ranks,utilities=[2.5-r for r in ranks]))
    return dict(blocks=2,fixed_seed_family=seeds,arms=[dict(name=a) for a in names],matches=jobs),records

def test_correlated_seats_and_seeds_remain_two_blocks():
    execution,records=panel();result=m.analyze(execution,records,bootstrap_replicates=100)
    assert result['independent_blocks']==2 and result['games']==64
    first=result['primary']['learned_tie_vs_reference']
    assert first['block_utility_differences']==[1,-1]
    assert first['mean_utility_difference']==0 and first['standard_error']==pytest.approx(1)
    assert result['primary']['small_vs_learned_tie']['bootstrap_9875_interval']==[0,0]
    assert all(v['games']==8 for v in result['by_arm'].values())

@pytest.mark.parametrize('change',['duplicate','missing','wrong_utility','wrong_role'])
def test_invalid_observations_cannot_enter_summary(change):
    execution,records=panel()
    if change=='duplicate':records[-1]=records[0]
    elif change=='missing':records.pop()
    elif change=='wrong_utility':records[0]['utilities'][0]=0
    else:records[0]['training_eligible']=True
    with pytest.raises(ValueError):m.analyze(execution,records,bootstrap_replicates=100)
