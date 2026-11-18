"""Fixed gates, no seed selection, exact bounded corrections and wall pairing."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from riichi_ai import stage3i_evaluation as m
from riichi_ai.stage3i_model import correction_choices


def fixture(tmp_path, *, blocks=2, failed=(), zero=()):
    seeds = [11,12,13]
    candidates = []
    for seed in seeds:
        path = tmp_path/f'{seed}.pt'
        path.write_bytes(str(seed).encode())
        candidates.append(dict(seed=seed,checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               gate=dict(passed=seed not in failed,reasons=['test_gate'] if seed in failed else []),
                               calibration=dict(lambda_=0 if seed in zero else 1)))
    reset = tmp_path/'reset.pt'
    reset.write_bytes(b'reset')
    plan = dict(training_seeds=seeds,evaluation_seeds=list(range(blocks)),preflight_evaluation_seeds=[100,101],
                reset_checkpoint=dict(path=str(reset),sha256=hashlib.sha256(b'reset').hexdigest()))
    calibration = dict(schema=m.CALIBRATION_SCHEMA,complete=True,plan_sha256='a'*64,seeds=candidates)
    config = dict(reference_checkpoint='/reference.pt',reference_sha256='b'*64,
                  call_checkpoint='/call.pt',call_sha256='c'*64,workers=14,evaluation_blocks=blocks,
                  preflight_evaluation_blocks=2,reset_blocks=64,include_reset_controls=True)
    return plan,calibration,config


def prepare(tmp_path,**kwargs):
    plan,calibration,config = fixture(tmp_path,**kwargs)
    return m.prepare_execution(plan,calibration,config,plan_sha256='a'*64,calibration_sha256='d'*64)


def grid(rates=None):
    rates = rates or {}
    return [dict(**{'lambda':strength},rows=100,change_rate=rates.get(strength,0),
                 reference_discard_change_rate=rates.get(strength,0)) for strength in m.LAMBDA_GRID]


def test_calibration_zero_fallback_and_ties():
    assert m.calibration_choice(grid())['lambda_']==0
    assert m.calibration_choice(grid({1:.02,2:.02,4:.03,8:.04}))['lambda_']==4
    assert m.calibration_choice(grid({1:.02,2:.02}))['lambda_']==1


def test_calibration_discard_ceiling_and_boundaries():
    rows = grid({1:.03,2:.02})
    rows[1]['reference_discard_change_rate'] = .1000000001
    assert m.calibration_choice(rows)['lambda_']==2
    rows[1]['reference_discard_change_rate'] = .1
    assert m.calibration_choice(rows)['lambda_']==1


@pytest.mark.parametrize('change',['missing_lambda','nonidentity_zero','different_rows','nonfinite','empty_rows'])
def test_invalid_calibration_rejected(change):
    rows = grid()
    if change=='missing_lambda': rows.pop()
    elif change=='nonidentity_zero': rows[0]['change_rate']=.001
    elif change=='different_rows': rows[1]['rows']=99
    elif change=='nonfinite': rows[1]['change_rate']=float('nan')
    else: rows[0]['rows']=0
    with pytest.raises(ValueError): m.calibration_choice(rows)


def test_gate_is_decision_weighted_strict_and_has_state_control():
    rows = [dict(match_id='a',rows=100,v_squared_error_sum=200.,va_squared_error_sum=100.,vb_squared_error_sum=150.),
            dict(match_id='b',rows=1,v_squared_error_sum=0.,va_squared_error_sum=2.,vb_squared_error_sum=1.)]
    result = m.predictive_gate(rows)
    assert result['passed']
    assert result['row_mse']['va']==102/101
    assert result['match_mse']['va']==1.5
    rows[0]['vb_squared_error_sum']=101.
    assert not m.predictive_gate(rows)['passed']


def test_fixed_panel_full_ambitious_count(tmp_path):
    execution = prepare(tmp_path,blocks=470)
    assert execution['planned_matches']==9912
    assert len([j for j in execution['matches'] if j['arm']=='reference_wrapped'])==1880
    assert len([j for j in execution['matches'] if j['arm']=='reset_wrapped'])==256
    assert len(execution['arms'])==3
    for block in range(470):
        jobs = [j for j in execution['matches'] if j['block']==block]
        assert {job['seed'] for job in jobs}=={block}
    for job in execution['matches']:
        focal=job['focal_seat']
        assert [job['lineup'][(focal+k)%4] for k in (1,2,3)]==list(m.OPPONENTS)


def test_all_failed_or_zero_candidates_preserve_controls_reset(tmp_path):
    execution = prepare(tmp_path,blocks=470,failed=(11,12),zero=(13,))
    assert execution['planned_matches']==4272
    assert execution['arms']==[]
    assert [item['seed'] for item in execution['skipped_seeds']]==[11,12,13]
    assert {job['arm'] for job in execution['matches']}==set(m.CONTROLS+m.RESETS)


@pytest.mark.parametrize('change',['seed_order','duplicate_walls','hash','benchmark','plan_hash'])
def test_invalid_execution_identity_rejected(tmp_path,change):
    plan,calibration,config=fixture(tmp_path)
    if change=='seed_order': calibration['seeds'].reverse()
    elif change=='duplicate_walls': plan['evaluation_seeds']=[0,0]
    elif change=='hash': calibration['seeds'][0]['sha256']='0'*64
    elif change=='benchmark': calibration['benchmark_only']=True
    else: calibration['plan_sha256']='0'*64
    with pytest.raises(ValueError):
        m.prepare_execution(plan,calibration,config,plan_sha256='a'*64,calibration_sha256='d'*64)


def test_grid_choices_equal_shared_deployed_formula():
    arrays = dict(target=np.array([.1,.2,.3]),candidate_offsets=np.array([0,3,5,7]),
                  logits=np.array([1.,1.,-20.,4.,3.,2.,0.],np.float32),
                  explorer_probabilities=np.array([.49,.50,.01,.99,.01,.4,.6]),
                  kinds=np.array([m.KINDS['DISCARD']]*3+[m.KINDS['PON'],m.KINDS['PASS'],m.KINDS['PASS'],m.KINDS['RON']]))
    residuals=np.array([0.,.001,100.,0.,100.,100.,-100.],np.float32)
    choices,raw,wrapped=m._all_grid_choices(arrays,residuals)
    for row in range(3):
        start,end=arrays['candidate_offsets'][row:row+2]
        for i,strength in enumerate(m.LAMBDA_GRID):
            actual=correction_choices(arrays['logits'][start:end],residuals[start:end],
                                      arrays['explorer_probabilities'][start:end],arrays['kinds'][start:end],strength)
            assert choices[i,row]==actual['selected_index']
            assert raw[row]==actual['raw_reference_index']
            assert wrapped[row]==actual['wrapped_reference_index']
    assert np.all(choices[:,2]==1)
    assert np.all(choices[:,1]==0)


def records(execution):
    return [dict(**job,completed=True,failure=None,training_eligible=False,ppo_eligible=False,
                 terminal_result_verified=True,utilities=[1.5,.5,-.5,-1.5],
                 ordinal_wall_fingerprints=['first','second']) for job in execution['matches']]


def cheap_statistics(monkeypatch):
    monkeypatch.setattr(m,'summarize_blocks',lambda values,**kwargs:dict(
        mean_utility_difference=float(np.mean(values)),independent_blocks=len(values)))


def test_summary_never_selects_winner_and_pools_inside_block(tmp_path,monkeypatch):
    cheap_statistics(monkeypatch)
    execution=prepare(tmp_path)
    result=m.summarize(execution,records(execution))
    assert result['selected_candidate'] is None
    assert result['seed_selection'] is False
    assert result['pooled_eligible_seeds']['seeds']==[11,12,13]
    assert result['pooled_eligible_seeds']['block_utility_differences']==[0,0]
    assert result['contrasts']['seed_11_vs_reference_wrapped']['statistics']['family_size_preregistered']==3


@pytest.mark.parametrize('change',['wall','missing','duplicate','failed','eligibility','utility'])
def test_summary_rejects_unpaired_incomplete_or_mislabelled_games(tmp_path,monkeypatch,change):
    cheap_statistics(monkeypatch)
    execution=prepare(tmp_path)
    saved=records(execution)
    if change=='wall': saved[1]['ordinal_wall_fingerprints'][0]='different'
    elif change=='missing': saved.pop()
    elif change=='duplicate': saved.append(deepcopy(saved[0]))
    elif change=='failed': saved[0]['completed']=False
    elif change=='eligibility': saved[0]['training_eligible']=True
    else: saved[0]['utilities']=[0,0,0,0]
    with pytest.raises(ValueError): m.summarize(execution,saved)
