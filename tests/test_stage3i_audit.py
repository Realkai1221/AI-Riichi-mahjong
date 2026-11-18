"""Independent audit rejects altered eligibility, identity and saved arithmetic."""
import importlib.util
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('stage3i_audit',ROOT/'scripts/stage3i_audit.py')
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def example():
    from riichi_ai.semantics import ACTION_KINDS
    kinds={name:i+1 for i,name in enumerate(ACTION_KINDS)}
    cache=dict(target=np.array([.1,.2],np.float32),candidate_offsets=np.array([0,3,5]),
               row_indices=np.array([4,9]),chosen_index=np.array([0,1]),
               logits=np.array([2,1,-20,0,5],np.float32),
               explorer_probabilities=np.array([.6,.39,.01,.4,.6]),
               kinds=np.array([kinds['DISCARD']]*3+[kinds['RON'],kinds['PASS']]),hand_ids=np.array([1,9]))
    choices=np.zeros((len(m.GRID),2),np.int16)
    for index,strength in enumerate(m.GRID):
        if strength>=128:choices[index,0]=1
    pred=dict(value=np.zeros(2,np.float32),state=np.zeros(2,np.float32),
              action=np.array([0,.01,100,-10,10],np.float32),choices_by_lambda=choices,
              raw_reference=np.array([0,1]),wrapped_reference=np.array([0,0]),row_indices=np.array([4,9]))
    return cache,pred


def test_independent_prediction_arithmetic_checks_wrapper_and_support():
    cache,pred=example()
    result=m.predictions(cache,pred)
    assert result['rows']==2
    assert result['grid'][0]['changed']==0
    assert result['grid'][-1]['changed']==1
    assert result['grid'][-1]['discard_changed']==1
    assert result['grid'][-1]['slices']['win']==dict(rows=1,changed=0)
    assert result['errors']['va']>result['errors']['v']


@pytest.mark.parametrize('mutation',['unsupported','win','row','raw_reference','nan','shape','distribution'])
def test_independent_prediction_audit_rejects_mutation(mutation):
    cache,pred=example()
    if mutation=='unsupported':pred['choices_by_lambda'][-1,0]=2
    elif mutation=='win':pred['choices_by_lambda'][0,1]=1
    elif mutation=='row':pred['row_indices'][0]=5
    elif mutation=='raw_reference':pred['raw_reference'][0]=1
    elif mutation=='nan':pred['value'][0]=np.nan
    elif mutation=='shape':pred['action']=pred['action'][:-1]
    else:cache['explorer_probabilities'][0]=.8
    with pytest.raises(ValueError):m.predictions(cache,pred)


def calibration_grid():
    return [dict(**{'lambda':strength},rows=100,changed=0,reference_discard_rows=50,
                 reference_discard_changed=0,change_rate=0.,reference_discard_change_rate=0.) for strength in m.GRID]


def test_independent_lambda_selection_tie_and_fallback():
    rows=calibration_grid()
    assert m.select_lambda(rows)==0
    for i in (1,2):
        rows[i].update(changed=3,change_rate=.03,reference_discard_changed=5,reference_discard_change_rate=.1)
    assert m.select_lambda(rows)==1
    rows[1].update(reference_discard_changed=6,reference_discard_change_rate=.12)
    assert m.select_lambda(rows)==2


def test_independent_lambda_selection_rejects_rounded_rate_claim():
    rows=calibration_grid();rows[1].update(changed=4,change_rate=.03)
    with pytest.raises(ValueError):m.select_lambda(rows)


def test_replay_sample_independent_of_record_order_and_outcomes():
    jobs=[dict(match_id=f'block-{i}',scores=[0,1,2,3]) for i in range(100)]
    left=m.replay_selection(dict(matches=jobs))
    changed=deepcopy(jobs[::-1])
    for job in changed:job['scores']=[100,0,0,0]
    right=m.replay_selection(dict(matches=changed))
    assert len(left)==32
    assert [job['match_id'] for job in left]==[job['match_id'] for job in right]


def test_statistics_keeps_blocks_as_units_and_zero_variance():
    result=m.statistics([0,0,0,0],20261024031)
    assert result['independent_blocks']==4
    assert result['mean_utility_difference']==0
    assert result['standard_error']==0
    assert result['bootstrap_percentile_95_exploratory']==[0,0]
    assert result['fixed_budget_hoeffding_95'][0]<0<result['fixed_budget_hoeffding_95'][1]


def test_statistics_matches_historical_fixed_bootstrap_random_stream():
    from riichi_ai.evaluation import summarize_blocks
    values=[-.5,1.5,0,-1,.25,.5]
    actual=m.statistics(values,20261024031)
    expected=summarize_blocks(values,bootstrap_seed=20261024031)
    for key,value in actual.items():
        assert np.allclose(value,expected[key],atol=1e-12,rtol=1e-12)


def decision_row(policy='seed_11'):
    keys=['discard-a','discard-b','discard-c']
    return dict(row_index=0,seat=0,semantic_keys=keys,action_kinds=['DISCARD']*3,
                representative_tokens=['a','b','c'],selected_index=1,selected_semantic_key='discard-b',
                selected_kind='DISCARD',selected_physical_token='b',probability=1.,training_eligible=False,
                ppo_eligible=False,legal_scores=[2.,1.,-20.],
                correction=dict(explorer_probabilities=[.6,.39,.01],action_residuals=[0.,.01,100.],
                                strength=128,raw_reference_index=0,wrapped_reference_index=0,
                                raw_corrected_index=1,eligible=[True,True,False],prior=[0.,-1.,-8.]))


def test_full_game_independent_candidate_choice():
    m.audit_decision(decision_row(),'seed_11')


@pytest.mark.parametrize('field',['token','semantic','selected','support','prior','strength','nonfinite'])
def test_full_game_decision_corruption_is_rejected(field):
    row=decision_row()
    if field=='token':row['selected_physical_token']='x'
    elif field=='semantic':row['selected_semantic_key']='x'
    elif field=='selected':row['selected_index']=2
    elif field=='support':row['correction']['eligible'][2]=True
    elif field=='prior':row['correction']['prior'][2]=-20.
    elif field=='strength':row['correction']['strength']=2
    else:row['legal_scores'][0]=float('nan')
    with pytest.raises(ValueError):m.audit_decision(row,'seed_11')


def response_fixture():
    policies=[dict(policy_id=f'policy-{seat}') for seat in range(4)]
    request=dict(step=0,player_views={'0':'hash-zero','1':'hash-one'},rows=[0,1],submitted=True)
    import hashlib
    identity=hashlib.sha256(m.encoded(dict(step=0,player_views=request['player_views']))).hexdigest()
    request['pre_response_id']=identity
    rows=[dict(row_index=seat,step=0,seat=seat,pre_response_id=identity,
               player_view_sha256=request['player_views'][str(seat)],policy_id=policies[seat]['policy_id']) for seat in (0,1)]
    return rows,[request],policies


def test_all_simultaneous_responses_share_committed_state():
    m.audit_response_links(*response_fixture())


@pytest.mark.parametrize('field',['state','membership','seat','policy','step'])
def test_simultaneous_response_corruption_rejected(field):
    rows,requests,policies=response_fixture()
    if field=='state':rows[1]['pre_response_id']='changed-after-first-response'
    elif field=='membership':requests[0]['rows']=[0]
    elif field=='seat':rows[1]['seat']=0
    elif field=='policy':rows[1]['policy_id']='different-policy'
    else:rows[1]['step']=1
    with pytest.raises(ValueError):m.audit_response_links(rows,requests,policies)
