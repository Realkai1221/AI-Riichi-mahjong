"""Independent evidence reconstruction detects tampering and denominator errors."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

_spec=importlib.util.spec_from_file_location('stage3h_audit_for_test',Path(__file__).resolve().parents[1]/'scripts/audit_stage3h.py')
audit=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def plan_fixture():
    corpus={'splits':{split:[split+str(i) for i in range(160)] for split in ('training','development')}}
    selected={split:audit.ordered(names,split)[:128] for split,names in corpus['splits'].items()}
    models=[dict(id='reference',kind='reference',arm=None,update=0),dict(id='initial',kind='value',arm='mc_a0',update=0)]
    arms=('mc_a0','mc_a1','delta_a0','delta_a1')
    models.extend(dict(id=f'{arm}_u{update}',kind='value',arm=arm,update=update) for arm in arms for update in (100,200,300,400))
    conditions=[dict(id='initial_mc',checkpoint_id='initial',target='terminal_utility',trained_alpha=None),
                dict(id='initial_delta',checkpoint_id='initial',target='one_hand_delta',trained_alpha=None)]
    conditions.extend(dict(id=arm,checkpoint_id=arm+'_u400',target='terminal_utility' if arm.startswith('mc') else 'one_hand_delta',trained_alpha=int(arm[-1])) for arm in arms)
    return dict(schema='stage3h-diagnostic-plan-v1',execution_admitted=True,audit_protocol=deepcopy(audit.AUDIT_PROTOCOL),
        optimizer_steps=0,new_matches=0,policy_promotion=False,selected_matches=selected,
        gradient_groups=[selected['training'][i:i+16] for i in range(0,64,16)],models=models,
        reference_model_id='reference',initial_model_id='initial',gradient_conditions=conditions),corpus


def row_fixture():
    # Rows 1 (frozen opponent) and 2 (forced explorer action) are deliberately excluded.
    kinds=np.asarray([audit.TOKENS[n] for n in ('PASS','RON','DISCARD','DISCARD','DISCARD','CHI')],dtype=np.int8)
    actor={'candidate_offsets':np.asarray([0,2,3,4,6]),'action_tokens':kinds[:,None]}
    labels={'behavior_id':np.asarray([3,0,3,3]),'seat':np.asarray([2,0,2,2]),'step':np.asarray([1,2,3,4]),
            'hand_id':np.asarray([0,0,0,1]),'chosen_index':np.asarray([0,0,0,1]),
            'final_utility':np.asarray([.5,-1.5,.5,.5],dtype=np.float32),
            'behavior_probabilities':np.asarray([.8,.2,1.,1.,.95,.05])}
    numeric={'row_indices':np.asarray([0,3]),'seat':np.asarray([2,2]),'step':np.asarray([1,4]),
             'hand_id':np.asarray([0,1]),'logged_index':np.asarray([0,1]),
             'logged_kind':kinds[[0,5]],'candidate_offsets':np.asarray([0,2,4]),'candidate_kinds':kinds[[0,1,4,5]],
             'behavior_probabilities':labels['behavior_probabilities'][[0,1,4,5]],
             'terminal_targets':np.asarray([.5,.5],dtype=np.float32),'delta_targets':np.asarray([.25,-.25],dtype=np.float32),
             'opportunity_mask':np.asarray([audit.opportunity(kinds[:2]),audit.opportunity(kinds[4:])],dtype=np.uint8),
             'legal_scores':np.tile(np.asarray([2.,1.,0.,3.],dtype=np.float32),(18,1))}
    raw=np.tile(np.asarray([0,1]),(18,1));wrapped=np.tile(np.asarray([1,1]),(18,1));deployed=wrapped.copy();deployed[0]=raw[0]
    for key,indices in (('raw',raw),('wrapped',wrapped),('deployed',deployed)):
        numeric[key+'_index']=indices.copy()
        flat_indices=indices+np.asarray([0,2])
        numeric[key+'_kind']=numeric['candidate_kinds'][flat_indices]
        numeric[key+'_probability']=numeric['behavior_probabilities'][flat_indices]
    report=dict(match_id='match',rows=2,model_rows=36,candidate_values_per_model=4)
    mapping={'matches':{'match':{'explorer_seat':2,'hand_deltas':{'0':.25,'1':-.25}}}}
    return numeric,report,actor,labels,mapping


def test_selection_is_outcome_independent_and_fixed_prefix():
    plan,corpus=plan_fixture();audit.validate_selection(plan,corpus)
    corpus['irrelevant_outcomes']={'training0':1e99}
    audit.validate_selection(plan,corpus)
    for count in (64,96):
        reduced=deepcopy(plan)
        reduced['selected_matches']={s:n[:count] for s,n in plan['selected_matches'].items()}
        audit.validate_selection(reduced,corpus)
    plan['selected_matches']['training'][:2]=reversed(plan['selected_matches']['training'][:2])
    with pytest.raises(ValueError,match='selection'):audit.validate_selection(plan,corpus)


@pytest.mark.parametrize('field',['audit_protocol','gradient_groups','gradient_conditions','models'])
def test_fixed_protocol_changes_rejected(field):
    plan,corpus=plan_fixture()
    if field=='audit_protocol':plan[field]['eligible_rows_per_match']=3
    elif field=='gradient_groups':plan[field][1][0]=plan[field][0][0]
    elif field=='gradient_conditions':plan[field][0]['target']='one_hand_delta'
    else:plan[field][-1]['update']=300
    with pytest.raises(ValueError):audit.validate_selection(plan,corpus)


def test_original_rows_choices_labels_and_first_win_are_reconstructed():
    numeric,report,actor,labels,mapping=row_fixture()
    np.testing.assert_array_equal(audit.validate_rows({},report,numeric,actor,labels,mapping),[0,3])
    assert numeric['raw_index'][1,0]==0 and numeric['deployed_index'][1,0]==1
    assert numeric['deployed_index'][0,0]==0  # reference has its original deployment rule


@pytest.mark.parametrize('mutation',['membership','offset','propensity','kind','terminal','delta','choice','wrapper','mask','score_bound'])
def test_saved_row_tampering_is_rejected(mutation):
    a,r,x,y,m=row_fixture()
    if mutation=='membership':a['row_indices'][0]=2
    elif mutation=='offset':a['candidate_offsets'][1]=1
    elif mutation=='propensity':a['behavior_probabilities'][0]=.79
    elif mutation=='kind':a['candidate_kinds'][0]=audit.TOKENS['DISCARD']
    elif mutation=='terminal':a['terminal_targets'][0]=1.5
    elif mutation=='delta':a['delta_targets'][0]=1.
    elif mutation=='choice':a['raw_index'][1,1]=0
    elif mutation=='wrapper':a['deployed_index'][1,0]=0
    elif mutation=='mask':a['opportunity_mask'][0]=1
    else:a['legal_scores'][1,0]=101.
    with pytest.raises(ValueError):audit.validate_rows({},r,a,x,y,m)


def test_value_metrics_use_logged_score_and_do_not_treat_reference_logits_as_rewards():
    a,*_=row_fixture()
    value=audit.metrics_for(a,1,dict(kind='value',update=400,arm='mc_a1'))
    expected_ce=np.asarray([math.log1p(math.exp(-1)),math.log1p(math.exp(-3))])
    np.testing.assert_allclose(value['chosen_ce'],expected_ce)
    np.testing.assert_array_equal(value['mc_chosen_mse'],[2.25,6.25])
    np.testing.assert_allclose(value['declared_objective'],np.asarray([2.25,6.25])+expected_ce)
    np.testing.assert_array_equal(value['value_top_two_margin'],[1,3])
    np.testing.assert_array_equal(value['win_wrapper_changes_raw'],[1,0])
    reference=audit.metrics_for(a,0,dict(kind='reference',update=0,arm=None))
    assert 'logit_maximum' in reference and 'chosen_ce' not in reference
    assert not any('mse' in key for key in reference)


@pytest.mark.parametrize('p,bucket',[(0,0),(.01,1),(.05,2),(.2,3),(1.,4)])
def test_probability_buckets_include_declared_boundaries(p,bucket):
    a,*_=row_fixture()
    a['deployed_probability'][1]=p
    values=audit.metrics_for(a,1,dict(kind='value',update=0,arm='mc_a0'))
    for index,name in enumerate(audit.BUCKETS):
        np.testing.assert_array_equal(values['deployed_probability_bucket_'+name],np.full(2,float(index==bucket)))


def test_whole_match_and_row_denominators_are_distinct_and_empty_slices_preserved():
    a,*_=row_fixture();models=[dict(id='reference',kind='reference',update=0,arm=None)]
    totals={};audit.accumulate(totals,'training',a,models)
    # A second match contributes one row to 'all', retaining other arrays for metrics.
    b=deepcopy(a);b['opportunity_mask'][1]=0;b['raw_probability'][0,0]=0.
    audit.accumulate(totals,'training',b,models)
    all_rows=totals[('training','reference','all')]
    assert all_rows['rows']==3 and all_rows['matches']==2
    assert all_rows['row']['raw_probability']/3==pytest.approx(.85/3)
    assert all_rows['match']['raw_probability']/2==pytest.approx(.425/2)
    empty=totals[('training','reference','kan')]
    assert empty['rows']==empty['matches']==0 and not empty['row']
    transition=all_rows['transitions']/3
    assert transition.sum()==pytest.approx(1.)


def test_zero_orthogonal_opposed_gradient_geometry_and_clip_factor():
    result=audit.pair(np.asarray([3.,0.]),np.asarray([0.,4.]))
    assert result['mse_gradient_l2']==3. and result['ce_gradient_l2']==4.
    assert result['mse_ce_cosine']==0. and result['combined']['1']['gradient_l2']==5.
    assert result['combined']['1']['clip_factor']==pytest.approx(1/(5+1e-6))
    assert audit.pair(np.zeros(2),np.ones(2))['mse_ce_cosine'] is None
    values=audit.cancellation(np.asarray([[1.,0.],[-1.,0.],[1.,0.],[-1.,0.]]))
    assert values['mean_gradient_l2']==0 and values['mean_norm_over_mean_group_norm']==0
    assert values['group_to_mean_cosines']==[None]*4 and values['opposing_groups']==0
    assert values['pairwise_group_cosines'][0]['cosine']==-1.


def test_block_layout_is_rebuilt_from_actual_trainable_parameter_order():
    model=torch.nn.Module()
    model.backbone=torch.nn.Linear(2,1,bias=False)
    model.action_score=torch.nn.Linear(1,1,bias=False)
    model.efficiency_encoder=torch.nn.Linear(1,1,bias=False)
    model.rank_head=torch.nn.Linear(1,4)
    for p in model.rank_head.parameters():p.requires_grad_(False)
    layout={'trainable_parameters':4,'frozen_rank_parameters':8,'ordered_parameters':[
        dict(name='backbone.weight',shape=[1,2],start=0,stop=2,block='backbone'),
        dict(name='action_score.weight',shape=[1,1],start=2,stop=3,block='scorer'),
        dict(name='efficiency_encoder.weight',shape=[1,1],start=3,stop=4,block='efficiency')],
        'blocks':{'backbone':[dict(name='backbone.weight',shape=[1,2],parameters=2)],
                  'scorer':[dict(name='action_score.weight',shape=[1,1],parameters=1)],
                  'efficiency':[dict(name='efficiency_encoder.weight',shape=[1,1],parameters=1)]}}
    indices=audit.validate_layout(layout,model)
    np.testing.assert_array_equal(indices['backbone'],[0,1])
    layout['ordered_parameters'][0]['stop']=3
    with pytest.raises(ValueError,match='layout'):audit.validate_layout(layout,model)


def test_close_rejects_missing_fields_nonfinite_and_changed_counts():
    for value,expected in (({'a':1},{'a':1,'b':0}), (math.nan,1.),(math.inf,1.),(2,1)):
        with pytest.raises(ValueError):audit.close(value,expected,'fixture')
