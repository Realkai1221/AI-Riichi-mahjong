"""Independent saved-state metrics/gradient arithmetic and fixed CPU forwards.

Does not call the diagnostic workers' metric or gradient aggregation functions.
No optimizer, new simulator match, or checkpoint write is performed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from riichi_ai.model import PolicyNetwork, collate_encoded, _state_hash
from riichi_ai.semantics import ACTION_KINDS
from riichi_ai.stage3g_data import LearningDataset

SLICES=('all','discard','call','riichi','kan','win','call_and_noncall')
BUCKETS=('zero','(0,.01]','(.01,.05]','(.05,.2]','(.2,1]')
TOKENS={name:i+1 for i,name in enumerate(ACTION_KINDS)}
CALLS={TOKENS[k] for k in ('CHI','PON','DAIMINKAN','ANKAN','KAKAN')}
KANS={TOKENS[k] for k in ('DAIMINKAN','ANKAN','KAKAN')}
WINS={TOKENS[k] for k in ('RON','TSUMO')}
AUDIT_PROTOCOL=dict(matches_per_split=8,eligible_rows_per_match=2,model_count=18,device='cpu',batch_rows=1,
    selection='first selected matches, then first eligible row indices',score_atol=1e-6,score_rtol=1e-6)


def require(condition,message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def artifact(path,identity,byte_count=None):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path)==identity,'Artifact hash differs: '+str(path))
    require(byte_count is None or path.stat().st_size==byte_count,'Artifact byte count differs')
    return path


def deadline(at):
    if time.monotonic()>=at: raise TimeoutError('Stage3H independent audit deadline')


def close(actual,expected,label,*,atol=2e-8,rtol=2e-6):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+': keys differ')
        for key in expected: close(actual[key],expected[key],label+'/'+key,atol=atol,rtol=rtol)
    elif isinstance(expected,(list,tuple)):
        require(isinstance(actual,(list,tuple)) and len(actual)==len(expected),label+': length differs')
        for i,(a,e) in enumerate(zip(actual,expected)): close(a,e,label+'/'+str(i),atol=atol,rtol=rtol)
    elif expected is None or isinstance(expected,(str,bool)):
        require(actual==expected,label+': value differs')
    elif isinstance(expected,(int,np.integer)):
        require(actual==int(expected),label+': count differs')
    else:
        require(isinstance(actual,(int,float)) and math.isfinite(actual)
                and math.isclose(actual,float(expected),rel_tol=rtol,abs_tol=atol),label+': numeric mismatch')


def ordered(names,split):
    return sorted(names,key=lambda name:hashlib.sha256(('riichi-ai/stage3h/v1|'+split+'|'+name).encode()).hexdigest())


def validate_selection(plan,corpus):
    require(plan['schema']=='stage3h-diagnostic-plan-v1' and plan['execution_admitted'] is True,'Formal diagnostic plan required')
    require(plan['audit_protocol']==AUDIT_PROTOCOL,'Fixed reforward audit protocol differs')
    require(plan['optimizer_steps']==plan['new_matches']==0 and plan['policy_promotion'] is False,'Diagnostic scope changed')
    names=plan['selected_matches']
    require(set(names)=={'training','development'} and len(names['training'])==len(names['development'])
            and len(names['training']) in (64,96,128),'Unexpected fixed split size')
    for split,selected in names.items():
        require(selected==ordered(corpus['splits'][split],split)[:len(selected)],'Hashed split selection changed')
    require(not set(names['training'])&set(names['development']),'Diagnostic split overlap')
    expected=names['training'][:64]
    require(plan['gradient_groups']==[expected[i:i+16] for i in range(0,64,16)],'Gradient groups differ from fixed first64')
    require(len(plan['models'])==18 and len({m['id'] for m in plan['models']})==18,'Eighteen distinct model identities required')
    records=plan['models']
    require(records[0]['id']==plan['reference_model_id']=='reference' and records[0]['kind']=='reference'
            and records[1]['id']==plan['initial_model_id']=='initial' and records[1]['update']==0
            and [(m['arm'],m['update']) for m in records[2:]]==[(a,u) for a in ('mc_a0','mc_a1','delta_a0','delta_a1') for u in (100,200,300,400)],
            'All fixed arm boundaries and shared initialization required')
    conditions=[dict(id='initial_mc',checkpoint_id='initial',target='terminal_utility',trained_alpha=None),
                dict(id='initial_delta',checkpoint_id='initial',target='one_hand_delta',trained_alpha=None)]
    conditions.extend(dict(id=a,checkpoint_id=a+'_u400',target='terminal_utility' if a.startswith('mc') else 'one_hand_delta',
                           trained_alpha=int(a[-1])) for a in ('mc_a0','mc_a1','delta_a0','delta_a1'))
    require(plan['gradient_conditions']==conditions,'Fixed six gradient conditions differ')


def load_models(plan):
    result=[]
    for entry in plan['models']:
        path=artifact(entry['path'],entry['sha256'])
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=PolicyNetwork(saved['model_config'])
        model.load_state_dict(saved['model_state_dict'],strict=True)
        weight_hash=_state_hash(model.state_dict())
        rank_hash=_state_hash(model.rank_head.state_dict())
        require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),'Nonfinite archived checkpoint')
        if entry['kind']=='value':
            metadata=saved['metadata']
            require(metadata['weights_sha256']==weight_hash and metadata['frozen_rank_sha256']==rank_hash
                and metadata['identity']['arm']==entry['arm'] and metadata['scheduled_updates']==entry['update']
                and metadata['optimizer_updates']==entry['update'] and metadata['complete'] is (entry['update']==400)
                and metadata['identity']['reference_sha256']==plan['models'][0]['sha256']
                and metadata['identity']['manifest_sha256']==plan['corpus_manifest_sha256']
                and metadata['identity']['split']=='training' and metadata['identity']['purpose']=='formal-training',
                'Archived checkpoint metadata differs')
        for name,p in model.named_parameters(): p.requires_grad_(not name.startswith('rank_head.'))
        result.append((model.eval(),dict(id=entry['id'],weights_sha256=weight_hash,rank_sha256=rank_hash)))
    return result


def opportunity(kinds):
    present=set(map(int,kinds))
    flags=[True,TOKENS['DISCARD'] in present,bool(present&CALLS),TOKENS['RIICHI'] in present,
           bool(present&KANS),bool(present&WINS),bool(present&CALLS) and bool(present-CALLS)]
    return sum(1<<i for i,flag in enumerate(flags) if flag)


def validate_rows(plan,report,arrays,actor,labels,mapping):
    eligible=np.flatnonzero((labels['behavior_id']==3)&(np.diff(actor['candidate_offsets'])>1))
    require(np.array_equal(arrays['row_indices'],eligible),'Saved optional explorer row membership differs')
    n=len(eligible); counts=np.diff(actor['candidate_offsets'])[eligible]
    offsets=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    require(np.array_equal(arrays['candidate_offsets'],offsets),'Saved candidate offsets differ')
    require(report['rows']==n and report['model_rows']==18*n and report['candidate_values_per_model']==int(offsets[-1]),'Saved row/value counters differ')
    require(arrays['legal_scores'].shape==(18,int(offsets[-1])) and np.isfinite(arrays['legal_scores']).all(),'Invalid saved legal scores')
    require(not np.any(np.abs(arrays['legal_scores'][1:])>100),'Saved values exceed declared safety bound')
    for field in ('seat','step','hand_id'):
        require(np.array_equal(arrays[field],labels[field][eligible]),'Saved row identity differs: '+field)
    require(np.array_equal(arrays['logged_index'],labels['chosen_index'][eligible]),'Logged action differs')
    require(np.array_equal(arrays['terminal_targets'],labels['final_utility'][eligible].astype(np.float32)),'Terminal label differs')
    entry=mapping['matches'][report['match_id']]
    delta=np.asarray([entry['hand_deltas'][str(int(h))] for h in labels['hand_id'][eligible]],dtype=np.float32)
    require(np.array_equal(arrays['delta_targets'],delta) and np.all(arrays['seat']==entry['explorer_seat']),'Hand target/seat differs')
    for j,row in enumerate(eligible):
        a,b=map(int,actor['candidate_offsets'][row:row+2]); left,right=map(int,offsets[j:j+2])
        kinds=actor['action_tokens'][a:b,0]
        probabilities=labels['behavior_probabilities'][a:b]
        require(np.array_equal(arrays['candidate_kinds'][left:right],kinds)
                and np.array_equal(arrays['behavior_probabilities'][left:right],probabilities),'Candidate kind/support differs')
        logged=int(labels['chosen_index'][row])
        require(arrays['logged_kind'][j]==kinds[logged] and arrays['opportunity_mask'][j]==opportunity(kinds),'Logged kind/opportunity differs')
        scores=arrays['legal_scores'][:,left:right]
        raw=np.argmax(scores,axis=1)
        wins=[k for k,v in enumerate(kinds) if int(v) in WINS]
        wrapped=np.full(18,wins[0],dtype=np.int64) if wins else raw
        deployed=wrapped.copy();deployed[0]=raw[0]
        for label,indices in (('raw',raw),('wrapped',wrapped),('deployed',deployed)):
            require(np.array_equal(arrays[label+'_index'][:,j],indices)
                    and np.array_equal(arrays[label+'_kind'][:,j],kinds[indices])
                    and np.array_equal(arrays[label+'_probability'][:,j],probabilities[indices]),'Greedy/wrapped choice or propensity differs')
    return eligible


def metrics_for(arrays,index,model):
    raw,wrapped,chosen=(arrays[k+'_index'][index] for k in ('raw','wrapped','deployed'))
    logged=arrays['logged_index']; ref=arrays['deployed_index'][0]; initial=arrays['deployed_index'][1]
    result={}
    for label,selection in (('raw',raw),('wrapped',wrapped),('deployed',chosen)):
        result[label+'_matches_reference']=(selection==ref).astype(float)
        result[label+'_matches_logged']=(selection==logged).astype(float)
        probability=arrays[label+'_probability'][index]
        result[label+'_probability']=probability
        kinds=arrays[label+'_kind'][index]
        for kind,token in TOKENS.items(): result[label+'_kind_'+kind]=(kinds==token).astype(float)
        masks=(probability==0,(probability>0)&(probability<=.01),
               (probability>.01)&(probability<=.05),(probability>.05)&(probability<=.2),(probability>.2)&(probability<=1))
        for bucket,mask in zip(BUCKETS,masks): result[label+'_probability_bucket_'+bucket]=mask.astype(float)
    result['deployed_matches_initial']=(chosen==initial).astype(float)
    result['win_wrapper_changes_raw']=(raw!=wrapped).astype(float)
    minimum=[];maximum=[];margin=[];selected=[];cross_entropy=[]
    for j,(left,right) in enumerate(zip(arrays['candidate_offsets'][:-1],arrays['candidate_offsets'][1:])):
        values=arrays['legal_scores'][index,int(left):int(right)].astype(np.float64)
        ranked=np.sort(values); top=float(ranked[-1]); q=float(values[int(logged[j])])
        minimum.append(float(ranked[0]));maximum.append(top);margin.append(top-float(ranked[-2]));selected.append(q)
        cross_entropy.append(top+math.log(math.fsum(math.exp(float(v)-top) for v in values))-q)
    low,high,gap,taken,ce=map(np.asarray,(minimum,maximum,margin,selected,cross_entropy))
    prefix='logit' if model['kind']=='reference' else 'value'
    result.update({prefix+'_minimum':low,prefix+'_maximum':high,prefix+'_spread':high-low,
                   prefix+'_top_two_margin':gap,prefix+'_greedy_minus_logged':high-taken})
    if model['kind']=='value':
        result['chosen_ce']=ce
        for target,labels in (('mc',arrays['terminal_targets']),('delta',arrays['delta_targets'])):
            result[target+'_chosen_mse']=(taken-labels.astype(float))**2
            result[target+'_zero_prediction_mse']=labels.astype(float)**2
        if model['update']:
            target='mc' if model['arm'].startswith('mc') else 'delta'
            result['declared_objective']=result[target+'_chosen_mse']+int(model['arm'][-1])*ce
    return result


def accumulate(totals,split,arrays,models):
    objectives={}
    for index,model in enumerate(models):
        values=metrics_for(arrays,index,model)
        if model['kind']=='value' and model['update'] in (0,400):
            objectives[model['id']]={key:float(np.mean(values[key])) if len(values[key]) else 0.
                                    for key in ('mc_chosen_mse','delta_chosen_mse','chosen_ce')}
        for bit,slice_name in enumerate(SLICES):
            mask=(arrays['opportunity_mask']&(1<<bit))!=0
            count=int(mask.sum());key=(split,model['id'],slice_name)
            item=totals.setdefault(key,dict(rows=0,matches=0,row=defaultdict(float),match=defaultdict(float),
                transitions=np.zeros((len(ACTION_KINDS),len(ACTION_KINDS))),match_transitions=np.zeros((len(ACTION_KINDS),len(ACTION_KINDS)))))
            if not count: continue
            item['rows']+=count;item['matches']+=1
            for metric,vector in values.items():
                total=float(np.sum(vector[mask],dtype=np.float64))
                item['row'][metric]+=total;item['match'][metric]+=total/count
            transition=np.zeros_like(item['transitions'])
            for before,after in zip(arrays['deployed_kind'][0,mask],arrays['deployed_kind'][index,mask]):
                transition[int(before)-1,int(after)-1]+=1
            item['transitions']+=transition;item['match_transitions']+=transition/count
    return objectives


def compare_aggregates(totals,summary,counts):
    close(summary['splits'],counts,'Split summary')
    require(summary['action_kind_order']==list(ACTION_KINDS) and summary['opportunity_slices']==list(SLICES)
            and summary['probability_buckets']==list(BUCKETS),'Summary categories differ')
    expected={}
    for (split,model,slice_name),item in totals.items():
        n,g=item['rows'],item['matches']
        expected.setdefault(split,{}).setdefault(model,{})[slice_name]=dict(rows=n,supporting_matches=g,
            row_weighted={k:v/n for k,v in item['row'].items()} if n else {},
            match_weighted={k:v/g for k,v in item['match'].items()} if g else {},
            reference_to_deployed_kind_transition_row_weighted=(item['transitions']/n).tolist() if n else None,
            reference_to_deployed_kind_transition_match_weighted=(item['match_transitions']/g).tolist() if g else None)
    close(summary['by_split_model_slice'],expected,'Independent aggregate reconstruction')
    return len(totals)


def audit_inference(plan,report,models,at):
    require(report['schema']=='riichi-ai-stage3h-inference-v1' and report['phase']=='run'
            and report['complete'] is True and report['failure'] is None and report['models']==plan['models']
            and report['optimizer_steps']==report['new_matches']==0 and report['policy_promotion'] is False,'Invalid inference completion/scope')
    expected={(split,name) for split,names in plan['selected_matches'].items() for name in names}
    records=report['matches']
    require(len(records)==len(expected) and {(r['split'],r['match_id']) for r in records}==expected,'Inference coverage differs')
    dataset=LearningDataset(plan['corpus_path'],'training')
    mapping=read(plan['hand_targets_path'])
    totals={}; counts={};forward_checks=[];max_difference=0.;row_cache={}
    for record in records:
        deadline(at);split,name=record['split'],record['match_id']
        require(record['schema']=='riichi-ai-stage3h-inference-match-v1' and record['phase']=='run'
                and record['complete'] is True and record['delta_target_available'] is True
                and record['optimizer_steps']==record['new_matches']==0 and record['model_ids']==[m['id'] for m in plan['models']]
                and record['plan_sha256']==report['plan_sha256'],'Saved match diagnostic identity differs')
        source=dataset.reports[name]
        require(record['source_actor']==source['artifacts']['actor/'+name+'.npz']
                and record['source_targets']==source['artifacts']['targets/'+name+'.npz'],'Source shard identity differs')
        path=artifact(record['numeric']['path'],record['numeric']['sha256'],record['numeric']['bytes'])
        match_path=path.parent.parent/'matches'/f'{split}-{name}.json'
        require(read(match_path)==record,'Persisted match receipt differs from inference result')
        with np.load(path,allow_pickle=False) as stored: arrays={key:stored[key] for key in stored.files}
        require(all(not a.dtype.hasobject and np.isfinite(a).all() for a in arrays.values()),'Invalid diagnostic arrays')
        actor,labels,_=dataset._load(name)
        eligible=validate_rows(plan,record,arrays,actor,labels,mapping)
        require(plan['optional_rows_by_match'][split][name]==len(eligible),'Prepared optional row count differs')
        for item,(_,metadata) in zip(record['models_unchanged'],models):
            require(item==dict(model_id=metadata['id'],before=metadata['weights_sha256'],after=metadata['weights_sha256']),'Model immutability receipt differs')
        require(len(record['models_unchanged'])==18,'Missing model hash check')
        counter=counts.setdefault(split,dict(matches=0,optional_rows=0,empty_matches=0))
        counter['matches']+=1;counter['optional_rows']+=len(eligible);counter['empty_matches']+=int(not len(eligible))
        objectives=accumulate(totals,split,arrays,plan['models'])
        if split=='training' and name in plan['selected_matches']['training'][:64]:
            row_cache[name]=dict(rows=eligible.copy(),explorer_rows=int((labels['behavior_id']==3).sum()),objectives=objectives)
        if name in plan['selected_matches'][split][:8]:
            require(len(eligible)>=2,'Fixed audit panel lacks two eligible rows; no adaptive replacement')
            for position,row in enumerate(eligible[:2]):
                staged,_=LearningDataset._batch(actor,labels,np.asarray([row],dtype=np.int64))
                sample={key:value[0] for key,value in staged.items() if key!='candidate_mask'}
                batch=collate_encoded([sample],'cpu')
                left,right=map(int,arrays['candidate_offsets'][position:position+2])
                for i,(model,metadata) in enumerate(models):
                    deadline(at)
                    with torch.inference_mode(): scores,_=model(batch)
                    actual=scores[0].cpu().numpy();saved=arrays['legal_scores'][i,left:right]
                    require(actual.shape==saved.shape and np.isfinite(actual).all()
                            and np.allclose(actual,saved,atol=1e-6,rtol=1e-6),'Fixed CPU reforward differs')
                    difference=float(np.max(np.abs(actual.astype(float)-saved.astype(float))))
                    max_difference=max(max_difference,difference)
                    forward_checks.append(dict(split=split,match_id=name,row_index=int(row),model_id=metadata['id'],maximum_abs_difference=difference))
        del actor,labels,arrays
        dataset.loaded_match=None
    require(len(forward_checks)==576,'Fixed reforward count differs')
    aggregate_count=compare_aggregates(totals,report['summary'],counts)
    require(report['optional_rows']==sum(c['optional_rows'] for c in counts.values())
            and report['model_rows']==18*report['optional_rows'] and report['planned_saved_matches']==len(expected)
            and report['completed_saved_matches']==len(expected),'Inference overall counters differ')
    return dict(valid=True,saved_matches=len(records),optional_rows=report['optional_rows'],model_rows=report['model_rows'],
                reconstructed_split_model_slices=aggregate_count,cpu_reforward_rows=32,cpu_reforward_model_rows=576,
                maximum_reforward_absolute_difference=max_difference,reforward_checks=forward_checks,
                scope='All stored choices, recorded propensities, labels, opportunity flags and aggregate arithmetic; fixed576 CPU forwards. Unchosen outcomes remain unknown.'),row_cache


def vector_norm(x): return math.sqrt(float(np.dot(x,x)))


def alignment(a,b):
    aa,bb=vector_norm(a),vector_norm(b)
    return None if min(aa,bb)<=1e-12 else max(-1.,min(1.,float(np.dot(a,b))/(aa*bb)))


def pair(a,b):
    na,nb=vector_norm(a),vector_norm(b)
    combined={}
    for alpha in (0,1):
        norm=vector_norm(a+alpha*b)
        combined[str(alpha)]=dict(alpha=alpha,gradient_l2=norm,clip_factor=min(1.,1/(norm+1e-6)),clipped=norm>1.)
    return dict(mse_gradient_l2=na,ce_gradient_l2=nb,mse_ce_cosine=alignment(a,b),
                mse_ce_dot=float(np.dot(a,b)),combined=combined,
                norm_below_interpretation_threshold=dict(mse=na<=1e-12,ce=nb<=1e-12))


def block_pair(a,b,indices):
    return dict(all_actor_parameters=pair(a,b),parameter_blocks={name:pair(a[index],b[index]) for name,index in indices.items()})


def cancellation(vectors):
    mean=np.sum(vectors,axis=0,dtype=np.float64)/4
    norms=[vector_norm(v) for v in vectors];average=math.fsum(norms)/4;length=vector_norm(mean)
    return dict(mean_gradient_l2=length,mean_group_gradient_l2=average,
        mean_norm_over_mean_group_norm=length/average if average>1e-12 else None,
        group_to_mean_cosines=[alignment(v,mean) for v in vectors],
        opposing_groups=sum(float(np.dot(v,mean))<0 for v in vectors),
        pairwise_group_cosines=[dict(left_group=i,right_group=j,cosine=alignment(vectors[i],vectors[j])) for i in range(4) for j in range(i+1,4)])


def validate_layout(layout,model):
    expected=[];cursor=0;blocks={k:[] for k in ('backbone','scorer','efficiency')}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        block='scorer' if name.startswith('action_score.') else 'efficiency' if name.startswith('efficiency_encoder.') else 'backbone'
        expected.append(dict(name=name,shape=list(p.shape),start=cursor,stop=cursor+p.numel(),block=block))
        blocks[block].extend(range(cursor,cursor+p.numel()));cursor+=p.numel()
    require(layout['ordered_parameters']==expected and layout['trainable_parameters']==cursor
            and layout['frozen_rank_parameters']==sum(p.numel() for p in model.rank_head.parameters()),'Gradient parameter layout differs from archived model')
    for block in blocks:
        members=[dict(name=p['name'],shape=p['shape'],parameters=p['stop']-p['start']) for p in expected if p['block']==block]
        require(layout['blocks'][block]==members,'Parameter block layout differs')
    return {key:np.asarray(value,dtype=np.int64) for key,value in blocks.items()}


def audit_gradients(plan,report,models,row_cache,at):
    require(report['schema']=='riichi-ai-stage3h-gradient-diagnostic-v1' and report['complete'] is True and report['passed'] is True
            and report['phase']=='formal' and report['optimizer_steps']==report['new_matches']==report['checkpoint_writes']==0
            and report['policy_promotion'] is False and report['interpretation_threshold']==1e-12,'Gradient diagnostic scope/completion differs')
    by_id={m['id']:(m,model,meta) for m,(model,meta) in zip(plan['models'],models)}
    conditions=plan['gradient_conditions'];groups=plan['gradient_groups']
    expected_identity=dict(corpus_manifest_sha256=plan['corpus_manifest_sha256'],hand_targets_sha256=plan['hand_targets_sha256'],
        selected_models={c['checkpoint_id']:by_id[c['checkpoint_id']][0]['sha256'] for c in conditions},
        gradient_groups=groups,gradient_conditions=conditions)
    require(report['input_identity']==expected_identity and len(report['conditions'])==len(conditions)==6,'Gradient input identity/condition count differs')
    layout=report['parameter_layout'];indices=validate_layout(layout,by_id['initial'][1]);dimension=layout['trainable_parameters']
    results=[]
    for definition,item in zip(conditions,report['conditions']):
        deadline(at)
        require(all(item[k]==v for k,v in definition.items()),'Gradient condition definition differs')
        entry,model,metadata=by_id[definition['checkpoint_id']]
        require(item['checkpoint_sha256']==entry['sha256'],'Gradient checkpoint differs')
        identity=item['vectors'];path=artifact(identity['path'],identity['sha256'],identity['bytes'])
        require(identity['dtype']=='float32' and identity['shape_per_component']==[4,dimension]
                and identity['array_keys']==['mse','ce'] and identity['parameter_layout_sha256']==hashlib.sha256(canonical(layout)).hexdigest(),
                'Raw gradient vector identity/layout differs')
        require(read(path.with_suffix('.json'))==item,'Completed gradient condition receipt differs')
        with np.load(path,allow_pickle=False) as data:
            require(set(data.files)=={'mse','ce'} and all(data[k].dtype==np.float32 and data[k].shape==(4,dimension)
                    and np.isfinite(data[k]).all() for k in data.files),'Invalid raw gradient arrays')
            mse,ce=data['mse'].astype(np.float64),data['ce'].astype(np.float64)
        require(len(item['groups'])==4,'Missing gradient group')
        for number,(group,saved) in enumerate(zip(groups,item['groups'])):
            require(saved['matches']==group and saved['weights_unchanged'] is True and saved['optimizer_steps']==0
                    and saved['weights_sha256_before']==saved['weights_sha256_after']==metadata['weights_sha256']
                    and saved['rank_weights_sha256']==metadata['rank_sha256'],'Gradient group or immutable weights differ')
            optional=sum(len(row_cache[n]['rows']) for n in group)
            require(saved['optional_rows']==optional and saved['explorer_rows']==sum(row_cache[n]['explorer_rows'] for n in group)
                    and saved['empty_matches']==sum(len(row_cache[n]['rows'])==0 for n in group),'Gradient row/explorer counts differ')
            h=hashlib.sha256()
            for name in group:
                ids=row_cache[name]['rows']
                seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'big')
                order=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261023012,0,seed,1]))).permutation(len(ids))
                h.update(name.encode());h.update(ids[order].astype(np.int64).tobytes())
            require(h.hexdigest()==saved['row_order_sha256'],'Gradient row order differs')
            target_key='mc_chosen_mse' if definition['target']=='terminal_utility' else 'delta_chosen_mse'
            for metric,key in (('mse',target_key),('ce','chosen_ce')):
                expected=math.fsum(row_cache[name]['objectives'][definition['checkpoint_id']][key] for name in group)/16
                # CPU batch-one scores versus MPS padded batches use the frozen preflight objective tolerance.
                close(saved[metric],expected,'Gradient objective from independently stored scores',atol=.001,rtol=.001)
            close(saved['gradient_summary'],block_pair(mse[number],ce[number],indices),'Gradient group summary')
        average_m=mse.sum(axis=0)/4;average_c=ce.sum(axis=0)/4
        summary=item['summary']
        close(summary['mean_of_four_group_gradients'],block_pair(average_m,average_c,indices),'Mean gradient pair')
        close(summary['cancellation'],dict(mse=cancellation(mse),ce=cancellation(ce),mse_plus_ce=cancellation(mse+ce)),'Global cancellation')
        for block,index in indices.items():
            close(summary['parameter_block_cancellation'][block],dict(mse=cancellation(mse[:,index]),ce=cancellation(ce[:,index]),
                  mse_plus_ce=cancellation((mse+ce)[:,index])),'Block cancellation/'+block)
        for metric in ('mse','ce'):
            close(summary['mean_'+metric],math.fsum(g[metric] for g in item['groups'])/4,'Group objective average')
        results.append(dict(id=definition['id'],vectors_sha256=identity['sha256'],groups=4,parameters=dimension,
                            checkpoint_sha256=entry['sha256'],raw_vector_arithmetic_valid=True))
    return dict(valid=True,conditions=results,groups=24,parameter_layout_verified=True,
                scope='Group objectives checked against saved CPU scores; raw persisted vector norms, MSE/CE alignment, alpha0/1 combination/clip factors, block decomposition and four-group cancellation independently recomputed. Gradients themselves are not independently differentiated; the separate CPU/MPS preflight checks one fixed group.')


def run(args):
    from stage3h import read_config,verify_sources
    config,config_sha=read_config();data=Path(config['data_root'])
    active=[op for op in read(data/'budget-ledger.json')['operations'] if op['status']=='active']
    require(len(active)==1 and active[0]['name'].startswith('audit') and active[0]['name']==os.environ.get('RIICHI_STAGE3H_OPERATION')
            and active[0]['token']==os.environ.get('RIICHI_STAGE3H_TOKEN'),'Active matching Stage3H audit lease required')
    require(math.isfinite(args.max_seconds) and 0<args.max_seconds<=120,'Audit deadline outside cap')
    remaining=active[0]['reserved_seconds']-(time.time()-active[0]['started_unix'])
    at=time.monotonic()+min(args.max_seconds,remaining-10)
    plan=read(args.plan);plan_sha=sha(args.plan)
    source_path=args.source_manifest or args.plan;sources=read(source_path)['sources']
    require('scripts/audit_stage3h.py' in sources and 'tests/test_stage3h_audit.py' in sources,'Audit source/test membership missing')
    verify_sources(sources);verify_sources(plan['sources'])
    for path,identity in plan['inputs'].items(): artifact(path,identity)
    corpus=read(Path(plan['corpus_path'])/'manifest.json')
    require(sha(Path(plan['corpus_path'])/'manifest.json')==plan['corpus_manifest_sha256']
            and sha(plan['hand_targets_path'])==plan['hand_targets_sha256'],'Diagnostic input identities differ')
    validate_selection(plan,corpus);torch.set_num_threads(1)
    inference,gradients=read(args.inference_report),read(args.gradients_report)
    require(inference['plan_sha256']==gradients['plan_sha256']==plan_sha and gradients['config_sha256']==config_sha,'Diagnostic report plan/config differs')
    require(gradients['source_manifest_sha256']==plan_sha,'Gradient runtime source binding differs')
    previous=Path(config['reports_root'])/'gradient-preflight.json'
    artifact(previous,gradients['preflight_report_sha256'])
    preflight=read(previous)
    require(preflight['schema']=='riichi-ai-stage3h-gradient-preflight-v1' and preflight['complete'] is True
            and preflight['passed'] is True and preflight['backend_comparison']['passed'] is True
            and preflight['input_identity']==gradients['input_identity'],'Successful bound gradient parity preflight missing')
    models=load_models(plan)
    inferred,row_cache=audit_inference(plan,inference,models,at)
    differentiated=audit_gradients(plan,gradients,models,row_cache,at)
    for model,meta in models:
        require(_state_hash(model.state_dict())==meta['weights_sha256'] and _state_hash(model.rank_head.state_dict())==meta['rank_sha256'],'Audit changed archived model state')
    deadline(at);verify_sources(sources)
    for path,identity in plan['inputs'].items(): artifact(path,identity)
    return dict(schema='riichi-ai-stage3h-evidence-audit-v1',valid=True,config_sha256=config_sha,plan_sha256=plan_sha,
                source_manifest_sha256=sha(source_path),inference_report_sha256=sha(args.inference_report),
                gradients_report_sha256=sha(args.gradients_report),input_files_verified=len(plan['inputs']),
                inference=inferred,gradients=differentiated,optimizer_steps=0,new_matches=0,checkpoint_writes=0,
                policy_promotion=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','inference-report','gradients-report','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--source-manifest')
    parser.add_argument('--max-seconds',type=float,default=90)
    args=parser.parse_args();output=Path(args.output)
    require(not output.exists(),'Audit report is immutable')
    started=time.monotonic()
    try: result=run(args)
    except Exception as error: result=dict(schema='riichi-ai-stage3h-evidence-audit-v1',valid=False,
        failure=dict(type=type(error).__name__,message=str(error)),optimizer_steps=0,new_matches=0,checkpoint_writes=0)
    result['elapsed_seconds']=time.monotonic()-started
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps(dict(valid=result['valid'],elapsed_seconds=result['elapsed_seconds'],failure=result.get('failure'))),flush=True)
    return 0 if result['valid'] else 1


if __name__=='__main__': raise SystemExit(main())
