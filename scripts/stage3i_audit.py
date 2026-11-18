"""Independent Stage 3I evidence arithmetic and reserved deterministic replays.

Run only under the Stage 3I audit lease. This verifies the saved experiment;
it neither promotes a policy nor certifies the complete rules implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import heapq
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import numpy as np
import torch

GRID = (0,1,2,4,8,16,32,64,128,256,512,1024)
HEAD_ORDER = ('value','action','state')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:
        handle.write(encoded(value))


def array_file(path):
    with np.load(path,allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def near(left,right,label):
    require(np.allclose(left,right,rtol=1e-12,atol=1e-12),label)


def deadline_check(deadline):
    if time.monotonic()>=deadline:
        raise TimeoutError('Independent audit deadline reached')


def predictions(cache,pred):
    """Recompute every legal correction and chosen error without evaluator code."""
    from riichi_ai.semantics import ACTION_KINDS
    kind={name:i+1 for i,name in enumerate(ACTION_KINDS)}
    n=len(cache['target']); offsets=cache['candidate_offsets']
    require(pred['value'].shape==(n,) and pred['state'].shape==(n,)
            and pred['action'].shape==(offsets[-1],) and pred['choices_by_lambda'].shape==(len(GRID),n),
            'Prediction artifact dimensions differ')
    require(all(np.isfinite(value).all() and not value.dtype.hasobject for value in pred.values()),'Nonfinite prediction artifact')
    require(np.array_equal(cache['row_indices'],pred['row_indices']),'Prediction source row identity differs')
    reference=np.empty(n,np.int64); choices=np.empty((len(GRID),n),np.int64)
    raw=np.empty(n,np.int64); selected_kind=np.empty(n,np.int64)
    opportunities={name:np.zeros(n,bool) for name in ('call','riichi','kan','win')}
    for row in range(n):
        lo,hi=map(int,offsets[row:row+2])
        z=cache['logits'][lo:hi].astype(np.float64)
        residual=pred['action'][lo:hi].astype(np.float64)
        q=cache['explorer_probabilities'][lo:hi]
        kinds=cache['kinds'][lo:hi]
        require(np.all((q>=0)&(q<=1)) and abs(float(q.sum())-1)<1e-12,'Invalid recorded explorer probabilities')
        canonical=int(np.argmax(z)); raw[row]=canonical
        eligible=q>=.02; eligible[canonical]=True
        prior=np.clip(z-z[canonical],-8,None)
        scores=prior[None,:]+np.array(GRID,np.float64)[:,None]*residual[None,:]
        scores[:,~eligible]=-np.inf
        indices=np.argmax(scores,axis=1); indices[0]=canonical
        wins=np.flatnonzero((kinds==kind['RON'])|(kinds==kind['TSUMO']))
        reference[row]=int(wins[0]) if len(wins) else canonical
        if len(wins): indices[:]=wins[0]
        choices[:,row]=indices
        selected_kind[row]=kinds[reference[row]]
        for name,codes in {'call':[kind['CHI'],kind['PON'],kind['DAIMINKAN']],
                           'riichi':[kind['RIICHI']],
                           'kan':[kind['DAIMINKAN'],kind['ANKAN'],kind['KAKAN']],
                           'win':[kind['RON'],kind['TSUMO']]}.items():
            opportunities[name][row]=bool(np.isin(kinds,codes).any())
    require(np.array_equal(pred['choices_by_lambda'],choices),'Saved lambda choices differ from independent arithmetic')
    require(np.array_equal(pred['raw_reference'],raw) and np.array_equal(pred['wrapped_reference'],reference),
            'Saved raw/wrapped reference decisions differ')
    taken=offsets[:-1]+cache['chosen_index']
    y=cache['target'].astype(np.float64)
    v=pred['value'].astype(np.float64); b=pred['state'].astype(np.float64)
    a=pred['action'][taken].astype(np.float64)
    squared={'v':np.square(v-y),'va':np.square(v+a-y),'vb':np.square(v+b-y),'zero':np.square(y)}
    slices={'all':np.ones(n,bool),'reference_discard':selected_kind==kind['DISCARD'],**opportunities,
            'hand_ordinal_1_4':cache['hand_ids']<=4,
            'hand_ordinal_5_8':(cache['hand_ids']>4)&(cache['hand_ids']<=8),
            'hand_ordinal_9_plus':cache['hand_ids']>8}
    for name,code in kind.items(): slices['reference_kind_'+name.lower()]=selected_kind==code
    if 'public_globals' in cache:
        g=cache['public_globals']
        slices.update(east_round=g[:,32]==1,south_round=g[:,33]==1,
                      west_or_north_round=(g[:,34]==1)|(g[:,35]==1),
                      self_riichi_declared=g[:,4]==1,opponent_riichi_declared=np.any(g[:,5:8]==1,axis=1))
    counts=[]
    for index,strength in enumerate(GRID):
        changed=choices[index]!=reference
        counts.append(dict(lambda_=strength,rows=n,changed=int(changed.sum()),
                           discard_rows=int(slices['reference_discard'].sum()),
                           discard_changed=int((changed&slices['reference_discard']).sum()),
                           slices={name:dict(rows=int(mask.sum()),changed=int((mask&changed).sum())) for name,mask in slices.items()}))
    return dict(rows=n,errors={name:float(values.sum()) for name,values in squared.items()},grid=counts)


def select_lambda(grid):
    """Independent discrete constrained selection, including lambda zero."""
    require(len(grid)==len(GRID),'Missing lambda table rows')
    choices=[]
    for strength,row in zip(GRID,grid):
        require(row['lambda']==strength and row['rows']>0,'Lambda grid identity differs')
        change=row['changed']/row['rows']
        discard=row['reference_discard_changed']/row['reference_discard_rows'] if row['reference_discard_rows'] else 0.
        near(change,row['change_rate'],'Reported overall change rate differs')
        near(discard,row['reference_discard_change_rate'],'Reported discard change rate differs')
        if change<=.03 and discard<=.10:
            choices.append((change,-strength,strength))
    require(grid[0]['changed']==0,'Zero correction changed reference actions')
    return max(choices)[2]


def audit_cache(plan,deadline):
    manifest_path=Path(plan['cache_manifest_path'])
    require(sha(manifest_path)==plan['cache_manifest_sha256'],'Cache manifest changed')
    manifest=read(manifest_path)
    require(manifest['schema']=='riichi-ai-stage3i-feature-cache-v1' and manifest['complete'] is True
            and manifest['purpose']=='formal' and manifest['labels_are_actor_inputs'] is False,'Cache purpose/schema mismatch')
    require(manifest['partitions']==plan['partitions'],'Cache exact partition membership mismatch')
    require({key:len(value) for key,value in manifest['partitions'].items()}==dict(training=3200,selection=200,calibration=200,gate=400),
            'Cache partition counts changed')
    corpus_path=Path(plan['corpus_path'])
    require(sha(corpus_path/'manifest.json')==plan['corpus_manifest_sha256'],'Historical corpus changed')
    require(sha(plan['hand_targets_path'])==plan['hand_targets_sha256'],'Historical one-hand targets changed')
    source=read(corpus_path/'manifest.json'); targets=read(plan['hand_targets_path'])['matches']
    originals={row['match_id']:row for row in source['matches']}
    members=[name for names in plan['partitions'].values() for name in names]
    require(len(members)==len(set(members))==4000,'Cache has overlap or missing matches')
    by_name={row['match_id']:row for row in manifest['shards']}
    require(len(by_name)==len(manifest['shards']) and set(by_name)==set(members),'Cache shard coverage differs')
    counts={key:dict(matches=0,rows=0,candidates=0) for key in plan['partitions']}
    reforward=[]
    for partition,names in plan['partitions'].items():
        for position,name in enumerate(names):
            deadline_check(deadline)
            record=by_name[name]
            require(record['partition']==partition,'Cache shard role mismatch')
            path=Path(record['path'])
            if not path.is_absolute(): path=manifest_path.parent/path
            require(sha(path)==record['sha256'],'Cache shard hash differs')
            cached=array_file(path)
            origin=originals[name]
            area={}
            for part in ('actor','targets'):
                relative=f'{part}/{name}.npz'
                require(sha(corpus_path/relative)==origin['artifacts'][relative]['sha256'],'Source numerical shard hash differs')
                area[part]=array_file(corpus_path/relative)
            actor,label=area['actor'],area['targets']
            rows=np.flatnonzero((label['behavior_id']==3)&(np.diff(actor['candidate_offsets'])>1))
            require(np.array_equal(cached['row_indices'],rows),'Optional explorer cache exposure differs')
            require(len(rows)==record['rows'],'Cache row count differs')
            starts=actor['candidate_offsets'][rows]; ends=actor['candidate_offsets'][rows+1]
            packed=np.concatenate([np.arange(lo,hi,dtype=np.int64) for lo,hi in zip(starts,ends)])
            offsets=np.r_[0,np.cumsum(ends-starts)]
            require(np.array_equal(cached['candidate_offsets'],offsets),'Cache candidate packing differs')
            require(np.array_equal(cached['chosen_index'],label['chosen_index'][rows]),'Cached selected actions differ')
            require(np.array_equal(cached['kinds'],actor['action_tokens'][packed,0]),'Cached action kinds differ')
            require(np.array_equal(cached['explorer_probabilities'],label['behavior_probabilities'][packed]),'Cached explorer probabilities differ')
            for output,input_name in [('steps','step'),('seats','seat'),('hand_ids','hand_id')]:
                require(np.array_equal(cached[output],label[input_name][rows]),'Cache public row provenance differs')
            require(np.array_equal(cached['public_globals'],actor['globals'][rows]),'Cache public context differs')
            mapping=targets[name]
            require(np.all(label['seat'][rows]==mapping['explorer_seat']),'Cached outcome belongs to a different explorer seat')
            expected=np.array([mapping['hand_deltas'][str(int(h))] for h in label['hand_id'][rows]],np.float32)
            require(np.array_equal(cached['target'],expected),'Cached one-hand labels differ')
            require(cached['state_features'].shape==(len(rows),256)
                    and cached['action_features'].shape==(len(packed),128),'Frozen cache feature dimensions differ')
            require(all(not a.dtype.hasobject and np.isfinite(a).all() for a in cached.values()),'Invalid cache array')
            counts[partition]['matches']+=1;counts[partition]['rows']+=len(rows);counts[partition]['candidates']+=len(packed)
            for local,row in enumerate(rows):
                key=int.from_bytes(hashlib.sha256(f'stage3i-cache-audit|{name}|{int(row)}'.encode()).digest(),'big')
                candidate=(-key,name,int(row),local)
                if len(reforward)<64:
                    heapq.heappush(reforward,candidate)
                elif candidate[0]>reforward[0][0]:
                    heapq.heapreplace(reforward,candidate)
            if (position+1)%400==0:
                print(json.dumps(dict(audit='cache',partition=partition,matches=position+1)),flush=True)
    require(counts['training']['rows']==471627,'Training exposure changed')
    # Independent production forward hook captures exactly the scorer's visible
    # inputs, rather than trusting the feature extractor's own implementation.
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.stage3f_stream import StreamDataset,StreamBatch
    policy=NeuralPolicy.from_checkpoint(plan['reference_checkpoint'],device='cpu')
    require(policy.checkpoint_hash==plan['reference_sha256'],'Original reference identity differs')
    captured=[]
    hook=policy.model.action_score.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    verified=0
    try:
        for _,name,row,local in sorted(reforward,reverse=True):
            deadline_check(deadline)
            path=Path(by_name[name]['path'])
            if not path.is_absolute(): path=manifest_path.parent/path
            cached=array_file(path)
            actor=array_file(corpus_path/f'actor/{name}.npz')
            label=array_file(corpus_path/f'targets/{name}.npz')
            tensors,labels=StreamDataset._batch(actor,label,np.array([row],np.int64))
            batch=StreamBatch(tensors,labels,name,np.array([row]),{},0).torch_actor('cpu')
            with torch.inference_mode(): logits,_=policy.model(batch)
            representation=captured.pop()
            lo,hi=map(int,cached['candidate_offsets'][local:local+2])
            require(np.array_equal(logits[0].numpy(),cached['logits'][lo:hi]),'Cached logits differ from independent original actor forward')
            require(np.array_equal(representation[0,0,:256].numpy(),cached['state_features'][local]),'Cached context differs from original scorer input')
            require(np.array_equal(representation[0,:,256:].numpy(),cached['action_features'][lo:hi]),'Cached candidates differ from original scorer input')
            verified+=1
    finally:
        hook.remove()
    require(verified==64,'Fixed cache re-forward sample incomplete')
    return dict(complete=True,partitions=counts,reforward_rows=verified,
                all_optional_explorer_rows_preserved=True,all_logged_probabilities_exact=True),manifest,by_name


def audit_training(plan,training,deadline):
    from riichi_ai.model import _state_hash
    require(training['complete'] is True and training['phase']=='run','Formal training report incomplete')
    require([item['seed'] for item in training['seeds']]==plan['training_seeds'],'Training seed family changed')
    epochs=plan['epochs']; row_count=471627; per_epoch=math.ceil(row_count/1024)
    exposure=updates=checkpoints=0
    for seed_record in training['seeds']:
        seed=seed_record['seed'];bundle_path=seed_record['checkpoint']
        require(sha(bundle_path)==seed_record['sha256'],'Selected head bundle file changed')
        bundle=torch.load(bundle_path,map_location='cpu',weights_only=True)
        require(bundle['complete'] is True and not bundle.get('benchmark_only',False),'Preflight head entered formal results')
        combined={name+'.'+key:value for name,values in bundle['head_state_dicts'].items() for key,value in values.items()}
        require(_state_hash(combined)==bundle['weights_sha256']==seed_record['weights_sha256'],
                'Selected full head bundle tensor hash differs')
        require(read(Path(bundle_path).parent/'seed-report.json')==seed_record,'Seed report differs from training manifest')
        for key in ('reference_sha256','call_sha256','protocol_sha256','cache_manifest_sha256'):
            require(bundle['identity'][key]==plan[key],'Selected head bundle identity changed')
        for head_index,name in enumerate(HEAD_ORDER):
            head=seed_record['heads'][name]
            records=head['epochs']; require(len(records)==epochs,'Head epoch coverage differs')
            best=min(records,key=lambda item:(item['selection']['row_mse'],item['epoch']))
            require(head['selected_epoch']==best['epoch']==bundle['selected_epochs'][name], 'Head checkpoint selection rule differs')
            require(head['row_presentations']==epochs*row_count and head['optimizer_updates']==epochs*per_epoch,'Head training exposure differs')
            for epoch,record in enumerate(records,1):
                deadline_check(deadline)
                require(record['epoch']==epoch and record['head']==name and record['seed']==seed,'Epoch identity differs')
                require(record['training']['rows']==row_count and record['optimizer_updates']==epoch*per_epoch,'Epoch is not a complete full pass')
                order=np.random.default_rng(np.random.SeedSequence([seed,head_index,epoch,20261022])).permutation(row_count)
                require(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()==record['order_sha256'],'Saved epoch permutation differs')
                cp=record['checkpoint']; require(sha(cp['path'])==cp['sha256'],'Epoch checkpoint file changed')
                require(read(Path(cp['path']).with_suffix('.json'))==record,'Epoch report file differs from training manifest')
                saved=torch.load(cp['path'],map_location='cpu',weights_only=True);metadata=saved['metadata']
                require(metadata['head']==name and metadata['seed']==seed and metadata['epoch']==epoch
                        and metadata['optimizer_updates']==epoch*per_epoch and metadata['complete_epoch'] is True,'Epoch checkpoint metadata differs')
                require(metadata['identity']==bundle['identity'],'Epoch checkpoint provenance differs from selected bundle')
                require(_state_hash(saved['head_state_dict'])==metadata['weights_sha256']==cp['weights_sha256'],'Epoch tensor hash differs')
                require(metadata['selection']==record['selection'],'Checkpoint selection evidence differs')
                optimizer=saved['optimizer_state_dict']
                require(len(optimizer['state'])==len(saved['head_state_dict']),'Optimizer parameter coverage differs')
                require(all(int(state['step'])==epoch*per_epoch for state in optimizer['state'].values()),'Optimizer exposure differs')
                if name!='value':
                    require(metadata['fixed_value_sha256']==_state_hash(bundle['head_state_dicts']['value']),'Residual target was not selected frozen V')
                if epoch==best['epoch']:
                    require(_state_hash(saved['head_state_dict'])==_state_hash(bundle['head_state_dicts'][name])==head['weights_sha256'],
                            'Selected bundle head differs from declared epoch')
                checkpoints+=1
            exposure+=head['row_presentations']; updates+=head['optimizer_updates']
    require(exposure==training['row_presentations'] and updates==training['optimizer_updates'],'Campaign head exposure differs')
    require(training['frozen_actor_checkpoint_sha256_before']==training['frozen_actor_checkpoint_sha256_after']==plan['reference_sha256'],
            'Frozen actor before/after identities differ')
    return dict(complete=True,seeds=len(training['seeds']),heads=9,epochs_per_head=epochs,
                checkpoints=checkpoints,optimizer_updates=updates,row_presentations=exposure,
                all_selected_epochs_follow_fixed_rule=True,all_residual_targets_bind_selected_value=True)


def audit_calibration(plan,calibration,cache_manifest,cache_records,output_dir,deadline):
    require(calibration['complete'] is True and calibration['schema']=='riichi-ai-stage3i-calibration-v1','Calibration incomplete')
    require(calibration['cache_manifest_sha256']==plan['cache_manifest_sha256'],'Calibration cache identity differs')
    require([item['seed'] for item in calibration['seeds']]==plan['training_seeds'],'Calibration seed family differs')
    parent=Path(plan['cache_manifest_path']).parent
    artifacts={(row['seed'],row['partition'],row['match_id']):row for row in calibration['artifacts']}
    require(len(artifacts)==len(calibration['artifacts'])==1800,'Calibration prediction artifact coverage differs')
    reported={item['seed']:item for item in calibration['seeds']}
    summaries={seed:{part:[] for part in ('calibration','gate')} for seed in plan['training_seeds']}
    head_samples={part:[] for part in ('calibration','gate')}
    total_rows=0
    for partition in ('calibration','gate'):
        for index,name in enumerate(cache_manifest['partitions'][partition]):
            deadline_check(deadline)
            source=cache_records[name];path=Path(source['path'])
            if not path.is_absolute():path=parent/path
            cache=array_file(path)
            for local,row in enumerate(cache['row_indices']):
                key=int.from_bytes(hashlib.sha256(f'stage3i-prediction-audit|{partition}|{name}|{int(row)}'.encode()).digest(),'big')
                sample=(-key,name,local)
                if len(head_samples[partition])<32:heapq.heappush(head_samples[partition],sample)
                elif sample[0]>head_samples[partition][0][0]:heapq.heapreplace(head_samples[partition],sample)
            for seed in plan['training_seeds']:
                artifact=artifacts[(seed,partition,name)]
                require(artifact['source_sha256']==source['sha256'],'Prediction artifact source identity differs')
                path=Path(output_dir)/artifact['path']
                require(not path.is_symlink() and path.resolve().is_relative_to(Path(output_dir).resolve())
                        and sha(path)==artifact['sha256'] and path.stat().st_size==artifact['bytes'],
                        'Prediction file changed or escaped its evidence root')
                calculated=predictions(cache,array_file(path))
                saved=reported[seed]['match_metrics'][partition][index]
                require(saved['match_id']==name and saved['rows']==calculated['rows'],'Prediction report match membership differs')
                for key,value in calculated['errors'].items():
                    near(saved[key+'_squared_error_sum'],value,'Saved match prediction MSE differs')
                require(saved['grid']==calculated['grid'],'Saved match change/slice arithmetic differs')
                summaries[seed][partition].append(calculated)
                total_rows+=calculated['rows']
            if (index+1)%100==0:
                print(json.dumps(dict(audit='prediction_arithmetic',partition=partition,matches=index+1)),flush=True)
    reforward=0
    for seed in plan['training_seeds']:
        item=reported[seed]
        require(sha(item['checkpoint'])==item['sha256'],'Prediction head bundle identity changed')
        bundle=torch.load(item['checkpoint'],map_location='cpu',weights_only=True)
        states=bundle['head_state_dicts']
        def forward(name,features):
            layer=states[name]
            hidden=torch.nn.functional.linear(features,layer['layers.0.weight'],layer['layers.0.bias'])
            return torch.nn.functional.linear(torch.relu(hidden),layer['layers.2.weight'],layer['layers.2.bias']).squeeze(-1)
        for partition in ('calibration','gate'):
            for _,name,local in sorted(head_samples[partition],reverse=True):
                deadline_check(deadline)
                cache_path=Path(cache_records[name]['path'])
                if not cache_path.is_absolute():cache_path=parent/cache_path
                cached=array_file(cache_path)
                pred=array_file(Path(output_dir)/artifacts[(seed,partition,name)]['path'])
                lo,hi=map(int,cached['candidate_offsets'][local:local+2])
                context=torch.from_numpy(cached['state_features'][local:local+1]).clone()
                candidates=torch.from_numpy(cached['action_features'][None,lo:hi]).clone()
                joined=torch.cat((context.unsqueeze(1).expand(-1,candidates.shape[1],-1),candidates),dim=-1)
                with torch.inference_mode():
                    v=forward('value',context)[0].numpy()
                    b=forward('state',context)[0].numpy()
                    a=forward('action',joined)[0].numpy()
                require(np.array_equal(v,pred['value'][local]) and np.array_equal(b,pred['state'][local])
                        and np.array_equal(a,pred['action'][lo:hi]),'Saved prediction differs from independent selected-head forward')
                reforward+=1
    require(reforward==192,'Fixed selected-head re-forward sample incomplete')
    results=[]
    for seed in plan['training_seeds']:
        saved=reported[seed];calibration_matches=summaries[seed]['calibration'];gate_matches=summaries[seed]['gate']
        for index,strength in enumerate(GRID):
            records=[item['grid'][index] for item in calibration_matches]
            expected=saved['lambda_grid'][index]
            require(expected['rows']==sum(item['rows'] for item in records)
                    and expected['changed']==sum(item['changed'] for item in records)
                    and expected['reference_discard_rows']==sum(item['discard_rows'] for item in records)
                    and expected['reference_discard_changed']==sum(item['discard_changed'] for item in records),
                    'Aggregate lambda counts differ')
            for key in records[0]['slices']:
                require(expected['slices'][key]==dict(rows=sum(item['slices'][key]['rows'] for item in records),
                                                      changed=sum(item['slices'][key]['changed'] for item in records)),
                        'Aggregate public slice counts differ')
        strength=select_lambda(saved['lambda_grid'])
        require(saved['calibration']['lambda_']==strength and saved['calibration']['fallback_zero']==(strength==0),
                'Selected lambda differs from independent constrained calibration')
        row_count=sum(item['rows'] for item in gate_matches)
        mse={key:sum(item['errors'][key] for item in gate_matches)/row_count for key in ('v','va','vb')}
        match_mse={key:float(np.mean([item['errors'][key]/item['rows'] for item in gate_matches])) for key in ('v','va','vb')}
        for key in mse:
            near(saved['gate']['row_mse'][key],mse[key],'Final gate row MSE differs')
            near(saved['gate']['match_mse'][key],match_mse[key],'Final gate match MSE differs')
        passed=mse['va']<mse['v'] and mse['va']<mse['vb']
        require(saved['gate']['passed']==passed and saved['admitted_for_games']==(passed and strength>0),
                'Predictive game admission differs')
        reasons=[]
        if not mse['va']<mse['v']: reasons.append('action_residual_does_not_improve_value_chosen_row_mse')
        if not mse['va']<mse['vb']: reasons.append('action_residual_does_not_improve_state_residual_chosen_row_mse')
        require(saved['gate']['reasons']==reasons,'Gate failure reasons differ')
        if strength==0:reasons.append('zero_calibrated_correction_no_candidate_games')
        require(saved['skip_reasons']==reasons,'Candidate skip reasons differ')
        results.append(dict(seed=seed,lambda_=strength,gate_passed=passed,admitted_for_games=passed and strength>0,
                            row_mse=mse,match_mse=match_mse))
    return dict(complete=True,artifacts=len(artifacts),prediction_rows=total_rows,all_choices_exact=True,seeds=results,
                independent_head_reforwards=reforward,head_reforward_selection='First32 SHA256(stage3i-prediction-audit|partition|match_id|row_index) per calibration/gate partition, all3seeds')


def statistics(values,seed):
    values=np.asarray(values,np.float64);count=len(values);mean=float(values.mean())
    require(count>0 and np.all(np.isfinite(values)) and np.all(np.abs(values)<=3),'Invalid paired block differences')
    rng=np.random.default_rng(seed)
    samples=np.empty(5000,np.float64)
    for start in range(0,5000,100):
        samples[start:start+100]=rng.choice(values,size=(100,count),replace=True).mean(axis=1)
    radius=6*math.sqrt(math.log(40)/(2*count))
    return dict(independent_blocks=count,mean_utility_difference=mean,mean_rank_improvement=mean,
                standard_error=float(values.std(ddof=1)/math.sqrt(count)) if count>1 else None,
                fixed_budget_hoeffding_95=[max(-3.,mean-radius),min(3.,mean+radius)],
                bootstrap_percentile_95_exploratory=np.percentile(samples,[2.5,97.5]).tolist(),bootstrap_replicates=5000)


def audit_decision(row,policy):
    """Independent scalar arithmetic on persisted legal choices, no model call."""
    keys,kinds,tokens=row['semantic_keys'],row['action_kinds'],row['representative_tokens']
    count=len(keys);chosen=row['selected_index']
    require(count>0 and len(kinds)==len(tokens)==count and type(chosen) is int and 0<=chosen<count,
            'Invalid semantic choice dimensions or index')
    require(len({encoded(key) for key in keys})==count,'Duplicate semantic legal action')
    require(row['selected_semantic_key']==keys[chosen] and row['selected_kind']==kinds[chosen]
            and row['selected_physical_token']==tokens[chosen],'Selected semantic/physical identity differs')
    require(row['probability']==1. and row['training_eligible'] is False and row['ppo_eligible'] is False,
            'Evaluation decision probability or training role differs')
    if policy in ('shanten_v2','defense_v2'):
        require(row['legal_scores'] is None and row['correction'] is None,'Heuristic decision carries unexpected learned scores')
        return
    if count==1:
        require(chosen==0 and row['legal_scores'] is None and row['correction']==dict(forced=True),
                'Forced deterministic action differs')
        return
    logits=row['legal_scores'];detail=row['correction']
    require(len(logits)==count and all(math.isfinite(z) for z in logits),'Invalid persisted legal scores')
    raw=max(range(count),key=lambda index:logits[index])
    wins=[index for index,kind in enumerate(kinds) if kind in ('RON','TSUMO')]
    wrapped=wins[0] if wins else raw
    if policy.startswith('reference_'):
        require(detail['raw_reference_index']==raw and chosen==(wrapped if policy=='reference_wrapped' else raw),
                'Reference argmax or win wrapper differs')
        return
    if policy.startswith('reset_'):
        require(all(abs(z)<=100 for z in logits) and detail['raw_index']==raw
                and chosen==(wrapped if policy=='reset_wrapped' else raw),'Reset argmax or win wrapper differs')
        return
    q,residual,strength=detail['explorer_probabilities'],detail['action_residuals'],detail['strength']
    require(len(q)==len(residual)==count and all(math.isfinite(a) for a in residual)
            and all(math.isfinite(p) and 0<=p<=1 for p in q) and abs(sum(q)-1)<=1e-12
            and strength in GRID,'Candidate probability/residual dimensions or values differ')
    eligible=[probability>=.02 or index==raw for index,probability in enumerate(q)]
    prior=[max(value-logits[raw],-8.) for value in logits]
    scores=[p+strength*a if allowed else -math.inf for p,a,allowed in zip(prior,residual,eligible)]
    require(all(math.isfinite(score) for score,allowed in zip(scores,eligible) if allowed),'Candidate corrected score overflow')
    corrected=raw if strength==0 else max(range(count),key=lambda index:scores[index])
    expected=wins[0] if wins else corrected
    require(detail['raw_reference_index']==raw and detail['wrapped_reference_index']==wrapped
            and detail['raw_corrected_index']==corrected and chosen==expected,'Candidate corrected score or wrapper choice differs')
    require(detail['eligible']==eligible and detail['prior']==prior,'Candidate eligibility or clipped prior differs')


def audit_response_links(rows,requests,policies):
    covered=[]
    for step,request in enumerate(requests):
        require(request['step']==step and request['submitted'] is True,'Response step order or submission differs')
        expected=hashlib.sha256(encoded(dict(step=step,player_views=request['player_views']))).hexdigest()
        require(request['pre_response_id']==expected,'Pre-response information state hash differs')
        seats=[]
        for index in request['rows']:
            require(type(index) is int and 0<=index<len(rows),'Response points outside decision rows')
            row=rows[index];seat=row['seat']
            require(row['row_index']==index and row['step']==step and row['pre_response_id']==expected,
                    'Decision response identity differs')
            require(row['player_view_sha256']==request['player_views'][str(seat)]
                    and row['policy_id']==policies[seat]['policy_id'],'Decision player view or behavior provenance differs')
            seats.append(seat);covered.append(index)
        require(len(seats)==len(set(seats)),'Duplicate response by one seat in a simultaneous request')
    require(covered==list(range(len(rows))),'Decision response coverage is not complete and ordered')


def _audit_game_decisions(item):
    deadline_check(item['deadline'])
    record=item['record'];root=Path(item['root']);artifact=record['artifact']
    raw=(root/artifact['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==artifact['sha256'],'Auditor game changed before full inspection')
    data=json.loads(gzip.decompress(raw))
    require(data['completed'] is True and data['failure'] is None and data['plan']['match_id']==record['match_id'],
            'Auditor completion or match identity differs')
    require(data['policies']==record['policies'] and data['scores']==record['scores'] and len(data['rows'])==record['rows']
            and len(data['requests'])==record['steps'],'Auditor row/policy/request counts differ')
    audit_response_links(data['rows'],data['requests'],data['policies'])
    counts=Counter();focal=Counter();optional=corrected=0
    for row in data['rows']:
        policy=data['policies'][row['seat']]['name']
        audit_decision(row,policy)
        counts[row['selected_kind']]+=1
        if row['seat']==record['focal_seat']:
            focal[row['selected_kind']]+=1
            optional+=len(row['semantic_keys'])>1
            if policy.startswith('seed_') and len(row['semantic_keys'])>1:
                require(row['correction']['strength']==data['policies'][row['seat']]['manifest']['strength'],
                        'Per-decision correction strength differs from policy provenance')
                corrected+=row['selected_index']!=row['correction']['wrapped_reference_index']
    require(dict(counts)==record['chosen_action_counts'] and dict(focal)==record['focal_action_counts'],'Reported action counts differ')
    require(data['ordinal_wall_fingerprints']==record['ordinal_wall_fingerprints']
            and [row['hand_id'] for row in data['hand_starts']]==list(range(1,len(data['ordinal_wall_fingerprints'])+1)),
            'Explicit hand/wall ordinal coverage differs')
    require(sum(event.get('type')=='start_kyoku' for event in data['canonical_events'])==len(data['hand_starts']),
            'Canonical events omit an explicit hand continuation')
    require(hashlib.sha256(encoded(data['canonical_events'])).hexdigest()==record['game_sha256'],
            'Canonical engine event stream hash differs')
    from riichi_ai.stage3c import terminal_result
    terminal=terminal_result(record['scores'],record['ranks'],data['canonical_events'],completed=True)
    require(terminal==data['terminal'] and terminal['utilities']==record['utilities'],
            'Reconstructed terminal scoring/rank utility differs')
    deadline_check(item['deadline'])
    return dict(match_id=record['match_id'],arm=record['arm'],rows=record['rows'],requests=record['steps'],
                focal_optional_rows=optional,focal_corrected_rows=corrected,complete=True)


def audit_all_game_decisions(root,records,deadline):
    from riichi_ai.evaluation import _terminate_evaluation_pool
    jobs=iter(dict(root=str(root),record=record,deadline=deadline) for record in records.values())
    pool=ProcessPoolExecutor(max_workers=14,mp_context=multiprocessing.get_context('spawn'))
    result=dict(games=0,rows=0,requests=0,focal_optional_rows=0,focal_corrected_rows=0)
    by_arm={}
    try:
        pending=deque()
        for _ in range(14):
            item=next(jobs,None)
            if item:pending.append(pool.submit(_audit_game_decisions,item))
        while pending:
            deadline_check(deadline)
            checked=pending.popleft().result(timeout=max(.001,deadline-time.monotonic()))
            result['games']+=1
            for key in ('rows','requests','focal_optional_rows','focal_corrected_rows'):result[key]+=checked[key]
            arm=by_arm.setdefault(checked['arm'],dict(games=0,focal_optional_rows=0,focal_corrected_rows=0))
            arm['games']+=1
            for key in ('focal_optional_rows','focal_corrected_rows'):arm[key]+=checked[key]
            item=next(jobs,None)
            if item:pending.append(pool.submit(_audit_game_decisions,item))
            if result['games']%512==0:print(json.dumps(dict(audit='full_game_decisions',games=result['games'])),flush=True)
    except BaseException:
        _terminate_evaluation_pool(pool)
        raise
    else:pool.shutdown(wait=True)
    require(result['games']==len(records),'Full game decision audit incomplete')
    return dict(complete=True,**result,by_arm=by_arm,all_semantic_tokens_and_simultaneous_response_links_verified=True,
                all_model_choices_recomputed_from_persisted_scores=True,all_terminal_scores_reconstructed=True)


def audit_games(plan,calibration,evaluation,deadline):
    require(evaluation['complete'] is True,'Formal game panel incomplete')
    path=Path(evaluation['manifest_path']); require(sha(path)==evaluation['manifest_sha256'],'Game manifest changed')
    manifest=read(path);execution=manifest['execution_plan'];root=path.parent
    require(execution['plan_sha256']==calibration['plan_sha256']
            and execution['calibration_sha256']==hashlib.sha256(encoded(calibration)).hexdigest()
            and execution['protocol_sha256']==plan['protocol_sha256']
            and execution['cache_manifest_sha256']==plan['cache_manifest_sha256']
            and execution['fixed_seed_family']==plan['training_seeds'],'Game execution plan/calibration/protocol identities differ')
    require(manifest['complete'] is True and execution['phase']=='formal' and execution['blocks']==470
            and execution['rotations']==4 and execution['reset_blocks']==list(range(64)), 'Formal fixed game panel dimensions differ')
    require(sha(root/'execution-plan.json')==manifest['execution_plan_sha256'] and read(root/'execution-plan.json')==execution,
            'Persisted game execution plan differs')
    eligible=[item for item in calibration['seeds'] if item['gate']['passed'] and item['calibration']['lambda_']>0]
    names=[f'seed_{item["seed"]}' for item in eligible]
    require([item['name'] for item in execution['arms']]==names,'Game candidate membership differs from fixed gates')
    for arm,item in zip(execution['arms'],eligible):
        require(arm['seed']==item['seed'] and arm['checkpoint']==item['checkpoint']
                and arm['sha256']==item['sha256'] and arm['strength']==item['calibration']['lambda_'],
                'Game checkpoint or calibrated correction strength differs')
    records={item['match_id']:item for item in manifest['matches']}
    expected={};expected_jobs=[]
    for block,seed in enumerate(plan['evaluation_seeds']):
        for focal in range(4):
            for arm in ['reference_raw','reference_wrapped',*names,*(('reset_raw','reset_wrapped') if block<64 else ())]:
                name=f'block-{block:03d}-seat-{focal}-{arm}'
                lineup=['']*4;lineup[focal]=arm
                for offset,opponent in enumerate(('shanten_v2','defense_v2','reference_raw'),1):lineup[(focal+offset)%4]=opponent
                expected[name]=dict(match_id=name,block=block,focal_seat=focal,arm=arm,seed=seed,lineup=lineup)
                expected_jobs.append(expected[name])
    require(execution['matches']==expected_jobs and set(records)==set(expected) and len(records)==len(manifest['matches']),
            'Complete game membership differs from fixed candidate/seat/block plan')
    require(execution['planned_matches']==len(records)==evaluation['completed_games']==evaluation['attempted_games'],
            'Game counts differ or failed attempts disappeared')
    total_rows=0
    for name,job in expected.items():
        deadline_check(deadline);record=records[name]
        require(all(record.get(key)==value for key,value in job.items()),'Saved game provenance differs')
        require(record['completed'] is True and record['failure'] is None and record['terminal_result_verified'] is True
                and record['training_eligible'] is False and record['ppo_eligible'] is False,'Incomplete or role-unsafe game record')
        require(sorted(record['ranks'])==[1,2,3,4] and record['utilities']==[2.5-rank for rank in record['ranks']],
                'Terminal average-rank utility differs')
        require(read(root/'reports'/f'{name}.json')==record,'Game report file differs from manifest')
        payload={key:value for key,value in record.items() if key not in ('payload_sha256','elapsed_seconds')}
        require(hashlib.sha256(encoded(payload)).hexdigest()==record['payload_sha256'],'Game deterministic payload hash differs')
        require(read(root/'lifecycle'/f'{name}.started.json')['plan']==job,'Game lifecycle start differs')
        finish=read(root/'lifecycle'/f'{name}.finished.json')
        require(finish['completed'] is True and finish['payload_sha256']==record['payload_sha256']
                and finish['report_sha256']==hashlib.sha256(encoded(record)).hexdigest(),'Game completion receipt differs')
        artifact=record['artifact'];binary=root/artifact['path']
        require(not binary.is_symlink() and binary.resolve().is_relative_to(root.resolve())
                and binary.stat().st_size==artifact['bytes'] and sha(binary)==artifact['sha256'],
                'Auditor game log artifact changed')
        require(len(record['ordinal_wall_fingerprints'])>0,'Missing physical wall commitment')
        require([item['name'] for item in record['policies']]==job['lineup'],'Seat policy provenance differs')
        total_rows+=record['rows']
    require(len(list((root/'lifecycle').glob('*.started.json')))==len(records)
            and len(list((root/'lifecycle').glob('*.finished.json')))==len(records),'Unreported game attempt or lifecycle omission')
    decisions=audit_all_game_decisions(root,records,deadline)
    saved=manifest['statistics'];contrasts={};pairing=[]
    comparisons=[(name,baseline,range(470)) for name in names for baseline in ('reference_wrapped','reference_raw')]
    comparisons.extend([('reference_wrapped','reference_raw',range(470)),('reset_raw','reference_raw',range(64)),
                        ('reset_wrapped','reference_wrapped',range(64)),('reset_wrapped','reset_raw',range(64))])
    for arm,baseline,blocks in comparisons:
        block_values=[];arm_utilities=[];control_utilities=[]
        for block in blocks:
            differences=[]
            for focal in range(4):
                a=records[f'block-{block:03d}-seat-{focal}-{arm}'];b=records[f'block-{block:03d}-seat-{focal}-{baseline}']
                n=min(len(a['ordinal_wall_fingerprints']),len(b['ordinal_wall_fingerprints']))
                require(n>0 and a['ordinal_wall_fingerprints'][:n]==b['ordinal_wall_fingerprints'][:n],
                        'Equal seeds lacked matching explicit physical walls')
                pairing.append(dict(arm=arm,baseline=baseline,block=block,focal_seat=focal,common_hands=n,paired_common_walls_match=True))
                differences.append(a['utilities'][focal]-b['utilities'][focal])
                arm_utilities.append(a['utilities'][focal]);control_utilities.append(b['utilities'][focal])
            block_values.append(float(np.mean(differences)))
        key=arm+'_vs_'+baseline;contrasts[key]=block_values;reported=saved['contrasts'][key]
        require(reported['block_utility_differences']==block_values,'Paired block arithmetic differs')
        near(reported['mean_utility'],np.mean(arm_utilities),'Candidate mean utility differs')
        near(reported['mean_rank'],2.5-np.mean(arm_utilities),'Candidate mean rank differs')
        near(reported['baseline_mean_utility'],np.mean(control_utilities),'Control mean utility differs')
        near(reported['baseline_mean_rank'],2.5-np.mean(control_utilities),'Control mean rank differs')
        computed=statistics(block_values,20261024031)
        for label,value in computed.items():
            if value is None: require(reported['statistics'][label] is None,'Undefined standard error differs')
            else: near(reported['statistics'][label],value,'Reported paired statistics differ: '+label)
        if arm in names and baseline=='reference_wrapped':
            family=3;radius=6*math.sqrt(math.log(2*family/.05)/(2*470));mean=float(np.mean(block_values))
            near(reported['statistics']['family_adjusted_hoeffding_95'],[max(-3.,mean-radius),min(3.,mean+radius)],
                 'Family-adjusted bounded interval differs')
            require(reported['statistics']['family_size_preregistered']==3,'Skipped seed was removed from multiplicity family')
    require(saved['pairing']==pairing,'Wall pairing audit coverage differs')
    if names:
        values=np.mean([contrasts[name+'_vs_reference_wrapped'] for name in names],axis=0)
        pooled=saved['pooled_eligible_seeds'];near(pooled['block_utility_differences'],values,'Pooled seeds were not averaged inside blocks')
        require(pooled['seeds']==[item['seed'] for item in eligible],'Pooled seed membership differs')
        for key,value in statistics(values,20261024032).items():
            if value is not None:near(pooled['statistics'][key],value,'Pooled block statistics differ')
    else:require(saved['pooled_eligible_seeds'] is None,'Pooled comparison fabricated without a candidate')
    require(saved['selected_candidate'] is None and saved['seed_selection'] is False
            and saved['policy_promotion'] is False,'Game outcome used for seed selection or promotion')
    require(saved==evaluation['statistics'],'Summary statistics differ from immutable game manifest')
    return dict(complete=True,games=len(records),rows=total_rows,contrasts=len(contrasts),
                physical_pair_checks=len(pairing),candidate_seeds=[item['seed'] for item in eligible],
                all_lifecycle_and_artifact_hashes_verified=True,no_seed_selection=True,decisions=decisions),execution,records


def replay_selection(execution):
    return sorted(execution['matches'],key=lambda job:hashlib.sha256(
        ('stage3i-audit-replay|'+job['match_id']).encode()).digest())[:32]


def _replay_worker(item):
    from riichi_ai.stage3i_evaluation import run_one
    return run_one(item['job'],item['execution'],item['output'],deadline=item['deadline'])


def replay(plan,evaluation,output_dir,deadline,reserved):
    manifest=read(evaluation['manifest_path']);execution=manifest['execution_plan']
    require(evaluation['complete'] and manifest['complete'],'Cannot replay incomplete formal game panel')
    require(sha(evaluation['manifest_path'])==evaluation['manifest_sha256'],'Replay game manifest changed')
    jobs=replay_selection(execution)
    require(len(jobs)==32 and reserved>=32,'Exactly 32 replay attempts require an explicit reservation')
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    write(root/'replay-plan.json',dict(schema='stage3i-replay-plan-v1',games=jobs,
          source_manifest_sha256=evaluation['manifest_sha256'],selection='SHA256(stage3i-audit-replay|match_id) ascending first32'))
    original={item['match_id']:item for item in manifest['matches']}
    from riichi_ai.evaluation import _terminate_evaluation_pool
    pool=ProcessPoolExecutor(max_workers=14,mp_context=multiprocessing.get_context('spawn'))
    try:
        futures=[pool.submit(_replay_worker,dict(job=job,execution=execution,output=str(root),deadline=deadline)) for job in jobs]
        results=[]
        for future in futures:
            deadline_check(deadline)
            result=future.result(timeout=max(.001,deadline-time.monotonic()))
            require(result['completed'] is True and result['failure'] is None,'Replay failed; no replacement allowed')
            old=original[result['match_id']]
            for key in ('game_sha256','scores','ranks','utilities','ordinal_wall_fingerprints','focal_action_counts','rows','steps','policies'):
                require(result[key]==old[key],'Replay deterministic result differs: '+key)
            require(result['payload_sha256']==old['payload_sha256'],'Replay full deterministic payload differs')
            results.append(dict(match_id=result['match_id'],game_sha256=result['game_sha256'],rows=result['rows'],exact=True))
    except BaseException:
        _terminate_evaluation_pool(pool)
        raise
    else:
        pool.shutdown(wait=True)
    require(len(list((root/'lifecycle').glob('*.started.json')))==32,'Replay attempt count differs')
    return dict(schema='stage3i-independent-replays-v1',complete=True,replays=results,attempts=32,
                all_game_and_action_artifact_payloads_exact=True,
                limitation='Same pinned engine deterministic reproducibility, not cross-engine full rule certification')


def lease(config,mode,max_seconds):
    ledger=read(Path(config['data_root'])/'budget-ledger.json')
    active=[item for item in ledger['operations'] if item['status']=='active']
    require(len(active)==1 and active[0]['name']==os.environ.get('RIICHI_STAGE3I_OPERATION')
            and active[0]['token']==os.environ.get('RIICHI_STAGE3I_TOKEN'),'Matching exclusive Stage 3I audit lease required')
    require(active[0]['name'].startswith('audit'),'Independent audit requires an audit operation')
    receipt=read(Path(config['reports_root'])/('operation-'+active[0]['name']+'.json'))
    reserved=int(os.environ.get('RIICHI_STAGE3I_MATCH_ATTEMPTS','0'))
    require(reserved==receipt['reserved_match_attempts'],'Replay attempt reservation differs from receipt')
    require((mode=='replay' and reserved>=32) or (mode=='evidence' and reserved==0),'Audit operation has incorrect game reservation')
    remaining=active[0]['reserved_seconds']-(time.time()-active[0]['started_unix'])
    require(math.isfinite(max_seconds) and max_seconds>0 and remaining>12,'Invalid audit cooperative deadline')
    return time.monotonic()+min(max_seconds,remaining-10),reserved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('evidence','replay'))
    parser.add_argument('--plan',required=True)
    parser.add_argument('--training-report')
    parser.add_argument('--calibration-report')
    parser.add_argument('--calibration-dir')
    parser.add_argument('--evaluation-report',required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--report',required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    args=parser.parse_args()
    from stage3i import read_config,verify_sources
    config,identity=read_config();deadline,reserved=lease(config,args.mode,args.max_seconds)
    torch.set_num_threads(1)
    plan=read(args.plan);require(plan['config_sha256']==identity,'Audit plan configuration changed')
    require(sha(plan['protocol_path'])==plan['protocol_sha256'],'Audit written protocol changed')
    verify_sources(plan['sources'])
    started=time.monotonic();evaluation=read(args.evaluation_report)
    report=Path(args.report).resolve()
    require(report.is_relative_to(Path(config['reports_root'])) and not report.exists(),'Audit report must be a fresh file in reports root')
    if args.mode=='replay':
        require(args.output_dir is not None,'Replay evidence needs its fresh directory')
        output=Path(args.output_dir).resolve()
        require(output.is_relative_to(Path(config['data_root'])) and output!=Path(config['data_root']) and not output.exists(),
                'Replay directory must be fresh and external')
        result=replay(plan,evaluation,output,deadline,reserved)
    else:
        require(args.training_report and args.calibration_report and args.calibration_dir,'Complete evidence audit needs training/calibration evidence')
        training=read(args.training_report);calibration=read(args.calibration_report)
        require(training['plan_sha256']==sha(args.plan)==calibration['plan_sha256'],'Formal evidence plan binding differs')
        require(calibration['training_report_sha256']==sha(args.training_report),'Calibration training report binding differs')
        cache_summary,cache_manifest,records=audit_cache(plan,deadline)
        train_summary=audit_training(plan,training,deadline)
        calibration_summary=audit_calibration(plan,calibration,cache_manifest,records,args.calibration_dir,deadline)
        game_summary,_,_=audit_games(plan,calibration,evaluation,deadline)
        result=dict(schema='stage3i-independent-evidence-audit-v1',complete=True,cache=cache_summary,
                    training=train_summary,calibration=calibration_summary,games=game_summary)
    for stem in ('reference','call'):
        require(sha(plan[stem+'_checkpoint'])==plan[stem+'_sha256'],'Historical frozen checkpoint changed')
    verify_sources(plan['sources']);deadline_check(deadline)
    result.update(elapsed_seconds=time.monotonic()-started,plan_sha256=sha(args.plan),policy_promotion=False)
    write(report,result)
    print(json.dumps(dict(complete=True,mode=args.mode,elapsed_seconds=result['elapsed_seconds'])),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
