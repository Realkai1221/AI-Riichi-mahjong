"""Prepare explorer-only, immutable CPU-batch-one features under Stage 3I lease."""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stage3i import read_config, digest, verify_sources

def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def lease(config,seconds,phase):
    active=[o for o in read(Path(config['data_root'])/'budget-ledger.json')['operations'] if o['status']=='active']
    if (len(active)!=1 or os.environ.get('RIICHI_STAGE3I_OPERATION')!=active[0]['name'] or os.environ.get('RIICHI_STAGE3I_TOKEN')!=active[0]['token'] or not active[0]['name'].startswith(phase)):
        raise ValueError('Feature preparation requires matching exclusive numerical lease')
    remain=active[0]['reserved_seconds']-(time.time()-active[0]['started_unix'])
    if not math.isfinite(seconds) or seconds<=0 or remain<15: raise ValueError('Invalid cooperative deadline')
    return time.monotonic()+min(seconds,remain-12)

def validate_partitions(partitions,training_ids,development_ids):
    if not isinstance(partitions,dict) or set(partitions)!={'training','selection','calibration','gate'}:
        raise ValueError('Feature plan must name all four fixed whole-match partitions')
    if any(not isinstance(names,list) or not names or any(not isinstance(name,str) for name in names) for names in partitions.values()):
        raise ValueError('Feature partitions require nonempty match-ID lists')
    members=[name for names in partitions.values() for name in names]
    if len(members)!=len(set(members)):
        raise ValueError('Feature partitions overlap or repeat matches')
    if set(partitions['training'])!=set(training_ids) or set(members)-set(partitions['training'])!=set(development_ids):
        raise ValueError('Feature partition source roles or complete corpus coverage differ')

def initialize(plan):
    global REFERENCE,DATA,TARGETS,PLAN
    import torch
    from riichi_ai.stage3i_model import FrozenReference
    from riichi_ai.stage3g_data import LearningDataset,HandTargets
    torch.set_num_threads(1)
    PLAN=plan
    REFERENCE=FrozenReference.from_checkpoints(plan['reference_checkpoint'],plan['reference_sha256'],plan['call_checkpoint'],plan['call_sha256'],'cpu')
    DATA={split:LearningDataset(plan['corpus_path'],split) for split in ('training','development')}
    validate_partitions(plan['partitions'],DATA['training'].match_ids,DATA['development'].match_ids)
    TARGETS=HandTargets(plan['hand_targets_path'],DATA['training'])
    if DATA['training'].manifest_sha256!=plan['corpus_manifest_sha256'] or TARGETS.sha256!=plan['hand_targets_sha256']: raise ValueError('Historical input identity changed')

def worker(job):
    import numpy as np
    from riichi_ai.stage3f_stream import StreamBatch
    from riichi_ai.stage3i_model import correction_choices
    name,partition,output,deadline=job
    if time.monotonic()>=deadline: raise TimeoutError('Feature deadline')
    dataset=DATA['training' if partition=='training' else 'development']
    if name not in dataset.match_ids or name not in PLAN['partitions'][partition]: raise ValueError('Feature job escaped its declared source role')
    actor,labels,size=dataset._load(name)
    indices=np.flatnonzero((labels['behavior_id']==3)&(np.diff(actor['candidate_offsets'])>1))
    if not len(indices): raise ValueError('Empty explorer match')
    states=[];actions=[];logits=[];kinds=[];probs=[];targets=[];offsets=[0]; max_error=0.
    for row in indices:
        if time.monotonic()>=deadline: raise TimeoutError('Feature deadline')
        aa,tt=dataset._batch(actor,labels,np.asarray([row],dtype=np.int64))
        batch=StreamBatch(aa,tt,name,np.asarray([row],dtype=np.int64),{},0)
        tensors=batch.torch_actor('cpu')
        z,s,a=REFERENCE.frozen_representations(tensors)
        q=REFERENCE.explorer_probabilities(z,s,a,tensors['action_tokens'][...,0],tensors['candidate_mask'])[0]
        original=tt['behavior_probabilities'][0]
        error=float(np.max(np.abs(q-original)));max_error=max(max_error,error)
        if not np.array_equal(q,original): raise ValueError(f'Explorer marginal differs from recorded collection: {name}:{row}, max_error={error}')
        # Explicit identity checks on every retained row, after both kinds of wrapper.
        k=aa['action_tokens'][0,:,0]
        choices=correction_choices(z[0].numpy(),np.zeros(len(q),dtype=np.float32),q,k,0.)
        if choices['raw_corrected_index']!=choices['raw_reference_index'] or choices['selected_index']!=choices['wrapped_reference_index']: raise ValueError('Zero correction identity failed')
        states.append(s[0].numpy());actions.append(a[0].numpy());logits.append(z[0].numpy());kinds.append(k);probs.append(q);offsets.append(offsets[-1]+len(q));targets.append(TARGETS.for_batch(batch)[0])
    REFERENCE.assert_unchanged()
    arrays=dict(state_features=np.stack(states),action_features=np.concatenate(actions),candidate_offsets=np.asarray(offsets,dtype=np.int64),chosen_index=labels['chosen_index'][indices].astype(np.int64),target=np.asarray(targets,dtype=np.float32),row_indices=indices,kinds=np.concatenate(kinds),logits=np.concatenate(logits),explorer_probabilities=np.concatenate(probs),steps=labels['step'][indices],hand_ids=labels['hand_id'][indices],seats=labels['seat'][indices],public_globals=actor['globals'][indices],behavior_probability=labels['behavior_probability'][indices],final_utility=labels['final_utility'][indices])
    path=Path(output)/'shards'/f'{name}.npz'
    with path.open('xb') as stream: np.savez(stream,**arrays)
    return dict(match_id=name,partition=partition,path=str(path.resolve()),sha256=digest(path),bytes=path.stat().st_size,rows=len(indices),candidates=offsets[-1],max_probability_error=max_error,actor_sha256=dataset.reports[name]['artifacts'][f'actor/{name}.npz']['sha256'],targets_sha256=dataset.reports[name]['artifacts'][f'targets/{name}.npz']['sha256'])

def run(plan,output,deadline,mode):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'shards').mkdir()
    parts=plan['partitions']
    if mode=='preflight': parts={key:(value[:16] if key in ('training','selection') else []) for key,value in parts.items()}
    jobs=[(name,part,str(output),deadline) for part,names in parts.items() for name in names]
    results=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=14,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(plan,)) as pool:
        pending=deque(); queue=iter(jobs)
        for _ in range(14):
            job=next(queue,None)
            if job: pending.append(pool.submit(worker,job))
        while pending:
            results.append(pending.popleft().result())
            if len(results)%128==0 or len(results)==len(jobs): print(json.dumps(dict(phase='features',completed=len(results),planned=len(jobs))),flush=True)
            job=next(queue,None)
            if job: pending.append(pool.submit(worker,job))
    rows=sum(r['rows'] for r in results);size=sum(r['bytes'] for r in results);elapsed=time.perf_counter()-started
    result=dict(schema='riichi-ai-stage3i-feature-cache-v1',complete=True,purpose='preflight-only' if mode=='preflight' else 'formal',reference_sha256=plan['reference_sha256'],call_sha256=plan['call_sha256'],corpus_manifest_sha256=plan['corpus_manifest_sha256'],hand_targets_sha256=plan['hand_targets_sha256'],feature_sources=plan['sources'],partitions=parts,shards=results,target='one_hand_delta',behavior_id=3,feature_method='CPU float32, one explorer optional row per actor forward; exact archived collection probabilities',rows=rows,bytes=size,elapsed_seconds=elapsed,rows_per_second=rows/elapsed,max_probability_error=max(r['max_probability_error'] for r in results),zero_correction_identity_rows=rows,full_4000_match_time_forecast_seconds=elapsed*4000/len(jobs),full_4000_match_size_forecast_bytes=size*4000/len(jobs),labels_are_actor_inputs=False)
    write(output/'manifest.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','run']);p.add_argument('--plan',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--report',required=True)
    a=p.parse_args();config,identity=read_config();deadline=lease(config,a.max_seconds,'preflight' if a.mode=='preflight' else 'features')
    plan=read(a.plan)
    if plan['config_sha256']!=identity or plan['schema']!='stage3i-plan-v1': raise ValueError('Plan identity mismatch')
    verify_sources(plan['sources'])
    if digest(Path(plan['corpus_path'])/'manifest.json')!=plan['corpus_manifest_sha256'] or digest(plan['hand_targets_path'])!=plan['hand_targets_sha256']: raise ValueError('Historical source identity mismatch')
    if a.mode=='run' and not plan.get('feature_execution_admitted'): raise ValueError('Measured feature admission required')
    result=run(plan,a.output_dir,deadline,a.mode);write(a.report,result)
if __name__=='__main__':main()
