"""Fixed development pilot: scheduling-independent four-seat collection."""
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
from stage3f import read_config, digest, verify_sources


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle: json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')


def read(path): return json.loads(Path(path).read_text())


def worker(job):
    import torch
    from riichi_ai.stage3f_collect import run_one
    torch.set_num_threads(1)
    config=job['config']
    if time.monotonic()>=job['deadline']:
        raise TimeoutError('No new worker admission after deadline')
    return run_one(job['plan'],config['reference_checkpoint'],config['reference_sha256'],
                   config['call_checkpoint'],job['output'],expected_call_sha256=config['call_sha256'],
                   deadline=job['deadline'])


def frozen_sources():
    paths=list((ROOT/'src/riichi_ai').glob('*.py'))
    paths += [ROOT/p for p in ('scripts/stage3f.py','scripts/stage3f_collect.py',
        'scripts/package_a_stability.py','scripts/stage3f_stream_benchmark.py',
        'tests/test_stage3f_controller.py','tests/test_stage3f_collect.py','tests/test_stage3f_stream.py','tests/test_stage3f_scheduling.py',
        'configs/stage3f.json')]
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def prepare(config):
    if (config.get('parity_matches')!=16 or config.get('worker_grid')!=[1,4,8,14]
            or config.get('matches_per_worker_probe')!=32 or config.get('pilot_matches')!=1000
            or config.get('max_match_executions')!=1200):
        raise ValueError('Fixed development counts or worker grid changed')
    from riichi_ai.stage3f_collect import derive_plan
    plans=[derive_plan(config['root_seed'],i) for i in range(1144)]
    new=[p['seed'] for p in plans]+[seed for p in plans for seed in p['sampling_seeds']]
    if len(new)!=len(set(new)):raise ValueError('Development seeds collide')
    # Compare all integer-valued old seed fields without opening historical games.
    historical=set()
    def walk(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items(): walk(v,k)
        elif isinstance(value,list):
            for v in value: walk(v,key)
        elif type(value)is int and 'seed' in key: historical.add(value)
    for path in (ROOT/'configs').glob('*.json'):
        if path.name!='stage3f.json':walk(read(path))
    for path in (ROOT/'reports').rglob('plan.json'):
        if 'stage3f' not in path.parts:walk(read(path))
    if set(new)&historical:raise ValueError('Development seeds overlap historical seeds')
    source=frozen_sources()
    plan=dict(schema='stage3f-fixed-development-plan-v1',config_sha256=digest(ROOT/'configs/stage3f.json'),
        sources=source,parity=plans[:16],worker_probes=[dict(workers=w,matches=plans[16+i*32:48+i*32])
            for i,w in enumerate(config['worker_grid'])],pilot=plans[144:],
        source_role='development-only',training_updates=0,training_eligible=False,ppo_eligible=False,
        fresh_seed_count=len(new),historical_seed_disjoint=True,
        historical_seed_check_scope='Integer seed fields in project configurations and historical reports/**/plan.json; not an exhaustive scan of all historic trajectory files',
        parity_executions=32,worker_probe_executions=128,pilot_executions=1000,
        replay_indices=list(range(0,1000,100)),max_match_executions=config['max_match_executions'])
    return plan


def collect(config,plan,name,matches,workers,deadline):
    if time.monotonic()>=deadline:raise TimeoutError('No new match admission after deadline')
    starts=list(Path(config['data_root']).glob('*/lifecycle/*.started.json'))
    if len(starts)+len(matches)>config['max_match_executions']:
        raise ValueError('Full-match execution ceiling would be exceeded')
    output=Path(config['data_root'])/name
    output.mkdir(parents=True,exist_ok=False)
    reports=[];failure=None
    started=time.perf_counter()
    jobs=[dict(config=config,plan=p,output=str(output),deadline=deadline) for p in matches]
    try:
        if workers==1:
            source=map(worker,jobs)
            for result in source:
                reports.append(result)
                if not result['completed']:raise RuntimeError('Incomplete formal match; preserve and stop')
                if len(reports)%16==0:print(json.dumps({'phase':name,'completed':len(reports),'planned':len(matches)}),flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                pending=deque();source=iter(jobs)
                for _ in range(workers):
                    if time.monotonic()>=deadline:raise TimeoutError('No new match admission after deadline')
                    job=next(source,None)
                    if job:pending.append(pool.submit(worker,job))
                while pending:
                    result=pending.popleft().result();reports.append(result)
                    if not result['completed']:raise RuntimeError('Incomplete formal match; preserve and stop')
                    if len(reports)%16==0:print(json.dumps({'phase':name,'completed':len(reports),'planned':len(matches)}),flush=True)
                    if time.monotonic()>=deadline:raise TimeoutError('No new match admission after deadline')
                    job=next(source,None)
                    if job:pending.append(pool.submit(worker,job))
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error)}
    # Workers may have durably completed additional in-flight jobs before failure.
    saved=[read(path) for path in sorted((output/'reports').glob('match-*.json'))]
    complete=failure is None and len(saved)==len(matches) and all(x['completed'] for x in saved)
    manifest=dict(schema='riichi-ai-stage3f-corpus-v1',complete=complete,source_role='development-only',
        config_sha256=plan['config_sha256'],plan_sha256=digest(Path(config['reports_root'])/'plan.json'),
        phase=name,workers=workers,planned=matches,matches=saved,
        splits={'development':[r['match_id'] for r in saved]},failure=failure,
        elapsed_seconds=time.perf_counter()-started,optimizer_steps=0,training_eligible=False,ppo_eligible=False)
    write_new(output/'manifest.json',manifest)
    summary=dict(phase=name,complete=complete,failure=failure,workers=workers,planned_matches=len(matches),
        completed_matches=sum(r['completed'] for r in saved),rows=sum(r['rows'] for r in saved),
        elapsed_seconds=manifest['elapsed_seconds'],matches_per_second=len(saved)/manifest['elapsed_seconds'],
        compressed_artifact_bytes=sum(a['bytes'] for r in saved for a in r['artifacts'].values()),
        uncompressed_array_bytes=sum(r['uncompressed_actor_bytes']+r['uncompressed_target_bytes'] for r in saved),
        manifest_path=str(output/'manifest.json'),manifest_sha256=digest(output/'manifest.json'))
    write_new(Path(config['reports_root'])/(name+'.json'),summary)
    if not complete:raise RuntimeError('Collection incomplete: '+str(failure))
    return summary


def parity(config):
    data=Path(config['data_root'])
    left,right=read(data/'parity-serial/manifest.json'),read(data/'parity-parallel/manifest.json')
    if not left['complete'] or not right['complete']:raise ValueError('Incomplete parity inputs')
    if len(left['matches'])!=16 or len(right['matches'])!=16:raise ValueError('Wrong parity count')
    for a,b in zip(left['matches'],right['matches']):
        if (a['match_id']!=b['match_id'] or a['payload_sha256']!=b['payload_sha256']
                or a['artifacts']!=b['artifacts'] or a['game_sha256']!=b['game_sha256']):
            raise ValueError('Serial/parallel deterministic payload mismatch')
    result=dict(valid=True,matched_games=16,identical_payloads=True,execution_metadata_excluded=True)
    write_new(Path(config['reports_root'])/'parity.json',result)
    return result


def admission(config,plan):
    reports=Path(config['reports_root'])
    if read(reports/'parity.json')['valid']is not True:raise ValueError('Parity gate required')
    grid=[read(reports/f'workers-{w}.json') for w in config['worker_grid']]
    if not all(r['complete'] and r['completed_matches']==32 for r in grid):raise ValueError('All fixed worker probes required')
    best=max(grid,key=lambda r:(r['matches_per_second'],-r['workers']))
    # Twofold small-pilot uncertainty allowance, plus 30s startup/drain overhead.
    forecast_seconds=2*1000/best['matches_per_second']+30
    per_match=max(r['compressed_artifact_bytes']/r['completed_matches'] for r in grid)
    forecast_bytes=math.ceil(2*per_match*1000)
    existing=sum(p.stat().st_size for p in Path(config['data_root']).rglob('*') if p.is_file())
    budget=read(Path(config['data_root'])/'budget-ledger.json')
    charged=sum(op['reserved_seconds'] if op['status']=='active' else op['elapsed_seconds'] for op in budget['operations'])
    time_fits=forecast_seconds+600+config['reserve_seconds']<=config['limit_seconds']-charged
    storage_fits=existing+forecast_bytes+50_000_000<config['artifact_bytes']
    result=dict(valid=time_fits and storage_fits,workers=best['workers'],pilot_matches=1000,
        basis='Max observed 32-match complete-collector throughput; twofold time/storage allowance, no match-outcome selection',
        forecast_seconds=forecast_seconds,pilot_reservation_seconds=math.ceil(forecast_seconds),
        protected_post_pilot_seconds=600,forecast_new_artifact_bytes=forecast_bytes,
        existing_artifact_bytes=existing,time_fits=time_fits,storage_fits=storage_fits,
        worker_probe_reports={str(r['workers']):digest(reports/f"workers-{r['workers']}.json") for r in grid})
    write_new(reports/'pilot-admission.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','parity-serial','parity-parallel','parity-check','workers-1','workers-4','workers-8','workers-14','admission','pilot'])
    parser.add_argument('--max-seconds',type=float,default=500)
    args=parser.parse_args();config,_=read_config();reports=Path(config['reports_root']);path=reports/'plan.json'
    ledger=read(Path(config['data_root'])/'budget-ledger.json')
    active=[op for op in ledger['operations'] if op['status']=='active']
    if (len(active)!=1 or os.environ.get('RIICHI_STAGE3F_TOKEN')!=active[0]['token']
            or os.environ.get('RIICHI_STAGE3F_OPERATION')!=active[0]['name']):
        raise RuntimeError('Collector requires an active matching numerical lease')
    if args.phase=='prepare':
        result=prepare(config);write_new(path,result)
        print(json.dumps({'phase':'prepare','plans':1144,'sources':len(result['sources'])}));return
    plan=read(path);verify_sources(plan['sources'])
    if plan['config_sha256']!=digest(ROOT/'configs/stage3f.json'):raise ValueError('Configuration changed')
    preflight=read(reports/'operation-release-preflight.json')
    if (preflight['status']!='completed' or any(preflight['sources'].get(k)!=v for k,v in plan['sources'].items())):
        raise ValueError('Source-bound final preflight required')
    order=['parity-serial','parity-parallel','parity-check','workers-1','workers-4','workers-8','workers-14','admission','pilot']
    prior=order[:order.index(args.phase)]
    for name in prior:
        if read(reports/('operation-'+name+'.json'))['status']!='completed':
            raise ValueError('Prior formal phase did not complete')
    remaining=active[0]['reserved_seconds']-(time.time()-active[0]['started_unix'])
    deadline=time.monotonic()+min(args.max_seconds,remaining-10)
    if args.phase=='parity-check':result=parity(config)
    elif args.phase=='admission':result=admission(config,plan)
    elif args.phase.startswith('parity-'):
        result=collect(config,plan,args.phase,plan['parity'],1 if args.phase=='parity-serial' else 4,deadline)
    elif args.phase.startswith('workers-'):
        if read(reports/'parity.json')['valid']is not True:raise ValueError('Parity gate required')
        w=int(args.phase.split('-')[1]);probe=next(x for x in plan['worker_probes'] if x['workers']==w)
        result=collect(config,plan,args.phase,probe['matches'],w,deadline)
    else:
        admission_result=read(reports/'pilot-admission.json')
        if admission_result['valid']is not True:raise ValueError('Pilot admission failed')
        result=collect(config,plan,'pilot',plan['pilot'],admission_result['workers'],deadline)
    verify_sources(plan['sources'])
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
