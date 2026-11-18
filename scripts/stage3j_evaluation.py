"""Execute the fixed Stage 3J paired panel with saved Stage 3I heads."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    from riichi_ai.stage3g_collect import _write
    from riichi_ai.stage3j_evaluation import json_bytes
    _write(Path(path),json_bytes(value))


def require_lease(config,max_seconds,mode):
    active = [op for op in read(Path(config['data_root'])/'budget-ledger.json')['operations'] if op['status']=='active']
    if (len(active)!=1 or os.environ.get('RIICHI_STAGE3J_OPERATION')!=active[0]['name']
            or os.environ.get('RIICHI_STAGE3J_TOKEN')!=active[0]['token']):
        raise ValueError('Evaluation requires its matching active Stage 3J numerical lease')
    expected = {'preflight':('preflight',),'formal':('evaluat',)}[mode]
    if not active[0]['name'].startswith(expected):
        raise ValueError('Evaluation mode does not match active phase')
    if not math.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError('Invalid cooperative deadline')
    remaining = active[0]['reserved_seconds']-(time.time()-active[0]['started_unix'])
    receipt = read(Path(config['reports_root'])/('operation-'+active[0]['name']+'.json'))
    if int(os.environ.get('RIICHI_STAGE3J_MATCH_ATTEMPTS','0')) != receipt.get('reserved_match_attempts',0):
        raise ValueError('Game attempt reservation differs from active ledger receipt')
    if remaining<=15:
        raise TimeoutError('Insufficient evaluation lease remains')
    return time.monotonic()+min(max_seconds,remaining-12)


_WORKER_CONTEXT = None


def initialize_worker(execution, output, deadline):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = dict(execution=execution, output=output, deadline=deadline)


def worker(job):
    from riichi_ai.stage3j_evaluation import run_one
    item = _WORKER_CONTEXT
    if item is None or time.monotonic()>=item['deadline']:
        raise TimeoutError('No new game may start after deadline')
    return run_one(job,item['execution'],item['output'],deadline=item['deadline'])


def run(execution,output,config,deadline):
    from riichi_ai.evaluation import _terminate_evaluation_pool
    from riichi_ai.stage3j_evaluation import summarize,digest
    root = Path(config['data_root'])
    starts = sum(1 for _ in root.glob('**/lifecycle/*.started.json'))
    reserved_attempts = int(os.environ.get('RIICHI_STAGE3J_MATCH_ATTEMPTS', '0'))
    if reserved_attempts < execution['planned_matches']:
        raise ValueError('Game count exceeds this operation\'s reserved attempt ceiling')
    if starts+execution['planned_matches']>config['max_match_executions']:
        raise ValueError('Games would exceed the global attempt ceiling')
    if time.monotonic()>=deadline:
        raise TimeoutError('Expired game admission')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'execution-plan.json',execution)
    jobs = iter(execution['matches'])
    started,failure,finished = time.perf_counter(),None,0
    pool = ProcessPoolExecutor(max_workers=execution['workers'],mp_context=multiprocessing.get_context('spawn'),
                               initializer=initialize_worker,initargs=(execution,str(output),deadline))
    try:
        pending = deque()
        for _ in range(execution['workers']):
            if time.monotonic()>=deadline:
                raise TimeoutError('No new submission after deadline')
            job = next(jobs,None)
            if job:
                pending.append(pool.submit(worker,job))
        while pending:
            remaining = deadline-time.monotonic()
            if remaining<=0:
                raise TimeoutError('Game panel deadline')
            result = pending.popleft().result(timeout=remaining)
            finished+=1
            if not result['completed']:
                raise RuntimeError('Game failed: preserve failure, stop, never replace')
            if finished%128==0 or finished==execution['planned_matches']:
                print(json.dumps(dict(phase='evaluation',completed=finished,planned=execution['planned_matches'])),flush=True)
            if time.monotonic()>=deadline:
                raise TimeoutError('No new submission after deadline')
            job = next(jobs,None)
            if job:
                pending.append(pool.submit(worker,job))
    except BaseException as error:
        failure = dict(type=type(error).__name__,message=str(error))
        _terminate_evaluation_pool(pool)
    else:
        pool.shutdown(wait=True)
    records = [read(path) for path in sorted((output/'reports').glob('block-*.json'))]
    summary = None
    if failure is None:
        try:
            summary = summarize(execution,records)
        except Exception as error:
            failure = dict(type=type(error).__name__,message=str(error))
    complete = failure is None and summary is not None
    elapsed = time.perf_counter()-started
    manifest = dict(schema='riichi-ai-stage3j-evaluation-v1',complete=complete,
                    execution_plan_sha256=digest(output/'execution-plan.json'),execution_plan=execution,
                    matches=records,failure=failure,statistics=summary,elapsed_seconds=elapsed,
                    training_eligible=False,ppo_eligible=False,policy_promotion=False,seed_selection=False)
    write_new(output/'manifest.json',manifest)
    return dict(schema='stage3j-evaluation-summary-v1',complete=complete,failure=failure,
                planned_games=execution['planned_matches'],completed_games=sum(r['completed'] for r in records),
                attempted_games=sum(1 for _ in (output/'lifecycle').glob('*.started.json')),
                rows=sum(r['rows'] for r in records),elapsed_seconds=elapsed,
                matches_per_second=len(records)/elapsed,statistics=summary,
                manifest_path=str(output/'manifest.json'),manifest_sha256=digest(output/'manifest.json'),
                training_eligible=False,policy_promotion=False,seed_selection=False,confirmation_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','formal'))
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    args = parser.parse_args()
    from stage3j import read_config,digest,verify_sources
    from riichi_ai.stage3j_evaluation import prepare_execution,validate_policies,validate_dependencies
    config,identity = read_config()
    deadline = require_lease(config,args.max_seconds,args.mode)
    plan = read(args.plan)
    if plan.get('config_sha256')!=identity:
        raise ValueError('Frozen plan or configuration changed')
    if args.mode != 'preflight' and plan.get('execution_admitted') is not True:
        raise ValueError('Formal evaluation has not been admitted')
    if digest(plan['protocol_path']) != plan['protocol_sha256']:
        raise ValueError('Written protocol identity changed')
    verify_sources(plan['sources'])
    validate_dependencies(plan,config)
    output,report = Path(args.output_dir).resolve(),Path(args.report).resolve()
    if (output==Path(config['data_root']) or not output.is_relative_to(Path(config['data_root']))
            or not report.is_relative_to(Path(config['reports_root'])) or report.exists()):
        raise ValueError('Fresh outputs must remain in the declared evidence roots')
    execution = prepare_execution(plan,config,plan_sha256=digest(args.plan),preflight=args.mode=='preflight')
    execution['authenticated_policies'] = validate_policies(execution)
    result = run(execution,output,config,deadline)
    verify_sources(plan['sources'])
    write_new(report,result)
    print(json.dumps(dict(complete=result['complete'],phase=args.mode,
                         games=result.get('completed_games'),seeds=len(result.get('seeds',[])))),flush=True)
    if not result['complete']:
        raise RuntimeError('Incomplete Stage 3J evidence: '+str(result.get('failure')))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
