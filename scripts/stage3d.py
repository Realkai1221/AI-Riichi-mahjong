"""Fixed, evaluation-only Stage 3D recovery; no training or checkpoint writes."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from package_a_stability import Ledger, atomic_json, digest, disk_guard as base_disk_guard, require_worker_lease

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_PROTOCOL = '1fb067d7b76a02eb286c9ab560678a78f8623cc6f59a75cbfa64016fe2936776'
REFERENCE_SHA = '5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f'
BOUNDARIES = {
    'candidate': {'path': '/Users/example/Library/Application Support/RiichiAI/stage3c/boundaries/batch-3.pt', 'sha256': '4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967', 'updates': 4},
    'baseline': {'path': '/Users/example/Library/Application Support/RiichiAI/stage3c/boundaries/initial.pt', 'sha256': '3baca74ddcb011ea7346396621c709e7499835c2cf4b9e6e201688aa52e75bf2', 'updates': 0}}
CONSTRAINTS = dict(optimizer_updates=0, checkpoint_writes=0, book_training=False,
    test_split_loading=False, external_acquisition=False, paid_compute=False,
    policy_promotion=False, failed_block_replacement=False, old_evaluation_pooling=False)
SOURCE_PATHS = ('pyproject.toml', 'requirements-macos-arm64.lock', 'scripts/package_a_stability.py', 'scripts/stage3c.py', 'scripts/stage3d.py', 'src/riichi_ai/__init__.py', 'src/riichi_ai/__main__.py', 'src/riichi_ai/auditing.py', 'src/riichi_ai/benchmark.py', 'src/riichi_ai/cli.py', 'src/riichi_ai/dataset.py', 'src/riichi_ai/dataset_validation.py', 'src/riichi_ai/diagnostics.py', 'src/riichi_ai/efficiency.py', 'src/riichi_ai/evaluation.py', 'src/riichi_ai/features.py', 'src/riichi_ai/model.py', 'src/riichi_ai/package_a_continuation.py', 'src/riichi_ai/package_a_diagnosis_fit.py', 'src/riichi_ai/package_a_diagnosis_precision.py', 'src/riichi_ai/package_a_stability.py', 'src/riichi_ai/player.py', 'src/riichi_ai/policies.py', 'src/riichi_ai/rules.py', 'src/riichi_ai/semantics.py', 'src/riichi_ai/simulator.py', 'src/riichi_ai/stage2b_data.py', 'src/riichi_ai/stage2b_runtime.py', 'src/riichi_ai/stage2b_training.py', 'src/riichi_ai/stage3a.py', 'src/riichi_ai/stage3b.py', 'src/riichi_ai/stage3c.py', 'src/riichi_ai/stage3d.py', 'src/riichi_ai/statistics.py', 'src/riichi_ai/training.py', 'tests/test_stage3c.py', 'tests/test_stage3c_controller.py', 'tests/test_stage3c_postcall.py', 'tests/test_stage3d_controller.py', 'tests/test_stage3d_lifecycle.py')


def seed_for(kind, index):
    return int.from_bytes(hashlib.sha256(f'riichi-ai-stage3d-20261021:{kind}:{index}'.encode()).digest()[:8], 'big')


def planned_evaluation():
    return [{'block': block, 'wall_seed': seed_for('evaluation-wall', block), 'pairs': [
        {'learner_seat': seat, 'sampling_seed': seed_for('evaluation-sampling', block*4+seat),
         'candidate_match_index': 320+block*8+seat*2, 'baseline_match_index':321+block*8+seat*2}
        for seat in range(4)]} for block in range(32)]


def settings(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    fixed = dict(schema='stage3d-evaluation-config-v1', limit_seconds=400, preparation_limit_seconds=45,
        memory_bytes=32*1024**3, artifact_bytes=500_000_000, reserve_seconds=30,
        per_block_allowance_seconds=30, prepare_allowance_seconds=20, test_allowance_seconds=40,
        device='cpu', torch_threads=1, evaluation_blocks=32, maximum_matches=256,
        max_steps=100000, behavior_protocol_sha256=BEHAVIOR_PROTOCOL, reference_sha256=REFERENCE_SHA,
        boundaries=BOUNDARIES, constraints=CONSTRAINTS, source_paths=list(SOURCE_PATHS),
        evaluation=planned_evaluation())
    if any(config.get(k) != v for k,v in fixed.items()):
        raise ValueError('Fixed Stage 3D recovery protocol changed')
    for key in ('data_root','reports_root','reference_checkpoint'):
        config[key] = str((ROOT/config[key]).resolve())
    return path, config, digest(path)


def hash_sources(root, members):
    """Hash ONLY declared paths; unrelated additions never change membership."""
    if not members or len(set(members)) != len(members):
        raise ValueError('Source paths must be nonempty and unique')
    result = {}
    for name in members:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix()!=name:
            raise ValueError('Unsafe declared source path')
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Linked or escaping declared source')
        result[name] = digest(path)
    return result


def implementation(scope='full'):
    members = SOURCE_PATHS if scope=='full' else ('scripts/stage3d.py','scripts/package_a_stability.py')
    return hash_sources(ROOT,members)


def verify_source_identity(expected,root=ROOT):
    if hash_sources(root,tuple(expected)) != expected:
        raise RuntimeError('A declared numerical source changed')


def disk_guard(config):
    total = base_disk_guard(config)
    total += sum(p.stat().st_size for p in (ROOT/'Books/stage3d').rglob('*') if p.is_file())
    if total >= config['artifact_bytes']:
        raise RuntimeError('Stage 3D artifact ceiling reached')
    return total



def write_once_json(path,value):
    """Publish an exclusive lifecycle marker and sync its directory links."""
    path=Path(path)
    directories=[path.parent]
    parent=path.parent
    while not parent.exists():
        parent=parent.parent
        directories.append(parent)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    for directory in directories:
        descriptor=os.open(directory,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)

def verify_inputs(config):
    if digest(config['reference_checkpoint']) != REFERENCE_SHA:
        raise ValueError('Original frozen opponent changed')
    for boundary in BOUNDARIES.values():
        if digest(boundary['path']) != boundary['sha256']:
            raise ValueError('Prespecified Stage 3C endpoint changed')
    previous = json.loads((ROOT/'reports/stage3d/previous-stage3c-snapshot.json').read_text())
    if previous.get('valid') is not True or previous.get('verified_files') != 383:
        raise ValueError('Historical Stage 3C snapshot verification required')


def verify_preflight(reports):
    receipt = json.loads((reports/'checks-release-preflight.json').read_text())
    if receipt['status']!='completed' or receipt['implementation']!=implementation():
        raise ValueError('Current source-bound preflight required')
    xml = ET.parse(reports/'release-preflight.xml')
    if any(xml.findall('.//'+kind) for kind in ('error','failure','skipped')):
        raise ValueError('Preflight failed or skipped')
    cases=xml.findall('.//testcase')
    classes={c.get('classname','').split('.')[-1] for c in cases}
    if not {'test_stage3d_controller','test_stage3d_lifecycle'} <= classes:
        raise ValueError('Missing source-freeze or interruption preflight')
    return {'receipt_sha256':digest(reports/'checks-release-preflight.json'),
            'junit_sha256':digest(reports/'release-preflight.xml'),'passing_cases':len(cases)}


def check_chain(reports,block):
    if type(block) is not int or not 0<=block<32:
        raise ValueError('Exactly 32 sequential blocks')
    if json.loads((reports/'prepare-attempt.json').read_text())['status']!='completed':
        raise ValueError('Successful preparation required')
    if len(list(reports.glob('eval-attempt-*.json'))) != block:
        raise ValueError('No repeated, skipped or replacement blocks')
    for i in range(block):
        if (json.loads((reports/f'eval-attempt-{i}.json').read_text())['status']!='completed'
                or not json.loads((reports/f'eval-{i}.json').read_text())['complete']):
            raise ValueError('Failed block stops this campaign')


def execute(command,allowance,config,evidence,source):
    import psutil  # Numerical/resource runtime imports occur only after reservation.
    started=time.perf_counter();process=None
    evidence.update(sampled_peak_tree_rss_bytes=0,artifact_bytes_peak=0,sample_interval_seconds=.2)
    try:
        process=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
        while True:
            parent=psutil.Process(os.getpid());rss=0
            for child in [parent,*parent.children(recursive=True)]:
                try:rss+=child.memory_info().rss
                except psutil.NoSuchProcess:pass
            evidence['sampled_peak_tree_rss_bytes']=max(evidence['sampled_peak_tree_rss_bytes'],rss)
            evidence['artifact_bytes_peak']=max(evidence['artifact_bytes_peak'],disk_guard(config))
            if rss>=config['memory_bytes']:raise MemoryError('Process-tree RSS ceiling reached')
            verify_source_identity(source)
            if process.poll() is not None:
                if process.returncode:raise RuntimeError(f'Worker returned {process.returncode}')
                return
            if time.perf_counter()-started>=max(.1,allowance-5):
                raise TimeoutError('Numerical operation watchdog deadline reached')
            time.sleep(.2)
    finally:
        if process is not None:
            if process.poll() is None:
                evidence['termination_requested']=True
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    evidence['forced_kill']=True;os.killpg(process.pid,signal.SIGKILL);process.wait()
            evidence['exit_code']=process.returncode
        evidence['elapsed_seconds']=time.perf_counter()-started


def load_states(config):
    from riichi_ai.stage3c import load_boundary,state_manifest
    states={arm:load_boundary(b['path'],expected_protocol_sha256=BEHAVIOR_PROTOCOL,device='cpu') for arm,b in BOUNDARIES.items()}
    manifests={arm:state_manifest(state) for arm,state in states.items()}
    for arm,state in states.items():
        if state.updates!=BOUNDARIES[arm]['updates'] or not manifests[arm]['base_unchanged']:
            raise ValueError('Endpoint update/base identity differs')
    return states,manifests


def worker(config,identity,allowance,phase,block=None):
    started=time.monotonic();deadline=started+allowance-10
    source=implementation();verify_inputs(config)
    preflight=verify_preflight(Path(config['reports_root']))
    import torch
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.stage3c import state_manifest
    torch.set_num_threads(1)
    reports,data=Path(config['reports_root']),Path(config['data_root'])
    states,manifests=load_states(config)
    opponent=NeuralPolicy.from_checkpoint(config['reference_checkpoint'],device='cpu')
    plan_path=reports/'plan.json'
    if phase=='prepare':
        if plan_path.exists():raise FileExistsError('Preserve frozen campaign plan')
        old=json.loads((ROOT/'configs/stage3c.json').read_text())
        old_seeds={m[k] for b in old['batches'] for m in b['matches'] for k in ('seed','sampling_seed')}
        old_seeds|={b['wall_seed'] for b in old['evaluation']}
        old_seeds|={p['sampling_seed'] for b in old['evaluation'] for p in b['pairs']}
        fresh=[b['wall_seed'] for b in config['evaluation']]+[p['sampling_seed'] for b in config['evaluation'] for p in b['pairs']]
        if len(set(fresh))!=160 or old_seeds & set(fresh):raise ValueError('Recovery seeds not fresh and disjoint')
        result=dict(schema='stage3d-plan-v1',config_sha256=identity,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,
            implementation=source,source_membership='explicit immutable list; unrelated additions ignored',
            frozen_before_formal_collection=True,boundaries=BOUNDARIES,states=manifests,opponent=opponent.manifest(),
            evaluation=config['evaluation'],constraints=CONSTRAINTS,preflight_receipt=preflight,
            old_stage3c_seeds_disjoint=True,python=platform.python_version(),platform=platform.platform(),
            versions={n:importlib.metadata.version(n) for n in ('torch','numpy','riichienv','mahjong','psutil')},
            implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        atomic_json(plan_path,result)
        print(json.dumps({'phase':'prepare','plan_sha256':digest(plan_path)}),flush=True)
        return
    plan=json.loads(plan_path.read_text())
    if (plan['config_sha256']!=identity or plan['implementation']!=source or plan['states']!=manifests
            or plan['evaluation']!=config['evaluation']):raise ValueError('Frozen campaign changed')
    from riichi_ai.stage3d import collect_evaluation_match, persist_evaluation_match
    cancelled=[False]
    def request_stop(signum,frame):cancelled[0]=True
    previous_handler=signal.signal(signal.SIGTERM,request_stop)
    planned=config['evaluation'][block]
    result=dict(schema='stage3d-evaluation-block-v1',block=block,config_sha256=identity,
        behavior_protocol_sha256=BEHAVIOR_PROTOCOL,plan_sha256=digest(plan_path),implementation=source,
        complete=False,planned=planned,boundaries=BOUNDARIES,states_before=manifests,
        matches=[],outcomes=[],pairs=[],mean_utility_difference=None)
    lifecycle_root=data/'lifecycle'
    try:
        for pair in planned['pairs']:
            records={}
            for arm in ('candidate','baseline'):
                if cancelled[0] or time.monotonic()>=deadline:raise TimeoutError('Stop before next match admission')
                index=pair[f'{arm}_match_index']
                match=dict(match_index=index,seed=planned['wall_seed'],sampling_seed=pair['sampling_seed'],learner_seat=pair['learner_seat'])
                start_path=lifecycle_root/f'match-{index}-start.json'
                if start_path.exists():raise FileExistsError('Match was already admitted')
                start=dict(schema='stage3d-match-start-v1',campaign_sha256=identity,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,
                    plan_sha256=digest(plan_path),block=block,arm=arm,collection_batch_index=1000+block,
                    planned_match=match,state=manifests[arm],started_unix=time.time())
                write_once_json(start_path,start)
                trajectory=collect_evaluation_match(states[arm],opponent,batch_index=1000+block,**match,
                    max_steps=config['max_steps'],deadline=deadline,stop_requested=lambda:cancelled[0])
                trajectory.record.update(campaign_sha256=identity,campaign_plan_sha256=digest(plan_path),campaign_block=block,campaign_arm=arm)
                evidence=persist_evaluation_match(trajectory,data/'evaluation'/f'block-{block}'/f'match-{index}')
                finish_path=lifecycle_root/f'match-{index}-finish.json'
                finish=dict(schema='stage3d-match-finish-v1',campaign_sha256=identity,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,
                    start_path=str(start_path),start_sha256=digest(start_path),block=block,arm=arm,
                    match_index=index,completed=trajectory.record['completed'],interrupted=trajectory.record.get('interrupted',False),
                    failure=trajectory.record['failure'],evidence=evidence,finished_unix=time.time())
                write_once_json(finish_path,finish)
                entry=dict(arm=arm,learner_seat=pair['learner_seat'],**evidence,
                    lifecycle_start_path=str(start_path),lifecycle_start_sha256=digest(start_path),
                    lifecycle_finish_path=str(finish_path),lifecycle_finish_sha256=digest(finish_path))
                result['matches'].append(entry);records[arm]=trajectory.record
                atomic_json(reports/f'eval-{block}.json',result)
                if not trajectory.record['completed']:raise RuntimeError('Incomplete fixed match; no replacement')
                result['outcomes'].append(dict(arm=arm,learner_seat=pair['learner_seat'],utility=trajectory.record['learner_utility'],rank=trajectory.record['ranks'][pair['learner_seat']]))
                disk_guard(config)
            a,b=records['candidate'],records['baseline'];common=min(len(a['ordinal_wall_fingerprints']),len(b['ordinal_wall_fingerprints']))
            if common==0 or a['ordinal_wall_fingerprints'][:common]!=b['ordinal_wall_fingerprints'][:common]:raise ValueError('Physical paired walls differ')
            result['pairs'].append(dict(learner_seat=pair['learner_seat'],common_hands=common,common_walls_verified=True,
                utility_difference=a['learner_utility']-b['learner_utility']))
        result['states_after']={arm:state_manifest(state) for arm,state in states.items()}
        if result['states_after']!=manifests:raise ValueError('Evaluation altered model or optimizer')
        verify_inputs(config);verify_source_identity(source)
        result['mean_utility_difference']=statistics.mean(p['utility_difference'] for p in result['pairs'])
        result['complete']=True
    except BaseException as error:
        result['failure']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['termination_requested']=cancelled[0]
        atomic_json(reports/f'eval-{block}.json',result)
        signal.signal(signal.SIGTERM,previous_handler)
    print(json.dumps({'phase':'evaluation','block':block,'difference':result['mean_utility_difference']}),flush=True)


def finalize(reports):
    blocks=[]
    for i in range(32):
        receipt=json.loads((reports/f'eval-attempt-{i}.json').read_text())
        block=json.loads((reports/f'eval-{i}.json').read_text())
        if receipt['status']!='completed' or not block['complete']:raise ValueError('All 32 closed valid blocks required')
        blocks.append(block)
    values=[b['mean_utility_difference'] for b in blocks];mean=statistics.mean(values);width=6*math.sqrt(math.log(40)/(2*32))
    result=dict(schema='stage3d-evaluation-v1',blocks=32,matches=256,block_differences=values,mean_utility_difference=mean,
        interval_method='fixed two-sided 95% Hoeffding; independent seeded blocks; D in [-3,3]',halfwidth=width,
        confidence_interval=[max(-3.,mean-width),min(3.,mean+width)],
        candidate_mean_rank=statistics.mean(o['rank'] for b in blocks for o in b['outcomes'] if o['arm']=='candidate'),
        baseline_mean_rank=statistics.mean(o['rank'] for b in blocks for o in b['outcomes'] if o['arm']=='baseline'),
        candidate_promoted=False,old_evaluation_pooled=False,optimizer_updates=0)
    atomic_json(reports/'evaluation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=str(ROOT/'configs/stage3d.json'))
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('budget');commands.add_parser('prepare')
    ev=commands.add_parser('evaluate');ev.add_argument('--block',type=int,required=True)
    check=commands.add_parser('budgeted');check.add_argument('--name',required=True);check.add_argument('--max-seconds',type=float,default=30);check.add_argument('arguments',nargs=argparse.REMAINDER)
    hidden=commands.add_parser('_worker');hidden.add_argument('--phase',choices=('prepare','evaluate'),required=True);hidden.add_argument('--block',type=int);hidden.add_argument('--allowance',type=float,required=True);hidden.add_argument('--lease-token',required=True)
    args=parser.parse_args();path,config,identity=settings(args.config)
    ledger=Ledger(Path(config['data_root'])/'budget-ledger.json',identity,400)
    if args.command=='budget':print(json.dumps(ledger.snapshot(),indent=2));return
    if args.command=='_worker':
        name='prepare' if args.phase=='prepare' else f'eval:{args.block}'
        require_worker_lease(ledger,args.lease_token,args.allowance,name)
        worker(config,identity,args.allowance,args.phase,args.block);return
    reports=Path(config['reports_root']);reports.mkdir(parents=True,exist_ok=True)
    scope='full';block=None
    if args.command=='budgeted':
        if not args.name.replace('-','').replace('_','').isalnum():raise ValueError('Unsafe operation name')
        command=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
        if not command:raise ValueError('Explicit check command required')
        name=f'check:{args.name}';maximum=min(args.max_seconds,config['test_allowance_seconds']);marker=reports/f'checks-{args.name}.json'
        if args.name.startswith('book-'):scope='controller-only'
    elif args.command=='prepare':name='prepare';maximum=config['prepare_allowance_seconds'];marker=reports/'prepare-attempt.json';command=None
    else:
        block=args.block;verify_preflight(reports);check_chain(reports,block)
        name=f'eval:{block}';maximum=config['per_block_allowance_seconds'];marker=reports/f'eval-attempt-{block}.json';command=None
    if marker.exists():raise FileExistsError('Preserve attempts; no overwrite or retry')
    snapshot=ledger.snapshot()
    if not list(reports.glob('eval-attempt-*.json')):
        preparation_remaining=config['preparation_limit_seconds']-snapshot['charged_seconds']
        if args.command=='evaluate':
            if preparation_remaining<0:raise RuntimeError('Preparation cap exceeded')
        else:maximum=min(maximum,preparation_remaining)
    if maximum<=5 or snapshot['remaining_seconds']-config['reserve_seconds']<maximum:
        raise RuntimeError('Insufficient budget for the full declared reservation')
    source=implementation(scope)
    record=dict(name=name,config_sha256=identity,implementation=source,source_scope=scope,status='reserved',started_unix=time.time(),watchdog={})
    atomic_json(marker,record);failure=None
    try:
        with ledger.operation(name,maximum,config['reserve_seconds']) as allowance:
            if command is None:
                lease=next(x for x in ledger.snapshot()['operations'] if x['status']=='active')
                command=[sys.executable,str(Path(__file__).resolve()),'--config',str(path),'_worker','--phase',args.command,'--allowance',str(allowance),'--lease-token',lease['token']]
                if block is not None:command+=['--block',str(block)]
            record['command']=command;execute(command,allowance,config,record['watchdog'],source)
            if digest(path)!=identity:raise RuntimeError('Configuration changed during operation')
            verify_source_identity(source);disk_guard(config)
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error));raise
    finally:
        record.update(status='failed' if failure else 'completed',failure=failure,finished_unix=time.time(),budget=ledger.snapshot())
        atomic_json(marker,record);atomic_json(reports/'budget.json',ledger.snapshot())
    if block==31:print(json.dumps(finalize(reports)),flush=True)
    print(json.dumps({'phase':name,'charged_seconds':ledger.snapshot()['charged_seconds']}),flush=True)


if __name__=='__main__':main()
