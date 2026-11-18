"""Fixed update-4 to update-32 continuation, then one fresh paired evaluation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from package_a_stability import Ledger, atomic_json, digest, disk_guard as base_disk_guard, require_worker_lease
from stage3d import hash_sources, verify_source_identity, write_once_json, SOURCE_PATHS as PREVIOUS_SOURCES

ROOT=Path(__file__).resolve().parents[1]
BEHAVIOR_PROTOCOL='1fb067d7b76a02eb286c9ab560678a78f8623cc6f59a75cbfa64016fe2936776'
REFERENCE_SHA='5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f'
INITIAL_BOUNDARY=dict(path='/Users/example/Library/Application Support/RiichiAI/stage3c/boundaries/batch-3.pt',sha256='4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967',updates=4)
SOURCE_PATHS=tuple(sorted({*PREVIOUS_SOURCES,'scripts/stage3e.py','src/riichi_ai/stage3e.py','tests/test_stage3e_controller.py','tests/test_stage3e_runtime.py'}))
CONSTRAINTS=dict(book_training=False,test_split_loading=False,external_acquisition=False,paid_compute=False,policy_promotion=False,failed_work_replacement=False,old_evaluation_pooling=False,opponent_actions_eligible=False,baseline='zero',entropy_bonus=0.0,reward_shaping=False,replay_reuse=False,base_frozen=True,rank_head_frozen=True,best_checkpoint_selection=False)


def seed_for(kind,index):
    return int.from_bytes(hashlib.sha256(f'riichi-ai-stage3e-20261021:{kind}:{index}'.encode()).digest()[:8],'big')


def planned_batches():
    return [dict(batch_index=index,matches=[dict(match_index=576+(index-4)*16+offset,
        seed=seed_for('training-wall',(index-4)*16+offset),sampling_seed=seed_for('training-sampling',(index-4)*16+offset),
        learner_seat=offset%4) for offset in range(16)]) for index in range(4,32)]


def planned_evaluation():
    return [dict(block=block,wall_seed=seed_for('evaluation-wall',block),pairs=[dict(learner_seat=seat,
        sampling_seed=seed_for('evaluation-sampling',block*4+seat),candidate_match_index=1024+block*8+seat*2,
        baseline_match_index=1025+block*8+seat*2) for seat in range(4)]) for block in range(32)]


def fixed_settings():
    return dict(schema='stage3e-continuation-config-v1',limit_seconds=1200,preparation_limit_seconds=60,
        memory_bytes=32*1024**3,artifact_bytes=1_500_000_000,reserve_seconds=60,
        protected_evaluation_seconds=400,protected_audit_seconds=40,
        per_batch_allowance_seconds=45,per_block_allowance_seconds=35,prepare_allowance_seconds=20,test_allowance_seconds=45,
        device='cpu',torch_threads=1,starting_updates=4,maximum_updates=32,maximum_batches=28,matches_per_batch=16,
        training_matches=448,evaluation_matches=256,evaluation_blocks=32,maximum_matches=704,
        learning_rate=1e-4,weight_decay=0.0,gradient_clip=1.0,microbatch_size=1,replay_atol=0.,replay_rtol=0.,replay_log_atol=0.,
        max_steps=100000,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,reference_sha256=REFERENCE_SHA,
        initial_boundary=INITIAL_BOUNDARY,constraints=CONSTRAINTS,source_paths=list(SOURCE_PATHS),
        batches=planned_batches(),evaluation=planned_evaluation())


def settings(path):
    path=Path(path).resolve();config=json.loads(path.read_text())
    if any(config.get(k)!=v for k,v in fixed_settings().items()):raise ValueError('Fixed Stage 3E protocol changed')
    for key in ('data_root','reports_root','reference_checkpoint'):config[key]=str((ROOT/config[key]).resolve())
    return path,config,digest(path)


def implementation(scope='full'):
    return hash_sources(ROOT,SOURCE_PATHS if scope=='full' else ('scripts/stage3e.py','scripts/stage3d.py','scripts/package_a_stability.py'))


def disk_guard(config):
    total=base_disk_guard(config)+sum(p.stat().st_size for p in (ROOT/'Books/stage3e').rglob('*') if p.is_file())
    if total>=config['artifact_bytes']:raise RuntimeError('Stage 3E artifact ceiling reached')
    return total


def operation_reserve(config,phase):
    return config['reserve_seconds']+(config['protected_evaluation_seconds']+config['protected_audit_seconds'] if phase=='train'
        else config['protected_audit_seconds'] if phase=='evaluate' else 0)


def verify_inputs(config):
    if digest(config['reference_checkpoint'])!=REFERENCE_SHA or digest(INITIAL_BOUNDARY['path'])!=INITIAL_BOUNDARY['sha256']:
        raise ValueError('Original checkpoint changed')
    old=json.loads((ROOT/'reports/stage3e/previous-stage3d-snapshot.json').read_text())
    if old.get('valid') is not True or old.get('verified_files')!=1216:raise ValueError('Historical Stage 3D snapshot must verify')


def verify_preflight(reports):
    path=reports/'checks-release-preflight.json';receipt=json.loads(path.read_text())
    if receipt['status']!='completed' or receipt['implementation']!=implementation():raise ValueError('Current source-bound preflight required')
    xml=ET.parse(reports/'release-preflight.xml')
    if any(xml.findall('.//'+k) for k in ('error','failure','skipped')):raise ValueError('Preflight failed or skipped')
    cases=xml.findall('.//testcase');classes={c.get('classname','').split('.')[-1] for c in cases}
    if not {'test_stage3e_controller','test_stage3e_runtime','test_stage3c_postcall'}<=classes:raise ValueError('Missing continuation/runtime/postcall coverage')
    return dict(receipt_sha256=digest(path),junit_sha256=digest(reports/'release-preflight.xml'),passing_cases=len(cases))


def verify_fresh_seeds(config):
    def seeds(c):
        values=[m[k] for b in c.get('batches',[]) for m in b['matches'] for k in ('seed','sampling_seed')]
        values += [b['wall_seed'] for b in c['evaluation']]
        values += [p['sampling_seed'] for b in c['evaluation'] for p in b['pairs']]
        return values
    previous=set()
    for stage in ('stage3c','stage3d'):previous.update(seeds(json.loads((ROOT/f'configs/{stage}.json').read_text())))
    fresh=seeds(config)
    if len(fresh)!=1056 or len(set(fresh))!=len(fresh) or previous&set(fresh):raise ValueError('Fresh seeds overlap or repeat')
    return dict(fresh_seed_count=len(fresh),stage3c_stage3d_disjoint=True)


def check_chain(reports,phase,index):
    if json.loads((reports/'prepare-attempt.json').read_text())['status']!='completed':raise ValueError('Successful frozen preparation required')
    def require_batch(i):
        if (json.loads((reports/f'attempt-{i}.json').read_text())['status']!='completed'
            or not json.loads((reports/f'batch-{i}.json').read_text())['complete']):raise ValueError('Failed training batch stops campaign')
    attempts=list(reports.glob('attempt-*.json'))
    if phase=='train':
        if type(index)is not int or not 4<=index<32 or len(attempts)!=index-4:raise ValueError('No repeat, replacement or skipped training batch')
        if list(reports.glob('eval-attempt-*.json')):raise ValueError('No training after evaluation')
        for i in range(4,index):require_batch(i)
    else:
        if type(index)is not int or not 0<=index<32 or len(attempts)!=28:raise ValueError('Fixed endpoint required before evaluation')
        for i in range(4,32):require_batch(i)
        if len(list(reports.glob('eval-attempt-*.json')))!=index:raise ValueError('No repeat, replacement or skipped evaluation block')
        for i in range(index):
            if (json.loads((reports/f'eval-attempt-{i}.json').read_text())['status']!='completed'
                or not json.loads((reports/f'eval-{i}.json').read_text())['complete']):raise ValueError('Failed evaluation stops campaign')


def execute(command,allowance,config,evidence,source):
    import psutil
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
            if time.perf_counter()-started>=max(.1,allowance-5):raise TimeoutError('Numerical operation watchdog deadline reached')
            time.sleep(.2)
    finally:
        if process is not None:
            if process.poll() is None:
                evidence['termination_requested']=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:evidence['forced_kill']=True;os.killpg(process.pid,signal.SIGKILL);process.wait()
            evidence['exit_code']=process.returncode
        evidence['elapsed_seconds']=time.perf_counter()-started


def load_fixed_boundary(boundary):
    from riichi_ai.stage3c import load_boundary,state_manifest
    if digest(boundary['path'])!=boundary['sha256']:raise ValueError('Boundary file changed')
    state=load_boundary(boundary['path'],expected_protocol_sha256=BEHAVIOR_PROTOCOL,device='cpu')
    manifest=state_manifest(state)
    if not manifest['base_unchanged'] or state.updates!=boundary['updates']:raise ValueError('Boundary update/base mismatch')
    return state,manifest


def worker(config,identity,allowance,phase,index=None):
    deadline=time.monotonic()+allowance-10;source=implementation();verify_inputs(config)
    reports,data=Path(config['reports_root']),Path(config['data_root']);preflight=verify_preflight(reports)
    import torch
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.stage3c import state_manifest,save_boundary
    from riichi_ai.stage3e import collect_match,persist_match,update_batch
    torch.set_num_threads(1)
    opponent=NeuralPolicy.from_checkpoint(config['reference_checkpoint'],device='cpu')
    plan_path=reports/'plan.json'
    if phase=='prepare':
        if plan_path.exists():raise FileExistsError('Preserve frozen continuation plan')
        state,manifest=load_fixed_boundary(INITIAL_BOUNDARY)
        if (manifest['retired_batches']!=list(range(4)) or manifest['optimizer_steps_executed']!=4
            or state.optimizer.param_groups[0]['lr']!=1e-4 or state.optimizer.param_groups[0]['weight_decay']!=0.):raise ValueError('Original optimizer continuation differs')
        old=json.loads((ROOT/'reports/stage3c/batch-3.json').read_text())
        if manifest!=old['output_state']:raise ValueError('Restored model/optimizer differs from saved update 4')
        plan=dict(schema='stage3e-plan-v1',config_sha256=identity,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,
            implementation=source,frozen_before_formal_collection=True,initial_boundary=INITIAL_BOUNDARY,initial_state=manifest,
            batches=config['batches'],evaluation=config['evaluation'],constraints=CONSTRAINTS,opponent=opponent.manifest(),
            seed_audit=verify_fresh_seeds(config),preflight_receipt=preflight,python=platform.python_version(),platform=platform.platform(),
            versions={n:importlib.metadata.version(n) for n in ('torch','numpy','riichienv','mahjong','psutil')},
            implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        atomic_json(plan_path,plan);print(json.dumps({'phase':'prepare','plan_sha256':digest(plan_path)}),flush=True);return
    plan=json.loads(plan_path.read_text());plan_sha=digest(plan_path)
    if plan['config_sha256']!=identity or plan['implementation']!=source or plan['batches']!=config['batches'] or plan['evaluation']!=config['evaluation']:
        raise ValueError('Frozen continuation plan differs')
    cancelled=[False]
    def stop(signum,frame):cancelled[0]=True
    old_handler=signal.signal(signal.SIGTERM,stop)
    def admit(state,manifest,match,*,role,batch_index,arm,block=None):
        if cancelled[0] or time.monotonic()>=deadline:raise TimeoutError('Stop before next match admission')
        mid=match['match_index'];start_path=data/'lifecycle'/f'match-{mid}-start.json'
        start=dict(schema='stage3e-match-start-v1',campaign_sha256=identity,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,
            plan_sha256=plan_sha,phase=role,block=block,arm=arm,collection_batch_index=batch_index,planned_match=match,state=manifest,started_unix=time.time())
        write_once_json(start_path,start)
        trajectory=collect_match(state,opponent,collection_role=role,batch_index=batch_index,**match,
            max_steps=config['max_steps'],deadline=deadline,stop_requested=lambda:cancelled[0])
        trajectory.record.update(campaign_sha256=identity,campaign_plan_sha256=plan_sha,campaign_phase=role,
            campaign_batch_index=batch_index,campaign_block=block,campaign_arm=arm)
        directory=data/('trajectories' if role=='training' else 'evaluation')/(f'batch-{batch_index}' if role=='training' else f'block-{block}')/f'match-{mid}'
        evidence=persist_match(trajectory,directory)
        finish_path=data/'lifecycle'/f'match-{mid}-finish.json'
        finish=dict(schema='stage3e-match-finish-v1',campaign_sha256=identity,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,
            start_path=str(start_path),start_sha256=digest(start_path),phase=role,block=block,arm=arm,
            match_index=mid,completed=trajectory.record['completed'],interrupted=trajectory.record['interrupted'],
            failure=trajectory.record['failure'],evidence=evidence,finished_unix=time.time())
        write_once_json(finish_path,finish)
        entry=dict(arm=arm,learner_seat=match['learner_seat'],**evidence,lifecycle_start_path=str(start_path),
            lifecycle_start_sha256=digest(start_path),lifecycle_finish_path=str(finish_path),lifecycle_finish_sha256=digest(finish_path))
        return trajectory,entry
    result=None;output_path=None
    try:
        if phase=='train':
            previous=INITIAL_BOUNDARY if index==4 else json.loads((reports/f'batch-{index-1}.json').read_text())['output_boundary']
            state,manifest=load_fixed_boundary(previous)
            if state.updates!=index or state.retired_batches!=list(range(index)):raise ValueError('Continuation chain is not exact')
            planned=config['batches'][index-4]['matches'];output_path=reports/f'batch-{index}.json'
            result=dict(schema='stage3e-batch-v1',batch_index=index,config_sha256=identity,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,
                plan_sha256=plan_sha,implementation=source,complete=False,input_boundary=previous,input_state=manifest,
                planned_matches=planned,opponent=opponent.manifest(),matches=[],update=None,output_boundary=None)
            trajectories=[]
            for match in planned:
                trajectory,evidence=admit(state,manifest,match,role='training',batch_index=index,arm='learner')
                result['matches'].append(evidence);trajectories.append(trajectory);atomic_json(output_path,result)
                if not trajectory.record['completed']:raise RuntimeError('Incomplete fixed training match; no update/replacement')
            if cancelled[0]:raise TimeoutError('Cancellation before update')
            if any(t.record['campaign_sha256']!=identity or t.record['campaign_plan_sha256']!=plan_sha for t in trajectories):raise ValueError('Foreign training campaign')
            result['update']=update_batch(state,trajectories,batch_index=index,planned_matches=planned,gradient_clip=1.,
                replay_atol=0.,replay_rtol=0.,replay_log_atol=0.,microbatch_size=1,deadline=deadline)
            result['output_boundary']=save_boundary(state,data/'boundaries'/f'batch-{index}.pt')
            result['output_boundary'].update(updates=state.updates)
            with Path(result['output_boundary']['path']).open('rb') as stream:os.fsync(stream.fileno())
            for directory in (data/'boundaries',data):
                descriptor=os.open(directory,os.O_RDONLY)
                try:os.fsync(descriptor)
                finally:os.close(descriptor)
            restored,after=load_fixed_boundary(result['output_boundary'])
            result['output_state']=state_manifest(state);result['boundary_roundtrip_verified']=after==result['output_state']
            if not result['boundary_roundtrip_verified'] or state.updates!=index+1:raise RuntimeError('Updated boundary round trip differs')
            if not result['update']['distribution_diagnostic_complete']:raise RuntimeError('Post-update diagnostics incomplete; stop campaign')
            result['complete']=True
        else:
            final=json.loads((reports/'batch-31.json').read_text())['output_boundary'];boundaries=dict(candidate=final,baseline=INITIAL_BOUNDARY)
            loaded={a:load_fixed_boundary(b) for a,b in boundaries.items()};states={a:v[0] for a,v in loaded.items()};manifests={a:v[1] for a,v in loaded.items()}
            if states['candidate'].updates!=32 or states['baseline'].updates!=4:raise ValueError('Wrong fixed endpoints')
            block=index;planned=config['evaluation'][block];output_path=reports/f'eval-{block}.json'
            result=dict(schema='stage3e-evaluation-block-v1',block=block,config_sha256=identity,behavior_protocol_sha256=BEHAVIOR_PROTOCOL,
                plan_sha256=plan_sha,implementation=source,complete=False,planned=planned,boundaries=boundaries,states_before=manifests,
                matches=[],outcomes=[],pairs=[],mean_utility_difference=None)
            for pair in planned['pairs']:
                records={}
                for arm in ('candidate','baseline'):
                    match=dict(match_index=pair[f'{arm}_match_index'],seed=planned['wall_seed'],sampling_seed=pair['sampling_seed'],learner_seat=pair['learner_seat'])
                    trajectory,evidence=admit(states[arm],manifests[arm],match,role='evaluation',batch_index=2000+block,arm=arm,block=block)
                    result['matches'].append(evidence);records[arm]=trajectory.record;atomic_json(output_path,result)
                    if not trajectory.record['completed']:raise RuntimeError('Incomplete fixed evaluation match; no replacement')
                    result['outcomes'].append(dict(arm=arm,learner_seat=pair['learner_seat'],rank=trajectory.record['ranks'][pair['learner_seat']],utility=trajectory.record['learner_utility'],scores=trajectory.record['scores']))
                first,second=records['candidate'],records['baseline'];common=min(len(first['ordinal_wall_fingerprints']),len(second['ordinal_wall_fingerprints']))
                if not common or first['ordinal_wall_fingerprints'][:common]!=second['ordinal_wall_fingerprints'][:common]:raise ValueError('Paired physical walls differ')
                result['pairs'].append(dict(learner_seat=pair['learner_seat'],utility_difference=first['learner_utility']-second['learner_utility'],common_ordinal_walls=common))
            # Also verify shared physical walls across the four seat rotations.
            all_records=[json.loads(Path(e['json_path']).read_text()) for e in result['matches']]
            walls=all_records[0]['ordinal_wall_fingerprints']
            for record in all_records[1:]:
                common=min(len(walls),len(record['ordinal_wall_fingerprints']))
                if not common or walls[:common]!=record['ordinal_wall_fingerprints'][:common]:raise ValueError('Rotated-seat wall commitments differ')
            result['states_after']={a:state_manifest(s) for a,s in states.items()}
            if result['states_after']!=manifests:raise ValueError('Fixed evaluation endpoints changed')
            for b in boundaries.values():
                if digest(b['path'])!=b['sha256']:raise ValueError('Checkpoint file changed')
            result['mean_utility_difference']=statistics.mean(p['utility_difference'] for p in result['pairs']);result['complete']=True
    except BaseException as error:
        if result is not None:result.update(complete=False,failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        signal.signal(signal.SIGTERM,old_handler)
        if result is not None:
            if phase=='train':
                result['final_in_memory_state']=state_manifest(state)
                result['optimizer_steps_executed_delta']=state.optimizer_steps_executed-manifest['optimizer_steps_executed']
                result['completed_update_count_delta']=state.updates-manifest['updates']
            result['sources_unchanged']=implementation()==source
            result['termination_requested']=cancelled[0];atomic_json(output_path,result)


def finalize(reports):
    blocks=[]
    for i in range(32):
        receipt=json.loads((reports/f'eval-attempt-{i}.json').read_text());block=json.loads((reports/f'eval-{i}.json').read_text())
        if receipt['status']!='completed' or not block['complete']:raise ValueError('All32successfulclosedblocks required')
        blocks.append(block)
    values=[b['mean_utility_difference'] for b in blocks];mean=statistics.mean(values);width=6*math.sqrt(math.log(40)/64)
    result=dict(schema='stage3e-evaluation-v1',blocks=32,matches=256,block_differences=values,mean_utility_difference=mean,
        interval_method='fixed two-sided 95% Hoeffding; independent seeded blocks; D in [-3,3]',halfwidth=width,
        confidence_interval=[max(-3.,mean-width),min(3.,mean+width)],
        candidate_mean_rank=statistics.mean(o['rank'] for b in blocks for o in b['outcomes'] if o['arm']=='candidate'),
        baseline_mean_rank=statistics.mean(o['rank'] for b in blocks for o in b['outcomes'] if o['arm']=='baseline'),
        candidate_promoted=False,old_evaluation_pooled=False,training_updates=28,evaluation_optimizer_updates=0)
    atomic_json(reports/'evaluation.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=str(ROOT/'configs/stage3e.json'))
    commands=parser.add_subparsers(dest='command',required=True);commands.add_parser('budget');commands.add_parser('prepare')
    train=commands.add_parser('train');train.add_argument('--batch',type=int,required=True)
    evaluate=commands.add_parser('evaluate');evaluate.add_argument('--block',type=int,required=True)
    check=commands.add_parser('budgeted');check.add_argument('--name',required=True);check.add_argument('--max-seconds',type=float,default=30);check.add_argument('arguments',nargs=argparse.REMAINDER)
    hidden=commands.add_parser('_worker');hidden.add_argument('--phase',choices=('prepare','train','evaluate'),required=True);hidden.add_argument('--index',type=int);hidden.add_argument('--allowance',type=float,required=True);hidden.add_argument('--lease-token',required=True)
    args=parser.parse_args();path,config,identity=settings(args.config);reports=Path(config['reports_root'])
    ledger=Ledger(Path(config['data_root'])/'budget-ledger.json',identity,1200)
    if args.command=='budget':print(json.dumps(ledger.snapshot(),indent=2));return
    if args.command=='_worker':
        name='prepare' if args.phase=='prepare' else f"{'batch' if args.phase=='train' else 'eval'}:{args.index}"
        require_worker_lease(ledger,args.lease_token,args.allowance,name);worker(config,identity,args.allowance,args.phase,args.index);return
    reports.mkdir(parents=True,exist_ok=True);index=None;command=None
    if args.command=='budgeted':
        if not args.name.replace('-','').replace('_','').isalnum():raise ValueError('Unsafe check name')
        command=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
        if not command:raise ValueError('Explicit check command required')
        name=f'check:{args.name}';maximum=min(args.max_seconds,config['test_allowance_seconds']);marker=reports/f'checks-{args.name}.json'
    elif args.command=='prepare':name='prepare';maximum=config['prepare_allowance_seconds'];marker=reports/'prepare-attempt.json'
    else:
        index=args.batch if args.command=='train' else args.block;verify_preflight(reports);check_chain(reports,args.command,index)
        name=f"{'batch' if args.command=='train' else 'eval'}:{index}";maximum=config['per_batch_allowance_seconds' if args.command=='train' else 'per_block_allowance_seconds']
        marker=reports/(f'attempt-{index}.json' if args.command=='train' else f'eval-attempt-{index}.json')
    if marker.exists():raise FileExistsError('Preserve every attempt; no overwrite or retry')
    snapshot=ledger.snapshot();reserve=operation_reserve(config,args.command)
    if not list(reports.glob('attempt-*.json')):
        remaining=config['preparation_limit_seconds']-snapshot['charged_seconds']
        if args.command=='train':
            if remaining<0:raise RuntimeError('Preparation cap exceeded')
        else:maximum=min(maximum,remaining)
    if maximum<=5 or snapshot['remaining_seconds']-reserve<maximum:raise RuntimeError('Insufficient reserved capacity; stop without substitution')
    source=implementation();record=dict(name=name,config_sha256=identity,implementation=source,source_scope='full',status='reserved',
        protected_reserve_seconds=reserve,started_unix=time.time(),watchdog={})
    atomic_json(marker,record);failure=None
    try:
        with ledger.operation(name,maximum,reserve) as allowance:
            if command is None:
                lease=next(o for o in ledger.snapshot()['operations'] if o['status']=='active')
                command=[sys.executable,str(Path(__file__).resolve()),'--config',str(path),'_worker','--phase',args.command,'--allowance',str(allowance),'--lease-token',lease['token']]
                if index is not None:command+=['--index',str(index)]
            record['command']=command;execute(command,allowance,config,record['watchdog'],source)
            if digest(path)!=identity:raise RuntimeError('Configuration changed')
            verify_source_identity(source);disk_guard(config)
    except BaseException as error:failure=dict(type=type(error).__name__,message=str(error));raise
    finally:
        record.update(status='failed' if failure else 'completed',failure=failure,finished_unix=time.time(),budget=ledger.snapshot())
        atomic_json(marker,record);atomic_json(reports/'budget.json',ledger.snapshot())
    if args.command=='evaluate' and index==31:print(json.dumps(finalize(reports)),flush=True)
    print(json.dumps(dict(phase=name,charged_seconds=ledger.snapshot()['charged_seconds'])),flush=True)

if __name__=='__main__':main()
