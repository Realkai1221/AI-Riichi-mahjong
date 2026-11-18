"""Leased Stage 3A: four fixed full-match policy-gradient batches.

Only standard-library code runs before a durable numerical lease. Failed formal
batches are preserved and stop the chain; this launcher never substitutes games.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from package_a_stability import Ledger, atomic_json, digest, disk_guard, require_worker_lease

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA = '5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f'


def seed_for(kind, index):
    return int.from_bytes(hashlib.sha256(f'riichi-ai-stage3a-20261018:{kind}:{index}'.encode()).digest()[:8], 'big')


def planned_batches():
    return [{'batch_index': batch, 'matches': [
        {'match_index': batch * 16 + offset, 'seed': seed_for('wall', batch * 16 + offset),
         'sampling_seed': seed_for('sampling', batch * 16 + offset), 'learner_seat': offset % 4}
        for offset in range(16)]} for batch in range(4)]


def settings(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    exact = dict(limit_seconds=600, memory_bytes=32 * 1024**3, artifact_bytes=1_000_000_000,
        reserve_seconds=45, per_batch_allowance_seconds=100, prepare_allowance_seconds=30,
        test_allowance_seconds=90, device='cpu', torch_threads=1, maximum_batches=4,
        matches_per_batch=16, maximum_matches=64, maximum_updates=4, learning_rate=1e-5,
        weight_decay=0.0, gradient_clip=1.0, replay_atol=1e-6, replay_rtol=1e-5, replay_log_atol=1e-4,
        microbatch_size=1, max_steps=100000, reference_sha256=REFERENCE_SHA)
    if any(config.get(key) != value for key, value in exact.items()):
        raise ValueError('Authorized Stage 3A limits or fixed recipe changed')
    if config['batches'] != planned_batches():
        raise ValueError('Fixed seed, count, seat or batch order changed')
    if config['constraints'] != dict(book_training=False, test_split_loading=False,
            external_acquisition=False, paid_compute=False, fresh_strength_evaluation=False,
            opponent_actions_eligible=False, baseline='zero', entropy_bonus=0.0,
            reward_shaping=False, replay_reuse=False, rank_head_frozen=True):
        raise ValueError('Stage 3A scope changed')
    for key in ('data_root', 'reports_root', 'reference_checkpoint'):
        config[key] = str((ROOT / config[key]).resolve())
    return path, config, digest(path)


def implementation():
    paths = sorted((ROOT / 'src/riichi_ai').glob('*.py'))
    paths += [Path(__file__), ROOT / 'scripts/package_a_stability.py', ROOT / 'pyproject.toml']
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def verify_inputs(config):
    if digest(config['reference_checkpoint']) != config['reference_sha256']:
        raise ValueError('Frozen reference checkpoint changed')


def check_chain(reports, index):
    if not 0 <= index < 4:
        raise ValueError('At most four fixed batches are authorized')
    preparation = reports / 'prepare-attempt.json'
    if (not preparation.exists() or json.loads(preparation.read_text())['status'] != 'completed'
            or not (reports / 'plan.json').is_file()):
        raise ValueError('Successful preparation and a frozen plan are required before collection')
    attempted = sorted(reports.glob('attempt-*.json'))
    if len(attempted) != index:
        raise ValueError('Fixed batch order; no retries or replacements')
    for prior in range(index):
        marker = json.loads((reports / f'attempt-{prior}.json').read_text())
        result = json.loads((reports / f'batch-{prior}.json').read_text())
        if marker['status'] != 'completed' or not result['complete']:
            raise ValueError('A failed or incomplete formal batch stops this experiment')


def execute(command, allowance, config, evidence):
    """Record watchdog evidence on success, worker error and resource timeout."""
    import psutil
    started = time.perf_counter()
    source = implementation()
    process = None
    evidence.update(sampled_peak_tree_rss_bytes=0, elapsed_seconds=0.0,
                    artifact_bytes_peak=0, exit_code=None, sample_interval_seconds=0.2)
    try:
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        while True:
            rss = 0
            try:
                parent = psutil.Process(os.getpid())
                processes = [parent, *parent.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            for child in processes:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            evidence['sampled_peak_tree_rss_bytes'] = max(evidence['sampled_peak_tree_rss_bytes'], rss)
            if rss >= config['memory_bytes']:
                raise MemoryError('Stage 3A sampled process-tree memory ceiling reached')
            evidence['artifact_bytes_peak'] = max(evidence['artifact_bytes_peak'], disk_guard(config))
            if source != implementation():
                raise RuntimeError('Stage 3A implementation changed during leased work')
            if process.poll() is not None:
                if process.returncode:
                    raise RuntimeError(f'Worker returned {process.returncode}')
                return
            if time.perf_counter() - started >= max(0.1, allowance - 10):
                raise TimeoutError('Stage 3A subprocess wall-time ceiling reached')
            time.sleep(0.2)
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            evidence['exit_code'] = process.returncode
        evidence['elapsed_seconds'] = time.perf_counter() - started


def worker(config, identity, allowance, index):
    started = time.perf_counter()
    deadline = time.monotonic() + allowance - 20
    source = implementation()
    verify_inputs(config)
    import torch
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.stage3a import (initialize, load_boundary, save_boundary, state_manifest,
                                 collect_match, persist_match, update_batch)
    torch.set_num_threads(config['torch_threads'])
    reports, data = Path(config['reports_root']), Path(config['data_root'])
    plan_path = reports / 'plan.json'
    if index is None:
        if plan_path.exists():
            raise FileExistsError('Do not replace frozen Stage 3A plan')
        state = initialize(config['reference_checkpoint'], config['reference_sha256'],
            protocol_sha256=identity, device=config['device'],
            learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
        boundary = save_boundary(state, data / 'boundaries' / 'initial.pt')
        restored = load_boundary(data / 'boundaries' / 'initial.pt',
            expected_protocol_sha256=identity, device=config['device'])
        if state_manifest(state) != state_manifest(restored):
            raise RuntimeError('Initial boundary round trip differs')
        opponent = NeuralPolicy.from_checkpoint(config['reference_checkpoint'], device=config['device'])
        plan = dict(schema='stage3a-plan-v1', config_sha256=identity, implementation=source,
            frozen_before_formal_collection=True, batches=config['batches'], constraints=config['constraints'],
            reference_checkpoint=config['reference_checkpoint'], reference_sha256=config['reference_sha256'],
            initial_boundary=boundary, initial_state=state_manifest(state), opponent=opponent.manifest(),
            versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'riichienv', 'mahjong', 'psutil')},
            python=platform.python_version(), platform=platform.platform(),
            implementation_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                check=True, capture_output=True, text=True).stdout.strip())
        atomic_json(plan_path, plan)
        return
    plan = json.loads(plan_path.read_text())
    if (plan['config_sha256'] != identity or plan['implementation'] != source
            or plan['batches'] != config['batches']):
        raise ValueError('Frozen Stage 3A protocol or implementation changed')
    previous = plan['initial_boundary'] if index == 0 else json.loads(
        (reports / f'batch-{index - 1}.json').read_text())['output_boundary']
    boundary_path = data / 'boundaries' / ('initial.pt' if index == 0 else f'batch-{index - 1}.pt')
    if digest(boundary_path) != previous['sha256']:
        raise ValueError('Incoming boundary identity changed')
    state = load_boundary(boundary_path, expected_protocol_sha256=identity, device=config['device'])
    if state.updates != index:
        raise ValueError('Incoming update count differs from fixed chain')
    opponent = NeuralPolicy.from_checkpoint(config['reference_checkpoint'], device=config['device'])
    output = dict(schema='stage3a-batch-v1', batch_index=index, config_sha256=identity,
        plan_sha256=digest(plan_path), implementation=source, complete=False,
        input_boundary=previous, input_state=state_manifest(state), planned_matches=config['batches'][index]['matches'],
        opponent=opponent.manifest(), matches=[], update=None, output_boundary=None)
    trajectories = []
    try:
        for match in config['batches'][index]['matches']:
            trajectory = collect_match(state, opponent, batch_index=index, **match,
                max_steps=config['max_steps'], deadline=deadline)
            evidence = persist_match(trajectory, data / 'trajectories' / f'batch-{index}' / f"match-{match['match_index']}")
            output['matches'].append(evidence)
            atomic_json(reports / f'batch-{index}.json', output)
            trajectories.append(trajectory)
            if not trajectory.record['completed']:
                raise RuntimeError('Incomplete or failed planned match; batch preserved without an update')
            disk_guard(config)
            print(json.dumps({'phase': 'collection', 'batch_index': index,
                'persisted_matches': len(trajectories)}), flush=True)
        output['update'] = update_batch(state, trajectories, batch_index=index,
            planned_matches=config['batches'][index]['matches'], gradient_clip=config['gradient_clip'],
            replay_atol=config['replay_atol'], replay_rtol=config['replay_rtol'],
            replay_log_atol=config['replay_log_atol'],
            microbatch_size=config['microbatch_size'], deadline=deadline)
        output['output_boundary'] = save_boundary(state, data / 'boundaries' / f'batch-{index}.pt')
        restored = load_boundary(data / 'boundaries' / f'batch-{index}.pt',
            expected_protocol_sha256=identity, device=config['device'])
        output['output_state'] = state_manifest(state)
        output['boundary_roundtrip_verified'] = state_manifest(restored) == output['output_state']
        if not output['boundary_roundtrip_verified']:
            raise RuntimeError('Updated boundary round trip differs')
        verify_inputs(config)
        output['sources_unchanged'] = source == implementation()
        if not output['sources_unchanged']:
            raise RuntimeError('Implementation changed during formal batch')
        output['complete'] = True
    except BaseException as error:
        output['failure'] = dict(type=type(error).__name__, message=str(error))
        # The collector attaches its partial evidence on exceptions when available.
        partial = getattr(error, 'trajectory', None)
        if partial is not None:
            output['partial_match'] = persist_match(partial, data / 'trajectories' / f'batch-{index}' / 'partial')
        raise
    finally:
        output['final_in_memory_state'] = state_manifest(state)
        output['completed_update_count_delta'] = state.updates - index
        output['optimizer_steps_executed_delta'] = state.optimizer_steps_executed - index
        output['elapsed_seconds'] = time.perf_counter() - started
        atomic_json(reports / f'batch-{index}.json', output)
    print(json.dumps({'phase': 'update', 'batch_index': index, 'complete': output['complete'],
        'elapsed_seconds': output['elapsed_seconds'], 'update': output['update']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/stage3a.json'))
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('budget'); commands.add_parser('prepare')
    leased = commands.add_parser('budgeted')
    leased.add_argument('--name', required=True)
    leased.add_argument('--max-seconds', type=float, default=90)
    leased.add_argument('arguments', nargs=argparse.REMAINDER)
    run = commands.add_parser('run'); run.add_argument('--index', type=int, required=True)
    hidden = commands.add_parser('_worker')
    hidden.add_argument('--index', type=int)
    hidden.add_argument('--allowance', type=float, required=True)
    hidden.add_argument('--lease-token', required=True)
    args = parser.parse_args()
    path, config, identity = settings(args.config)
    ledger = Ledger(Path(config['data_root']) / 'budget-ledger.json', identity, config['limit_seconds'])
    if args.command == 'budget':
        print(json.dumps(ledger.snapshot(), indent=2)); return
    if args.command == '_worker':
        name = 'prepare' if args.index is None else f'batch:{args.index}'
        require_worker_lease(ledger, args.lease_token, args.allowance, name)
        worker(config, identity, args.allowance, args.index); return
    reports = Path(config['reports_root']); reports.mkdir(parents=True, exist_ok=True)
    source = implementation()
    index = None
    if args.command == 'budgeted':
        if not args.name.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Unsafe check name')
        command = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        if not command:
            raise ValueError('Need an explicit numerical check command')
        name, maximum = f'check:{args.name}', min(args.max_seconds, config['test_allowance_seconds'])
        marker = reports / f'checks-{args.name}.json'
    elif args.command == 'prepare':
        name, maximum = 'prepare', config['prepare_allowance_seconds']
        marker, command = reports / 'prepare-attempt.json', None
    else:
        index = args.index
        check_chain(reports, index)
        name, maximum = f'batch:{index}', config['per_batch_allowance_seconds']
        marker, command = reports / f'attempt-{index}.json', None
    if marker.exists():
        raise FileExistsError('Operation already attempted; preserve evidence')
    record = dict(name=name, config_sha256=identity, implementation=source,
        status='reserved', started_unix=time.time(), watchdog={})
    atomic_json(marker, record)
    failure = None
    try:
        with ledger.operation(name, maximum, config['reserve_seconds']) as allowance:
            if command is None:
                lease = [x for x in ledger.snapshot()['operations'] if x['status'] == 'active'][0]
                command = [sys.executable, str(Path(__file__).resolve()), '--config', str(path),
                    '_worker', '--allowance', str(allowance), '--lease-token', lease['token']]
                if index is not None:
                    command += ['--index', str(index)]
            record['command'] = command
            execute(command, allowance, config, record['watchdog'])
            if digest(path) != identity or source != implementation():
                raise RuntimeError('Source/config changed during numerical operation')
            disk_guard(config)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        record.update(status='failed' if failure else 'completed', failure=failure,
            finished_unix=time.time(), budget=ledger.snapshot())
        atomic_json(marker, record)
        atomic_json(reports / 'budget.json', ledger.snapshot())
    print(json.dumps({'phase': 'budget', **ledger.snapshot()}), flush=True)


if __name__ == '__main__':
    main()
