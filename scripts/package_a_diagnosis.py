"""Budgeted inference-only follow-up; existing experiments are never modified.

All numerical imports, loading, tests and inference run in leased subprocesses.
The tested Package A ledger/watchdog are reused with a separate identity/root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from package_a_stability import Ledger, atomic_json, digest, disk_guard, execute, require_worker_lease

ROOT = Path(__file__).resolve().parents[1]


def settings(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config['limit_seconds'] != 600 or config['maximum_model_attempts'] != 7:
        raise ValueError('This follow-up is bounded at 600 seconds and seven checkpoint attempts')
    if config['constraints'] != {'optimizer_updates': 0, 'new_matches': 0,
            'action_test_predictions': 0, 'test_split_loading': False}:
        raise ValueError('Inference-only scope changed')
    for key in ('data_root', 'reports_root', 'stage2b_config'):
        config[key] = str((ROOT / config[key]).resolve())
    return path, config, digest(path)


def implementation(config):
    files = sorted((ROOT / 'src/riichi_ai').glob('*.py'))
    files += [Path(__file__), ROOT / 'scripts/package_a_stability.py', Path(config['stage2b_config'])]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def verify_checkpoints(config):
    for item in config['checkpoints']:
        if digest(item['path']) != item['sha256']:
            raise ValueError(f"Frozen checkpoint changed: {item['name']}")


def load_data(config):
    from riichi_ai.stage2b_data import load_efficiency_split
    stage = json.loads(Path(config['stage2b_config']).read_text())
    if digest(Path(stage['stage2a_dataset']) / 'manifest.json') != stage['stage2a_manifest_sha256']:
        raise ValueError('Historical training manifest changed')
    if json.loads((Path(stage['data_root']) / 'dataset/manifest.json').read_text())['config'] != stage:
        raise ValueError('Validation corpus configuration changed')
    training = load_efficiency_split(stage['stage2a_dataset'], Path(stage['data_root']) / 'derived-stage2a',
        'train', assisted=True, max_matches=stage['training']['training_matches'],
        memory_limit_bytes=config['memory_bytes'] // 4, allow_test=False)
    validation = load_efficiency_split(Path(stage['data_root']) / 'dataset',
        Path(stage['data_root']) / 'derived-stage2b', 'validation', assisted=True,
        memory_limit_bytes=config['memory_bytes'] // 4, allow_test=False)
    if len(training) != config['expected_training_rows'] or len(validation) != config['expected_validation_rows']:
        raise ValueError('Dataset size differs from frozen scope')
    if len(training.stratum_indices['student']) or len(training.stratum_indices['teacher']) != len(training):
        raise ValueError('Expected teacher-only historical training distribution')
    return training, validation


def worker(config, identity, allowance, index):
    started = time.perf_counter()
    source = implementation(config)
    import numpy as np
    import torch
    from riichi_ai.stage2b_training import _index_hash, stratified_indices
    from riichi_ai.training import _data_manifest, _state_digest, _rss, collate_training
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    peak = 0

    def check():
        nonlocal peak
        if time.perf_counter() - started >= max(.1, allowance - 20):
            raise TimeoutError('Cooperative diagnosis deadline; preserve partial outputs')
        driver = torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0
        peak = max(peak, _rss() + driver)
        if peak >= config['memory_bytes']:
            raise MemoryError('Conservative worker memory ceiling')
        disk_guard(config)

    training, validation = load_data(config)
    check()
    reports = Path(config['reports_root'])
    plan_path = reports / 'plan.json'
    if index is None:
        if plan_path.exists():
            raise FileExistsError('The frozen row plan already exists')
        train_indices = sorted(map(int, np.random.default_rng(config['training_sample_seed']).choice(
            len(training), config['training_rows'], replace=False)))
        val_indices = list(range(len(validation)))
        precision = stratified_indices(validation, per_stratum=config['precision_per_stratum'],
                                       seed=config['precision_seed'])
        old = json.loads((ROOT / 'reports/package_a/a2/run-control-202610171.json').read_text())
        if _index_hash(precision) != old['settings']['diagnostic_indices_sha256']:
            raise ValueError('Precision states differ from Package A fixed diagnostics')
        verify_checkpoints(config)
        plan = {'schema': 'package-a-diagnosis-frozen-plan-v1', 'config_sha256': identity,
            'frozen_before_model_outputs': True, 'implementation': source,
            'checkpoints': config['checkpoints'], 'train_indices': train_indices,
            'validation_indices': val_indices, 'precision_indices': precision,
            'index_sha256': {name: _index_hash(rows) for name, rows in
                [('training', train_indices), ('validation', val_indices), ('precision', precision)]},
            'training_manifest': _data_manifest(training), 'validation_manifest': _data_manifest(validation),
            'interpretation': config['interpretation'], 'constraints': config['constraints'],
            'sampled_peak_conservative_memory_bytes': peak, 'elapsed_seconds': time.perf_counter() - started}
        if source != implementation(config):
            raise ValueError('Implementation changed during preparation')
        atomic_json(plan_path, plan)
        print(json.dumps({'phase': 'frozen_plan', 'training_sample_rows': len(train_indices),
                         'validation_rows': len(val_indices), 'precision_rows': len(precision)}), flush=True)
        return
    plan = json.loads(plan_path.read_text())
    if (plan['implementation'] != source or plan['config_sha256'] != identity
            or plan['checkpoints'] != config['checkpoints']):
        raise ValueError('Frozen protocol or source changed')
    if plan['training_manifest'] != _data_manifest(training) or plan['validation_manifest'] != _data_manifest(validation):
        raise ValueError('Dataset identity changed after row freeze')
    item = config['checkpoints'][index]
    if digest(item['path']) != item['sha256']:
        raise ValueError('Checkpoint identity changed')
    from riichi_ai.model import NeuralPolicy
    from riichi_ai.package_a_diagnosis_fit import evaluate_fit
    from riichi_ai.package_a_diagnosis_precision import compare_precision
    model = NeuralPolicy.from_checkpoint(item['path'], device='cpu').model
    before = _state_digest(model.state_dict())
    if not torch.backends.mps.is_available():
        raise RuntimeError('Declared MPS comparison unavailable; do not silently fall back')
    output = {'schema': 'package-a-diagnosis-model-v1', 'checkpoint': item, 'config_sha256': identity,
        'plan_sha256': digest(plan_path), 'implementation': source, 'complete': False,
        'constraints': config['constraints'], 'model_state_sha256_before': before, 'fit': {}, 'arrays': {}}
    arrays_root = Path(config['data_root']) / 'arrays'
    arrays_root.mkdir(parents=True, exist_ok=True)
    def save_arrays(name, arrays):
        path = arrays_root / f"{item['name']}-{name}.npz"
        if path.exists():
            raise FileExistsError('Do not overwrite existing inference evidence')
        np.savez_compressed(path, **arrays)
        output['arrays'][name] = {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
        check()
    try:
        records = [validation[i] for i in plan['precision_indices']]
        batch, labels = collate_training(records, 'cpu')
        precision_summary, arrays = compare_precision(model, batch, labels,
            row_ids=plan['precision_indices'], records=records, include_mps=True, check=check)
        if (not precision_summary['mps_available']
                or not all(branch['finite'] for branch in precision_summary['branches'].values())
                or any(value is False for value in precision_summary['checks'].values())):
            raise RuntimeError('Precision inference invalid or caller state changed')
        output['precision'] = precision_summary
        save_arrays('precision', arrays)
        model.to(config['device']).eval()
        for split, dataset, indices in [('training', training, plan['train_indices']),
                                       ('validation', validation, plan['validation_indices'])]:
            print(json.dumps({'phase': 'fit', 'checkpoint': item['name'], 'split': split,
                             'rows': len(indices)}), flush=True)
            summary, arrays = evaluate_fit(model, dataset, indices=indices, device=config['device'],
                batch_size=config['batch_size'], epsilon=config['epsilon'], check=check)
            output['fit'][split] = summary
            save_arrays(split, arrays)
        output.update(model_state_sha256_after=_state_digest(model.state_dict()),
            sources_unchanged=source == implementation(config),
            checkpoint_file_unchanged=digest(item['path']) == item['sha256'])
        if (output['model_state_sha256_after'] != before or not output['sources_unchanged']
                or not output['checkpoint_file_unchanged']):
            raise RuntimeError('Inference mutated model, checkpoint or implementation')
        output['complete'] = True
    except BaseException as error:
        output['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        output.update(elapsed_seconds=time.perf_counter() - started,
                      sampled_peak_conservative_memory_bytes=peak)
        atomic_json(reports / f"model-{item['name']}.json", output)
    print(json.dumps({'phase': 'model_completed', 'name': item['name'],
                     'elapsed_seconds': output['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/package_a_diagnosis.json'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('budget')
    sub.add_parser('prepare')
    test = sub.add_parser('budgeted')
    test.add_argument('--name', required=True)
    test.add_argument('--max-seconds', type=float, default=90)
    test.add_argument('arguments', nargs=argparse.REMAINDER)
    run = sub.add_parser('run')
    run.add_argument('--index', type=int, required=True)
    hidden = sub.add_parser('_worker')
    hidden.add_argument('--index', type=int)
    hidden.add_argument('--allowance', type=float, required=True)
    hidden.add_argument('--lease-token', required=True)
    args = parser.parse_args()
    config_path, config, identity = settings(args.config)
    ledger = Ledger(Path(config['data_root']) / 'budget-ledger.json', identity, config['limit_seconds'])
    if args.command == 'budget':
        print(json.dumps(ledger.snapshot()), flush=True)
        return
    if args.command == '_worker':
        name = 'prepare' if args.index is None else f"checkpoint:{config['checkpoints'][args.index]['name']}"
        require_worker_lease(ledger, args.lease_token, args.allowance, name)
        worker(config, identity, args.allowance, args.index)
        return
    reports = Path(config['reports_root'])
    reports.mkdir(parents=True, exist_ok=True)
    source = implementation(config)
    if args.command == 'budgeted':
        command = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        if not command:
            raise ValueError('Need explicit subprocess command')
        name, maximum = args.name, args.max_seconds
        marker = reports / f"checks-{name}.json"
    else:
        index = None if args.command == 'prepare' else args.index
        if index is not None:
            if not 0 <= index < len(config['checkpoints']):
                raise ValueError('Undeclared checkpoint index')
            done = sorted(reports.glob('attempt-*.json'))
            if index != len(done) or len(done) >= config['maximum_model_attempts']:
                raise ValueError('Fixed checkpoint order; no replacement attempts')
            name = f"checkpoint:{config['checkpoints'][index]['name']}"
            maximum = config['per_model_allowance_seconds']
            marker = reports / f'attempt-{index}.json'
        else:
            name, maximum = 'prepare', config['prepare_allowance_seconds']
            marker = reports / 'prepare-attempt.json'
        command = None
    if marker.exists():
        raise FileExistsError('Operation was already attempted; preserve it')
    record = {'name': name, 'config_sha256': identity, 'implementation': source,
              'status': 'reserved', 'started_unix': time.time()}
    atomic_json(marker, record)
    failure = None
    try:
        with ledger.operation(name, maximum, config['reserve_seconds']) as allowed:
            if command is None:
                active = [x for x in ledger.snapshot()['operations'] if x['status'] == 'active'][0]
                command = [sys.executable, str(Path(__file__).resolve()), '--config', str(config_path),
                           '_worker', '--allowance', str(allowed), '--lease-token', active['token']]
                if index is not None:
                    command += ['--index', str(index)]
            record['watchdog'] = execute(command, allowed, config)
            if digest(config_path) != identity or source != implementation(config):
                raise RuntimeError('Source/configuration changed during operation')
            disk_guard(config)
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        record.update(status='failed' if failure else 'completed', failure=failure,
                      finished_unix=time.time(), budget=ledger.snapshot())
        atomic_json(marker, record)
    print(json.dumps({'phase': 'budget', **ledger.snapshot()}), flush=True)


if __name__ == '__main__':
    main()
