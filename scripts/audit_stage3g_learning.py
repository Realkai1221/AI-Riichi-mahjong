#!/usr/bin/env python3
"""Independent saved-evidence arithmetic audit; no optimization or new games.

Checks all value checkpoints/update exposures; independently rebuilds public
features, OOF membership, proper-score/calibration gate and all hand deltas;
recomputes six saved predictors on 96 prespecified games; recomputes every
paired development outcome and selection. This does not replay all gradients
or independently certify Riichi rules. Full corpus/action audits are separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch

UTILITY = np.asarray([1.5, .5, -.5, -1.5], dtype=np.float64)
ARMS = ('mc_a0', 'mc_a1', 'delta_a0', 'delta_a1')
ORDER_SEED = 20261023012


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def hash_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def check_deadline(deadline):
    require(time.monotonic() < deadline, 'Evidence audit cooperative deadline reached')


def artifact(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked artifact: ' + str(path))
    require(hash_file(path) == expected, 'Artifact hash differs: ' + str(path))
    return path


def close(actual, expected, label, *, atol=1e-10, rtol=1e-9):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, atol=atol, rtol=rtol), label)


def public_features(hand, seat):
    """Independent explicit whitelist; outcomes/behavior are never features."""
    order = [(seat + i) % 4 for i in range(4)]
    scores = hand['scores']
    result = [scores[i] / 10000 for i in order]
    result += [(scores[i] - scores[seat]) / 10000 for i in order]
    result += [float(hand['bakaze'] == wind) for wind in 'ESWN']
    result += [float(hand['kyoku'] == k) for k in range(1, 5)]
    result += [hand['honba'] / 10, hand['kyotaku'] / 10]
    result += [float((hand['oya'] - seat) % 4 == i) for i in range(4)]
    result += [float(seat == i) for i in range(4)]
    result += [i / 3 for i in order]
    result += [hand['hand_id'] / 20]
    return np.asarray(result, dtype=np.float32)


def baseline_probabilities(relative_scores):
    """Vectorized independent exact 24-permutation Plackett-Luce marginal."""
    weights = np.exp(relative_scores - relative_scores.max(axis=1, keepdims=True))
    result = np.zeros((len(weights), 4), dtype=np.float64)
    for order in itertools.permutations(range(4)):
        p = np.ones(len(weights), dtype=np.float64)
        for place in range(3):
            p *= weights[:, order[place]] / weights[:, list(order[place:])].sum(axis=1)
        result[:, order.index(0)] += p
    return result


def gate_arithmetic(probabilities, baseline, ranks, weights, groups, stages):
    """Independent point-gate arithmetic, with the original fixed thresholds."""
    require(probabilities.shape == baseline.shape == (len(ranks), 4), 'Probability shape mismatch')
    for values in (probabilities, baseline):
        require(np.isfinite(values).all() and np.all(values >= 0), 'Invalid placement probabilities')
        close(values.sum(axis=1), np.ones(len(values)), 'Placement probabilities do not normalize')
    labels = np.eye(4)[ranks]
    proper = {}
    for key, values in [('model', probabilities), ('baseline', baseline)]:
        proper[key] = {'nll': float(np.dot(weights, -np.log(np.maximum(values[np.arange(len(ranks)), ranks], np.finfo(float).tiny))) / weights.sum()),
            'brier': float(np.dot(weights, np.square(values - labels).sum(axis=1)) / weights.sum())}
    means, truths = probabilities @ UTILITY, UTILITY[ranks]
    indices = np.clip(np.searchsorted([-1.5, -.9, -.3, .3, .9, 1.5], means, side='right') - 1, 0, 4)
    strata = {}
    for number, name in enumerate(('overall', 'early', 'middle', 'late')):
        mask = np.ones(len(ranks), dtype=bool) if number == 0 else stages == number - 1
        total = float(weights[mask].sum())
        supported_mass = absolute = maximum = 0.
        bins = []
        for index in range(5):
            chosen = mask & (indices == index)
            unique = len(np.unique(groups[chosen]))
            mass = float(weights[chosen].sum())
            bias = float(np.dot(weights[chosen], means[chosen] - truths[chosen]) / mass) if mass else None
            supported = unique >= 50
            if supported:
                supported_mass += mass
                absolute += mass * abs(bias)
                maximum = max(maximum, abs(bias))
            bins.append({'unique_games': unique, 'supported': supported, 'bias': bias, 'mass': mass / total if total else 0.})
        unique = len(np.unique(groups[mask]))
        coverage = supported_mass / total if total else 0.
        error = absolute / supported_mass if supported_mass else None
        passed = unique >= 100 and coverage >= .9 and error is not None and error <= .15 and maximum <= .3
        strata[name] = {'passed': passed, 'unique_games': unique, 'supported_mass': coverage,
            'weighted_absolute_bias': error, 'max_supported_bin_bias': maximum if supported_mass else None, 'bins': bins}
    proper_passed = all(proper['model'][k] <= proper['baseline'][k] for k in ('nll', 'brier'))
    return {'passed': proper_passed and all(s['passed'] for s in strata.values()),
        'proper_score_gate_passed': proper_passed, 'proper': proper, 'strata': strata}


def audit_grp(data, corpus, deadline):
    directory = data / 'grp'
    report, target, assignment = (read(directory / name) for name in ('report.json', 'hand-targets.json', 'fold-assignment.json'))
    corpus_sha = hash_file(data / 'corpus/manifest.json')
    require(report.get('complete') is True and report['corpus_manifest_sha256'] == corpus_sha == target['corpus_manifest_sha256'], 'GRP corpus identity/completion differs')
    artifact(directory / 'hand-targets.json', report['target_sha256'])
    train, dev = corpus['splits']['training'], corpus['splits']['development']
    ordered = sorted(train, key=lambda name: hashlib.sha256(canonical([20261023021, name])).hexdigest())
    folds = {name: i % 5 for i, name in enumerate(ordered)}
    fold_sha = hashlib.sha256(canonical(folds)).hexdigest()
    require(folds == assignment['folds'] == target['folds'] and assignment['assignment_sha256'] == fold_sha == report['crossfit_assignment_sha256'] == target['crossfit_assignment_sha256'], 'Whole-match OOF assignment changed')
    require(target['splits'] == corpus['splits'] and target['target'] == 'one_hand_placement_potential_difference', 'Hand target scope differs')
    records = {item['match_id']: item for item in corpus['matches']}
    names = sorted(records)
    xs, ys, weights, groups, stages, starts, relative_scores = [], [], [], [], [], {}, []
    sources = {}
    for group, name in enumerate(names):
        check_deadline(deadline)
        record = records[name]
        path = data / 'corpus/targets' / (name + '.json')
        relative = f'targets/{name}.json'
        artifact(path, record['artifacts'][relative]['sha256'])
        sources[relative] = hash_file(path)
        public = read(path)
        seat = record['seat_behaviors'].index('explorer')
        hands = public['hand_starts']
        require([h['hand_id'] for h in hands] == list(range(1, len(hands) + 1)), 'GRP explicit hand sequence missing')
        require(public['ranks'] == record['ranks'] and public['utilities'] == record['utilities'], 'Public terminal labels differ')
        require(public['source_role'] == record['source_role'], 'Public role differs')
        starts[name] = (seat, hands, public['utilities'][seat])
        for hand in hands:
            xs.append(public_features(hand, seat))
            ys.append(public['ranks'][seat] - 1)
            weights.append(1 / len(hands))
            groups.append(group)
            round_index = 'ESWN'.index(hand['bakaze']) * 4 + hand['kyoku'] - 1
            stages.append(0 if round_index < 2 else 1 if round_index < 6 else 2)
            relative_scores.append([hand['scores'][(seat + i) % 4] / 10000 for i in range(4)])
    x, ranks, weights, groups, stages = (np.asarray(v, dtype=d) for v, d in
        [(xs, np.float32), (ys, np.int64), (weights, np.float64), (groups, np.int64), (stages, np.int8)])
    require(sources == target['source_target_hashes'] == report['source_target_hashes'], 'GRP public source membership/hash differs')
    with np.load(directory / 'public-predictions.npz', allow_pickle=False) as saved:
        require(saved['match_ids'].tolist() == names, 'Prediction match order changed')
        for key, expected in [('public_features', x), ('ranks', ranks), ('weights', weights), ('groups', groups), ('stages', stages)]:
            require(np.array_equal(saved[key], expected), 'Independent GRP reconstruction differs: ' + key)
        probabilities, baseline = saved['probabilities'].copy(), saved['baseline'].copy()
    close(baseline, baseline_probabilities(np.asarray(relative_scores)), 'Independent PL baseline differs', atol=2e-12, rtol=2e-12)
    fit_records = {item['name']: item for item in report['fits']}
    require(set(fit_records) == {'fold-0', 'fold-1', 'fold-2', 'fold-3', 'fold-4', 'final'}, 'Six completed GRP fits required')
    predictor_ids, samples = {}, []
    for fold in [0, 1, 2, 3, 4, None]:
        check_deadline(deadline)
        name = f'fold-{fold}' if fold is not None else 'final'
        path = directory / f'predictor-{name}.pt'
        identity = fit_records[name]['predictor_sha256']
        artifact(path, identity)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        expected_train = sorted([match for match in train if fold is None or folds[match] != fold])
        expected_predict = sorted(dev if fold is None else [match for match in train if folds[match] == fold])
        require(saved['training_match_ids'] == expected_train and saved['prediction_match_ids'] == expected_predict and not set(expected_train) & set(expected_predict), 'GRP fit leaks held-out matches')
        require(saved['fold'] == fold and saved['epochs'] == 80 and saved['initialization_seed'] == 20261023022
                and saved['crossfit_assignment_sha256'] == fold_sha and saved['corpus_manifest_sha256'] == corpus_sha, 'GRP recipe/identity changed')
        expected_rows = sum(len(starts[item][1]) for item in expected_train)
        require(fit_records[name]['epochs'] == 80 and fit_records[name]['optimizer_steps'] == 80 * math.ceil(expected_rows / 512)
                and fit_records[name]['rows'] == expected_rows, 'GRP fixed exposure changed')
        chosen = expected_predict[:16]
        selected = np.isin(groups, [names.index(item) for item in chosen])
        network = torch.nn.Sequential(torch.nn.Linear(31, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
        network.load_state_dict({key.removeprefix('layers.'): value for key, value in saved['model_state_dict'].items()}, strict=True)
        network.eval()
        with torch.inference_mode():
            recomputed = torch.softmax(network(torch.from_numpy(x[selected].copy())), dim=-1).numpy().astype(np.float64)
        recomputed /= recomputed.sum(axis=1, keepdims=True)
        close(probabilities[selected], recomputed, 'Held-out saved predictor inference differs', atol=2e-6, rtol=2e-6)
        predictor_ids[name] = identity
        samples.append({'predictor': name, 'match_ids': chosen, 'hand_states': int(selected.sum()),
                        'maximum_abs_probability_difference': float(np.max(np.abs(probabilities[selected] - recomputed)))})
    devmask = np.isin(groups, [names.index(item) for item in dev])
    gate = gate_arithmetic(probabilities[devmask], baseline[devmask], ranks[devmask], weights[devmask], groups[devmask], stages[devmask])
    require(gate['passed'] == report['calibration_gate_passed'] == target['calibration_gate_passed'] == target['gate']['passed'] == target['eligible_for_delta_training'], 'Calibration fallback decision differs')
    require(report['gate'] == target['gate'], 'GRP report/target gate evidence differs')
    require(report['gate']['thresholds'] == {'minimum_unique_games': 100, 'minimum_bin_unique_games': 50,
        'minimum_supported_mass': .9, 'maximum_weighted_absolute_bias': .15,
        'maximum_supported_bin_bias': .3, 'utility_bins': [-1.5, -.9, -.3, .3, .9, 1.5]},
        'Calibration thresholds differ from the frozen gate')
    require(gate['proper_score_gate_passed'] == report['gate']['proper_score_gate_passed'], 'Proper-score gate changed')
    for key in ('nll', 'brier'):
        close(gate['proper']['model'][key], report['gate']['proper_scores'][key], 'GRP proper score differs')
        close(gate['proper']['baseline'][key], report['gate']['baseline_proper_scores'][key], 'GRP baseline proper score differs')
    for name, expected in gate['strata'].items():
        actual = report['gate']['strata'][name]
        for key in ('passed', 'unique_games'):
            require(actual[key] == expected[key], 'GRP stratum gate/support differs')
        for key in ('supported_mass', 'weighted_absolute_bias', 'max_supported_bin_bias'):
            if expected[key] is None:
                require(actual[key] is None, 'Empty stratum arithmetic differs')
            else:
                close(actual[key], expected[key], 'GRP stratum calibration arithmetic differs')
        for a, b in zip(actual['bins'], expected['bins']):
            require(a['unique_games'] == b['unique_games'] and a['supported'] == b['supported'], 'Calibration bin support differs')
            if b['bias'] is not None:
                close(a['bias'], b['bias'], 'Calibration bin bias differs')
    require(set(target['matches']) == set(names), 'Hand delta match coverage differs')
    values = probabilities @ UTILITY
    hand_count = 0
    for group, name in enumerate(names):
        seat, hands, terminal = starts[name]
        entry = target['matches'][name]
        expected_model = predictor_ids[f'fold-{folds[name]}'] if name in folds else predictor_ids['final']
        require(entry['predictor_sha256'] == expected_model and entry['explorer_seat'] == seat
                and entry['source_role'] == records[name]['source_role'] and entry['fold'] == folds.get(name), 'Hand potential predictor/role identity differs')
        current = values[groups == group]
        expected = np.append(current[1:], terminal) - current
        require(set(entry['hand_deltas']) == set(entry['hands']) == {str(i) for i in range(1, len(hands) + 1)}, 'Hand target ordinal coverage differs')
        for i, delta in enumerate(expected):
            detail = entry['hands'][str(i + 1)]
            close(entry['hand_deltas'][str(i + 1)], delta, 'One-hand delta arithmetic differs')
            close(detail['current_value'], current[i], 'Current potential differs')
            close(detail['next_value'], terminal if i + 1 == len(hands) else current[i + 1], 'Next potential/terminal substitution differs')
            close(detail['target'], delta, 'Detailed hand target differs')
            require(detail['terminal_next'] is (i + 1 == len(hands)) and detail['predictor_sha256'] == expected_model, 'Boundary uses wrong predictor or terminal flag')
            hand_count += 1
    return {'valid': True, 'calibration_gate_passed': gate['passed'], 'matches': len(names),
        'hand_targets_recomputed': hand_count, 'models_verified': 6, 'inference_sample_games': sum(len(v['match_ids']) for v in samples),
        'inference_samples': samples, 'predictor_probability_tolerance': {'atol': 2e-6, 'rtol': 2e-6},
        'target_sha256': report['target_sha256'], 'scope': 'All public features, labels, source hashes, six OOF fit memberships, fixed calibration point gate and every hand difference; 96 fixed held-out games re-forwarded. GRP optimizer trace and bootstrap intervals are not independently replayed.'}


def exposure_schedule(corpus, directory, deadline):
    """Read only compact candidate offsets and seat/behavior identity arrays."""
    records = {item['match_id']: item for item in corpus['matches']}
    names = corpus['splits']['training']
    per_match = {}
    behaviors = ['shanten_v2', 'defense_v2', 'reference', 'explorer']
    for name in names:
        check_deadline(deadline)
        report = records[name]
        require(report['role'] == 'training' and report['source_role'] == 'offline-training' and report['training_eligible'] is True and report['ppo_eligible'] is False, 'Training report role changed')
        for area in ('actor', 'targets'):
            artifact(directory / area / (name + '.npz'), report['artifacts'][f'{area}/{name}.npz']['sha256'])
        with np.load(directory / 'actor' / (name + '.npz'), allow_pickle=False) as actor:
            counts = np.diff(actor['candidate_offsets'])
        with np.load(directory / 'targets' / (name + '.npz'), allow_pickle=False) as targets:
            ids, seats = targets['behavior_id'], targets['seat']
        require(len(ids) == len(counts) == report['rows'], 'Numeric row count differs')
        expected = np.asarray([behaviors.index(item) for item in report['seat_behaviors']])[seats]
        require(np.array_equal(ids, expected), 'Numeric behavior differs from seat assignment')
        eligible = np.flatnonzero((ids == 3) & (counts > 1))
        per_match[name] = {'indices': eligible, 'rows': len(ids), 'explorer_rows': int((ids == 3).sum())}
    schedule = []
    for epoch in range(2):
        order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([ORDER_SEED, epoch, 0]))).permutation(len(names))
        ordered = [names[int(i)] for i in order]
        for offset in range(0, 3200, 16):
            group = ordered[offset:offset + 16]
            h = hashlib.sha256()
            for name in group:
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'big')
                indices = per_match[name]['indices']
                permutation = np.random.Generator(np.random.PCG64(np.random.SeedSequence([ORDER_SEED, epoch, seed, 1]))).permutation(len(indices))
                h.update(name.encode())
                h.update(indices[permutation].tobytes())
            schedule.append({'epoch': epoch, 'match_ids': group, 'row_order_sha256': h.hexdigest(),
                'optional_rows': sum(len(per_match[n]['indices']) for n in group),
                'explorer_rows': sum(per_match[n]['explorer_rows'] for n in group),
                'all_seat_rows': sum(per_match[n]['rows'] for n in group),
                'empty_matches': sum(len(per_match[n]['indices']) == 0 for n in group)})
    return schedule


def check_update_records(records, schedule):
    require(len(records) == len(schedule) == 400, 'Exactly 400 complete-group updates required')
    totals = {'optional_row_presentations': 0, 'all_seat_rows_loaded': 0, 'explorer_rows_loaded': 0,
        'empty_match_presentations': 0, 'clipped_updates': 0, 'match_presentations': 0}
    prefixes = [dict(totals)]
    steps = 0
    for index, (record, expected) in enumerate(zip(records, schedule), 1):
        for key, value in expected.items():
            require(record.get(key) == value, 'Update exposure/order differs: ' + key)
        position = index % 200 * 16
        cursor = {'epoch': index // 200, 'next_match_position': position}
        steps += int(expected['optional_rows'] > 0)
        require(record['scheduled_update'] == index and record['cursor'] == cursor and record['optimizer_updates'] == steps
                and record['optimizer_step'] is (expected['optional_rows'] > 0) and record['matches'] == 16, 'Update counter/cursor differs')
        require(record['clipped'] is (record['gradient_l2_before_clip'] > 1)
                and all(math.isfinite(record[k]) for k in ('objective', 'mse', 'conservative', 'gradient_l2_before_clip')),
                'Nonfinite update evidence or clipping flag changed')
        totals['optional_row_presentations'] += expected['optional_rows']
        totals['all_seat_rows_loaded'] += expected['all_seat_rows']
        totals['explorer_rows_loaded'] += expected['explorer_rows']
        totals['empty_match_presentations'] += expected['empty_matches']
        totals['clipped_updates'] += int(record['clipped'])
        totals['match_presentations'] += 16
        prefixes.append(dict(totals))
    return totals, prefixes, steps


def audit_learning(data, corpus, arms, target_sha, deadline):
    corpus_sha = hash_file(data / 'corpus/manifest.json')
    config = read(ROOT / 'configs/stage3g.json')
    reference = torch.load(artifact(config['reference_checkpoint'], config['reference_sha256']), map_location='cpu', weights_only=True)
    reference_state = reference['model_state_dict']
    rank_reference = {k.removeprefix('rank_head.'): v for k, v in reference_state.items() if k.startswith('rank_head.')}
    rank_hash = state_hash(rank_reference)
    schedule = exposure_schedule(corpus, data / 'corpus', deadline)
    common_initial, result_records = None, []
    for arm in arms['arms']:
        check_deadline(deadline)
        directory = Path(arm['checkpoint']).parent
        artifact(arm['checkpoint'], arm['sha256'])
        result = read(directory / 'result.json')
        records = [json.loads(line) for line in (directory / 'updates.jsonl').read_text().splitlines()]
        totals, prefixes, steps = check_update_records(records, schedule)
        require(result['complete'] is True and result['policy_promotion'] is False and result['rank_head_unchanged'] is True
                and result['scheduled_updates'] == 400 and result['optimizer_updates'] == steps and result['statistics'] == totals,
                'Arm result exposure/counters differ')
        require(result['identity']['manifest_sha256'] == corpus_sha and result['identity']['split'] == 'training'
                and result['identity']['purpose'] == 'formal-training' and result['identity']['arm'] == arm['name']
                and result['identity']['reference_sha256'] == config['reference_sha256'], 'Training identity differs')
        require(result['identity']['hand_targets_sha256'] == (target_sha if arm['name'].startswith('delta_') else None), 'Arm target identity changed')
        close(result['first_objective'], records[0]['objective'], 'First objective differs')
        close(result['last_objective'], records[-1]['objective'], 'Last objective differs')
        expected_names = ['initial.pt', 'update-0100.pt', 'update-0200.pt', 'update-0300.pt', 'final.pt']
        require([Path(item['path']).name for item in result['checkpoints']] == expected_names, 'Checkpoint boundary inventory differs')
        checkpoints = []
        for checkpoint, count in zip(result['checkpoints'], (0, 100, 200, 300, 400)):
            check_deadline(deadline)
            path = artifact(checkpoint['path'], checkpoint['sha256'])
            require(path.parent == directory and path.stat().st_size == checkpoint['bytes'], 'Checkpoint path/size differs')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            meta, state, optimizer = saved['metadata'], saved['model_state_dict'], saved['optimizer_state_dict']
            actual_hash = state_hash(state)
            frozen = {k.removeprefix('rank_head.'): v for k, v in state.items() if k.startswith('rank_head.')}
            require(actual_hash == meta['weights_sha256'] == checkpoint['weights_sha256']
                    and state_hash(frozen) == meta['frozen_rank_sha256'] == rank_hash
                    and all(bool(torch.isfinite(v).all()) for v in state.values()), 'Checkpoint tensor hash/rank/finiteness differs')
            cursor = {'epoch': count // 200, 'next_match_position': count % 200 * 16}
            expected_steps = sum(record['optimizer_step'] for record in records[:count])
            require(meta['identity'] == result['identity'] and meta['cursor'] == cursor and meta['scheduled_updates'] == count
                    and meta['optimizer_updates'] == expected_steps and meta['complete'] is (count == 400)
                    and meta['statistics'] == prefixes[count] and meta['policy_promotion'] is False,
                    'Checkpoint cursor/exposure/completion differs')
            require(meta['initial_weights_sha256'] == result['initial_weights_sha256'], 'Initial identity differs')
            groups = optimizer['param_groups']
            require(len(groups) == 1 and groups[0]['lr'] == .0001 and tuple(groups[0]['betas']) == (.9, .999)
                    and groups[0]['eps'] == 1e-8 and groups[0]['weight_decay'] == 0, 'Adam recipe changed')
            states = optimizer['state']
            require((not states) if count == 0 else bool(states), 'Adam initialization/final state absent')
            require(set(states) <= set(groups[0]['params']), 'Adam state contains unknown parameter')
            for state_value in states.values():
                require(float(state_value['step']) == expected_steps, 'Adam step differs from update count')
                require(all(bool(torch.isfinite(value).all()) for value in state_value.values() if isinstance(value, torch.Tensor)), 'Nonfinite Adam state')
            if count == 0:
                common_initial = actual_hash if common_initial is None else common_initial
                require(actual_hash == common_initial == result['initial_weights_sha256'], 'Arms lack identical initialization')
                require(all(torch.equal(value, reference_state[key]) for key, value in state.items() if not key.startswith('action_score.')), 'Initial backbone/reference differs')
            if count == 400:
                require(actual_hash == result['final_weights_sha256'] and checkpoint['sha256'] == arm['sha256'], 'Final checkpoint identity differs')
            checkpoints.append({'name': path.name, 'sha256': checkpoint['sha256'], 'scheduled_updates': count,
                                'optimizer_steps': expected_steps, 'tensor_hash_verified': True})
        result_records.append({'arm': arm['name'], 'scheduled_updates': 400, 'optimizer_updates': steps,
            'epochs': 2, 'training_matches': 3200, 'statistics': totals, 'checkpoints': checkpoints})
    counts = {item['statistics']['optional_row_presentations'] for item in result_records}
    require(len(counts) == 1, 'Arms saw unequal explorer optional row exposure')
    return {'valid': True, 'arms': result_records, 'common_initial_weights_sha256': common_initial,
        'frozen_reference_rank_sha256': rank_hash, 'training_matches': 3200,
        'common_group_and_row_order': True, 'optional_row_presentations_per_arm': next(iter(counts)),
        'scope': 'All 3200 training shard identities and explorer eligibility masks; every update order/count and five checkpoints per arm. No gradient, Adam trajectory or loss recomputation; preflight and separate corpus audit cover those distinct mechanisms.'}


def development_arithmetic(execution, records):
    expected = {job['match_id']: job for job in execution['matches']}
    actual = {record['match_id']: record for record in records}
    require(len(actual) == len(records) == len(expected) and set(actual) == set(expected), 'Development panel incomplete or duplicated')
    names = execution['tie_order']
    require(execution['blocks'] == 64 and execution['rotations'] == 4 and names in [list(ARMS), list(ARMS[:2])], 'Development panel shape changed')
    for name, record in actual.items():
        require(all(record.get(k) == v for k, v in expected[name].items()) and record['completed'] is True
                and record['failure'] is None and record['training_eligible'] is False and record['ppo_eligible'] is False
                and record['source_role'] == 'development-only' and record['terminal_result_verified'] is True, 'Development record identity/role differs')
        ranks = [0] * 4
        for rank, seat in enumerate(sorted(range(4), key=lambda seat: (-record['scores'][seat], seat)), 1):
            ranks[seat] = rank
        require(record['ranks'] == ranks and record['utilities'] == [float(UTILITY[r - 1]) for r in ranks], 'Development rank/utility reconstruction differs')
    summaries, walls = {}, 0
    for arm in names:
        differences, candidate, baseline = [], [], []
        for block in range(64):
            seat_differences = []
            for seat in range(4):
                c = actual[f'block-{block:03d}-seat-{seat}-{arm}']
                b = actual[f'block-{block:03d}-seat-{seat}-baseline']
                n = min(len(c['ordinal_wall_fingerprints']), len(b['ordinal_wall_fingerprints']))
                require(n > 0 and c['ordinal_wall_fingerprints'][:n] == b['ordinal_wall_fingerprints'][:n], 'Shared physical walls differ')
                walls += n
                candidate.append(c['utilities'][seat])
                baseline.append(b['utilities'][seat])
                seat_differences.append(candidate[-1] - baseline[-1])
            differences.append(sum(seat_differences) / 4)
        values = np.asarray(differences)
        mean = float(values.mean())
        half = 6 * math.sqrt(math.log(40) / (2 * 64))
        rng = np.random.default_rng(20261023031)
        bootstrap = [float(rng.choice(values, size=64, replace=True).mean()) for _ in range(5000)]
        summaries[arm] = {'mean_utility': float(np.mean(candidate)), 'baseline_mean_utility': float(np.mean(baseline)),
            'mean_rank': 2.5 - float(np.mean(candidate)), 'block_utility_differences': differences,
            'mean_utility_difference': mean, 'standard_error': float(values.std(ddof=1) / 8),
            'fixed_budget_hoeffding_95': [max(-3., mean - half), min(3., mean + half)],
            'bootstrap_percentile_95_exploratory': np.percentile(bootstrap, [2.5, 97.5]).tolist()}
    best = max(names, key=lambda name: (summaries[name]['mean_utility'], -names.index(name)))
    selected = best if summaries[best]['mean_utility_difference'] > 0 else None
    return {'arms': summaries, 'selected_candidate': selected, 'best_observed_arm': best,
            'shared_physical_wall_pairs_checked': walls}


def audit_development(report_path, arms, arms_sha, deadline):
    report = read(report_path)
    require(report['complete'] is True and report['policy_promotion'] is False and report['confirmation_started'] is False, 'Development report incomplete or promoted')
    manifest_path = artifact(report['manifest_path'], report['manifest_sha256'])
    manifest = read(manifest_path)
    execution = manifest['execution_plan']
    plan_path = ROOT / 'reports/stage3g/plan.json'
    plan = read(plan_path)
    require(execution['plan_sha256'] == hash_file(plan_path) and execution['arms_manifest_sha256'] == arms_sha,
            'Development is not bound to the frozen plan and completed arms')
    expected_jobs = []
    for block, seed in enumerate(plan['evaluation_seeds']):
        for seat in range(4):
            for arm in ['baseline', *[item['name'] for item in arms['arms']]]:
                lineup = [''] * 4
                lineup[seat] = 'reference' if arm == 'baseline' else arm
                for offset, opponent in enumerate(('shanten_v2', 'defense_v2', 'reference'), 1):
                    lineup[(seat + offset) % 4] = opponent
                expected_jobs.append({'match_id': f'block-{block:03d}-seat-{seat}-{arm}', 'block': block,
                    'focal_seat': seat, 'arm': arm, 'seed': seed, 'lineup': lineup})
    require(execution['matches'] == expected_jobs, 'Development jobs differ from the prespecified seeds/rotations')
    require(manifest['complete'] is True and manifest['failure'] is None and manifest['policy_promotion'] is False, 'Development manifest incomplete')
    artifact(manifest_path.parent / 'execution-plan.json', manifest['execution_plan_sha256'])
    require(read(manifest_path.parent / 'execution-plan.json') == execution, 'Development execution evidence differs')
    require([item['name'] for item in execution['arms']] == [item['name'] for item in arms['arms']]
            and execution['skipped_arms'] == arms['skipped_arms'], 'Completed/skipped arm panel differs')
    for actual, declared in zip(execution['arms'], arms['arms']):
        require(actual['sha256'] == declared['sha256'], 'Development used a different checkpoint')
    for record in manifest['matches']:
        check_deadline(deadline)
        path = manifest_path.parent / 'reports' / (record['match_id'] + '.json')
        require(read(path) == record, 'Development standalone and manifest records differ')
        artifact(manifest_path.parent / record['artifact']['path'], record['artifact']['sha256'])
        deterministic = {k: v for k, v in record.items() if k not in ('payload_sha256', 'elapsed_seconds')}
        require(hashlib.sha256(canonical(deterministic)).hexdigest() == record['payload_sha256'], 'Development deterministic payload differs')
    arithmetic = development_arithmetic(execution, manifest['matches'])
    saved = manifest['statistics']
    require(saved == report['statistics'] and saved['selected_candidate'] == arithmetic['selected_candidate']
            and saved['best_observed_arm'] == arithmetic['best_observed_arm'] and saved['blocks'] == 64
            and saved['games'] == len(manifest['matches']) and saved['confirmation_started'] is False
            and saved['policy_promotion'] is False, 'Development selection or reporting differs')
    for name, summary in arithmetic['arms'].items():
        stored = saved['arms'][name]
        for key in ('mean_utility', 'baseline_mean_utility', 'mean_rank', 'block_utility_differences'):
            close(stored[key], summary[key], 'Development arm outcome arithmetic differs')
        for key in ('mean_utility_difference', 'standard_error', 'fixed_budget_hoeffding_95', 'bootstrap_percentile_95_exploratory'):
            close(stored['statistics'][key], summary[key], 'Development uncertainty arithmetic differs')
    return {'valid': True, 'games': len(manifest['matches']), 'blocks': 64, 'rotations': 4,
        'selected_candidate': arithmetic['selected_candidate'], 'best_observed_arm': arithmetic['best_observed_arm'],
        'shared_physical_wall_pairs_checked': arithmetic['shared_physical_wall_pairs_checked'],
        'manifest_sha256': report['manifest_sha256'], 'policy_promotion': False, 'confirmation_started': False,
        'scope': 'All saved report identities and terminal rank/utility/block/selection/interval arithmetic; artifact hashes and common ordinal physical-wall commitments. Action-by-action evaluation log reconstruction is separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms-manifest', required=True)
    parser.add_argument('--evaluation-report', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-seconds', type=float, default=120)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), 'Audit output already exists')
    started = time.monotonic()
    result = {'schema': 'riichi-ai-stage3g-learning-audit-v1', 'valid': False,
        'optimizer_steps_performed': 0, 'new_matches': 0}
    try:
        from stage3g import read_config, verify_sources
        config, config_sha = read_config()
        data = Path(config['data_root'])
        active = [item for item in read(data / 'budget-ledger.json')['operations'] if item['status'] == 'active']
        require(len(active) == 1 and active[0]['name'].startswith('audit')
                and active[0]['name'] == os.environ.get('RIICHI_STAGE3G_OPERATION')
                and active[0]['token'] == os.environ.get('RIICHI_STAGE3G_TOKEN'), 'Matching active audit lease required')
        require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 120, 'Audit deadline outside fixed 120-second allowance')
        remaining = active[0]['reserved_seconds'] - (time.time() - active[0]['started_unix'])
        deadline = started + min(args.max_seconds, remaining - 10)
        sources = read(args.source_manifest)['sources']
        require('scripts/audit_stage3g_learning.py' in sources and 'tests/test_stage3g_evidence.py' in sources, 'Audit source/test membership missing')
        verify_sources(sources)
        plan = read(ROOT / 'reports/stage3g/plan.json')
        verify_sources(plan['sources'])
        torch.set_num_threads(1)
        arms = read(args.arms_manifest)
        require(arms['schema'] == 'stage3g-arms-v1', 'Unknown completed arm manifest')
        names = [item['name'] for item in arms['arms']]
        require(names in [list(ARMS), list(ARMS[:2])] and arms['skipped_arms'] == [name for name in ARMS if name not in names], 'Require fixed complete or calibration-fallback factorial')
        corpus = read(data / 'corpus/manifest.json')
        require(corpus['complete'] is True and corpus['schema'] == 'riichi-ai-stage3g-corpus-v1'
                and len(corpus['splits']['training']) == 3200 and len(corpus['splits']['development']) == 800, 'Formal corpus shape/identity differs')
        result['grp'] = audit_grp(data, corpus, deadline)
        require(arms['delta_gate_passed'] == result['grp']['calibration_gate_passed']
                and len(names) == (4 if arms['delta_gate_passed'] else 2), 'Wrong calibration fallback arm exposure')
        result['learning'] = audit_learning(data, corpus, arms, result['grp']['target_sha256'], deadline)
        result['development'] = audit_development(args.evaluation_report, arms, hash_file(args.arms_manifest), deadline)
        check_deadline(deadline)
        verify_sources(sources)
        result.update(valid=True, config_sha256=config_sha, source_manifest_sha256=hash_file(args.source_manifest),
            arms_manifest_sha256=hash_file(args.arms_manifest), evaluation_report_sha256=hash_file(args.evaluation_report),
            corpus_manifest_sha256=hash_file(data / 'corpus/manifest.json'),
            development_manifest_sha256=result['development']['manifest_sha256'])
    except Exception as error:
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
    result['elapsed_seconds'] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'valid': result['valid'], 'scope': 'Saved training, GRP and paired-development evidence; no optimization or games', 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
