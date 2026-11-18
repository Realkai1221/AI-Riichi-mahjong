"""Independent, stdlib-only integrity audit of the bounded Stage 3A pilot.

No model, numerical package or engine is imported. Stored tensors are inspected
as NPY byte records; numerical model replay remains separately reported worker
evidence. Integrity and the engineering outcome are deliberately separate.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'stage3a-complete-match-reinforce-v1'
REFERENCE = '5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f'
# Original resolved profile, preserved in reports/stage1/manifest.json and the
# immutable continuation snapshot; no engine import is needed to check identity.
RULE_PROFILE = 'c3f043568af2dcefc1be8de65fd241311e53347e56401f3ac09f5edcde3dbe98'
ACTION_KINDS = ('DISCARD', 'CHI', 'PON', 'DAIMINKAN', 'ANKAN', 'KAKAN', 'RIICHI',
                'RON', 'TSUMO', 'KYUSHU_KYUHAI', 'PASS')
FIELDS = {'tiles', 'globals', 'history', 'history_mask', 'action_tokens', 'candidate_efficiency'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def object_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class Checks:
    def __init__(self):
        self.counts = Counter()

    def require(self, condition, message, category='integrity'):
        if not condition:
            raise ValueError(message)
        self.counts[category] += 1


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_artifact(path, expected_sha, data_root, checked):
    path = Path(path)
    checked.require(path.is_absolute() and path.resolve().is_relative_to(data_root.resolve()),
                    'Artifact outside configured external root', 'identity')
    checked.require(path.is_file() and digest(path) == expected_sha,
                    f'Artifact missing or changed: {path}', 'identity')
    return path


def parse_npy(raw):
    """Read only safe, C-order primitive arrays; never deserialize objects."""
    if raw[:6] != b'\x93NUMPY' or len(raw) < 10:
        raise ValueError('Invalid NPY magic/header')
    version = tuple(raw[6:8])
    if version == (1, 0):
        size, offset = struct.unpack('<H', raw[8:10])[0], 10
    elif version in ((2, 0), (3, 0)) and len(raw) >= 12:
        size, offset = struct.unpack('<I', raw[8:12])[0], 12
    else:
        raise ValueError('Unsupported NPY version')
    if size > 1_000_000 or offset + size > len(raw):
        raise ValueError('Invalid NPY header size')
    header = ast.literal_eval(raw[offset:offset + size].decode('utf-8' if version == (3, 0) else 'latin1'))
    if (not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}
            or header['fortran_order'] is not False):
        raise ValueError('Unexpected NPY header or non-C array')
    dtype = {'<f4': ('float32', 4), '<i2': ('int16', 2), '|b1': ('bool', 1)}.get(header['descr'])
    shape = header['shape']
    if dtype is None or not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('Unexpected actor dtype or array shape')
    data = raw[offset + size:]
    if len(data) != math.prod(shape) * dtype[1]:
        raise ValueError('NPY payload length does not match shape')
    return dtype[0], shape, data


def semantic_tokens(key):
    value = json.loads(key)
    if not isinstance(value, list) or len(value) != 6 or value[0] != 'semantic-actions-v1':
        raise ValueError('Invalid semantic candidate key')
    _, kind, tile, red, drawn, consumed = value
    if (kind not in ACTION_KINDS or (tile is not None and (type(tile) is not int or not 0 <= tile < 34))
            or type(red) is not bool or (drawn is not None and type(drawn) is not bool)
            or not isinstance(consumed, list) or len(consumed) > 4):
        raise ValueError('Malformed semantic action fields')
    suffix = []
    for item in consumed:
        if (not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int
                or not 0 <= item[0] < 34 or type(item[1]) is not bool):
            raise ValueError('Malformed consumed semantic tile')
        suffix.append(1 + item[0] + 34 * item[1])
    return [ACTION_KINDS.index(kind) + 1, 0 if tile is None else tile + 1, int(red),
            0 if drawn is None else 2 if drawn else 1, *suffix, *([0] * (4 - len(suffix)))]


def audit_encoded(path, rows, checked):
    expected = {f'row_{i:05d}__{key}.npy' for i in range(len(rows)) for key in FIELDS}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        checked.require(len(names) == len(set(names)) and set(names) == expected,
                        'Actor archive has missing, duplicate or unexpected fields', 'actor_contract')
        for index, row in enumerate(rows):
            count = len(row['semantic_keys'])
            shapes = {'tiles': ('float32', (34, 23)), 'globals': ('float32', (45,)),
                'history': ('int16', (128, 10)), 'history_mask': ('bool', (128,)),
                'action_tokens': ('int16', (count, 8)), 'candidate_efficiency': ('float32', (count, 4))}
            result = hashlib.sha256()
            for key in sorted(FIELDS):
                dtype, shape, payload = parse_npy(archive.read(f'row_{index:05d}__{key}.npy'))
                checked.require((dtype, shape) == shapes[key], f'Actor tensor contract differs: {key}', 'actor_contract')
                result.update(canonical([key, dtype, list(shape)]).encode())
                result.update(payload)
                if dtype == 'float32':
                    checked.require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', payload)),
                                    'Nonfinite encoded feature', 'actor_contract')
                if key == 'action_tokens':
                    tokens = list(struct.unpack('<' + 'h' * (8 * count), payload))
                    wanted = [v for action in row['semantic_keys'] for v in semantic_tokens(action)]
                    checked.require(tokens == wanted, 'Semantic candidates and action tensors differ', 'actor_contract')
                elif key == 'history_mask':
                    checked.require(set(payload) <= {0, 1} and payload == bytes(sorted(payload, reverse=True)),
                                    'History mask is not a valid prefix', 'actor_contract')
            checked.require(result.hexdigest() == row['encoded_sha256'], 'Encoded row byte hash mismatch', 'actor_contract')


def audit_row(row, checked):
    probabilities = row['probabilities']
    count = len(row['semantic_keys'])
    selected = row['selected_index']
    checked.require(count > 0 and len(set(row['semantic_keys'])) == count
        and len(row['representative_tokens']) == len(probabilities) == count
        and type(selected) is int and 0 <= selected < count, 'Invalid legal candidate dimensions', 'behavior')
    checked.require(all(finite(p) and 0 <= p <= 1 for p in probabilities)
        and math.isclose(math.fsum(probabilities), 1, abs_tol=1e-12, rel_tol=0),
        'Malformed behavior probabilities', 'behavior')
    chosen = probabilities[selected]
    checked.require(chosen > 0 and row['probability'] == chosen
        and math.isclose(row['log_probability'], math.log(chosen), abs_tol=1e-12, rel_tol=0),
        'Selected behavior probability/log mismatch', 'behavior')
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p)
    checked.require(math.isclose(row['entropy'], entropy, abs_tol=1e-10, rel_tol=1e-10),
                    'Behavior entropy mismatch', 'behavior')
    checked.require(row['forced'] is (count == 1) and row['eligible'] is True
        and row['fallback'] is False and row['overridden'] is False, 'Ineligible or overridden learner action', 'behavior')
    checked.require(row['selected_semantic_key'] == row['semantic_keys'][selected]
        and row['selected_physical_token'] == row['representative_tokens'][selected],
        'Selected semantic/physical candidate mismatch', 'behavior')
    if count == 1:
        checked.require(probabilities == [1.0] and row['uniform_draw'] is None and row['log_probability'] == 0,
                        'Forced decision must have probability one and no RNG draw', 'behavior')
    else:
        draw = row['uniform_draw']
        checked.require(finite(draw) and 0 <= draw < 1, 'Invalid categorical draw', 'behavior')
        # Sequential addition matches the declared CDF, including its explicit
        # last-positive-bin rounding rule. Zero bins never receive a draw.
        total, expected = 0.0, None
        for index, probability in enumerate(probabilities):
            total += probability
            if draw < total:
                expected = index
                break
        if expected is None:
            expected = max(i for i, probability in enumerate(probabilities) if probability > 0)
        checked.require(selected == expected, 'CDF draw and selected action differ', 'behavior')


def audit_score_chain(events, checked):
    """Independent ledger of the pinned RiichiEnv 0.4.10 MJAI score schema.

    The upstream terminal event is a bare marker. Accepted riichi payments,
    result transfers and successive start-hand snapshots provide the evidence.
    Its exhaustive/nagashi score_delta quirk is corrected only in the explicit
    branches documented in pinned state/mod.rs::_trigger_ryukyoku.
    """
    checked.require(bool(events) and events[-1].get('type') == 'end_game'
        and sum(e.get('type') == 'end_game' for e in events) == 1,
        'Score ledger requires exactly one final end_game marker', 'terminal')
    balances, pot, paid = None, 0, [0] * 4
    hands = declarations = results = 0
    corrections = []
    for event in events:
        kind = event.get('type')
        if kind == 'start_kyoku':
            values, sticks = event.get('scores'), event.get('kyotaku')
            checked.require(isinstance(values, list) and len(values) == 4
                and all(type(v) is int for v in values) and type(sticks) is int and sticks >= 0,
                'Malformed start-hand score ledger', 'terminal')
            checked.require(sum(values) + sticks * 1000 == 100000,
                'Start-hand scores/deposits do not conserve', 'terminal')
            if balances is not None:
                checked.require(values == balances and sticks * 1000 == pot,
                    'Successive hand score/deposit snapshots disagree', 'terminal')
            balances, pot, paid = values.copy(), sticks * 1000, [0] * 4
            hands += 1
        elif kind == 'reach_accepted':
            seat = event.get('actor')
            checked.require(balances is not None and type(seat) is int and seat in range(4)
                and paid[seat] == 0, 'Invalid or duplicate accepted riichi payment', 'terminal')
            balances[seat] -= 1000
            pot += 1000
            paid[seat] = -1000
            declarations += 1
        elif kind in ('hora', 'ryukyoku'):
            raw = event.get('deltas')
            checked.require(balances is not None and isinstance(raw, list) and len(raw) == 4
                and all(type(v) is int for v in raw), 'Malformed hand-result transfers', 'terminal')
            transfer = raw.copy()
            if kind == 'ryukyoku':
                reason = event.get('reason')
                if reason == 'exhaustive_draw' and sum(raw) < 0:
                    checked.require(raw == paid, 'Unexplained exhaustive-draw deductions', 'terminal')
                    transfer = [0] * 4
                elif reason == 'nagashimangan':
                    transfer = [raw[seat] - paid[seat] for seat in range(4)]
                checked.require(sum(transfer) == 0, 'Draw transfers must be zero-sum after declared correction', 'terminal')
                if transfer != raw:
                    corrections.append({'hand': hands, 'reason': reason,
                        'raw_deltas': raw, 'effective_deltas': transfer})
            balances = [balance + delta for balance, delta in zip(balances, transfer)]
            pot -= sum(transfer)
            checked.require(pot >= 0 and pot % 1000 == 0 and sum(balances) + pot == 100000,
                'Result transfers conflict with outstanding deposits', 'terminal')
            results += 1
    checked.require(balances is not None, 'Missing starting-hand score evidence', 'terminal')
    recipient = sorted(range(4), key=lambda seat: (-balances[seat], seat))[0]
    balances[recipient] += pot
    if 'scores' in events[-1]:
        checked.require(events[-1]['scores'] == balances, 'Optional terminal scores differ from score ledger', 'terminal')
    return balances, {'method': 'start-hand scores + accepted riichi + result deltas + final deposit award',
        'hands': hands, 'accepted_riichi': declarations, 'hand_results': results,
        'upstream_log_delta_corrections': corrections, 'terminal_deposit_award': pot,
        'terminal_deposit_recipient': recipient if pot else None}


def audit_match(evidence, planned, batch_index, before, opponent, protocol_sha,
                data_root, checked):
    record_path = checked_artifact(evidence['json_path'], evidence['json_sha256'], data_root, checked)
    array_path = checked_artifact(evidence['npz_path'], evidence['npz_sha256'], data_root, checked)
    record = read(record_path)
    checked.require(record['schema'] == SCHEMA and record['protocol_sha256'] == protocol_sha
        and record['batch_index'] == batch_index, 'Match schema/protocol/batch mismatch', 'identity')
    checked.require(all(record[key] == value for key, value in planned.items()), 'Match differs from frozen seed/seat plan', 'identity')
    checked.require(record['policy_id'] == before['policy_id'] and record['sampler'] == before['sampler'],
                    'Match has wrong behavior policy identity', 'identity')
    checked.require(record['rule_profile_sha256'] == RULE_PROFILE
        and record['view_hash_schema'] == 'riichi-ai-player-view-v2'
        and record['max_steps'] == 100000, 'Rules, player-view contract or match ceiling changed', 'identity')
    checked.require(record['encoded_artifact'] == {'path': str(array_path), 'sha256': evidence['npz_sha256'],
                    'rows': len(record['learner_rows'])}, 'Actor artifact association mismatch', 'identity')
    focal = record['learner_seat']
    policies = record['policies']
    checked.require(len(policies) == 4 and 0 <= focal < 4, 'Invalid focal seat/policy count', 'provenance')
    expected_opponent = {**opponent, 'role': 'frozen_opponent', 'policy_id': 'frozen-argmax:' + REFERENCE}
    for seat, policy in enumerate(policies):
        if seat != focal:
            checked.require(policy == expected_opponent, 'Frozen opponent identity changed', 'provenance')
        else:
            checked.require(policy['role'] == 'learner' and policy['policy_id'] == before['policy_id']
                and policy['weights_sha256'] == before['weights_sha256'] and policy['update'] == batch_index
                and policy['reference_sha256'] == REFERENCE and policy['sampling_seed'] == planned['sampling_seed']
                and policy['feature_schema'] == 'actor-features-v2', 'Learner seat manifest mismatch', 'provenance')
    log = record['auditor_log']
    checked.require(object_hash(log) == record['auditor_log_sha256'], 'Auditor-log hash mismatch', 'terminal')
    checked.require(record['hands'] == sum(e.get('type') == 'start_kyoku' for e in log),
                    'Hand count must include dealer continuations', 'terminal')
    if record['completed']:
        checked.require(record['terminated'] is True and record['truncated'] is False
            and record['failure'] is None and record['stop_reason'] == 'completed'
            and record['terminal_result_verified'] is True and record['conservation_verified'] is True,
            'Completed-match termination evidence differs', 'terminal')
        scores = record['scores']
        checked.require(len(scores) == 4 and all(type(x) is int for x in scores), 'Invalid terminal scores', 'terminal')
        order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
        ranks = [order.index(seat) + 1 for seat in range(4)]
        utilities = [2.5 - rank for rank in ranks]
        terminals = [event for event in log if event.get('type') == 'end_game']
        reconstructed_scores, reconstruction = audit_score_chain(log, checked)
        checked.require(len(terminals) == 1 and record['terminal_event'] == terminals[0]
            and reconstructed_scores == record['reconstructed_scores'] == scores
            and reconstruction == record['terminal_score_reconstruction'],
            'Full-match terminal score ledger missing or different', 'terminal')
        checked.require(record['ranks'] == record['reconstructed_ranks'] == ranks
            and record['utilities'] == utilities and record['learner_utility'] == utilities[focal],
            'Independent full-match rank utility mismatch', 'terminal')
        checked.require(sum(scores) + 1000 * record['riichi_sticks'] == 100000,
                        'Terminal points/deposits do not conserve', 'terminal')
    else:
        checked.require(record['terminated'] is False and record['learner_utility'] is None
            and record['ranks'] is None and record['utilities'] is None,
            'Incomplete match received terminal labels', 'terminal')
    rows, requests = record['learner_rows'], record['requests']
    rng = random.Random(planned['sampling_seed'])
    checked.require(object_hash(rng.getstate()) == object_hash(record['sampling_rng_start']),
                    'Initial per-match RNG state differs', 'behavior')
    for index, row in enumerate(rows):
        checked.require(row['row_index'] == index and row['match_index'] == planned['match_index']
            and row['batch_index'] == batch_index and row['seat'] == focal
            and row['policy_id'] == before['policy_id'], 'Learner row identity mismatch', 'provenance')
        checked.require(type(row['hand_id']) is int and 1 <= row['hand_id'] <= record['hands']
            and (index == 0 or row['hand_id'] >= rows[index - 1]['hand_id']),
            'Explicit learner hand identities are absent or nonsequential', 'provenance')
        audit_row(row, checked)
        if not row['forced']:
            checked.require(row['uniform_draw'] == rng.random(), 'Categorical RNG stream differs', 'behavior')
    checked.require(object_hash(rng.getstate()) == object_hash(record['sampling_rng_end']),
                    'Final per-match RNG state differs', 'behavior')
    linked = []
    for step, request in enumerate(requests):
        checked.require(request['step'] == step, 'Nonsequential pre-response batches', 'provenance')
        checked.require(set(request['choices']) == set(request['player_views']),
            'Pre-response player snapshots and submitted response seats differ', 'provenance')
        expected_id = object_hash({'match_index': planned['match_index'], 'step': step, 'views': request['player_views']})
        checked.require(request['pre_response_id'] == expected_id, 'Pre-response snapshot identity mismatch', 'provenance')
        for seat_text, choice in request['choices'].items():
            seat = int(seat_text)
            checked.require(seat in range(4) and seat_text in request['player_views']
                and choice['policy_id'] == policies[seat]['policy_id']
                and choice['fallback'] is False and choice['overridden'] is False,
                'Submitted response provenance differs', 'provenance')
            if seat == focal:
                index = choice['row_index']
                checked.require(type(index) is int and 0 <= index < len(rows), 'Unknown learner row link', 'provenance')
                row = rows[index]
                checked.require(choice['eligible'] is True and row['step'] == step
                    and row['pre_response_id'] == expected_id
                    and row['player_view_sha256'] == request['player_views'][seat_text]
                    and choice['semantic_key'] == row['selected_semantic_key']
                    and choice['physical_token'] == row['selected_physical_token']
                    and choice['probability'] == row['probability'], 'Learner row/submitted action mismatch', 'provenance')
                linked.append(index)
            else:
                checked.require(choice['eligible'] is False and choice['probability'] == 1
                    and 'row_index' not in choice, 'Frozen-opponent action entered learner rows', 'provenance')
    if record['completed']:
        checked.require(linked == list(range(len(rows))) and len(requests) == record['steps'],
                        'Completed match has omitted/duplicate learner rows or steps', 'provenance')
    checked.require(record['learner_last_step'] == (rows[-1]['step'] if rows else None)
        and record['opponent_steps_after_last_learner'] == (record['steps'] - 1 - rows[-1]['step'] if rows else record['steps']),
        'Post-learner terminal linkage differs', 'terminal')
    audit_encoded(array_path, rows, checked)
    return {'match_index': record['match_index'], 'completed': record['completed'],
        'learner_utility': record['learner_utility'], 'learner_rows': len(rows),
        'optional_rows': sum(not row['forced'] for row in rows),
        'forced_rows': sum(row['forced'] for row in rows),
        'stored_log_probability_sum': math.fsum(row['log_probability'] for row in rows),
        'optional_entropy_sum': math.fsum(row['entropy'] for row in rows if not row['forced']),
        'opponent_steps_after_last_learner': record['opponent_steps_after_last_learner'],
        'files': [str(record_path), str(array_path)]}


def close(actual, expected, tolerance=1e-8):
    return finite(actual) and finite(expected) and math.isclose(actual, expected,
        abs_tol=tolerance, rel_tol=1e-10)


def audit_state(state, updates, protocol_sha, checked, initial=None):
    checked.require(state['schema'] == SCHEMA and state['protocol_sha256'] == protocol_sha
        and state['reference_sha256'] == REFERENCE and state['device'] == 'cpu'
        and state['sampler'] == 'python-random-float64-cdf-softmax-v1',
        'Learner state protocol differs', 'identity')
    checked.require(state['updates'] == state['optimizer_steps_executed'] == updates
        and state['retired_batches'] == list(range(updates)),
        'Learner update/optimizer/retirement boundary differs', 'update')
    for key in ('weights_sha256', 'optimizer_sha256', 'rank_head_sha256'):
        checked.require(bool(re.fullmatch('[0-9a-f]{64}', state[key])), 'Malformed state digest', 'identity')
    checked.require(state['policy_id'] == f"{SCHEMA}:update-{updates}:{state['weights_sha256']}",
        'Policy identifier does not bind weights and update', 'identity')
    checked.require(state['optimizer_config'] == {'name': 'AdamW', 'learning_rate': 1e-5,
        'weight_decay': 0.0, 'betas': [.9, .999], 'eps': 1e-8, 'foreach': False,
        'fresh_optimizer': True}, 'Optimizer recipe changed', 'update')
    checked.require(state['model_config']['feature_schema'] == 'actor-features-v2',
        'Actor feature schema changed', 'actor_contract')
    if initial is not None:
        checked.require(state['rank_head_sha256'] == initial['rank_head_sha256']
            and state['model_config'] == initial['model_config'],
            'Frozen rank parameters or model architecture changed', 'update')


def audit_update(update, matches, before, planned, config, checked):
    count = config['matches_per_batch']
    checked.require(len(matches) == count and all(m['completed'] for m in matches)
        and len(planned) == count, 'Partial planned batch cannot update', 'update')
    checked.require(update['schema'] == SCHEMA and update['batch_index'] == before['updates']
        and update['planned_matches'] == planned and update['completed_matches'] == count
        and update['loss_denominator_matches'] == count and update['optimizer_steps'] == 1,
        'Update must average exactly the complete planned matches', 'objective')
    checked.require(update['baseline'] == 0 and update['discount'] == 1
        and update['opponent_rows_in_loss'] == 0, 'Reward or eligible-seat objective changed', 'objective')
    optional = sum(m['optional_rows'] for m in matches)
    checked.require(optional > 0 and update['optional_learner_rows'] == optional
        and update['forced_learner_rows'] == sum(m['forced_rows'] for m in matches),
        'Update learner row counts differ', 'objective')
    utilities = [m['learner_utility'] for m in matches]
    stored = [m['stored_log_probability_sum'] for m in matches]
    checked.require(update['match_utilities'] == utilities
        and len(update['stored_behavior_match_log_probability_sums']) == count
        and all(close(a, b) for a, b in zip(update['stored_behavior_match_log_probability_sums'], stored)),
        'Stored collection log probabilities or utilities differ', 'objective')
    stored_loss = -math.fsum(u * s for u, s in zip(utilities, stored)) / count
    checked.require(close(update['stored_behavior_loss'], stored_loss),
        'Original collection objective reconstruction differs', 'objective')
    replayed = update['match_log_probability_sums']
    error = update['selected_log_probability_replay_max_error']
    checked.require(finite(error) and 0 <= error <= config['replay_log_atol']
        and update['selected_log_probability_replay_atol'] == config['replay_log_atol']
        and len(replayed) == count and all(finite(x) for x in replayed),
        'Selected log-probability replay tolerance exceeded', 'replay')
    checked.require(all(abs(a - b) <= m['optional_rows'] * error + 1e-8
        for a, b, m in zip(replayed, stored, matches)),
        'Match replay log sums exceed recorded per-action error', 'replay')
    replayed_loss = -math.fsum(u * s for u, s in zip(utilities, replayed)) / count
    checked.require(close(update['behavior_reconstructed_loss'], replayed_loss)
        and close(update['differentiable_loss'], replayed_loss)
        and close(update['loss_numerical_difference'], abs(update['differentiable_loss'] - replayed_loss)),
        'Differentiable objective bookkeeping differs', 'objective')
    checked.require(update['probability_replay_atol'] == config['replay_atol']
        and update['probability_replay_rtol'] == config['replay_rtol']
        and finite(update['probability_replay_max_absolute_error'])
        and 0 <= update['probability_replay_max_absolute_error'] <= config['replay_atol'] + config['replay_rtol'],
        'Probability replay configuration or bound differs', 'replay')
    checked.require(update['before'] == before, 'Update starts from different state', 'update')
    audit_state(update['after'], before['updates'] + 1, before['protocol_sha256'], checked, before)
    checked.require(update['gradient_clip'] == config['gradient_clip']
        and update['gradients_finite'] is True and update['parameters_finite'] is True
        and update['rank_head_unchanged'] is True, 'Gradient/parameter guard evidence differs', 'update')
    for key in ('gradient_norm_before_clip', 'parameter_max_absolute_change', 'parameter_l2_change',
                'distribution_max_absolute_change', 'mean_optional_entropy_before'):
        checked.require(finite(update[key]) and update[key] >= 0, f'Invalid update diagnostic {key}', 'update')
    checked.require(close(update['mean_optional_entropy_before'],
        math.fsum(m['optional_entropy_sum'] for m in matches) / optional),
        'Original learner entropy differs', 'behavior')
    observed = update['distribution_diagnostic_rows']
    checked.require(type(observed) is int and 0 <= observed <= optional
        and (not update['distribution_diagnostic_complete'] or observed == optional),
        'Distribution diagnostic completion is overstated', 'update')
    for key in ('distribution_mean_l1_change', 'mean_optional_entropy_after'):
        checked.require((observed == 0 and update[key] is None)
            or (observed > 0 and finite(update[key]) and update[key] >= 0),
            'Invalid post-update distribution evidence', 'update')
    checked.require(update['numerical_no_op'] is (update['parameter_max_absolute_change'] == 0
        or update['distribution_max_absolute_change'] == 0), 'No-op classification differs', 'update')
    checked.require((update['parameter_max_absolute_change'] == 0)
        is (before['weights_sha256'] == update['after']['weights_sha256']),
        'Weight-change diagnostic conflicts with state hashes', 'update')
    return {'optimizer_steps': 1, 'stored_behavior_loss': stored_loss,
        'replayed_behavior_loss': replayed_loss, 'differentiable_loss': update['differentiable_loss'],
        'loss_numerical_difference': update['loss_numerical_difference'],
        'probability_replay_max_absolute_error': update['probability_replay_max_absolute_error'],
        'selected_log_probability_replay_max_error': error,
        'optional_rows': optional, 'forced_rows': update['forced_learner_rows'],
        'parameter_max_absolute_change': update['parameter_max_absolute_change'],
        'distribution_max_absolute_change': update['distribution_max_absolute_change'],
        'distribution_diagnostic_complete': update['distribution_diagnostic_complete'],
        'numerical_no_op': update['numerical_no_op']}


def audit_budget(config, protocol_sha, reports, data_root, checked):
    raw = read(data_root / 'budget-ledger.json')
    saved = read(reports / 'budget.json')
    checked.require(raw['schema'] == 'package-a2-budget-v1' and raw['identity'] == protocol_sha
        and raw['limit_seconds'] == 600 and all(saved.get(k) == v for k, v in raw.items()),
        'Live/saved ledger identity or operations differ', 'budget')
    operations = raw['operations']
    checked.require(len({item['name'] for item in operations}) == len(operations)
        and len({item['token'] for item in operations}) == len(operations),
        'Duplicate lease name or token', 'budget')
    charged = 0.0
    markers, peaks, artifacts = {}, [], []
    for index, operation in enumerate(operations):
        name = operation['name']
        checked.require(operation['id'] == index and operation['status'] in ('completed', 'failed')
            and finite(operation['elapsed_seconds']) and operation['elapsed_seconds'] >= 0,
            'Unclosed, malformed or nonsequential numerical lease', 'budget')
        maximum = config['test_allowance_seconds'] if name.startswith('check:') else (
            config['prepare_allowance_seconds'] if name == 'prepare' else config['per_batch_allowance_seconds'])
        checked.require(finite(operation['reserved_seconds']) and 0 < operation['reserved_seconds'] <= maximum
            and operation['reserved_seconds'] <= 600 - charged - config['reserve_seconds'] + 1e-8,
            'Lease exceeds reservation, operation or shutdown-reserve bounds', 'budget')
        charged += operation['elapsed_seconds']
        checked.require(operation['exceeded_reservation'] is (operation['elapsed_seconds'] > operation['reserved_seconds']),
            'Lease overrun flag differs', 'budget')
        if name == 'prepare':
            marker_path = reports / 'prepare-attempt.json'
        elif re.fullmatch(r'batch:[0-3]', name):
            marker_path = reports / f"attempt-{name.split(':')[1]}.json"
        elif re.fullmatch(r'check:[a-zA-Z0-9_-]+', name):
            marker_path = reports / f"checks-{name.split(':')[1]}.json"
        else:
            raise ValueError('Unexpected operation outside Stage 3A scope')
        marker = read(marker_path)
        checked.require(marker['name'] == name and marker['config_sha256'] == protocol_sha
            and marker['status'] == operation['status'] and marker['failure'] == operation['failure'],
            'Operation marker and durable lease differ', 'budget')
        prefix = marker['budget']
        checked.require(prefix['operations'] == operations[:index + 1]
            and prefix['identity'] == protocol_sha and prefix['limit_seconds'] == 600
            and close(prefix['charged_seconds'], charged)
            and close(prefix['remaining_seconds'], max(0, 600 - charged)),
            'Operation budget snapshot is not its closed ledger prefix', 'budget')
        watchdog = marker['watchdog']
        checked.require(finite(watchdog['elapsed_seconds']) and watchdog['elapsed_seconds'] >= 0
            and watchdog['elapsed_seconds'] <= operation['elapsed_seconds'] + 1e-8
            and type(watchdog['sampled_peak_tree_rss_bytes']) is int
            and watchdog['sampled_peak_tree_rss_bytes'] >= 0
            and type(watchdog['artifact_bytes_peak']) is int and watchdog['artifact_bytes_peak'] >= 0,
            'Malformed watchdog resource evidence', 'budget')
        if operation['status'] == 'completed':
            checked.require(watchdog['exit_code'] == 0 and operation['failure'] is None,
                'Completed operation has failure evidence', 'budget')
        else:
            checked.require(operation['failure'] is not None, 'Failed operation omitted failure evidence', 'budget')
        peaks.append(watchdog['sampled_peak_tree_rss_bytes'])
        artifacts.append(watchdog['artifact_bytes_peak'])
        markers[name] = marker
    expected_marker_names = {'prepare-attempt.json' if name == 'prepare' else
        f"attempt-{name.split(':')[1]}.json" if name.startswith('batch:') else
        f"checks-{name.split(':')[1]}.json" for name in markers}
    actual_marker_names = {p.name for pattern in ('prepare-attempt.json', 'attempt-*.json', 'checks-*.json')
        for p in reports.glob(pattern)}
    checked.require(expected_marker_names == actual_marker_names, 'Uncharged or omitted operation marker', 'budget')
    checked.require(charged <= 600 and close(saved['charged_seconds'], charged)
        and close(saved['remaining_seconds'], 600 - charged), 'Cumulative numerical time differs or exceeds ceiling', 'budget')
    return {'limit_seconds': 600, 'charged_seconds': charged, 'remaining_seconds': 600 - charged,
        'operations': len(operations), 'completed_operations': sum(x['status'] == 'completed' for x in operations),
        'failed_operations': sum(x['status'] == 'failed' for x in operations),
        'all_closed': True, 'sampled_peak_tree_rss_bytes': max(peaks, default=0),
        'artifact_bytes_peak': max(artifacts, default=0),
        'memory_within_limit': max(peaks, default=0) < config['memory_bytes'],
        'artifacts_within_sampled_limit': max(artifacts, default=0) < config['artifact_bytes'],
        'all_reservations_respected': all(not x['exceeded_reservation'] for x in operations)}, markers


def audit(config_path, expected_config_sha256, *, root=ROOT):
    checked = Checks()
    config_path, root = Path(config_path).resolve(), Path(root).resolve()
    protocol_sha = digest(config_path)
    checked.require(protocol_sha == expected_config_sha256, 'Config does not match pinned protocol SHA', 'identity')
    config = read(config_path)
    exact = {'schema': 'stage3a-config-v1', 'limit_seconds': 600, 'memory_bytes': 32 * 1024**3,
        'artifact_bytes': 1_000_000_000, 'reserve_seconds': 45, 'per_batch_allowance_seconds': 100,
        'prepare_allowance_seconds': 30, 'test_allowance_seconds': 90, 'device': 'cpu', 'torch_threads': 1,
        'maximum_batches': 4, 'matches_per_batch': 16, 'maximum_matches': 64, 'maximum_updates': 4,
        'learning_rate': 1e-5, 'weight_decay': 0., 'gradient_clip': 1., 'replay_atol': 1e-6,
        'replay_rtol': 1e-5, 'replay_log_atol': 1e-4, 'microbatch_size': 1,
        'max_steps': 100000, 'reference_sha256': REFERENCE}
    checked.require(all(config.get(k) == v for k, v in exact.items()), 'Authorized recipe or ceiling changed', 'scope')
    scope = {'book_training': False, 'test_split_loading': False, 'external_acquisition': False,
        'paid_compute': False, 'fresh_strength_evaluation': False, 'opponent_actions_eligible': False,
        'baseline': 'zero', 'entropy_bonus': 0., 'reward_shaping': False,
        'replay_reuse': False, 'rank_head_frozen': True}
    checked.require(config['constraints'] == scope, 'Experiment scope changed', 'scope')
    def seed(kind, index):
        return int.from_bytes(hashlib.sha256(f'riichi-ai-stage3a-20261018:{kind}:{index}'.encode()).digest()[:8], 'big')
    expected_batches = [{'batch_index': batch, 'matches': [{'match_index': index,
        'seed': seed('wall', index), 'sampling_seed': seed('sampling', index), 'learner_seat': index % 4}
        for index in range(batch * 16, batch * 16 + 16)]} for batch in range(4)]
    checked.require(config['batches'] == expected_batches, 'Prespecified match membership/order changed', 'scope')
    data_root = (root / config['data_root']).resolve()
    reports = (root / config['reports_root']).resolve()
    checked.require(data_root.is_dir() and reports.is_dir() and not data_root.is_relative_to(reports)
        and not reports.is_relative_to(data_root), 'Missing or overlapping evidence roots', 'identity')
    reference = (root / config['reference_checkpoint']).resolve()
    checked.require(digest(reference) == REFERENCE, 'Original reference checkpoint changed', 'identity')
    plan_path = reports / 'plan.json'
    plan, plan_sha = read(plan_path), digest(plan_path)
    checked.require(plan['schema'] == 'stage3a-plan-v1' and plan['config_sha256'] == protocol_sha
        and plan['frozen_before_formal_collection'] is True and plan['batches'] == expected_batches
        and plan['constraints'] == scope and plan['reference_sha256'] == REFERENCE
        and Path(plan['reference_checkpoint']).resolve() == reference, 'Frozen plan differs from protocol', 'identity')
    paths = sorted((root / 'src/riichi_ai').glob('*.py')) + [root / 'scripts/stage3a.py',
        root / 'scripts/package_a_stability.py', root / 'pyproject.toml']
    implementation = {str(p.relative_to(root)): digest(p) for p in paths}
    checked.require(plan['implementation'] == implementation, 'Frozen implementation changed', 'identity')
    budget, markers = audit_budget(config, protocol_sha, reports, data_root, checked)
    checked.require('prepare' in markers and markers['prepare']['status'] == 'completed'
        and markers['prepare']['implementation'] == implementation, 'Prepared state has no successful frozen lease', 'identity')
    initial = plan['initial_state']
    audit_state(initial, 0, protocol_sha, checked)
    boundary = plan['initial_boundary']
    checked.require({k: v for k, v in boundary.items() if k not in ('path', 'sha256')} == initial,
        'Initial boundary/state association differs', 'identity')
    checked_artifact(boundary['path'], boundary['sha256'], data_root, checked)
    opponent = plan['opponent']
    checked.require(opponent['checkpoint_sha256'] == REFERENCE
        and opponent['weights_sha256'] == initial['weights_sha256']
        and opponent['decision_rule'] == 'stable_semantic_argmax'
        and opponent['model_config'] == initial['model_config'],
        'Opponent does not use original reference', 'identity')
    formal_names = [name for name in markers if name.startswith('batch:')]
    checked.require(formal_names == [f'batch:{i}' for i in range(len(formal_names))]
        and len(formal_names) <= 4, 'Formal attempt order/count differs', 'scope')
    checked.require({p.name for p in reports.glob('batch-*.json')} <=
        {f'batch-{i}.json' for i in range(len(formal_names))}, 'Uncharged formal batch report', 'scope')
    summaries, all_matches, executed, unknown_steps = [], [], 0, 0
    state = initial
    for index, name in enumerate(formal_names):
        marker = markers[name]
        checked.require(marker['implementation'] == implementation,
            'Formal attempt used different implementation', 'identity')
        checked.require(marker['started_unix'] >= markers['prepare']['finished_unix'],
            'Formal collection began before preparation finished', 'identity')
        if index:
            checked.require(summaries[-1]['complete'] and markers[f'batch:{index - 1}']['status'] == 'completed',
                'Formal work continued after failed/incomplete batch', 'scope')
        path = reports / f'batch-{index}.json'
        if not path.exists():
            checked.require(marker['status'] == 'failed', 'Completed batch report missing', 'identity')
            unknown_steps += 1
            summaries.append({'batch_index': index, 'complete': False, 'failure': marker['failure'],
                'optimizer_steps_executed': None, 'missing_worker_report': True})
            continue
        batch = read(path)
        planned = expected_batches[index]['matches']
        checked.require(batch['schema'] == 'stage3a-batch-v1' and batch['batch_index'] == index
            and batch['config_sha256'] == protocol_sha and batch['plan_sha256'] == plan_sha
            and batch['implementation'] == implementation and batch['planned_matches'] == planned
            and batch['opponent'] == opponent, 'Batch plan/source identity differs', 'identity')
        checked.require(batch['input_boundary'] == boundary and batch['input_state'] == state,
            'Sequential checkpoint/optimizer boundary differs', 'update')
        checked.require(len(batch['matches']) <= 16, 'Extra or replacement formal matches', 'scope')
        matches = [audit_match(evidence, planned[offset], index, state, opponent,
            protocol_sha, data_root, checked) for offset, evidence in enumerate(batch['matches'])]
        if 'partial_match' in batch:
            checked.require(len(matches) < 16 and batch['complete'] is False,
                'Unexpected partial trajectory', 'scope')
            matches.append(audit_match(batch['partial_match'], planned[len(matches)], index,
                state, opponent, protocol_sha, data_root, checked))
        all_matches.extend(matches)
        update = None
        if batch['update'] is not None:
            update = audit_update(batch['update'], matches, state, planned, config, checked)
        final = batch.get('final_in_memory_state')
        if final is None:
            checked.require(marker['status'] == 'failed' and not batch['complete'],
                'Completed batch omitted final state', 'update')
            step_delta = None
            unknown_steps += 1
        else:
            step_delta = batch['optimizer_steps_executed_delta']
            checked.require(type(step_delta) is int and step_delta in (0, 1)
                and final['optimizer_steps_executed'] == state['optimizer_steps_executed'] + step_delta
                and batch['completed_update_count_delta'] == final['updates'] - state['updates']
                and 0 <= batch['completed_update_count_delta'] <= step_delta,
                'Actual optimizer-step accounting differs', 'update')
            executed += step_delta
            if step_delta == 0:
                checked.require(final == state and update is None,
                    'No-step batch changed model/optimizer state', 'update')
            if update is not None:
                checked.require(step_delta == 1 and final == batch['update']['after'],
                    'Update report/final in-memory state differs', 'update')
        if batch['complete']:
            checked.require(marker['status'] == 'completed' and update is not None
                and batch['boundary_roundtrip_verified'] is True and batch['sources_unchanged'] is True
                and batch['output_state'] == final, 'Completed batch lacks verified saved update', 'update')
            boundary = batch['output_boundary']
            checked.require({k: v for k, v in boundary.items() if k not in ('path', 'sha256')} == final,
                'Output checkpoint/state association differs', 'identity')
            checked_artifact(boundary['path'], boundary['sha256'], data_root, checked)
            state = final
        else:
            checked.require(marker['status'] == 'failed', 'Incomplete batch marked successful', 'update')
            # A failed post-step save is still an executed update; a killed
            # process without final state remains explicitly unknown.
            if batch.get('output_boundary') is not None:
                checked_artifact(batch['output_boundary']['path'], batch['output_boundary']['sha256'], data_root, checked)
        summaries.append({'batch_index': index, 'complete': batch['complete'],
            'matches_persisted': len(matches), 'matches_completed': sum(m['completed'] for m in matches),
            'optimizer_steps_executed': step_delta, 'update': update, 'failure': batch.get('failure'),
            'report_sha256': digest(path)})
    checked.require(executed + unknown_steps <= 4 and len(all_matches) <= 64,
        'Pilot exceeds authorized match/update count', 'scope')
    actual_bytes = sum(p.stat().st_size for directory in (data_root, reports)
        for p in directory.rglob('*') if p.is_file())
    budget['current_artifact_bytes'] = actual_bytes
    budget['current_artifacts_within_limit'] = actual_bytes < config['artifact_bytes']
    complete_updates = [x['update'] for x in summaries if x.get('complete') and x.get('update')]
    engineering_checks = {'at_least_one_verified_update': bool(complete_updates),
        'all_formal_attempts_completed': bool(summaries) and all(x['complete'] for x in summaries),
        'all_update_diagnostics_complete': bool(complete_updates) and all(x['distribution_diagnostic_complete'] for x in complete_updates),
        'nonzero_parameter_and_distribution_change': bool(complete_updates) and all(not x['numerical_no_op'] for x in complete_updates),
        'executed_step_count_known': unknown_steps == 0,
        'resource_bounds_respected': all(budget[k] for k in ('memory_within_limit',
            'artifacts_within_sampled_limit', 'current_artifacts_within_limit', 'all_reservations_respected'))}
    return {'schema': 'stage3a-independent-audit-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'valid': True, 'config_sha256': protocol_sha, 'plan_sha256': plan_sha,
        'implementation': implementation, 'reference_sha256': REFERENCE,
        'checks': dict(sorted(checked.counts.items())), 'check_count': sum(checked.counts.values()),
        'budget': budget, 'formal_attempts': len(formal_names),
        'completed_batches': sum(x['complete'] for x in summaries),
        'persisted_matches': len(all_matches), 'completed_matches': sum(m['completed'] for m in all_matches),
        'optimizer_steps_executed_known': executed, 'attempts_with_unknown_step_count': unknown_steps,
        'learner_rows': sum(m['learner_rows'] for m in all_matches),
        'optional_learner_rows': sum(m['optional_rows'] for m in all_matches),
        'forced_learner_rows': sum(m['forced_rows'] for m in all_matches),
        'matches_with_opponent_steps_after_last_learner': sum(m['opponent_steps_after_last_learner'] > 0 for m in all_matches),
        'batches': summaries, 'engineering_checks': engineering_checks,
        'engineering_evidence_pass': all(engineering_checks.values()),
        'fixture_checks': {name: {'status': marker['status'], 'command': marker['command'],
            'failure': marker['failure']} for name, marker in markers.items() if name.startswith('check:')},
        'limitations': ['No playing-strength inference; 64 matches and four updates are upper bounds.',
            'Tensor bytes, legal candidate association, provenance and objective arithmetic are independently audited; model replay and gradient execution are worker evidence.',
            'A whitelisted byte contract alone does not prove information hiding; canonical feature source and separate non-leakage fixtures remain necessary.',
            'Frozen rank-head parameter hashes do not imply unchanged rank outputs through the updated shared network.',
            'Sampled process-tree RSS is sampled evidence, not a continuous system-memory maximum.',
            'The engineering checklist covers formal evidence; repaired pre-freeze fixture failures remain visible and need review with their subsequent passing checks.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/stage3a.json'))
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = audit(args.config, args.expected_config_sha256)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
