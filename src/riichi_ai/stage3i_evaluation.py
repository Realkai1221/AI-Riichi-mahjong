"""Stage 3I fixed calibration and paired games; no game-based seed selection.

Predictions on saved explorer decisions are observational diagnostics. The
eligibility vector is the exact collector's explorer propensity, not empirical
coverage or a causal action-value guarantee. Engine state remains auditor-only.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .auditing import PlayerViewHasher, VIEW_HASH_SCHEMA
from .evaluation import _wall_fingerprint, summarize_blocks
from .features import encode_view
from .model import _weights_hash, collate_encoded
from .player import PlayerAdapter
from .policies import policy_manifest
from .rules import audit_physical_tiles, rule_profile_hash
from .semantics import ACTION_KINDS, semantic_actions, teacher_decision
from .simulator import INITIAL_SCORES, create_env, score_total
from .stage3c import terminal_result
from .stage3g_collect import _write

SCHEMA = 'riichi-ai-stage3i-evaluation-match-v1'
EXECUTION_SCHEMA = 'riichi-ai-stage3i-evaluation-execution-v1'
CALIBRATION_SCHEMA = 'riichi-ai-stage3i-calibration-v1'
OPPONENTS = ('shanten_v2', 'defense_v2', 'reference_raw')
CONTROLS = ('reference_raw', 'reference_wrapped')
RESETS = ('reset_raw', 'reset_wrapped')
LAMBDA_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
KINDS = {kind: index + 1 for index, kind in enumerate(ACTION_KINDS)}
WIN_KINDS = {KINDS['RON'], KINDS['TSUMO']}


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path: str | Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def wrapped_index(kinds, raw_index: int) -> int:
    wins = np.flatnonzero(np.isin(np.asarray(kinds), list(WIN_KINDS)))
    return int(wins[0]) if len(wins) else int(raw_index)


def calibration_choice(grid_reports: list[dict]) -> dict:
    """Maximize observed change under fixed ceilings; no positive-change floor."""
    if [row.get('lambda') for row in grid_reports] != list(LAMBDA_GRID):
        raise ValueError('Calibration requires the full frozen lambda grid in order')
    if any(type(row.get('rows')) is not int or row['rows'] < 1 for row in grid_reports):
        raise ValueError('Nonempty calibration rows required')
    if len({row['rows'] for row in grid_reports}) != 1:
        raise ValueError('Every lambda must use identical calibration rows')
    acceptable = []
    for row in grid_reports:
        if any(not math.isfinite(row[key]) or not 0 <= row[key] <= 1
               for key in ('change_rate', 'reference_discard_change_rate')):
            raise ValueError('Invalid calibration change rate')
        if row['change_rate'] <= .03 and row['reference_discard_change_rate'] <= .10:
            acceptable.append(row)
    zero = grid_reports[0]
    if zero['change_rate'] != 0 or zero['reference_discard_change_rate'] != 0:
        raise ValueError('Lambda-zero must preserve the wrapped reference')
    selected = max(acceptable, key=lambda row: (row['change_rate'], -row['lambda']))
    return dict(lambda_=selected['lambda'], change_rate=selected['change_rate'],
                reference_discard_change_rate=selected['reference_discard_change_rate'],
                fallback_zero=selected['lambda'] == 0,
                rule='Maximum observed change under 3% overall and 10% reference-discard ceilings; ties smallest lambda; zero is valid')


def predictive_gate(match_metrics: list[dict]) -> dict:
    """Fixed held-out chosen-action prediction test, never a strength claim."""
    if not match_metrics or len({row['match_id'] for row in match_metrics}) != len(match_metrics):
        raise ValueError('Unique nonempty gate matches required')
    fields = ('v_squared_error_sum', 'va_squared_error_sum', 'vb_squared_error_sum')
    if any(type(row.get('rows')) is not int or row['rows'] <= 0
           or any(not math.isfinite(row[key]) or row[key] < 0 for key in fields)
           for row in match_metrics):
        raise ValueError('Invalid gate errors or row counts')
    count = sum(row['rows'] for row in match_metrics)
    row_means = {key.removesuffix('_squared_error_sum'): sum(row[key] for row in match_metrics)/count for key in fields}
    match_means = {key.removesuffix('_squared_error_sum'): float(np.mean([row[key]/row['rows'] for row in match_metrics])) for key in fields}
    passed = row_means['va'] < row_means['v'] and row_means['va'] < row_means['vb']
    reasons = []
    if not row_means['va'] < row_means['v']:
        reasons.append('action_residual_does_not_improve_value_chosen_row_mse')
    if not row_means['va'] < row_means['vb']:
        reasons.append('action_residual_does_not_improve_state_residual_chosen_row_mse')
    return dict(passed=passed, rows=count, matches=len(match_metrics), row_mse=row_means,
                match_mse=match_means, reasons=reasons,
                rule='Strictly lower decision-weighted held-out MSE for V+A than both V and V+B; match-weighted MSE descriptive',
                interpretation='Predictive eligibility only; chosen logged outcomes cannot identify counterfactual move quality')


def _identity(path, sha):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or not path.is_file()
            or path.is_symlink() or digest(path) != sha):
        raise ValueError('Checkpoint path or hash identity failed')
    return str(path)


def line_up(arm: str, focal: int, candidate_names=()) -> tuple[str, ...]:
    if arm not in (*CONTROLS, *RESETS, *candidate_names) or type(focal) is not int or not 0 <= focal < 4:
        raise ValueError('Invalid evaluation arm or focal seat')
    result = [''] * 4
    result[focal] = arm
    for offset, opponent in enumerate(OPPONENTS, 1):
        result[(focal + offset) % 4] = opponent
    return tuple(result)


def prepare_execution(plan: dict, calibration: dict, config: dict, *, plan_sha256: str,
                      calibration_sha256: str, preflight: bool = False) -> dict:
    """Freeze all gate-passing seeds, controls and fixed reset subset before games."""
    if (calibration.get('schema') != CALIBRATION_SCHEMA or calibration.get('complete') is not True
            or calibration.get('plan_sha256') != plan_sha256):
        raise ValueError('Completed calibration must bind the frozen plan')
    if calibration.get('benchmark_only') is True and not preflight:
        raise ValueError('Benchmark heads cannot enter formal games')
    expected_seeds = plan['training_seeds']
    candidates = calibration.get('seeds', [])
    if [entry.get('seed') for entry in candidates] != expected_seeds:
        raise ValueError('Calibration must account for every fixed training seed in order')
    seeds = plan['preflight_evaluation_seeds' if preflight else 'evaluation_seeds']
    blocks = len(seeds)
    expected_count = config['preflight_evaluation_blocks' if preflight else 'evaluation_blocks']
    if (blocks != expected_count or len(set(seeds)) != blocks or blocks < 1
            or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds)):
        raise ValueError('Evaluation wall blocks differ from the preregistered count')
    normalized, skipped = [], []
    for entry in candidates:
        if entry.get('gate', {}).get('passed') is not True:
            skipped.append(dict(seed=entry['seed'], reason='predictive_gate_failed', details=entry['gate']['reasons']))
            continue
        strength = entry['calibration']['lambda_']
        if strength not in LAMBDA_GRID:
            raise ValueError('Candidate lambda differs from fixed calibration grid')
        if strength == 0:
            skipped.append(dict(seed=entry['seed'], reason='zero_calibrated_correction', details=[]))
            continue
        normalized.append(dict(name=f'seed_{entry["seed"]}', seed=entry['seed'],
                               checkpoint=_identity(entry['checkpoint'], entry['sha256']),
                               sha256=entry['sha256'], strength=strength))
    reset_blocks = list(range(min(config['reset_blocks'], blocks))) if config.get('include_reset_controls') else []
    names = [entry['name'] for entry in normalized]
    jobs = []
    for block, seed in enumerate(seeds):
        for focal in range(4):
            for arm in [*CONTROLS, *names, *(RESETS if block in reset_blocks else ())]:
                jobs.append(dict(match_id=f'block-{block:03d}-seat-{focal}-{arm}', block=block,
                                 focal_seat=focal, arm=arm, seed=seed,
                                 lineup=list(line_up(arm, focal, names))))
    reset = None
    if reset_blocks:
        reset = dict(plan['reset_checkpoint'])
        _identity(reset['path'], reset['sha256'])
    return dict(schema=EXECUTION_SCHEMA, plan_sha256=plan_sha256,
                calibration_sha256=calibration_sha256, phase='preflight' if preflight else 'formal',
                protocol_sha256=plan.get('protocol_sha256'),
                cache_manifest_sha256=plan.get('cache_manifest_sha256'),
                reference_checkpoint=config['reference_checkpoint'], reference_sha256=config['reference_sha256'],
                call_checkpoint=config['call_checkpoint'], call_sha256=config['call_sha256'],
                arms=normalized, skipped_seeds=skipped, reset_checkpoint=reset, reset_blocks=reset_blocks,
                blocks=blocks, rotations=4, workers=config['workers'], matches=jobs, planned_matches=len(jobs),
                fixed_seed_family=expected_seeds, training_eligible=False, ppo_eligible=False,
                policy_promotion=False, seed_selection=False,
                source_role='preflight-only' if preflight else 'development-only',
                wall_pairing='Pinned engine with explicit ordinal physical wall commitments compared across all arms; numeric seeds alone insufficient')


@lru_cache(maxsize=1)
def frozen_policy(reference_path, reference_sha, call_path, call_sha):
    from .stage3i_model import FrozenReference
    torch.set_num_threads(1)
    return FrozenReference.from_checkpoints(reference_path, reference_sha, call_path, call_sha, device='cpu')


@lru_cache(maxsize=3)
def candidate_policy(path, sha):
    from .stage3i_model import load_head_bundle
    torch.set_num_threads(1)
    return load_head_bundle(path, expected_sha256=sha, device='cpu')


@lru_cache(maxsize=1)
def reset_policy(record_json):
    from .stage3h_inference import load_checkpoint
    return load_checkpoint(json.loads(record_json), 'cpu')


def validate_policies(execution):
    frozen = frozen_policy(execution['reference_checkpoint'], execution['reference_sha256'],
                           execution['call_checkpoint'], execution['call_sha256'])
    result = dict(frozen=dict(frozen.metadata), candidates={})
    for item in execution['arms']:
        _, metadata = candidate_policy(item['checkpoint'], item['sha256'])
        if (metadata['seed'] != item['seed'] or metadata['identity']['reference_sha256'] != execution['reference_sha256']
                or metadata['identity']['call_sha256'] != execution['call_sha256']):
            raise ValueError('Candidate seed or frozen actor identity mismatch')
        if execution['phase'] == 'formal' and (metadata.get('complete') is not True
                or metadata.get('benchmark_only') is True
                or metadata['identity']['protocol_sha256'] != execution['protocol_sha256']
                or metadata['identity']['cache_manifest_sha256'] != execution['cache_manifest_sha256']):
            raise ValueError('Only completed heads from this frozen formal programme may play formal games')
        result['candidates'][item['name']] = metadata
    if execution['reset_checkpoint']:
        _, metadata = reset_policy(json.dumps(execution['reset_checkpoint'], sort_keys=True))
        result['reset'] = metadata
    return result


def decide(name, view, frozen, values, reset=None):
    from .stage3i_model import correction_choices
    actions = semantic_actions(view)
    if not actions:
        raise ValueError('Empty legal semantic action set')
    if name in OPPONENTS[:2]:
        decision = teacher_decision(view, defense=name == 'defense_v2')
        if decision.actions != actions:
            raise ValueError('Teacher semantic ordering differs')
        return actions, decision.chosen_index, None, None
    if len(actions) == 1:
        return actions, 0, None, dict(forced=True)
    kinds = np.asarray([KINDS[action.kind] for action in actions], dtype=np.int64)
    encoded = encode_view(view, actions, feature_schema=frozen.actor.config['feature_schema'])
    batch = collate_encoded([encoded], 'cpu')
    if name in RESETS:
        model = reset[0]
        with torch.inference_mode():
            scores, _ = model(batch)
        scores = scores[0].detach().cpu().numpy()
        if not np.isfinite(scores).all() or np.any(np.abs(scores) > 100):
            raise ValueError('Reset values invalid')
        raw = int(scores.argmax())
        index = wrapped_index(kinds, raw) if name == 'reset_wrapped' else raw
        return actions, index, scores.tolist(), dict(raw_index=raw, wrapper=name == 'reset_wrapped')
    with torch.inference_mode():
        logits, context, candidates = frozen.frozen_representations(batch)
        scores = logits[0].detach().cpu().numpy()
        if scores.shape != (len(actions),) or not np.isfinite(scores).all():
            raise ValueError('Reference logits invalid')
        raw = int(scores.argmax())
        if name in CONTROLS:
            index = wrapped_index(kinds, raw) if name == 'reference_wrapped' else raw
            return actions, index, scores.tolist(), dict(raw_reference_index=raw, wrapper=name == 'reference_wrapped')
        heads, metadata, strength = values[name]
        context_candidates = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        residuals = heads.action(torch.cat([context_candidates, candidates], dim=-1))[0].cpu().numpy()
        probabilities = frozen.explorer_probabilities(logits, context, candidates,
                            batch['action_tokens'][:, :, 0], batch['candidate_mask'])[0]
    decision = correction_choices(scores, residuals, probabilities, kinds, strength, wrapped=True)
    evidence = dict(raw_reference_index=decision['raw_reference_index'],
                    wrapped_reference_index=decision['wrapped_reference_index'],
                    raw_corrected_index=decision['raw_corrected_index'], strength=strength,
                    explorer_probabilities=probabilities.tolist(), action_residuals=residuals.tolist(),
                    eligible=np.asarray(decision['eligible']).tolist(), prior=np.asarray(decision['prior']).tolist())
    return actions, decision['selected_index'], scores.tolist(), evidence


def run_one(job: dict, execution: dict, output_dir: str | Path, *, deadline: float,
            max_steps: int = 100_000) -> dict:
    if type(max_steps) is not int or max_steps < 1 or time.monotonic() >= deadline:
        raise ValueError('Invalid match limit or expired admission')
    if job not in execution['matches'] or job['lineup'] != list(line_up(job['arm'], job['focal_seat'], [item['name'] for item in execution['arms']])):
        raise ValueError('Match differs from frozen execution plan')
    directory, name = Path(output_dir), job['match_id']
    started = time.monotonic()
    _write(directory / 'lifecycle' / f'{name}.started.json', json_bytes(dict(schema=SCHEMA, plan=job)))
    env = None
    rows, requests, walls, hand_starts, policies = [], [], [], [], []
    steps, failure = 0, None
    try:
        frozen = frozen_policy(execution['reference_checkpoint'], execution['reference_sha256'],
                               execution['call_checkpoint'], execution['call_sha256'])
        values = {item['name']: (*candidate_policy(item['checkpoint'], item['sha256']), item['strength'])
                  for item in execution['arms'] if item['name'] == job['arm']}
        reset = (reset_policy(json.dumps(execution['reset_checkpoint'], sort_keys=True))
                 if job['arm'] in RESETS else None)
        for seat, policy in enumerate(job['lineup']):
            if policy in OPPONENTS[:2]:
                manifest = policy_manifest(policy, 0)
            elif policy in CONTROLS:
                manifest = dict(**frozen.metadata, choice=policy)
            elif policy in RESETS:
                manifest = dict(checkpoint_sha256=reset[1]['checkpoint_sha256'],
                                weights_sha256=reset[1]['weights_sha256'], choice=policy)
            else:
                metadata = values[policy][1]
                manifest = dict(arm=policy, seed=metadata['seed'],
                                checkpoint_sha256=metadata['checkpoint_sha256'],
                                weights_sha256=metadata['weights_sha256'], strength=values[policy][2],
                                eligibility='actual explorer probability >=0.02 or raw reference argmax',
                                prior_floor=-8, choice='first legal win else clipped-prior-plus-residual argmax')
            policies.append(dict(seat=seat, name=policy, policy_id=policy+':'+hashlib.sha256(json_bytes(manifest)).hexdigest(), manifest=manifest))
        env, observations = create_env(job['seed'])
        adapters = [PlayerAdapter(seat) for seat in range(4)]
        hashers = [PlayerViewHasher() for _ in range(4)]
        while not env.done() and steps < max_steps:
            if time.monotonic() >= deadline:
                raise TimeoutError('Evaluation cooperative match deadline')
            audit_physical_tiles(env)
            if score_total(env) != sum(INITIAL_SCORES):
                raise ValueError('Evaluation point/deposit conservation failure')
            # Collect every snapshot and digest before obtaining any response.
            views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
            if not views or not any(view.legal_actions for view in views.values()):
                raise ValueError('Unfinished evaluation match has no legal request')
            hashes = {str(seat): hashers[seat].digest(view) for seat, view in views.items()}
            response_id = hashlib.sha256(json_bytes(dict(step=steps, player_views=hashes))).hexdigest()
            ordinal = max(view.hand_id for view in views.values())
            if ordinal > len(walls):
                if ordinal != len(walls) + 1:
                    raise ValueError('Missing explicit ordinal hand boundary')
                walls.append(_wall_fingerprint(env))
                view = next(view for view in views.values() if view.hand_id == ordinal)
                start = dict(next(event for event in view.history if event.kind == 'start_kyoku').data)
                hand_starts.append(dict(hand_id=ordinal, **{k: start[k] for k in ('bakaze', 'kyoku', 'honba', 'kyotaku', 'oya', 'scores')}))
            chosen, row_ids = {}, []
            for seat, view in views.items():
                if not view.legal_actions:
                    continue
                actions, index, scores, correction = decide(job['lineup'][seat], view, frozen, values, reset)
                action = actions[index]
                row_ids.append(len(rows))
                rows.append(dict(row_index=len(rows), step=steps, hand_id=view.hand_id, seat=seat,
                    policy_id=policies[seat]['policy_id'], player_view_sha256=hashes[str(seat)],
                    pre_response_id=response_id, semantic_keys=[a.key for a in actions],
                    action_kinds=[a.kind for a in actions], representative_tokens=[a.representative.token for a in actions],
                    selected_index=index, selected_semantic_key=action.key, selected_kind=action.kind,
                    selected_physical_token=action.representative.token, probability=1.,
                    legal_scores=scores, correction=correction, training_eligible=False, ppo_eligible=False))
                chosen[seat] = action.representative
            requests.append(dict(step=steps, pre_response_id=response_id, player_views=hashes, rows=row_ids, submitted=False))
            native = {seat: adapters[seat].resolve(action, observations[seat]) for seat, action in chosen.items()}
            observations = env.step(native)
            requests[-1]['submitted'] = True
            steps += 1
        audit_physical_tiles(env)
        if score_total(env) != sum(INITIAL_SCORES):
            raise ValueError('Terminal evaluation point/deposit conservation failure')
        if not env.done():
            raise RuntimeError('Truncated evaluation match')
        frozen.assert_unchanged()
        for heads, metadata, _ in values.values():
            from .stage3g_learning import _state_hash as state_hash
            if state_hash(heads.state_dict()) != metadata['weights_sha256']:
                raise ValueError('Residual head weights changed during evaluation match')
        if reset is not None and _weights_hash(reset[0]) != reset[1]['weights_sha256']:
            raise ValueError('Reset diagnostic weights changed during evaluation match')
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error), step=steps)
    log = env.mjai_log if env is not None else []
    scores = list(env.scores()) if env is not None else None
    completed = bool(env is not None and env.done() and failure is None)
    try:
        terminal = terminal_result(scores or [], list(env.ranks()) if completed else [], log, completed=completed)
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error), step=steps)
        completed = False
        terminal = terminal_result([], [], log, completed=False)
    auditor = dict(schema=SCHEMA, plan=job, canonical_events=log, ordinal_wall_fingerprints=walls,
                   rule_profile_sha256=rule_profile_hash(), policies=policies, rows=rows, requests=requests,
                   hand_starts=hand_starts, player_view_hash_schema=VIEW_HASH_SCHEMA,
                   terminal=terminal, scores=scores, completed=completed, failure=failure,
                   training_eligible=False, ppo_eligible=False)
    payload = gzip.compress(json_bytes(auditor), mtime=0)
    audit_relative = f'auditor/{name}.json.gz'
    _write(directory / audit_relative, payload)
    report = dict(schema=SCHEMA, **job, completed=completed, failure=failure, steps=steps,
                  rows=len(rows), policies=policies, scores=scores, ranks=terminal['ranks'],
                  utilities=terminal['utilities'], terminal_result_verified=terminal['terminal_result_verified'],
                  ordinal_wall_fingerprints=walls, game_sha256=hashlib.sha256(json_bytes(log)).hexdigest(),
                  chosen_action_counts=dict(sorted(Counter(row['selected_kind'] for row in rows).items())),
                  focal_action_counts=dict(sorted(Counter(row['selected_kind'] for row in rows if row['seat'] == job['focal_seat']).items())),
                  artifact=dict(path=audit_relative, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)),
                  training_eligible=False, ppo_eligible=False, source_role=execution['source_role'])
    report['payload_sha256'] = hashlib.sha256(json_bytes(report)).hexdigest()
    report['elapsed_seconds'] = time.monotonic()-started
    _write(directory / 'reports' / f'{name}.json', json_bytes(report))
    _write(directory / 'lifecycle' / f'{name}.finished.json', json_bytes(dict(schema=SCHEMA, match_id=name,
           completed=completed, payload_sha256=report['payload_sha256'], report_sha256=hashlib.sha256(json_bytes(report)).hexdigest())))
    return report



def _paired_contrast(execution, actual, arm, baseline, blocks):
    differences, arm_utilities, baseline_utilities, pairing = [], [], [], []
    for block in blocks:
        per_seat = []
        for focal in range(4):
            candidate = actual[f'block-{block:03d}-seat-{focal}-{arm}']
            control = actual[f'block-{block:03d}-seat-{focal}-{baseline}']
            a, b = candidate['ordinal_wall_fingerprints'], control['ordinal_wall_fingerprints']
            common = min(len(a), len(b))
            if common < 1 or a[:common] != b[:common]:
                raise ValueError('Shared ordinal physical walls differ across paired arms')
            pairing.append(dict(arm=arm, baseline=baseline, block=block, focal_seat=focal,
                                common_hands=common, paired_common_walls_match=True))
            c, r = candidate['utilities'][focal], control['utilities'][focal]
            per_seat.append(c-r)
            arm_utilities.append(c)
            baseline_utilities.append(r)
        differences.append(sum(per_seat)/4)
    statistics = summarize_blocks(differences, bootstrap_seed=20261024031)
    return dict(arm=arm, baseline=baseline, mean_utility=float(np.mean(arm_utilities)),
                baseline_mean_utility=float(np.mean(baseline_utilities)),
                mean_rank=2.5-float(np.mean(arm_utilities)),
                baseline_mean_rank=2.5-float(np.mean(baseline_utilities)),
                block_utility_differences=differences, statistics=statistics), pairing


def summarize(execution: dict, records: list[dict]) -> dict:
    expected = {job['match_id']: job for job in execution['matches']}
    actual = {record['match_id']: record for record in records}
    if len(actual) != len(records) or set(actual) != set(expected):
        raise ValueError('Evaluation coverage is incomplete or duplicated')
    for name, record in actual.items():
        if (any(record.get(k) != v for k, v in expected[name].items()) or record.get('completed') is not True
                or record.get('failure') is not None or record.get('training_eligible') is not False
                or record.get('ppo_eligible') is not False or record.get('terminal_result_verified') is not True):
            raise ValueError('Invalid completed evaluation record')
        if (not isinstance(record.get('utilities'), list) or sorted(record['utilities']) != [-1.5,-.5,.5,1.5]
                or len(record.get('ordinal_wall_fingerprints', [])) < 1):
            raise ValueError('Missing terminal utilities or physical wall commitments')
    contrasts, pairing = {}, []
    for entry in execution['arms']:
        for baseline in ('reference_wrapped', 'reference_raw'):
            key = entry['name'] + '_vs_' + baseline
            result, checks = _paired_contrast(execution, actual, entry['name'], baseline, range(execution['blocks']))
            result['role'] = 'primary' if baseline == 'reference_wrapped' else 'secondary'
            contrasts[key] = result
            pairing.extend(checks)
    result, checks = _paired_contrast(execution, actual, 'reference_wrapped', 'reference_raw', range(execution['blocks']))
    contrasts['reference_wrapped_vs_reference_raw'] = dict(result, role='wrapper_diagnostic')
    pairing.extend(checks)
    for arm, baseline in [('reset_raw', 'reference_raw'), ('reset_wrapped', 'reference_wrapped'),
                          ('reset_wrapped', 'reset_raw')]:
        if execution['reset_blocks']:
            result, checks = _paired_contrast(execution, actual, arm, baseline, execution['reset_blocks'])
            contrasts[arm + '_vs_' + baseline] = dict(result, role='initialization_wrapper_diagnostic')
            pairing.extend(checks)
    pooled = None
    if execution['arms']:
        # Shared walls induce cross-seed dependence: average seeds inside block,
        # then bootstrap blocks, never pretend seed x block are independent.
        values = np.asarray([contrasts[item['name']+'_vs_reference_wrapped']['block_utility_differences']
                             for item in execution['arms']], dtype=np.float64).mean(axis=0)
        pooled = dict(seeds=[item['seed'] for item in execution['arms']],
                      block_utility_differences=values.tolist(),
                      statistics=summarize_blocks(values, bootstrap_seed=20261024032),
                      interpretation='Mean across all eligible preregistered training seeds within each shared wall block; no best seed selection')
    family_size = len(execution['fixed_seed_family'])
    for entry in execution['arms']:
        item = contrasts[entry['name']+'_vs_reference_wrapped']
        mean, n = item['statistics']['mean_utility_difference'], execution['blocks']
        half = 6 * math.sqrt(math.log(2*family_size/.05)/(2*n))
        item['statistics']['family_adjusted_hoeffding_95'] = [max(-3., mean-half), min(3., mean+half)]
        item['statistics']['family_size_preregistered'] = family_size
    return dict(complete=True, blocks=execution['blocks'], games=len(records), contrasts=contrasts,
                pooled_eligible_seeds=pooled, pairing=pairing, skipped_seeds=execution['skipped_seeds'],
                selected_candidate=None, seed_selection=False, confirmation_started=False, policy_promotion=False,
                limitation='Fixed fresh development game comparison; no independent confirmation or policy promotion. All prediction gates use reused development data.')


def _load_cache_manifest(plan):
    path = Path(plan['cache_manifest_path'])
    if digest(path) != plan['cache_manifest_sha256']:
        raise ValueError('Frozen feature cache manifest hash changed')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'riichi-ai-stage3i-feature-cache-v1'
            or manifest.get('target') != 'one_hand_delta' or manifest.get('behavior_id') != 3
            or manifest.get('complete') is not True or manifest.get('purpose') != 'formal'
            or manifest.get('labels_are_actor_inputs') is not False
            or set(manifest.get('partitions', {})) != {'training', 'selection', 'calibration', 'gate'}):
        raise ValueError('Calibration needs the frozen explorer-only one-hand feature cache')
    partitions = manifest['partitions']
    if partitions != plan['partitions']:
        raise ValueError('Cache partitions differ from the frozen plan')
    if any(manifest.get(key) != plan[key] for key in ('reference_sha256','call_sha256',
                                                    'corpus_manifest_sha256','hand_targets_sha256')):
        raise ValueError('Cache historical feature/target identities differ from the frozen plan')
    feature_sources = manifest.get('feature_sources')
    if (not isinstance(feature_sources,dict) or not feature_sources
            or any(plan['sources'].get(name) != sha for name,sha in feature_sources.items())):
        raise ValueError('Cache feature-preparation source identities changed')
    combined = sum(partitions.values(), [])
    if len(combined) != len(set(combined)):
        raise ValueError('Cache partitions overlap')
    if len(partitions['calibration']) != 200 or len(partitions['gate']) != 400:
        raise ValueError('Exactly 200 calibration and 400 gate matches required')
    by_name = {item['match_id']: item for item in manifest['shards']}
    if len(by_name) != len(manifest['shards']) or set(by_name) != set(combined):
        raise ValueError('Cache shard membership differs from fixed partitions')
    return path, manifest, by_name


def _load_cache_shard(record, manifest_path):
    path = Path(record['path'])
    if not path.is_absolute():
        path = manifest_path.parent / path
    if digest(path) != record['sha256']:
        raise ValueError('Calibration cache shard identity changed')
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    required = {'state_features', 'action_features', 'candidate_offsets', 'chosen_index', 'target',
                'row_indices', 'kinds', 'logits', 'explorer_probabilities', 'steps', 'hand_ids', 'seats'}
    if not required <= arrays.keys() or any(a.dtype.hasobject or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Invalid calibration cache arrays')
    n, c = record['rows'], record['candidates']
    offsets = arrays['candidate_offsets']
    if (arrays['state_features'].shape != (n,256) or arrays['action_features'].shape != (c,128)
            or offsets.shape != (n+1,) or offsets[0] != 0 or offsets[-1] != c
            or np.any(np.diff(offsets) < 2) or arrays['chosen_index'].shape != (n,)
            or np.any(arrays['chosen_index'] < 0) or np.any(arrays['chosen_index'] >= np.diff(offsets))
            or arrays['target'].shape != (n,) or arrays['logits'].shape != (c,)
            or arrays['explorer_probabilities'].shape != (c,) or arrays['kinds'].shape != (c,)):
        raise ValueError('Malformed optional explorer cache shapes')
    return arrays


def _head_predictions(heads, arrays):
    """Same one-row head shapes used at deployment; no actor or critic leakage."""
    count = len(arrays['target'])
    values, states = np.empty(count, np.float32), np.empty(count, np.float32)
    residuals = np.empty(len(arrays['logits']), np.float32)
    with torch.inference_mode():
        for row in range(count):
            start, end = map(int, arrays['candidate_offsets'][row:row+2])
            context = torch.from_numpy(arrays['state_features'][row:row+1]).clone()
            candidates = torch.from_numpy(arrays['action_features'][None, start:end]).clone()
            predicted = heads.predictions(context, candidates)
            values[row] = float(predicted['value'][0])
            states[row] = float(predicted['state'][0])
            residuals[start:end] = predicted['action'][0].cpu().numpy()
    return values, states, residuals


def _all_grid_choices(arrays, residuals):
    """Vectorize lambdas only: stable choices are identical to correction_choices."""
    n = len(arrays['target'])
    choices = np.empty((len(LAMBDA_GRID), n), dtype=np.int16)
    raw_reference = np.empty(n, np.int16)
    wrapped_reference = np.empty(n, np.int16)
    for row in range(n):
        start, end = map(int, arrays['candidate_offsets'][row:row+2])
        z, a = arrays['logits'][start:end].astype(np.float64), residuals[start:end].astype(np.float64)
        q, kinds = arrays['explorer_probabilities'][start:end], arrays['kinds'][start:end]
        if np.any(q < 0) or np.any(q > 1) or not math.isclose(float(q.sum()), 1., abs_tol=1e-12, rel_tol=0):
            raise ValueError('Explorer probabilities are not a normalized distribution')
        raw = int(z.argmax())
        prior, eligible = np.maximum(z-z[raw], -8.), q >= .02
        eligible[raw] = True
        scores = prior[None, :] + np.asarray(LAMBDA_GRID, np.float64)[:,None]*a[None,:]
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite calibration corrected score')
        scores[:,~eligible] = -np.inf
        selected = scores.argmax(axis=1)
        selected[0] = raw
        wins = np.flatnonzero(np.isin(kinds, list(WIN_KINDS)))
        if len(wins):
            selected[:] = wins[0]
        choices[:,row] = selected
        raw_reference[row], wrapped_reference[row] = raw, wrapped_index(kinds, raw)
    return choices, raw_reference, wrapped_reference


def _match_prediction_report(name, arrays, values, states, residuals, choices, reference):
    offsets = arrays['candidate_offsets']
    taken = offsets[:-1] + arrays['chosen_index']
    labels = arrays['target'].astype(np.float64)
    v, b, a = values.astype(np.float64), states.astype(np.float64), residuals[taken].astype(np.float64)
    metrics = dict(match_id=name, rows=len(labels),
                   v_squared_error_sum=float(np.square(v-labels).sum()),
                   va_squared_error_sum=float(np.square(v+a-labels).sum()),
                   vb_squared_error_sum=float(np.square(v+b-labels).sum()),
                   zero_squared_error_sum=float(np.square(labels).sum()))
    reference_kinds = arrays['kinds'][offsets[:-1]+reference]
    discard = reference_kinds == KINDS['DISCARD']
    opportunity = {name: np.asarray([np.isin(arrays['kinds'][start:end], codes).any()
                     for start,end in zip(offsets[:-1],offsets[1:])], dtype=bool)
                   for name,codes in {'call':[KINDS[k] for k in ('CHI','PON','DAIMINKAN')],
                                      'riichi':[KINDS['RIICHI']],
                                      'kan':[KINDS[k] for k in ('DAIMINKAN','ANKAN','KAKAN')],
                                      'win':list(WIN_KINDS)}.items()}
    slices = {'all':np.ones(len(labels),bool),'reference_discard':discard,**opportunity,
              'hand_ordinal_1_4':arrays['hand_ids']<=4,
              'hand_ordinal_5_8':(arrays['hand_ids']>4)&(arrays['hand_ids']<=8),
              'hand_ordinal_9_plus':arrays['hand_ids']>8}
    for kind,code in KINDS.items():
        slices['reference_kind_'+kind.lower()] = reference_kinds == code
    if 'public_globals' in arrays:
        public = arrays['public_globals']
        if public.shape != (len(labels),45):
            raise ValueError('Public context slice shape differs from audited feature schema')
        slices['east_round'] = public[:,32] == 1
        slices['south_round'] = public[:,33] == 1
        slices['west_or_north_round'] = (public[:,34] == 1) | (public[:,35] == 1)
        slices['self_riichi_declared'] = public[:,4] == 1
        slices['opponent_riichi_declared'] = np.any(public[:,5:8] == 1,axis=1)
    grid = []
    for index,strength in enumerate(LAMBDA_GRID):
        changed = choices[index] != reference
        grid.append(dict(lambda_=strength, rows=len(labels), changed=int(changed.sum()),
                         discard_rows=int(discard.sum()), discard_changed=int((changed&discard).sum()),
                         slices={key:dict(rows=int(mask.sum()),changed=int((changed&mask).sum()))
                                 for key,mask in slices.items()}))
    return dict(**metrics, grid=grid)


def calibrate(plan, training, *, plan_sha256, training_sha256, output_dir, deadline):
    """All three fixed seeds; each owns distinct calibration and final gate rows."""
    if (training.get('complete') is not True
            or [item.get('seed') for item in training.get('seeds', [])] != plan['training_seeds']
            or any(item.get('complete') is not True for item in training['seeds'])):
        raise ValueError('Calibration requires all fixed seeds completed, without adaptive selection')
    from .stage3i_training import RECIPE
    if (any(training.get('identity',{}).get(key) != plan[key]
            for key in ('reference_sha256','call_sha256','protocol_sha256','cache_manifest_sha256'))
            or training.get('identity',{}).get('recipe') != RECIPE):
        raise ValueError('Training report recipe or historical identity differs from the frozen programme')
    path, manifest, records = _load_cache_manifest(plan)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    loaded = {}
    for entry in training['seeds']:
        heads, metadata = candidate_policy(entry['checkpoint'], entry['sha256'])
        if (metadata['seed'] != entry['seed']
                or metadata['identity']['cache_manifest_sha256'] != plan['cache_manifest_sha256']
                or metadata['identity']['reference_sha256'] != manifest['reference_sha256']
                or metadata['identity']['call_sha256'] != manifest['call_sha256']
                or metadata['identity']['protocol_sha256'] != plan['protocol_sha256']
                or metadata.get('complete') is not True or metadata.get('benchmark_only') is True):
            raise ValueError('Training checkpoint identity differs from calibration cache')
        loaded[entry['seed']] = (heads, metadata)
    summaries = {entry['seed']:{'calibration':[],'gate':[]} for entry in training['seeds']}
    artifacts = []
    for partition in ('calibration','gate'):
        for index,name in enumerate(manifest['partitions'][partition]):
            if time.monotonic() >= deadline:
                raise TimeoutError('Calibration deadline: no additional shard admitted')
            record = records[name]
            if record['partition'] != partition:
                raise ValueError('Calibration shard partition differs')
            arrays = _load_cache_shard(record, path)
            for seed,(heads,metadata) in loaded.items():
                values,states,residuals = _head_predictions(heads, arrays)
                choices,raw_reference,reference = _all_grid_choices(arrays,residuals)
                report = _match_prediction_report(name,arrays,values,states,residuals,choices,reference)
                summaries[seed][partition].append(report)
                payload = io.BytesIO()
                np.savez_compressed(payload, value=values,state=states,action=residuals,
                                    choices_by_lambda=choices,raw_reference=raw_reference,
                                    wrapped_reference=reference,row_indices=arrays['row_indices'])
                relative = f'seed-{seed}/{partition}/{name}.npz'
                binary = payload.getvalue()
                _write(output/relative,binary)
                artifacts.append(dict(seed=seed,partition=partition,match_id=name,path=relative,
                                      sha256=hashlib.sha256(binary).hexdigest(),bytes=len(binary),
                                      source_sha256=record['sha256']))
            if (index+1)%50 == 0:
                print(json.dumps(dict(phase='calibration',partition=partition,matches=index+1)),flush=True)
    results = []
    for entry in training['seeds']:
        seed = entry['seed']
        grid = []
        for index,strength in enumerate(LAMBDA_GRID):
            matches = [row['grid'][index] for row in summaries[seed]['calibration']]
            rows,changed = sum(row['rows'] for row in matches),sum(row['changed'] for row in matches)
            discard,discard_changed = sum(row['discard_rows'] for row in matches),sum(row['discard_changed'] for row in matches)
            grid.append(dict(lambda_=strength, **{'lambda':strength},rows=rows,changed=changed,
                             change_rate=changed/rows,reference_discard_rows=discard,
                             reference_discard_changed=discard_changed,
                             reference_discard_change_rate=discard_changed/discard if discard else 0.,
                             slices={key:dict(rows=sum(row['slices'][key]['rows'] for row in matches),
                                              changed=sum(row['slices'][key]['changed'] for row in matches))
                                     for key in matches[0]['slices']}))
        selection = calibration_choice(grid)
        gate = predictive_gate(summaries[seed]['gate'])
        reasons = list(gate['reasons'])
        if selection['lambda_'] == 0:
            reasons.append('zero_calibrated_correction_no_candidate_games')
        results.append(dict(seed=seed,checkpoint=entry['checkpoint'],sha256=entry['sha256'],
                            calibration=selection,lambda_grid=grid,gate=gate,
                            admitted_for_games=gate['passed'] and selection['lambda_']>0,
                            skip_reasons=reasons,match_metrics=summaries[seed]))
    result = dict(schema=CALIBRATION_SCHEMA,complete=True,plan_sha256=plan_sha256,
                  training_report_sha256=training_sha256,cache_manifest_sha256=plan['cache_manifest_sha256'],
                  seeds=results,artifacts=artifacts,elapsed_seconds=time.perf_counter()-started,
                  calibration_matches=200,gate_matches=400,policy_promotion=False,seed_selection=False,
                  source_role='reused-development-diagnostics',
                  limitation='No counterfactual regret or causal action advantage follows from a chosen logged target; fresh games remain necessary')
    _write(output/'manifest.json',json_bytes(result))
    return result
