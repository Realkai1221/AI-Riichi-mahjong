"""Fixed, paired development games for completed Stage 3G value-learning arms.

Every policy consumes only immutable audited PlayerViews. Full wall commitments
and engine logs belong exclusively to the auditor. Development selection is
not a confirmation test or policy promotion.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
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
from .model import NeuralPolicy, _weights_hash, collate_encoded
from .player import PlayerAdapter
from .policies import policy_manifest
from .rules import audit_physical_tiles, rule_profile_hash
from .semantics import semantic_actions, teacher_decision
from .simulator import INITIAL_SCORES, create_env, score_total
from .stage3c import terminal_result
from .stage3g_collect import _write

SCHEMA = 'riichi-ai-stage3g-development-match-v1'
EXECUTION_SCHEMA = 'riichi-ai-stage3g-development-execution-v1'
ARM_ORDER = ('mc_a0', 'mc_a1', 'delta_a0', 'delta_a1')
BASELINE = 'baseline'
OPPONENTS = ('shanten_v2', 'defense_v2', 'reference')


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path: str | Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def value_choice(actions, values) -> int:
    """Safety rule: first legal semantic win, otherwise finite guarded argmax."""
    values = np.asarray(values, dtype=np.float64)
    if (not actions or values.shape != (len(actions),) or not np.isfinite(values).all()
            or np.any(np.abs(values) > 100)):
        raise ValueError('Legal action values are nonfinite, mismatched or outside [-100,100]')
    wins = [index for index, action in enumerate(actions) if action.kind in ('RON', 'TSUMO')]
    return wins[0] if wins else int(values.argmax())


def line_up(arm: str, focal: int) -> tuple[str, ...]:
    if arm not in (*ARM_ORDER, BASELINE) or type(focal) is not int or not 0 <= focal < 4:
        raise ValueError('Invalid development arm or focal seat')
    result = [''] * 4
    result[focal] = 'reference' if arm == BASELINE else arm
    for offset, name in enumerate(OPPONENTS, 1):
        result[(focal + offset) % 4] = name
    return tuple(result)


def prepare_execution(seed_plan: dict, arms_manifest: dict, config: dict,
                      *, plan_sha256: str, arms_sha256: str) -> dict:
    seeds = seed_plan.get('evaluation_seeds')
    if (not isinstance(seeds, list) or len(seeds) != 64 or len(set(seeds)) != 64
            or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds)):
        raise ValueError('Exactly 64 unique fixed uint64 development wall seeds required')
    if arms_manifest.get('schema') != 'stage3g-arms-v1':
        raise ValueError('Unrecognized completed-arm manifest')
    arms, skipped = arms_manifest.get('arms'), arms_manifest.get('skipped_arms')
    if not isinstance(arms, list) or not isinstance(skipped, list):
        raise ValueError('Explicit completed and skipped arms required')
    names = [arm.get('name') for arm in arms]
    if names != [name for name in ARM_ORDER if name in names] or len(names) != len(set(names)):
        raise ValueError('Completed arms must have unique preregistered order')
    if names not in [list(ARM_ORDER), list(ARM_ORDER[:2])]:
        raise ValueError('Require both MC controls and either both or neither delta arms')
    if skipped != [name for name in ARM_ORDER if name not in names]:
        raise ValueError('Skipped arms must exactly account for the incomplete factorial')
    if skipped and arms_manifest.get('delta_gate_passed') is not False:
        raise ValueError('Only the predeclared calibration-gate fallback may skip delta arms')
    if not skipped and arms_manifest.get('delta_gate_passed') is not True:
        raise ValueError('Delta arms require a passed calibration gate')
    normalized = []
    for arm in arms:
        path, sha = Path(arm.get('checkpoint', '')), arm.get('sha256')
        if (not path.is_absolute() or path.resolve() != path or not path.is_file()
                or path.is_symlink() or not isinstance(sha, str) or len(sha) != 64
                or digest(path) != sha):
            raise ValueError('Completed arm checkpoint identity failed')
        normalized.append(dict(name=arm['name'], checkpoint=str(path), sha256=sha))
    if config.get('development_blocks') != 64 or config.get('workers') != 14:
        raise ValueError('Development panel or worker count changed')
    jobs = []
    for block, seed in enumerate(seeds):
        for focal in range(4):
            for arm in [BASELINE, *names]:
                jobs.append(dict(match_id=f'block-{block:03d}-seat-{focal}-{arm}', block=block,
                                 focal_seat=focal, arm=arm, seed=seed, lineup=list(line_up(arm, focal))))
    return dict(schema=EXECUTION_SCHEMA, plan_sha256=plan_sha256,
                arms_manifest_sha256=arms_sha256, reference_checkpoint=config['reference_checkpoint'],
                reference_sha256=config['reference_sha256'], arms=normalized, skipped_arms=skipped,
                delta_gate_passed=arms_manifest['delta_gate_passed'], blocks=64,
                rotations=4, workers=14, matches=jobs, planned_matches=len(jobs),
                tie_order=names, training_eligible=False, ppo_eligible=False,
                policy_promotion=False, source_role='development-only',
                wall_pairing='Same pinned engine and ordinal physical-wall commitments checked across arms; equal seeds alone are insufficient')


@lru_cache(maxsize=1)
def reference_policy(path: str, sha: str):
    torch.set_num_threads(1)
    reference = NeuralPolicy.from_checkpoint(path, device='cpu')
    if reference.checkpoint_hash != sha:
        raise ValueError('Reference checkpoint changed')
    return reference


@lru_cache(maxsize=4)
def value_policy(path: str, sha: str, arm: str, reference_sha: str):
    from .stage3g_learning import load_value_checkpoint
    torch.set_num_threads(1)
    if digest(path) != sha:
        raise ValueError('Value checkpoint changed')
    model, metadata = load_value_checkpoint(path, device='cpu')
    if (metadata['checkpoint_sha256'] != sha or metadata['identity']['arm'] != arm
            or metadata['identity']['reference_sha256'] != reference_sha
            or metadata['identity']['split'] != 'training' or metadata['scheduled_updates'] != 400):
        raise ValueError('Value checkpoint arm, reference, training split or exposure changed')
    return model, metadata


def validate_policies(execution: dict) -> dict:
    """Authenticate every completed model before admitting any game."""
    reference = reference_policy(execution['reference_checkpoint'], execution['reference_sha256'])
    result = {'reference': reference.manifest()}
    for item in execution['arms']:
        _, metadata = value_policy(item['checkpoint'], item['sha256'], item['name'], execution['reference_sha256'])
        result[item['name']] = metadata
    return result


def decide(name, view, reference, values):
    actions = semantic_actions(view)
    if not actions:
        raise ValueError('Empty legal semantic action set')
    if name in OPPONENTS[:2]:
        selection = teacher_decision(view, defense=name == 'defense_v2')
        if selection.actions != actions:
            raise ValueError('Teacher semantic ordering differs')
        index, scores = selection.chosen_index, None
    elif name == 'reference':
        index, scores = 0, None
        if len(actions) > 1:
            encoded = encode_view(view, actions, feature_schema=reference.model.config['feature_schema'])
            with torch.inference_mode():
                logits, _ = reference.model(collate_encoded([encoded], 'cpu'))
            scores = logits[0].detach().cpu().numpy()
            if scores.shape != (len(actions),) or not np.isfinite(scores).all():
                raise ValueError('Reference legal logits are nonfinite or mismatched')
            index = int(scores.argmax())
    else:
        model = values[name][0]
        encoded = encode_view(view, actions, feature_schema=model.config['feature_schema'])
        with torch.inference_mode():
            predicted, _ = model(collate_encoded([encoded], 'cpu'))
        scores = predicted[0].detach().cpu().numpy()
        index = value_choice(actions, scores)
    return actions, index, None if scores is None else scores.tolist()


def run_one(job: dict, execution: dict, output_dir: str | Path, *, deadline: float,
            max_steps: int = 100_000) -> dict:
    if type(max_steps) is not int or max_steps < 1 or time.monotonic() >= deadline:
        raise ValueError('Invalid match limit or expired admission')
    if job not in execution['matches'] or job['lineup'] != list(line_up(job['arm'], job['focal_seat'])):
        raise ValueError('Match differs from frozen execution plan')
    directory, name = Path(output_dir), job['match_id']
    started = time.monotonic()
    _write(directory / 'lifecycle' / f'{name}.started.json', json_bytes(dict(schema=SCHEMA, plan=job)))
    env = None
    rows, requests, walls, hand_starts, policies = [], [], [], [], []
    steps, failure = 0, None
    try:
        reference = reference_policy(execution['reference_checkpoint'], execution['reference_sha256'])
        values = {item['name']: value_policy(item['checkpoint'], item['sha256'], item['name'], execution['reference_sha256'])
                  for item in execution['arms'] if item['name'] == job['arm']}
        for seat, policy in enumerate(job['lineup']):
            if policy in OPPONENTS[:2]:
                manifest = policy_manifest(policy, 0)
            elif policy == 'reference':
                manifest = reference.manifest()
            else:
                metadata = values[policy][1]
                manifest = dict(arm=policy, checkpoint_sha256=metadata['checkpoint_sha256'],
                                weights_sha256=metadata['weights_sha256'],
                                choice='first legal semantic win else greedy finite scalar action value')
            policies.append(dict(seat=seat, name=policy, policy_id=policy+':'+hashlib.sha256(json_bytes(manifest)).hexdigest(), manifest=manifest))
        env, observations = create_env(job['seed'])
        adapters = [PlayerAdapter(seat) for seat in range(4)]
        hashers = [PlayerViewHasher() for _ in range(4)]
        while not env.done() and steps < max_steps:
            if time.monotonic() >= deadline:
                raise TimeoutError('Development cooperative match deadline')
            audit_physical_tiles(env)
            if score_total(env) != sum(INITIAL_SCORES):
                raise ValueError('Development point/deposit conservation failure')
            # Collect every snapshot and digest before obtaining any response.
            views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
            if not views or not any(view.legal_actions for view in views.values()):
                raise ValueError('Unfinished development match has no legal request')
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
                actions, index, scores = decide(job['lineup'][seat], view, reference, values)
                action = actions[index]
                row_ids.append(len(rows))
                rows.append(dict(row_index=len(rows), step=steps, hand_id=view.hand_id, seat=seat,
                    policy_id=policies[seat]['policy_id'], player_view_sha256=hashes[str(seat)],
                    pre_response_id=response_id, semantic_keys=[a.key for a in actions],
                    action_kinds=[a.kind for a in actions], representative_tokens=[a.representative.token for a in actions],
                    selected_index=index, selected_semantic_key=action.key, selected_kind=action.kind,
                    selected_physical_token=action.representative.token, probability=1.,
                    legal_scores=scores, training_eligible=False, ppo_eligible=False))
                chosen[seat] = action.representative
            requests.append(dict(step=steps, pre_response_id=response_id, player_views=hashes, rows=row_ids, submitted=False))
            native = {seat: adapters[seat].resolve(action, observations[seat]) for seat, action in chosen.items()}
            observations = env.step(native)
            requests[-1]['submitted'] = True
            steps += 1
        audit_physical_tiles(env)
        if score_total(env) != sum(INITIAL_SCORES):
            raise ValueError('Terminal development point/deposit conservation failure')
        if not env.done():
            raise RuntimeError('Truncated development match')
        if _weights_hash(reference.model) != reference.weights_hash:
            raise ValueError('Reference weights changed during development match')
        for model, metadata in values.values():
            if _weights_hash(model) != metadata['weights_sha256']:
                raise ValueError('Value weights changed during development match')
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
                  training_eligible=False, ppo_eligible=False, source_role='development-only')
    report['payload_sha256'] = hashlib.sha256(json_bytes(report)).hexdigest()
    report['elapsed_seconds'] = time.monotonic()-started
    _write(directory / 'reports' / f'{name}.json', json_bytes(report))
    _write(directory / 'lifecycle' / f'{name}.finished.json', json_bytes(dict(schema=SCHEMA, match_id=name,
           completed=completed, payload_sha256=report['payload_sha256'], report_sha256=hashlib.sha256(json_bytes(report)).hexdigest())))
    return report


def summarize(execution: dict, records: list[dict]) -> dict:
    expected = {job['match_id']: job for job in execution['matches']}
    actual = {record['match_id']: record for record in records}
    if len(actual) != len(records) or set(actual) != set(expected):
        raise ValueError('Development coverage is incomplete or duplicated')
    for name, record in actual.items():
        if (any(record.get(k) != v for k, v in expected[name].items()) or record.get('completed') is not True
                or record.get('failure') is not None or record.get('training_eligible') is not False
                or record.get('ppo_eligible') is not False or record.get('terminal_result_verified') is not True):
            raise ValueError('Invalid completed development record')
        if (not isinstance(record.get('utilities'), list) or sorted(record['utilities']) != [-1.5,-.5,.5,1.5]
                or len(record.get('ordinal_wall_fingerprints', [])) < 1):
            raise ValueError('Missing terminal utilities or physical wall commitments')
    names = execution['tie_order']
    summaries, pairing = {}, []
    for arm in names:
        differences, arm_utilities, baseline_utilities = [], [], []
        for block in range(execution['blocks']):
            per_seat = []
            for focal in range(4):
                candidate = actual[f'block-{block:03d}-seat-{focal}-{arm}']
                baseline = actual[f'block-{block:03d}-seat-{focal}-{BASELINE}']
                a, b = candidate['ordinal_wall_fingerprints'], baseline['ordinal_wall_fingerprints']
                common = min(len(a), len(b))
                if a[:common] != b[:common]:
                    raise ValueError('Shared ordinal physical walls differ across arms')
                pairing.append(dict(arm=arm, block=block, focal_seat=focal, common_hands=common,
                                    paired_common_walls_match=True))
                c, r = candidate['utilities'][focal], baseline['utilities'][focal]
                per_seat.append(c-r)
                arm_utilities.append(c)
                baseline_utilities.append(r)
            differences.append(sum(per_seat)/4)
        summaries[arm] = dict(mean_utility=float(np.mean(arm_utilities)),
                              baseline_mean_utility=float(np.mean(baseline_utilities)),
                              mean_rank=2.5-float(np.mean(arm_utilities)),
                              block_utility_differences=differences,
                              statistics=summarize_blocks(differences, bootstrap_seed=20261023031))
    best = max(names, key=lambda name: (summaries[name]['mean_utility'], -names.index(name)))
    selected = best if summaries[best]['statistics']['mean_utility_difference'] > 0 else None
    return dict(complete=True, blocks=execution['blocks'], games=len(records), arms=summaries,
                pairing=pairing, selected_candidate=selected, best_observed_arm=best,
                selection_rule='Highest development mean utility, declared tie order; no selection unless above original-reference baseline',
                skipped_arms=execution['skipped_arms'], factorial_complete=not execution['skipped_arms'],
                confirmation_started=False, policy_promotion=False,
                limitation='Descriptive fixed-panel development selection only; no independent confirmation, multiplicity-adjusted strength claim or cross-engine rule certification')
