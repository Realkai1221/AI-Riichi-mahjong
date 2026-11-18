"""Fixed fresh-game comparisons of saved heads, small corrections and exact ties.

No optimization, calibration, gate-based exclusion or policy promotion occurs
here. Full engine state is confined to the complete-match auditor payload.
"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .auditing import PlayerViewHasher, VIEW_HASH_SCHEMA
from .evaluation import _wall_fingerprint, summarize_blocks
from .features import encode_view
from .model import collate_encoded
from .player import PlayerAdapter
from .policies import policy_manifest
from .rules import audit_physical_tiles, rule_profile_hash
from .semantics import semantic_actions, teacher_decision
from .simulator import INITIAL_SCORES, create_env, score_total
from .stage3c import terminal_result
from .stage3g_collect import _write
from .stage3i_evaluation import (KINDS, OPPONENTS, candidate_policy, digest,
                                frozen_policy, json_bytes, _identity)
from .stage3j_model import HASH_TIE_SALT, actor_feature_fingerprint, stage3j_choices

SCHEMA = 'riichi-ai-stage3j-evaluation-match-v1'
EXECUTION_SCHEMA = 'riichi-ai-stage3j-evaluation-execution-v1'
TRAINING_SEEDS = (20261024011, 20261024012, 20261024013)
ARM_NAMES = ('reference_wrapped', 'hash_tie',
             *(f'tie_seed_{seed}' for seed in TRAINING_SEEDS),
             *(f'small_seed_{seed}' for seed in TRAINING_SEEDS))
FORWARD_SAMPLE_BLOCKS = (0, 32, 64, 96, 128, 160, 192, 224)


def validate_dependencies(plan, config):
    """Authenticate the source bundles and reconstruct the frozen history set."""
    dependencies = plan.get('dependencies', [])
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError('Explicit historical dependencies required')
    paths = set()
    for entry in dependencies:
        path = _identity(entry['path'], entry['sha256'])
        if path in paths or not isinstance(entry.get('role'), str):
            raise ValueError('Duplicate or malformed historical dependency')
        paths.add(path)
    for stem in ('reference', 'call'):
        if not any(entry['path'] == config[stem+'_checkpoint'] and entry['sha256'] == config[stem+'_sha256']
                   for entry in dependencies):
            raise ValueError('Historical actor or collector missing from dependencies')
    for bundle in plan['selected_bundles']:
        if not any(entry['path'] == bundle['checkpoint'] and entry['sha256'] == bundle['sha256']
                   for entry in dependencies):
            raise ValueError('Saved selected head missing from dependencies')
    freshness = plan.get('freshness', {})
    snapshots = freshness.get('historical_seed_sources', [])
    if not snapshots:
        raise ValueError('Frozen historical seed/wall inventory required')
    seeds, walls = set(), set()
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('seed', 'seeds') or key.endswith('_seeds'):
                    if type(item) is int: seeds.add(item)
                    elif isinstance(item, list): seeds.update(x for x in item if type(x) is int)
                if key == 'ordinal_wall_fingerprints' and isinstance(item, list) and item and isinstance(item[0], str):
                    walls.add(item[0])
                collect(item)
        elif isinstance(value, list):
            for item in value: collect(item)
    for entry in snapshots:
        path = _identity(entry['path'], entry['sha256'])
        if path not in paths:
            raise ValueError('Historical snapshot missing from dependency list')
        saved = json.loads(Path(path).read_text())
        if saved.get('schema') != 'stage3j-historical-inputs-v1' or not saved.get('inputs'):
            raise ValueError('Missing historical snapshot inputs')
        before_seeds, before_walls = set(seeds), set(walls)
        seeds, walls = set(), set()
        seen = set()
        for source in saved['inputs']:
            source_path = _identity(source['path'], source['sha256'])
            if source_path in seen:
                raise ValueError('Duplicate historical snapshot input')
            seen.add(source_path)
            collect(json.loads(Path(source_path).read_text()))
        if sorted(seeds) != saved.get('historical_seeds') or sorted(walls) != saved.get('historical_first_wall_fingerprints'):
            raise ValueError('Frozen historical seed/wall inventory does not match its inputs')
        seeds.update(before_seeds); walls.update(before_walls)
    if (sorted(seeds) != freshness.get('historical_seeds')
            or sorted(walls) != freshness.get('historical_first_wall_fingerprints')):
        raise ValueError('Plan history differs from authenticated historical inputs')
    if set(plan['evaluation_seeds'] + plan['preflight_evaluation_seeds']) & seeds:
        raise ValueError('Evaluation reuses a recorded historical seed')


def line_up(arm: str, focal: int, candidate_names=()) -> tuple[str, ...]:
    if arm not in ARM_NAMES or type(focal) is not int or not 0 <= focal < 4:
        raise ValueError('Invalid fixed Stage 3J arm or focal seat')
    result = [''] * 4
    result[focal] = arm
    for offset, opponent in enumerate(OPPONENTS, 1):
        result[(focal + offset) % 4] = opponent
    return tuple(result)


def prepare_execution(plan: dict, config: dict, *, plan_sha256: str,
                      preflight: bool = False) -> dict:
    """Freeze all eight arms and all three saved seeds before any game starts."""
    if plan.get('training_seeds') != list(TRAINING_SEEDS):
        raise ValueError('Every saved Stage 3I training seed is required in fixed order')
    bundles = plan.get('selected_bundles', [])
    if [entry.get('seed') for entry in bundles] != list(TRAINING_SEEDS):
        raise ValueError('All three selected bundles are required without selection')
    source = plan.get('source_identity', {})
    required = ('reference_sha256', 'call_sha256', 'protocol_sha256', 'cache_manifest_sha256')
    if (any(not isinstance(source.get(key), str) or len(source[key]) != 64 for key in required)
            or any(source[key] != config[key] for key in required[:2])):
        raise ValueError('Historical Stage 3I bundle identity is not explicitly bound')
    formal, bench = plan['evaluation_seeds'], plan['preflight_evaluation_seeds']
    if (len(formal) != config['evaluation_blocks'] or len(bench) != config['preflight_evaluation_blocks']
            or len(set(formal + bench)) != len(formal) + len(bench)
            or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in formal + bench)):
        raise ValueError('Fixed fresh formal and preflight wall blocks overlap or differ')
    if (formal != list(range(202610290000, 202610290000 + config['evaluation_blocks']))
            or bench != [202610289000, 202610289001]):
        raise ValueError('Evaluation wall seeds differ from the fixed Stage 3J ranges')
    seeds = bench if preflight else formal
    if not seeds:
        raise ValueError('Nonempty fixed game panel required')
    if plan.get('forward_sample_blocks') != list(FORWARD_SAMPLE_BLOCKS):
        raise ValueError('Fresh forward samples must retain the fixed eight block indices')
    arms = [dict(name='reference_wrapped', mode='reference_wrapped'),
            dict(name='hash_tie', mode='hash_tie')]
    for prefix, mode in [('tie', 'learned_tie'), ('small', 'fractional')]:
        for entry in bundles:
            arms.append(dict(name=f'{prefix}_seed_{entry["seed"]}', mode=mode,
                             seed=entry['seed'], checkpoint=_identity(entry['checkpoint'], entry['sha256']),
                             sha256=entry['sha256'], strength=.25 if mode == 'fractional' else 0.))
    jobs = [dict(match_id=f'block-{block:03d}-seat-{focal}-{arm}', block=block,
                 focal_seat=focal, arm=arm, seed=seed, lineup=list(line_up(arm, focal)))
            for block, seed in enumerate(seeds) for focal in range(4) for arm in ARM_NAMES]
    return dict(schema=EXECUTION_SCHEMA, plan_sha256=plan_sha256,
                protocol_sha256=plan['protocol_sha256'], source_identity=dict(source),
                phase='preflight' if preflight else 'formal',
                reference_checkpoint=config['reference_checkpoint'], reference_sha256=config['reference_sha256'],
                call_checkpoint=config['call_checkpoint'], call_sha256=config['call_sha256'],
                arms=arms, blocks=len(seeds), rotations=4, workers=config['workers'],
                matches=jobs, planned_matches=len(jobs), fixed_seed_family=list(TRAINING_SEEDS),
                forward_sample_blocks=[block for block in FORWARD_SAMPLE_BLOCKS if block < len(seeds)],
                historical_first_wall_fingerprints=plan.get('freshness', {}).get('historical_first_wall_fingerprints', []),
                training_eligible=False, ppo_eligible=False, policy_promotion=False, seed_selection=False,
                source_role='preflight-only' if preflight else 'development-only',
                wall_pairing='Pinned engine with explicit ordinal physical wall commitments compared across all arms; numeric seeds alone insufficient')


def validate_policies(execution):
    frozen = frozen_policy(execution['reference_checkpoint'], execution['reference_sha256'],
                           execution['call_checkpoint'], execution['call_sha256'])
    result = dict(frozen=dict(frozen.metadata), candidates={})
    for item in execution['arms']:
        if not item.get('checkpoint'):
            continue
        _, metadata = candidate_policy(item['checkpoint'], item['sha256'])
        if (metadata.get('seed') != item['seed'] or metadata.get('complete') is not True
                or metadata.get('benchmark_only') is True
                or any(metadata.get('identity', {}).get(key) != value
                       for key, value in execution['source_identity'].items())):
            raise ValueError('Saved complete Stage 3I head seed or historical source identity changed')
        result['candidates'][item['name']] = metadata
    return result


def decide(name, view, frozen, values):
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
    with torch.inference_mode():
        logits, context, candidates = frozen.frozen_representations(batch)
        scores = logits[0].detach().cpu().numpy()
        if scores.dtype != np.float32 or scores.shape != (len(actions),) or not np.isfinite(scores).all():
            raise ValueError('Reference logits must be actual finite float32 actor outputs')
        raw = int(scores.argmax())
        if name == 'reference_raw':
            return actions, raw, scores.tolist(), dict(raw_reference_index=raw, wrapper=False)
        probabilities = frozen.explorer_probabilities(logits, context, candidates,
                            batch['action_tokens'][:, :, 0], batch['candidate_mask'])[0]
        if name in ('reference_wrapped', 'hash_tie'):
            mode, residuals = name, np.zeros(len(actions), dtype=np.float32)
        else:
            heads, _, mode = values[name]
            expanded = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
            residuals = heads.action(torch.cat([expanded, candidates], dim=-1))[0].cpu().numpy()
    fingerprint = actor_feature_fingerprint(encoded) if mode == 'hash_tie' else None
    decision = stage3j_choices(scores, residuals, probabilities, kinds, mode=mode,
                              actor_fingerprint=fingerprint, semantic_keys=[a.key for a in actions])
    evidence = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in decision.items() if key != 'scores'}
    evidence['scores'] = [float(value) if np.isfinite(value) else None for value in decision['scores']]
    evidence.update(mode=mode, strength=.25 if mode == 'fractional' else 0.,
                    explorer_probabilities=probabilities.tolist(), action_residuals=residuals.tolist(),
                    actor_fingerprint=fingerprint)
    if mode == 'fractional':
        evidence['companion_tie_selected_index'] = stage3j_choices(
            scores, residuals, probabilities, kinds, mode='learned_tie')['selected_index']
    return actions, decision['selected_index'], scores.tolist(), evidence


def decision_diagnostics(rows: list[dict], focal: int) -> dict:
    optional = [row for row in rows if row['seat'] == focal and row['legal_scores'] is not None]
    changes = Counter(row['correction']['change_classification'] for row in optional)
    discard = [row for row in optional if row['action_kinds'][row['correction']['wrapped_reference_index']] == 'DISCARD']
    return dict(optional_decisions=len(optional), forced_decisions=sum(
                    row['seat'] == focal and bool((row['correction'] or {}).get('forced')) for row in rows),
                reference_discard_decisions=len(discard),
                changed=sum(changes[key] for key in ('exact_tie', 'positive_margin')),
                exact_tie_changes=changes['exact_tie'], positive_margin_changes=changes['positive_margin'],
                reference_discard_changes=sum(row['selected_index'] != row['correction']['wrapped_reference_index'] for row in discard),
                exact_top_tie_decisions=sum(sum(row['correction']['exact_tie_mask']) > 1 for row in optional),
                eligible_top_tie_decisions=sum(sum(row['correction']['eligible_tie_mask']) > 1 for row in optional),
                win_opportunity_decisions=sum(row['correction']['wrapper_active'] for row in optional),
                wrapper_override_decisions=sum(row['correction']['wrapper_applied'] for row in optional),
                small_vs_tie_decision_disagreements=sum(
                    row['selected_index'] != row['correction']['companion_tie_selected_index']
                    for row in optional if 'companion_tie_selected_index' in row['correction']))


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
    forward_sample = None
    steps, failure = 0, None
    try:
        frozen = frozen_policy(execution['reference_checkpoint'], execution['reference_sha256'],
                               execution['call_checkpoint'], execution['call_sha256'])
        values = {item['name']: (*candidate_policy(item['checkpoint'], item['sha256']), item['mode'])
                  for item in execution['arms'] if item['name'] == job['arm'] and item.get('checkpoint')}
        for seat, policy in enumerate(job['lineup']):
            if policy in OPPONENTS[:2]:
                manifest = policy_manifest(policy, 0)
            elif policy in ('reference_raw', 'reference_wrapped'):
                manifest = dict(**frozen.metadata, choice=policy)
            elif policy == 'hash_tie':
                manifest = dict(**frozen.metadata, choice=policy, mode='hash_tie',
                                salt=HASH_TIE_SALT, eligibility='actual explorer probability >=0.02 or raw reference argmax',
                                input='audited encoded actor features and semantic action key only')
            else:
                metadata = values[policy][1]
                manifest = dict(arm=policy, seed=metadata['seed'],
                                checkpoint_sha256=metadata['checkpoint_sha256'],
                                weights_sha256=metadata['weights_sha256'], mode=values[policy][2],
                                strength=.25 if values[policy][2] == 'fractional' else 0.,
                                source_identity=dict(execution['source_identity']),
                                eligibility='actual explorer probability >=0.02 or raw reference argmax',
                                prior_floor=-8, choice='first legal win else fixed Stage 3J mode')
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
                fingerprint = _wall_fingerprint(env)
                if ordinal == 1 and fingerprint in execution['historical_first_wall_fingerprints']:
                    raise ValueError('Realized first physical wall was used in historical evidence')
                walls.append(fingerprint)
                view = next(view for view in views.values() if view.hand_id == ordinal)
                start = dict(next(event for event in view.history if event.kind == 'start_kyoku').data)
                hand_starts.append(dict(hand_id=ordinal, **{k: start[k] for k in ('bakaze', 'kyoku', 'honba', 'kyotaku', 'oya', 'scores')}))
            chosen, row_ids = {}, []
            for seat, view in views.items():
                if not view.legal_actions:
                    continue
                actions, index, scores, correction = decide(job['lineup'][seat], view, frozen, values)
                action = actions[index]
                if (forward_sample is None and job['block'] in execution['forward_sample_blocks']
                        and seat == job['focal_seat'] and len(actions) > 1):
                    encoded = encode_view(view, actions, feature_schema=frozen.actor.config['feature_schema'])
                    forward_sample = dict(row_index=len(rows), encoded={key:dict(dtype=str(value.dtype),
                        shape=list(value.shape), values=value.tolist()) for key,value in encoded.items()})
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
                   forward_sample=forward_sample,
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
                  focal_diagnostics=decision_diagnostics(rows, job['focal_seat']),
                  forward_sample_rows=1 if forward_sample is not None else 0,
                  forward_sample_missing=job['block'] in execution['forward_sample_blocks'] and forward_sample is None,
                  artifact=dict(path=audit_relative, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)),
                  training_eligible=False, ppo_eligible=False, source_role=execution['source_role'])
    report['payload_sha256'] = hashlib.sha256(json_bytes(report)).hexdigest()
    report['elapsed_seconds'] = time.monotonic()-started
    _write(directory / 'reports' / f'{name}.json', json_bytes(report))
    _write(directory / 'lifecycle' / f'{name}.finished.json', json_bytes(dict(schema=SCHEMA, match_id=name,
           completed=completed, payload_sha256=report['payload_sha256'], report_sha256=hashlib.sha256(json_bytes(report)).hexdigest())))
    return report


def _paired_contrast(execution, actual, arm, baseline):
    differences, arm_utilities, baseline_utilities, pairing = [], [], [], []
    for block in range(execution['blocks']):
        per_seat = []
        for focal in range(4):
            candidate = actual[f'block-{block:03d}-seat-{focal}-{arm}']
            control = actual[f'block-{block:03d}-seat-{focal}-{baseline}']
            left, right = candidate['ordinal_wall_fingerprints'], control['ordinal_wall_fingerprints']
            common = min(len(left), len(right))
            if common < 1 or left[:common] != right[:common]:
                raise ValueError('Shared ordinal physical walls differ across paired arms')
            pairing.append(dict(arm=arm, baseline=baseline, block=block, focal_seat=focal,
                                common_hands=common, paired_common_walls_match=True))
            value, reference = candidate['utilities'][focal], control['utilities'][focal]
            per_seat.append(value-reference)
            arm_utilities.append(value)
            baseline_utilities.append(reference)
        differences.append(sum(per_seat)/4)
    return dict(arm=arm, baseline=baseline,
                mean_utility=float(np.mean(arm_utilities)), baseline_mean_utility=float(np.mean(baseline_utilities)),
                mean_rank=2.5-float(np.mean(arm_utilities)), baseline_mean_rank=2.5-float(np.mean(baseline_utilities)),
                block_utility_differences=differences,
                statistics=summarize_blocks(differences, bootstrap_seed=20261029031)), pairing


def summarize(execution: dict, records: list[dict]) -> dict:
    expected = {job['match_id']: job for job in execution['matches']}
    actual = {record['match_id']: record for record in records}
    if len(actual) != len(records) or set(actual) != set(expected):
        raise ValueError('Evaluation coverage is incomplete or duplicated')
    for name, record in actual.items():
        if (any(record.get(key) != value for key, value in expected[name].items())
                or record.get('completed') is not True or record.get('failure') is not None
                or record.get('training_eligible') is not False or record.get('ppo_eligible') is not False
                or record.get('terminal_result_verified') is not True):
            raise ValueError('Invalid completed evaluation record')
        if (not isinstance(record.get('utilities'), list) or sorted(record['utilities']) != [-1.5,-.5,.5,1.5]
                or len(record.get('ordinal_wall_fingerprints', [])) < 1):
            raise ValueError('Missing terminal utilities or physical wall commitments')
    contrasts, pairing = {}, []
    pairs = [(arm, 'reference_wrapped') for arm in ARM_NAMES[1:]]
    pairs.extend((f'tie_seed_{seed}', 'hash_tie') for seed in TRAINING_SEEDS)
    pairs.extend((f'small_seed_{seed}', f'tie_seed_{seed}') for seed in TRAINING_SEEDS)
    for arm, baseline in pairs:
        result, checks = _paired_contrast(execution, actual, arm, baseline)
        contrasts[arm+'_vs_'+baseline] = dict(result, role='fixed_descriptive_seed_or_control_contrast')
        pairing.extend(checks)
    by_arm = {}
    for arm in ARM_NAMES:
        rows = [record for record in records if record['arm'] == arm]
        utilities = [record['utilities'][record['focal_seat']] for record in rows]
        diagnostics = Counter()
        for record in rows:
            diagnostics.update(record.get('focal_diagnostics', {}))
        by_arm[arm] = dict(games=len(rows), mean_utility=float(np.mean(utilities)),
                           mean_rank=2.5-float(np.mean(utilities)), focal_diagnostics=dict(diagnostics))
    matched_tie_small = {}
    for seed in TRAINING_SEEDS:
        pairs = [(actual[f'block-{block:03d}-seat-{seat}-tie_seed_{seed}'],
                  actual[f'block-{block:03d}-seat-{seat}-small_seed_{seed}'])
                 for block in range(execution['blocks']) for seat in range(4)]
        matched_tie_small[str(seed)] = dict(games=len(pairs), identical_canonical_games=sum(
            left.get('game_sha256') == right.get('game_sha256') and left.get('game_sha256') is not None
            for left, right in pairs))
    return dict(complete=True, blocks=execution['blocks'], games=len(records), contrasts=contrasts,
                by_arm=by_arm, pairing=pairing, matched_tie_small=matched_tie_small,
                selected_candidate=None, seed_selection=False, confirmation_started=False, policy_promotion=False,
                limitation='Fixed fresh development comparison, with three previously selected observational residual heads; no independent confirmation, best-seed selection or policy promotion.')

