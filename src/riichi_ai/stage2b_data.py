"""Additive focal-policy collection and actor-only efficiency derivation.

Stage 2A files remain immutable. Behavior requests, teacher targets, and resolved
events are distinct: labels describe a permitted pre-action view, whereas the
terminal rank describes the policy that actually played the complete match.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import io
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any

import numpy as np

from .auditing import PlayerViewHasher, VIEW_HASH_SCHEMA
from .dataset import (
    ACTOR_SCHEMA, TARGET_SCHEMA, DATASET_SCHEMA, SplitDataset, _hash, _json,
    _pack_arrays, _positive_integer, _process_tree_rss, _rare_paths, _rss,
    _sources, _git_state, _write, load_split, validate_manifest,
)
from .features import encode_history_events, encode_view, MAX_HISTORY
from .player import PlayerAdapter
from .policies import make_policy
from .rules import audit_physical_tiles, rule_profile_hash, resolved_rule_profile
from .semantics import ACTION_KINDS, semantic_actions, teacher_decision
from .simulator import create_env, score_total, INITIAL_SCORES


FOCAL_SCHEMA = 'riichi-ai-stage2b-focal-data-v1'
FOCAL_AUDITOR_SCHEMA = 'riichi-ai-stage2b-focal-auditor-v1'
DERIVED_SCHEMA = 'riichi-ai-stage2b-efficiency-data-v1'
SEED_NAMESPACE = 'riichi-ai/stage2b/focal-collection/v1'
_SPLITS = ('train', 'diagnostic', 'validation', 'test')
AGGREGATION_SCHEMA = 'riichi-ai-stage2b-student-aggregation-v1'
_ACTION_EVENT = {'DISCARD': 'dahai', 'RIICHI': 'reach', 'CHI': 'chi', 'PON': 'pon',
                 'ANKAN': 'ankan', 'DAIMINKAN': 'daiminkan', 'KAKAN': 'kakan',
                 'RON': 'hora', 'TSUMO': 'hora', 'KYUSHU_KYUHAI': 'ryukyoku'}


def _earlier_seeds() -> set[int]:
    """Exact wall seeds and roots from earlier recorded campaign manifests."""
    root = Path(__file__).resolve().parents[2]
    seeds: set[int] = set()
    for directory in ('reports/stage1', 'reports/stage2'):
        for path in sorted((root / directory).glob('*.json')):
            record = json.loads(path.read_text())
            for source in (record, record.get('configuration', {})):
                if isinstance(source.get('root_seed'), int):
                    seeds.add(source['root_seed'])
            matches = record.get('matches', [])
            for match in matches if isinstance(matches, list) else []:
                if isinstance(match, dict) and isinstance(match.get('seed'), int):
                    seeds.add(match['seed'])
    return seeds


def plan_stage2b_matches(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Assign group, stratum, focal seat and unique seed before extraction."""
    groups = config['data_groups']
    names, roots, seeds = set(), set(), set()
    forbidden = _earlier_seeds()
    conditional = config.get('conditional_collection')
    reserved_roots = set(config.get('seed_namespaces', {}).values())
    forbidden.update(reserved_roots - ({config['seed_namespaces']['aggregation']} if conditional else set()))
    base_count = 0
    if conditional is not None:
        if conditional.get('schema') != AGGREGATION_SCHEMA:
            raise ValueError('Unknown conditional collection schema')
        base_dir = Path(conditional['base_dataset_directory'])
        base_raw = (base_dir / 'manifest.json').read_bytes()
        if _hash(base_raw) != conditional['base_manifest_sha256']:
            raise ValueError('Conditional collection base manifest changed')
        if json.loads(base_raw).get('config', {}).get('conditional_collection') is not None:
            raise ValueError('Conditional collection cannot recursively aggregate')
        base = validate_stage2b_manifest(base_dir)
        inherited = {k: v for k, v in config.items() if k not in ('conditional_collection', 'data_groups')}
        if inherited != {k: v for k, v in base['config'].items() if k != 'data_groups'}:
            raise ValueError('Conditional collection changed the main frozen protocol')
        base_count = base['completed_matches']
        forbidden.update(r['seed'] for r in base['matches'])
        forbidden.update(r['root_seed'] for r in base['matches'])
        if len(groups) != 1 or sum(g['matches'] for g in groups) > 300:
            raise ValueError('Conditional collection permits one group of at most 300 matches')
        if groups[0]['root_seed'] != config['seed_namespaces']['aggregation']:
            raise ValueError('Conditional collection must use its declared aggregation namespace')
    result = []
    for group in groups:
        name, root_seed = group['name'], group['root_seed']
        _positive_integer(group['matches'], 'group matches')
        if name in names or root_seed in roots:
            raise ValueError('Data groups need unique names and roots')
        if type(root_seed) is not int or not 0 <= root_seed < 2**64 or root_seed in forbidden:
            raise ValueError('Data group root is invalid or reuses an earlier seed')
        if group['split'] not in _SPLITS or group['stratum'] not in ('teacher', 'student'):
            raise ValueError('Unknown focal split or stratum')
        if conditional is not None:
            if group['split'] != 'train' or group['stratum'] != 'student':
                raise ValueError('Conditional collection is student training only')
            expected = 'frozen_candidate'
        else:
            if group['split'] == 'train':
                raise ValueError('Training collection requires frozen conditional configuration')
            expected = 'teacher' if group['stratum'] == 'teacher' else 'historical_student'
        if group['behavior'] != expected:
            raise ValueError('Behavior and stratum disagree')
        names.add(name)
        roots.add(root_seed)
        for local in range(group['matches']):
            seed = int(_hash(_json([SEED_NAMESPACE, name, root_seed, local]))[:16], 16)
            if seed in seeds or seed in forbidden or seed in reserved_roots:
                raise ValueError('Derived wall seed overlaps another recorded match')
            seeds.add(seed)
            index = len(result)
            result.append({'match_id': f'match-{index:06d}', 'index': index,
                           'group': name, 'group_index': local, 'split': group['split'],
                           'stratum': group['stratum'], 'behavior': group['behavior'],
                           'root_seed': root_seed, 'seed': seed, 'focal_seat': local % 4,
                           'independent_group': f'{SEED_NAMESPACE}:{name}:{root_seed}:{local}'})
    if base_count + len(result) > min(600, config['limits']['maximum_new_corpus_matches']):
        raise ValueError('Collection exceeds the configured new-corpus ceiling')
    return result


def make_aggregation_config(main_config: dict, base_dataset_dir: str | Path,
                            checkpoint_path: str | Path, checkpoint_sha256: str, *,
                            matches: int = 300) -> dict:
    """Freeze an explicitly triggered student-training collection, without running it."""
    _positive_integer(matches, 'matches')
    base_dir = Path(base_dataset_dir).expanduser().resolve()
    base = validate_stage2b_manifest(base_dir)
    if base['config'] != main_config or main_config.get('conditional_collection') is not None:
        raise ValueError('Aggregation must bind the original main collection configuration')
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if _hash(checkpoint.read_bytes()) != checkpoint_sha256:
        raise ValueError('Frozen candidate checkpoint hash mismatch')
    config = copy.deepcopy(main_config)
    config['conditional_collection'] = {
        'schema': AGGREGATION_SCHEMA, 'base_dataset_directory': str(base_dir),
        'base_manifest_sha256': _hash((base_dir / 'manifest.json').read_bytes()),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha256,
        'authorization': 'Conditional protocol only; caller must establish trigger before collection.'}
    config['data_groups'] = [{'name': 'training_student', 'split': 'train', 'stratum': 'student',
                              'matches': matches, 'root_seed': config['seed_namespaces']['aggregation'],
                              'behavior': 'frozen_candidate'}]
    plan_stage2b_matches(config)
    return config


def _behavior_checkpoint(config: dict) -> tuple[Path, str]:
    conditional = config.get('conditional_collection')
    if conditional is not None:
        return Path(conditional['checkpoint']), conditional['checkpoint_sha256']
    return Path(config['historical_checkpoint']), config['historical_checkpoint_sha256']


def _validate_policy_provenance(config: dict, plan: dict, provenance: list[dict]) -> None:
    if len(provenance) != 4:
        raise ValueError('Expected provenance for all four behavior seats')
    focal = plan['focal_seat']
    if plan['stratum'] == 'teacher':
        if provenance[focal].get('policy') != 'shanten_v2':
            raise ValueError('Focal teacher provenance mismatch')
    elif provenance[focal].get('checkpoint_sha256') != _behavior_checkpoint(config)[1]:
        raise ValueError('Loaded focal checkpoint differs from frozen behavior identity')
    for relative, name in enumerate(config['opponents'], 1):
        if provenance[(focal + relative) % 4].get('policy') != name:
            raise ValueError('Clockwise opponent provenance mismatch')


def _empty_output(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Output must be empty; historical artifacts are never overwritten')
    directory.mkdir(parents=True, exist_ok=True)


def _policies(config: dict, plan: dict) -> list[Any]:
    from .evaluation import _evaluation_policy
    focal = plan['focal_seat']
    names = [None] * 4
    names[focal] = ('shanten_v2' if plan['stratum'] == 'teacher'
                    else 'checkpoint:' + str(_behavior_checkpoint(config)[0]))
    for relative, name in enumerate(config['opponents'], 1):
        names[(focal + relative) % 4] = name
    policies = [_evaluation_policy(name, plan['seed'] ^ seat) for seat, name in enumerate(names)]
    _validate_policy_provenance(config, plan, [p.manifest() for p in policies])
    return policies


def _numeric_shards(rows: list[dict], histories: dict[int, list[np.ndarray]], ranks: list[int]) -> tuple[dict, dict]:
    starts, parts, cursor = {}, [], 0
    for hand, chunks in histories.items():
        value = np.concatenate(chunks)
        starts[hand] = cursor
        cursor += len(value)
        parts.append(value)
    counts = [len(row['encoding']['action_tokens']) for row in rows]
    actor = {
        'tiles': np.stack([row['encoding']['tiles'] for row in rows]).astype(np.float32),
        'globals': np.stack([row['encoding']['globals'] for row in rows]).astype(np.float32),
        'action_tokens': np.concatenate([row['encoding']['action_tokens'] for row in rows]).astype(np.int16),
        'candidate_offsets': np.asarray([0, *np.cumsum(counts)], dtype=np.int64),
        'history_tokens': np.concatenate(parts).astype(np.int16),
        'history_start': np.asarray([starts[r['hand_id']] + max(0, r['history_length'] - MAX_HISTORY) for r in rows], dtype=np.int64),
        'history_length': np.asarray([min(MAX_HISTORY, r['history_length']) for r in rows], dtype=np.int16),
    }
    targets = {
        'policy_target': np.concatenate([np.asarray(r['teacher'].soft_targets, dtype=np.float32) for r in rows]),
        'best_mask': np.concatenate([np.asarray(r['teacher'].best_mask, dtype=np.bool_) for r in rows]),
        'teacher_preference_keys': np.concatenate([np.asarray(r['teacher'].preference_keys, dtype=np.int16) for r in rows]),
        'chosen_index': np.asarray([r['teacher'].chosen_index for r in rows], dtype=np.int16),
        'final_rank': np.asarray([ranks[r['seat']] for r in rows], dtype=np.int8),
        'seat': np.asarray([r['seat'] for r in rows], dtype=np.int8),
        'step': np.asarray([r['step'] for r in rows], dtype=np.int32),
        'hand_id': np.asarray([r['hand_id'] for r in rows], dtype=np.int32),
        'physical_candidate_count': np.asarray([r['physical_candidate_count'] for r in rows], dtype=np.int16),
    }
    return actor, targets


def _annotate_resolution(requests: list[dict], events: list[dict]) -> None:
    """Bind requests to actual subsequent event spans without inventing labels."""
    for index, batch in enumerate(requests):
        end = requests[index + 1]['event_start'] if index + 1 < len(requests) else len(events)
        start = batch['event_start']
        if not 0 <= start <= end <= len(events):
            raise ValueError('Observation event cursors disagree with canonical event stream')
        batch['event_end'] = end
        for request in batch['requests']:
            kind = request['behavior_kind']
            matches = [i for i in range(start, end) if events[i].get('type') == _ACTION_EVENT.get(kind)
                       and events[i].get('actor') == request['seat']]
            request['resolved_event_indices'] = matches
            request['resolution'] = ('no_explicit_pass_event' if kind == 'PASS' else
                                     'resolved' if matches else 'not_emitted_or_preempted')


def _collect_match(job: dict) -> dict:
    config, plan = job['config'], job['plan']
    started = time.perf_counter()
    env, observations = create_env(plan['seed'])
    adapters = [PlayerAdapter(seat) for seat in range(4)]
    hashers = [PlayerViewHasher() for _ in range(4)]
    policies = _policies(config, plan)
    rows, requests, histories = [], [], {}
    lengths, event_cursors = {}, [0] * 4
    focal, steps, failure = plan['focal_seat'], 0, None
    peak = _rss()
    behavior_counts, teacher_counts, candidates = Counter(), Counter(), Counter()
    try:
        while not env.done() and steps < job['max_steps']:
            audit_physical_tiles(env)
            if score_total(env) != sum(INITIAL_SCORES):
                raise ValueError('Point/deposit conservation failure')
            # Returned event snapshots are consumed exactly once per seat.
            for seat, observation in observations.items():
                event_cursors[seat] += len(observation.events)
            event_start = max(event_cursors)
            views = {seat: adapters[seat].build(observations[seat]) for seat in sorted(observations)}
            selected, batch = {}, []
            for seat, view in views.items():
                if not view.legal_actions:
                    continue
                actions = semantic_actions(view)
                decision = teacher_decision(view) if seat == focal else None
                behavior = policies[seat].choose(view)
                behavior_index = next(i for i, action in enumerate(actions) if behavior.token in action.physical_tokens)
                row_index = None
                if decision is not None:
                    if decision.actions != actions:
                        raise ValueError('Teacher and behavior semantic candidate ordering differs')
                    encoding = encode_view(view, actions)
                    previous = lengths.get(view.hand_id, 0)
                    if len(view.history) < previous:
                        raise ValueError('Focal hand history shrank without a new hand identifier')
                    histories.setdefault(view.hand_id, []).append(encode_history_events(view.history[previous:], seat))
                    lengths[view.hand_id] = len(view.history)
                    # Fixed128 history is reconstructed later from shared prefixes.
                    encoding = {key: encoding[key] for key in ('tiles', 'globals', 'action_tokens')}
                    row_index = len(rows)
                    rows.append({'encoding': encoding, 'teacher': decision, 'seat': seat, 'step': steps,
                                 'hand_id': view.hand_id, 'history_length': len(view.history),
                                 'physical_candidate_count': len(view.legal_actions)})
                    behavior_counts[behavior.kind] += 1
                    teacher_counts[actions[decision.chosen_index].kind] += 1
                    candidates.update(action.kind for action in actions)
                record = {'seat': seat, 'row': row_index, 'hand_id': view.hand_id,
                          'view_sha256': hashers[seat].digest(view),
                          'physical_candidates': [action.token for action in view.legal_actions],
                          'semantic_keys': [action.key for action in actions],
                          'submitted_action': behavior.token, 'behavior_kind': behavior.kind,
                          'behavior_chosen_index': behavior_index, 'behavior_probability': 1.0}
                if decision is not None:
                    record['teacher'] = {'chosen_index': decision.chosen_index,
                                         'preference_keys': decision.preference_keys,
                                         'best_mask': decision.best_mask, 'soft_targets': decision.soft_targets}
                batch.append(record)
                selected[seat] = adapters[seat].resolve(behavior, observations[seat])
            if not selected:
                raise ValueError('Nonterminal match has no legal requests')
            requests.append({'step': steps, 'event_start': event_start, 'requests': batch})
            observations = env.step(selected)
            steps += 1
            if steps % 64 == 0:
                peak = max(peak, _rss())
                if peak > job['worker_memory_limit']:
                    raise MemoryError('Worker exceeded sampled memory share')
        audit_physical_tiles(env)
        if score_total(env) != sum(INITIAL_SCORES):
            raise ValueError('Terminal point/deposit conservation failure')
        if not env.done():
            failure = {'type': 'TruncatedMatch', 'message': 'max_steps reached', 'step': steps}
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error), 'step': steps}
    completed = env.done() and failure is None
    events = env.mjai_log
    ranks = list(env.ranks()) if completed else None
    if completed and (sorted(ranks) != [1, 2, 3, 4] or not rows):
        completed, ranks = False, None
        failure = {'type': 'InvalidTerminalMatch', 'message': 'Invalid ranks or no focal rows'}
    _annotate_resolution(requests, events)
    auditor = {'schema': FOCAL_AUDITOR_SCHEMA, 'plan': plan, 'rules_hash': rule_profile_hash(),
               'policy_provenance': [policy.manifest() for policy in policies],
               'view_hash_schema': VIEW_HASH_SCHEMA, 'canonical_events': events,
               'policy_requests': requests, 'completed': completed, 'failure': failure,
               'final_ranks': ranks, 'final_scores': list(env.scores()),
               'label_semantics': 'Teacher requests from focal pre-action views; final ranks belong to actual behavior. Resolved events are separate from requests.'}
    name = plan['match_id']
    artifacts = {f'auditor/{name}.json.gz': gzip.compress(_json(auditor), mtime=0)}
    uncompressed = 0
    if completed:
        actor, targets = _numeric_shards(rows, histories, ranks)
        artifacts[f'actor/{name}.npz'] = _pack_arrays(actor)
        artifacts[f'targets/{name}.npz'] = _pack_arrays(targets)
        uncompressed = sum(a.nbytes for a in (*actor.values(), *targets.values()))
    report = {**plan, 'completed': completed, 'failure': failure, 'decisions': len(rows), 'steps': steps,
              'all_seat_requests': sum(len(batch['requests']) for batch in requests),
              'hands': sum(event.get('type') == 'start_kyoku' for event in events),
              'final_ranks': ranks, 'final_scores': list(env.scores()), 'game_hash': _hash(_json(events)),
              'physical_forced': sum(r['physical_candidate_count'] == 1 for r in rows),
              'semantic_forced': sum(len(r['teacher'].actions) == 1 for r in rows),
              'behavior_action_counts': dict(behavior_counts), 'teacher_action_counts': dict(teacher_counts),
              'semantic_candidate_counts': dict(candidates), 'rare_paths': _rare_paths(events),
              'uncompressed_training_bytes': uncompressed, 'peak_worker_rss_bytes': max(peak, _rss()),
              'elapsed_seconds': time.perf_counter() - started,
              'artifacts': {path: {'sha256': _hash(raw), 'bytes': len(raw)} for path, raw in sorted(artifacts.items())}}
    return {'report': report, 'artifacts': artifacts}


def _ordered_results(executor, jobs, workers) -> Iterator[dict]:
    source, pending = iter(jobs), deque()
    for _ in range(workers):
        job = next(source, None)
        if job is not None:
            pending.append(executor.submit(_collect_match, job))
    while pending:
        yield pending.popleft().result()
        job = next(source, None)
        if job is not None:
            pending.append(executor.submit(_collect_match, job))


def generate_stage2b_dataset(output_dir: str | Path, config: dict, *, workers: int | None = None,
                            max_steps: int | None = None, disk_limit_bytes: int | None = None,
                            memory_limit_bytes: int | None = None) -> dict:
    """Collect declared focal corpora, including separately frozen conditional data."""
    plan = plan_stage2b_matches(config)
    if config['rule_profile_hash'] != rule_profile_hash() or config['teacher'] != 'shanten_v2':
        raise ValueError('Unexpected rules or teacher')
    if len(config['opponents']) != 3:
        raise ValueError('Exactly three clockwise opponents are required')
    checkpoint, checkpoint_sha = _behavior_checkpoint(config)
    if _hash(checkpoint.read_bytes()) != checkpoint_sha:
        raise ValueError('Frozen behavior checkpoint hash mismatch')
    workers = config['collection']['workers'] if workers is None else workers
    max_steps = config['collection']['max_steps'] if max_steps is None else max_steps
    disk_limit_bytes = config['limits']['new_artifact_bytes'] if disk_limit_bytes is None else disk_limit_bytes
    memory_limit_bytes = config['limits']['memory_gib'] * 1024**3 if memory_limit_bytes is None else memory_limit_bytes
    for value, name in ((workers, 'workers'), (max_steps, 'max_steps'), (disk_limit_bytes, 'disk_limit_bytes'),
                        (memory_limit_bytes, 'memory_limit_bytes')):
        _positive_integer(value, name)
    if workers > 8:
        raise ValueError('Focal collection supports at most eight bounded workers')
    directory = Path(output_dir).expanduser().resolve()
    _empty_output(directory)
    plan_record = {'schema': FOCAL_SCHEMA, 'seed_namespace': SEED_NAMESPACE, 'config': config, 'matches': plan}
    plan_raw = _json(plan_record)
    if len(plan_raw) + 100000 >= disk_limit_bytes:
        raise ValueError('Disk cap cannot accommodate plan and final manifest')
    _write(directory / 'plan.json', plan_raw)
    before, started, written = _sources(), time.perf_counter(), len(plan_raw)
    jobs = [{'plan': item, 'config': config, 'max_steps': max_steps,
             'worker_memory_limit': memory_limit_bytes // (workers + 1)} for item in plan]
    reports, failure, executor = [], None, None
    peak_tree = _process_tree_rss()
    try:
        if workers == 1:
            results = map(_collect_match, jobs)
        else:
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
            results = _ordered_results(executor, jobs, workers)
        for result in results:
            report, artifacts = result['report'], result['artifacts']
            incoming = sum(len(raw) for raw in artifacts.values())
            if written + incoming + max(100000, len(plan) * 6000) > disk_limit_bytes:
                failure = {'type': 'DiskBudgetExceeded', 'match_id': report['match_id']}
                break
            for name, raw in artifacts.items():
                _write(directory / name, raw)
            written += incoming
            reports.append(report)
            if not report['completed']:
                failure = report['failure']
                break
            sample = _process_tree_rss()
            if sample is not None:
                peak_tree = max(peak_tree or 0, sample)
                if peak_tree > memory_limit_bytes:
                    failure = {'type': 'MemoryBudgetExceeded'}
                    break
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    unchanged = before == _sources()
    checkpoint_unchanged = _hash(checkpoint.read_bytes()) == checkpoint_sha
    game_hashes = [report['game_hash'] for report in reports if report['completed']]
    duplicate_games = len(game_hashes) != len(set(game_hashes))
    manifest = {'schema': FOCAL_SCHEMA, 'actor_schema': ACTOR_SCHEMA, 'target_schema': TARGET_SCHEMA,
                'auditor_schema': FOCAL_AUDITOR_SCHEMA, 'config': config, 'config_sha256': _hash(_json(config)),
                'plan_sha256': _hash(plan_raw), 'seed_namespace': SEED_NAMESPACE,
                'provenance': before, 'sources_unchanged': unchanged, 'git': _git_state(),
                'rule_profile': resolved_rule_profile(), 'rule_profile_hash': rule_profile_hash(),
                'valid': failure is None and len(reports) == len(plan) and unchanged and checkpoint_unchanged and not duplicate_games,
                'behavior_checkpoint_unchanged': checkpoint_unchanged,
                'failure': failure, 'requested_matches': len(plan),
                'completed_matches': sum(r['completed'] for r in reports),
                'rejected_matches': [r['match_id'] for r in reports if not r['completed']],
                'not_written_matches': [r['match_id'] for r in plan[len(reports):]],
                'splits': {split: [r['match_id'] for r in reports if r['split'] == split and r['completed']] for split in _SPLITS},
                'matches': reports, 'decisions': sum(r['decisions'] for r in reports),
                'all_seat_requests': sum(r['all_seat_requests'] for r in reports),
                'hands': sum(r['hands'] for r in reports), 'duplicate_complete_games': duplicate_games,
                'workers': workers, 'max_steps': max_steps, 'disk_limit_bytes': disk_limit_bytes,
                'memory_target_bytes': memory_limit_bytes, 'sampled_peak_process_tree_rss_bytes': peak_tree,
                'memory_scope': 'Sampled RSS, not an OS hard limit; null tree RSS means unavailable. Per-worker checks remain active.',
                'artifact_bytes_before_manifest': written,
                'uncompressed_training_bytes': sum(r['uncompressed_training_bytes'] for r in reports),
                'elapsed_seconds': time.perf_counter() - started,
                'rank_target_semantics': 'True final rank under recorded behavior and opponents; not a counterfactual teacher value.',
                'test_policy': 'Integrity checks allowed; predictions/performance withheld until final selection.'}
    raw = _json(manifest)
    if written + len(raw) > disk_limit_bytes:
        raise RuntimeError('Final manifest exceeds reserved disk space')
    _write(directory / 'manifest.json', raw)
    return manifest


def validate_stage2b_manifest(output_dir: str | Path) -> dict:
    directory = Path(output_dir).expanduser().resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != FOCAL_SCHEMA or manifest.get('valid') is not True:
        raise ValueError('Only complete valid focal datasets can be loaded')
    if (manifest.get('actor_schema') != ACTOR_SCHEMA or manifest.get('target_schema') != TARGET_SCHEMA
            or manifest.get('auditor_schema') != FOCAL_AUDITOR_SCHEMA
            or manifest.get('seed_namespace') != SEED_NAMESPACE):
        raise ValueError('Focal numeric/auditor schema mismatch')
    raw_plan = (directory / 'plan.json').read_bytes()
    if _hash(raw_plan) != manifest['plan_sha256']:
        raise ValueError('Focal split plan hash mismatch')
    plan = json.loads(raw_plan)
    config = manifest['config']
    if (plan.get('schema') != FOCAL_SCHEMA or plan.get('seed_namespace') != SEED_NAMESPACE
            or plan['config'] != config or _hash(_json(config)) != manifest['config_sha256']):
        raise ValueError('Focal config identity mismatch')
    expected = plan_stage2b_matches(config)
    if plan['matches'] != expected or len(manifest['matches']) != len(expected):
        raise ValueError('Focal deterministic assignment mismatch')
    for report, item in zip(manifest['matches'], expected):
        if any(report.get(key) != value for key, value in item.items()) or not report['completed'] or report['failure']:
            raise ValueError('Focal report assignment/completion mismatch')
        name = item['match_id']
        if set(report['artifacts']) != {f'actor/{name}.npz', f'targets/{name}.npz', f'auditor/{name}.json.gz'}:
            raise ValueError('Focal artifact paths mismatch')
    splits = {split: [r['match_id'] for r in expected if r['split'] == split] for split in _SPLITS}
    if manifest['splits'] != splits:
        raise ValueError('Focal splits differ from exact disjoint plan')
    if (manifest['completed_matches'] != len(expected) or manifest['failure'] or manifest['rejected_matches']
            or manifest['not_written_matches'] or not manifest['sources_unchanged']
            or not manifest['behavior_checkpoint_unchanged']):
        raise ValueError('Focal completion/provenance metadata inconsistent')
    if len({r['game_hash'] for r in manifest['matches']}) != len(expected):
        raise ValueError('Duplicate complete focal games')
    return manifest


def _source_manifest(directory: Path) -> dict:
    schema = json.loads((directory / 'manifest.json').read_text()).get('schema')
    if schema == DATASET_SCHEMA:
        return validate_manifest(directory)
    if schema == FOCAL_SCHEMA:
        return validate_stage2b_manifest(directory)
    raise ValueError('Unsupported source dataset schema')


def _read_numeric(directory: Path, report: dict, area: str) -> dict[str, np.ndarray]:
    relative = f'{area}/{report["match_id"]}.npz'
    raw = (directory / relative).read_bytes()
    if _hash(raw) != report['artifacts'][relative]['sha256']:
        raise ValueError(f'Corrupted numeric artifact: {relative}')
    with np.load(io.BytesIO(raw), allow_pickle=False) as shard:
        return {key: shard[key] for key in shard.files}


def _derive_match(actor: dict, targets: dict) -> tuple[dict, dict]:
    from .efficiency import EFFICIENCY_SCALES, exact_efficiency_from_encoded
    from .features import efficiency_from_encoded
    values, exact_values = [], []
    discard_id = ACTION_KINDS.index('DISCARD') + 1
    applicable = 0
    for row, (start, end) in enumerate(zip(actor['candidate_offsets'][:-1], actor['candidate_offsets'][1:])):
        tokens = actor['action_tokens'][start:end]
        raw = exact_efficiency_from_encoded(actor['tiles'][row], tokens)
        scaled = raw.astype(np.float32) / np.asarray(EFFICIENCY_SCALES, dtype=np.float32)
        # The live public encoder is checked once per match; every candidate's
        # integer values and teacher alignment are still checked below.
        if row == 0 and not np.array_equal(scaled, efficiency_from_encoded(actor['tiles'][row], tokens)):
            raise ValueError('Live and precomputed efficiency scaling disagree')
        discard = tokens[:, 0] == discard_id
        if not np.array_equal(raw[:, 3], discard.astype(np.int16)) or np.any(raw[~discard]):
            raise ValueError('Efficiency applicability disagrees with candidate kind')
        keys = np.zeros((len(tokens), 5), dtype=np.int16)
        for index, token in enumerate(tokens):
            kind = ACTION_KINDS[int(token[0]) - 1]
            branch = 0 if kind in ('RON', 'TSUMO') else 1 if kind == 'RIICHI' else 2 if kind == 'DISCARD' else 3 if kind == 'PASS' else 4
            keys[index, 0] = branch
            if kind == 'DISCARD':
                keys[index, 2:] = raw[index, 0], -raw[index, 1], token[2]
        if not np.array_equal(keys, targets['teacher_preference_keys'][start:end]):
            raise ValueError(f'Exact teacher preference mismatch at row {row}')
        tuples = [tuple(map(int, key)) for key in keys]
        expected_best = np.asarray([key == min(tuples) for key in tuples])
        if not np.array_equal(expected_best, targets['best_mask'][start:end]):
            raise ValueError(f'Recomputed best semantic set mismatch at row {row}')
        if int(targets['chosen_index'][row]) != int(np.flatnonzero(expected_best)[0]):
            raise ValueError('Teacher deterministic representative differs')
        if not np.array_equal(targets['policy_target'][start:end], expected_best.astype(np.float32) / np.float32(expected_best.sum())):
            raise ValueError('Teacher soft target differs')
        expected_scaled = raw.astype(np.float32) / np.asarray([8, 136, 34, 1], dtype=np.float32)
        if not np.array_equal(scaled, expected_scaled):
            raise ValueError('Scaled feature does not exactly match integer feature schema')
        applicable += int(discard.sum())
        values.append(scaled)
        exact_values.append(raw)
    return {'candidate_efficiency': np.concatenate(values), 'exact_efficiency': np.concatenate(exact_values)}, {
        'checked_rows': len(values), 'checked_candidates': len(actor['action_tokens']), 'applicable_discard_candidates': applicable,
        'exact_preference_and_best_set_mismatches': 0,
    }


def derive_efficiency(source_dir: str | Path, output_dir: str | Path, *,
                      splits: Sequence[str] = ('train', 'validation'),
                      disk_limit_bytes: int = 5_000_000_000, memory_limit_bytes: int = 8_000_000_000,
                      max_seconds: float | None = None) -> dict:
    """Derive from actor inputs only; labels are read solely for exact audits."""
    source, directory = Path(source_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    source_sha = _hash((source / 'manifest.json').read_bytes())
    manifest = _source_manifest(source)
    if _hash((source / 'manifest.json').read_bytes()) != source_sha:
        raise ValueError('Source manifest changed while loading')
    splits = tuple(splits)
    if len(set(splits)) != len(splits) or not splits or any(split not in manifest['splits'] for split in splits):
        raise ValueError('Invalid derived split list')
    if manifest['schema'] == DATASET_SCHEMA and any(split not in ('train', 'validation') for split in splits):
        raise ValueError('Stage 2A test data must not be derived or reused')
    _positive_integer(disk_limit_bytes, 'disk_limit_bytes')
    _positive_integer(memory_limit_bytes, 'memory_limit_bytes')
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('max_seconds must be positive')
    _empty_output(directory)
    from .features import FEATURE_SCHEMA_V2
    before, started = _sources(), time.perf_counter()
    reports, failure, written, peak = [], None, 0, _rss()
    selected = [report for report in manifest['matches'] if report['split'] in splits]
    try:
        for report in selected:
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                raise TimeoutError('Feature derivation time budget exhausted')
            actor, targets = (_read_numeric(source, report, area) for area in ('actor', 'targets'))
            values, audit = _derive_match(actor, targets)
            raw = _pack_arrays(values)
            if written + len(raw) + max(100000, len(selected) * 1500) > disk_limit_bytes:
                raise OSError('Derived artifact disk cap exhausted')
            peak = max(peak, _rss())
            if peak > memory_limit_bytes:
                raise MemoryError('Derivation process exceeds sampled memory limit')
            relative = f'efficiency/{report["match_id"]}.npz'
            _write(directory / relative, raw)
            written += len(raw)
            reports.append({'match_id': report['match_id'], 'split': report['split'],
                            'source_game_hash': report['game_hash'], 'path': relative,
                            'sha256': _hash(raw), 'bytes': len(raw),
                            'uncompressed_bytes': sum(v.nbytes for v in values.values()), **audit})
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    unchanged = before == _sources()
    source_unchanged = source_sha == _hash((source / 'manifest.json').read_bytes())
    result = {'schema': DERIVED_SCHEMA, 'feature_schema': FEATURE_SCHEMA_V2,
              'source_directory': str(source), 'source_schema': manifest['schema'],
              'source_manifest_sha256': source_sha, 'splits': list(splits), 'matches': reports,
              'requested_matches': len(selected), 'completed_matches': len(reports),
              'not_written_matches': [r['match_id'] for r in selected[len(reports):]],
              'valid': failure is None and len(reports) == len(selected) and unchanged and source_unchanged,
              'source_manifest_unchanged': source_unchanged,
              'failure': failure, 'provenance': before, 'sources_unchanged': unchanged,
              'checked_rows': sum(r['checked_rows'] for r in reports),
              'checked_candidates': sum(r['checked_candidates'] for r in reports),
              'applicable_discard_candidates': sum(r['applicable_discard_candidates'] for r in reports),
              'peak_rss_bytes': peak, 'disk_limit_bytes': disk_limit_bytes,
              'memory_limit_bytes': memory_limit_bytes, 'artifact_bytes_before_manifest': written,
              'elapsed_seconds': time.perf_counter() - started,
              'information_boundary': 'Inputs are stored own/visible tile channels and legal semantic candidates. Teacher keys/masks are comparison targets only.'}
    raw = _json(result)
    if written + len(raw) > disk_limit_bytes:
        raise RuntimeError('Derived manifest exceeds reserved disk space')
    _write(directory / 'manifest.json', raw)
    return result


class EfficiencyDataset(Sequence[dict[str, Any]]):
    """Identical sample ordering for raw/assisted variants, metadata outside inputs."""

    def __init__(self, base: SplitDataset, manifest: dict, derived_dir: Path | None, *, assisted: bool):
        self.base, self.directory, self.split = base, base.directory, base.split
        self.match_ids, self.index = base.match_ids, base.index
        self.assisted, self.source_manifest = assisted, manifest
        self._reports = base._reports
        self._derived_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._derived_reports, self.derived_dir, self._derived_sha = {}, derived_dir, None
        self.stratum_indices = {'teacher': [], 'student': []}
        for index, (match_index, _) in enumerate(base.index):
            name = base.match_ids[int(match_index)]
            self.stratum_indices[self._reports[name].get('stratum', 'teacher')].append(index)
        self.stratum_indices = {key: np.asarray(value, dtype=np.int64) for key, value in self.stratum_indices.items()}
        self.manifest_sha256 = _hash((self.directory / 'manifest.json').read_bytes())
        if assisted:
            from .features import FEATURE_SCHEMA_V2
            if derived_dir is None:
                raise ValueError('Assisted data needs precomputed audited efficiency artifacts')
            raw = (derived_dir / 'manifest.json').read_bytes()
            derived = json.loads(raw)
            if (derived.get('schema') != DERIVED_SCHEMA or derived.get('feature_schema') != FEATURE_SCHEMA_V2
                    or derived.get('valid') is not True
                    or derived['source_manifest_sha256'] != self.manifest_sha256):
                raise ValueError('Derived manifest is invalid or belongs to another source')
            self._derived_sha = _hash(raw)
            self._derived_reports = {r['match_id']: r for r in derived['matches']}
            if len(self._derived_reports) != len(derived['matches']):
                raise ValueError('Duplicate derived match identities')
            for name in self.match_ids:
                record = self._derived_reports[name]
                if record['split'] != self.split or record['source_game_hash'] != self._reports[name]['game_hash']:
                    raise ValueError('Derived split or complete-game identity mismatch')
            if base.preload:
                for name in self.match_ids:
                    self._efficiency(name)

    def _efficiency(self, name: str) -> np.ndarray:
        if name in self._derived_cache:
            self._derived_cache.move_to_end(name)
            return self._derived_cache[name]
        record = self._derived_reports[name]
        expected_path = f'efficiency/{name}.npz'
        if record['path'] != expected_path:
            raise ValueError('Derived artifact path differs from match identity')
        raw = (self.derived_dir / expected_path).read_bytes()
        if _hash(raw) != record['sha256']:
            raise ValueError('Corrupted derived efficiency artifact')
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            value = archive['candidate_efficiency']
        actor, _ = self.base._load(name)
        if value.dtype != np.float32 or value.shape != (len(actor['action_tokens']), 4) or not np.isfinite(value).all():
            raise ValueError('Invalid derived efficiency shape/dtype/value')
        while len(self._derived_cache) >= self.base.cache_matches:
            self._derived_cache.popitem(last=False)
        cached_bytes = sum(a.nbytes for a in self._derived_cache.values())
        if self.base.loaded_bytes + cached_bytes + value.nbytes > self.base.memory_limit_bytes:
            raise MemoryError('Combined numeric/efficiency cache exceeds byte cap')
        value.setflags(write=False)
        self._derived_cache[name] = value
        return value

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        match_index, row = self.base.index[index]
        name = self.match_ids[int(match_index)]
        actor, targets = self.base._load(name)
        start, end = map(int, actor['candidate_offsets'][int(row):int(row) + 2])
        sample['teacher_preference_keys'] = targets['teacher_preference_keys'][start:end]
        sample['stratum'] = self._reports[name].get('stratum', 'teacher')
        if self.assisted:
            sample['encoded']['candidate_efficiency'] = self._efficiency(name)[start:end]
        return sample

    def dataset_identity(self) -> dict:
        from .features import FEATURE_SCHEMA, FEATURE_SCHEMA_V2
        return {'schema': 'riichi-ai-stage2b-sample-identity-v1', 'path': str(self.directory),
                'source_manifest_sha256': self.manifest_sha256, 'derived_manifest_sha256': self._derived_sha,
                'split': self.split, 'match_ids': list(self.match_ids), 'rows': len(self),
                'index_sha256': _hash(self.index.tobytes()), 'optional_only': self.base.optional_only,
                'feature_schema': FEATURE_SCHEMA_V2 if self.assisted else FEATURE_SCHEMA,
                'stratum_rows': {key: len(value) for key, value in self.stratum_indices.items()},
                'teacher_keys': 'targets artifact hashes in source manifest; never encoded actor inputs'}


def load_efficiency_split(source_dir: str | Path, derived_dir: str | Path | None, split: str, *,
                          assisted: bool = True, max_matches: int | None = None,
                          optional_only: bool = False, preload: bool = True, cache_matches: int = 2,
                          memory_limit_bytes: int = 8_000_000_000, allow_test: bool = False) -> EfficiencyDataset:
    source = Path(source_dir).expanduser().resolve()
    manifest = _source_manifest(source)
    split = 'validation' if split == 'val' else split
    if split not in manifest['splits']:
        raise ValueError('Unknown split')
    if split == 'test' and (manifest['schema'] == DATASET_SCHEMA or not allow_test):
        raise ValueError('Test predictions are sealed; old Stage 2A test is not reusable')
    _positive_integer(cache_matches, 'cache_matches')
    _positive_integer(memory_limit_bytes, 'memory_limit_bytes')
    base = SplitDataset(source, manifest, split, max_matches=max_matches, optional_only=optional_only,
                        preload=preload, cache_matches=cache_matches, memory_limit_bytes=memory_limit_bytes)
    return EfficiencyDataset(base, manifest, Path(derived_dir).expanduser().resolve() if derived_dir else None,
                             assisted=assisted)


load_stage2b_split = load_efficiency_split


def verify_stage2b_dataset(output_dir: str | Path, *, replay_matches_per_group: int = 1) -> dict:
    """Hash/schema-check all focal matches; reproduce selected full matches.

    Reproduction recomputes every behavior choice, teacher target, actor input,
    event span and final rank, comparing deterministic artifact bytes. It is an
    internal consistency check, not an independent rules or strength test.
    """
    from .dataset_validation import _validate_arrays
    if type(replay_matches_per_group) is not int or replay_matches_per_group < 0:
        raise ValueError('Replay count must be a nonnegative integer')
    directory = Path(output_dir).expanduser().resolve()
    started = time.perf_counter()
    result = {'schema': 'riichi-ai-stage2b-focal-verification-v1', 'valid': False, 'errors': [],
              'checked_matches': 0, 'checked_artifacts': 0, 'checked_rows': 0,
              'replayed_matches': 0, 'replayed_rows': 0, 'replayed_match_ids': []}
    try:
        manifest = validate_stage2b_manifest(directory)
        if manifest['rule_profile_hash'] != rule_profile_hash():
            raise ValueError('Replay rules differ from frozen profile')
        config = manifest['config']
        checkpoint, checkpoint_sha = _behavior_checkpoint(config)
        if _hash(checkpoint.read_bytes()) != checkpoint_sha:
            raise ValueError('Frozen behavior checkpoint changed')
        groups: dict[str, list[dict]] = {}
        for report in manifest['matches']:
            groups.setdefault(report['group'], []).append(report)
        replay_ids = set()
        for group in groups.values():
            indices = np.linspace(0, len(group) - 1, min(len(group), replay_matches_per_group), dtype=int)
            replay_ids.update(group[int(index)]['match_id'] for index in indices)
        for report in manifest['matches']:
            try:
                name = report['match_id']
                actor, targets = (_read_numeric(directory, report, area) for area in ('actor', 'targets'))
                relative = f'auditor/{name}.json.gz'
                raw = (directory / relative).read_bytes()
                if _hash(raw) != report['artifacts'][relative]['sha256']:
                    raise ValueError('Auditor artifact hash mismatch')
                auditor = json.loads(gzip.decompress(raw))
                _validate_policy_provenance(config, report, auditor['policy_provenance'])
                if (auditor['schema'] != FOCAL_AUDITOR_SCHEMA or not auditor['completed'] or auditor['failure']
                        or auditor['final_ranks'] != report['final_ranks']
                        or auditor['final_scores'] != report['final_scores']
                        or auditor['rules_hash'] != manifest['rule_profile_hash']):
                    raise ValueError('Auditor identity or terminal metadata differs')
                if any(auditor['plan'].get(key) != report.get(key) for key in auditor['plan']):
                    raise ValueError('Auditor plan assignment differs')
                if _hash(_json(auditor['canonical_events'])) != report['game_hash']:
                    raise ValueError('Canonical game hash mismatch')
                if len(auditor['policy_requests']) != report['steps']:
                    raise ValueError('Auditor request step count mismatch')
                if not np.all(targets['seat'] == report['focal_seat']):
                    raise ValueError('Non-focal target row included')
                row = 0
                for step, batch in enumerate(auditor['policy_requests']):
                    if batch['step'] != step:
                        raise ValueError('Request steps are not contiguous')
                    for request in batch['requests']:
                        if request['seat'] == report['focal_seat']:
                            if request['row'] != row or request.get('teacher') is None:
                                raise ValueError('Focal row/teacher request mismatch')
                            if (targets['step'][row] != step or targets['hand_id'][row] != request['hand_id']):
                                raise ValueError('Focal row/step/hand metadata mismatch')
                            row += 1
                        elif request['row'] is not None or 'teacher' in request:
                            raise ValueError('Opponent decision leaked into focal training rows')
                if row != report['decisions']:
                    raise ValueError('Focal row count mismatch')
                result['checked_rows'] += _validate_arrays(actor, targets, report)
                result['checked_artifacts'] += 3
                result['checked_matches'] += 1
                if name in replay_ids:
                    regenerated = _collect_match({'config': config, 'plan': auditor['plan'],
                                                  'max_steps': manifest['max_steps'],
                                                  'worker_memory_limit': manifest['memory_target_bytes']})
                    if not regenerated['report']['completed']:
                        raise ValueError(f'Replay did not complete: {regenerated["report"]["failure"]}')
                    if regenerated['report']['artifacts'] != report['artifacts']:
                        raise ValueError('Full replay actor/teacher/behavior/event artifact bytes differ')
                    result['replayed_matches'] += 1
                    result['replayed_rows'] += report['decisions']
                    result['replayed_match_ids'].append(name)
            except Exception as error:
                result['errors'].append({'match_id': report['match_id'], 'type': type(error).__name__, 'message': str(error)})
        result['valid'] = not result['errors'] and result['checked_matches'] == len(manifest['matches'])
    except Exception as error:
        result['errors'].append({'type': type(error).__name__, 'message': str(error)})
    result['elapsed_seconds'] = time.perf_counter() - started
    return result
