"""Independent Stage 3J decision arithmetic, fresh forwards and fixed replays.

Every numerical entry point requires the exclusive Stage 3J audit lease.
This is evidence validation for one development experiment, not full engine
certification or an independent playing-strength confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

SEEDS = (20261024011, 20261024012, 20261024013)
SAMPLE_BLOCKS = (0, 32, 64, 96, 128, 160, 192, 224)
HASH_SALT = 'riichi-ai-stage3j-hash-tie-v1'
MODES = ('reference_wrapped', 'hash_tie', 'learned_tie', 'fractional')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(encoded(value))


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Independent Stage 3J audit deadline reached')


def audit_decision(row, policy, mode=None):
    """Reconstruct legal choices without importing the production choice rule."""
    keys, kinds, tokens = row['semantic_keys'], row['action_kinds'], row['representative_tokens']
    count, chosen = len(keys), row['selected_index']
    require(count > 0 and len(kinds) == len(tokens) == count and type(chosen) is int
            and 0 <= chosen < count, 'Invalid semantic choice dimensions or index')
    require(len({encoded(key) for key in keys}) == count, 'Duplicate semantic legal action')
    require(row['selected_semantic_key'] == keys[chosen] and row['selected_kind'] == kinds[chosen]
            and row['selected_physical_token'] == tokens[chosen], 'Selected semantic/physical identity differs')
    require(row['probability'] == 1. and row['training_eligible'] is False
            and row['ppo_eligible'] is False, 'Evaluation probability or training role differs')
    if policy in ('shanten_v2', 'defense_v2'):
        require(row['legal_scores'] is None and row['correction'] is None,
                'Heuristic decision carries unexpected learned scores')
        return dict(optional=count > 1, changed=False, classification='unchanged')
    if count == 1:
        require(chosen == 0 and row['legal_scores'] is None and row['correction'] == dict(forced=True),
                'Forced deterministic action differs')
        return dict(optional=False, changed=False, classification='unchanged')
    logits, detail = row['legal_scores'], row['correction']
    require(len(logits) == count and all(math.isfinite(z) for z in logits), 'Invalid persisted legal scores')
    require(np.array_equal(np.asarray(logits, np.float32).astype(np.float64), np.asarray(logits, np.float64)),
            'Persisted reference scores are not exact float32 values')
    raw = max(range(count), key=lambda i: logits[i])
    wins = [i for i, kind in enumerate(kinds) if kind in ('RON', 'TSUMO')]
    wrapped = wins[0] if wins else raw
    if policy == 'reference_raw':
        require(detail['raw_reference_index'] == raw and chosen == raw,
                'Frozen opponent reference choice differs')
        return dict(optional=True, changed=False, classification='unchanged')
    mode = mode or detail.get('mode')
    require(mode in MODES and detail.get('mode') == mode, 'Undeclared decision mode')
    q, residual = detail['explorer_probabilities'], detail['action_residuals']
    require(len(q) == len(residual) == count and all(math.isfinite(a) for a in residual)
            and all(math.isfinite(p) and 0 <= p <= 1 for p in q) and abs(sum(q) - 1) <= 1e-12,
            'Invalid candidate probabilities or residuals')
    if mode in ('reference_wrapped', 'hash_tie'):
        require(all(value == 0 for value in residual), 'Control unexpectedly consumes learned residuals')
    if wins:
        require(q == [float(index == wins[0]) for index in range(count)],
                'Explorer first-win marginal must be exactly one-hot')
    eligible = [p >= .02 or i == raw for i, p in enumerate(q)]
    exact_ties = [z == logits[raw] for z in logits]
    tie_mask = [allowed and tied for allowed, tied in zip(eligible, exact_ties)]
    tie_indices = [i for i, tied in enumerate(tie_mask) if tied]
    prior = [max(z - logits[raw], -8.) for z in logits]
    strength = .25 if mode == 'fractional' else 0.
    require(detail['strength'] == strength, 'Correction strength changed from fixed protocol')
    hashes = [None] * count
    if mode == 'reference_wrapped':
        corrected = raw
        scores = [p if allowed else -math.inf for p, allowed in zip(prior, eligible)]
    elif mode == 'learned_tie':
        corrected = max(tie_indices, key=lambda i: residual[i])
        scores = [a if tied else -math.inf for a, tied in zip(residual, tie_mask)]
    elif mode == 'fractional':
        scores = [p + .25 * a if allowed else -math.inf for p, a, allowed in zip(prior, residual, eligible)]
        require(all(math.isfinite(s) for s, allowed in zip(scores, eligible) if allowed), 'Corrected score overflow')
        corrected = max(range(count), key=lambda i: scores[i])
        companion = wins[0] if wins else max(tie_indices, key=lambda i: residual[i])
        require(detail['companion_tie_selected_index'] == companion, 'Fractional companion tie choice differs')
    else:
        fingerprint = detail['actor_fingerprint']
        require(isinstance(fingerprint, str) and len(fingerprint) == 64
                and all(c in '0123456789abcdef' for c in fingerprint), 'Invalid audited actor fingerprint')
        for i in tie_indices:
            hashes[i] = hashlib.sha256(encoded([HASH_SALT, fingerprint, keys[i]])).hexdigest()
        corrected = min(tie_indices, key=lambda i: (hashes[i], i))
        order = sorted(tie_indices, key=lambda i: (hashes[i], i))
        scores = [-float(order.index(i)) if i in order else -math.inf for i in range(count)]
    expected = wins[0] if wins else corrected
    require(detail['raw_reference_index'] == raw and detail['wrapped_reference_index'] == wrapped
            and detail['raw_corrected_index'] == corrected and chosen == expected,
            'Independent correction or win wrapper choice differs')
    require(detail['eligible'] == eligible and detail['prior'] == prior
            and detail['exact_tie_mask'] == exact_ties and detail['eligible_tie_mask'] == tie_mask,
            'Eligibility, float32 tie membership or clipped prior differs')
    require(detail['tie_hashes'] == hashes, 'Nonlearned tie hash differs')
    require(detail['scores'] == [s if math.isfinite(s) else None for s in scores], 'Saved ranked scores differ')
    require(detail['selected_index'] == chosen, 'Duplicated selected index differs')
    if mode != 'hash_tie':
        require(detail['actor_fingerprint'] is None, 'Unexpected actor hash input in learned/reference mode')
    gap = logits[raw] - logits[chosen]
    classification = ('unchanged' if chosen == wrapped else 'exact_tie' if gap == 0 else 'positive_margin')
    require(detail['selected_reference_gap'] == gap and detail['change_classification'] == classification,
            'Reference margin/change classification differs')
    require(detail['wrapper_active'] == bool(wins) and detail['wrapper_applied'] == (bool(wins) and corrected != wins[0]),
            'Win wrapper classification differs')
    if mode in ('learned_tie', 'hash_tie') and chosen != wrapped:
        require(gap == 0 and tie_mask[chosen], 'Tie-only policy overrode a positive reference margin')
    return dict(optional=True, changed=chosen != wrapped, classification=classification,
                exact_top_tie=sum(exact_ties) > 1, eligible_top_tie=len(tie_indices) > 1,
                wrapper_active=bool(wins), fractional_companion_diff=(mode == 'fractional' and chosen != companion))


def audit_response_links(rows, requests, policies):
    covered = []
    for step, request in enumerate(requests):
        require(request['step'] == step and request['submitted'] is True, 'Response step order/submission differs')
        expected = hashlib.sha256(encoded(dict(step=step, player_views=request['player_views']))).hexdigest()
        require(request['pre_response_id'] == expected, 'Pre-response information state hash differs')
        seats = []
        for index in request['rows']:
            require(type(index) is int and 0 <= index < len(rows), 'Response points outside decision rows')
            row, seat = rows[index], rows[index]['seat']
            require(row['row_index'] == index and row['step'] == step and row['pre_response_id'] == expected,
                    'Decision response identity differs')
            require(row['player_view_sha256'] == request['player_views'][str(seat)]
                    and row['policy_id'] == policies[seat]['policy_id'], 'Player view or behavior provenance differs')
            seats.append(seat)
            covered.append(index)
        require(len(seats) == len(set(seats)), 'Duplicate simultaneous response by one seat')
    require(covered == list(range(len(rows))), 'Decision response coverage is not complete and ordered')


def _audit_game_decisions(item):
    deadline_check(item['deadline'])
    record, root = item['record'], Path(item['root'])
    artifact = record['artifact']
    raw = (root / artifact['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == artifact['sha256'], 'Auditor game changed before inspection')
    data = json.loads(gzip.decompress(raw))
    from riichi_ai.auditing import VIEW_HASH_SCHEMA
    from riichi_ai.rules import rule_profile_hash
    require(data['schema'] == 'riichi-ai-stage3j-evaluation-match-v1'
            and data['rule_profile_sha256'] == rule_profile_hash()
            and data['player_view_hash_schema'] == VIEW_HASH_SCHEMA,
            'Auditor rule profile, view commitment or schema identity differs')
    require(data['completed'] is True and data['failure'] is None and data['plan']['match_id'] == record['match_id'],
            'Auditor completion or match identity differs')
    require(data['policies'] == record['policies'] and data['scores'] == record['scores']
            and len(data['rows']) == record['rows'] and len(data['requests']) == record['steps'],
            'Auditor row/policy/request counts differ')
    audit_response_links(data['rows'], data['requests'], data['policies'])
    counts, focal, diagnostics = Counter(), Counter(), Counter()
    reported = dict(optional_decisions=0, forced_decisions=0, reference_discard_decisions=0,
                    changed=0, exact_tie_changes=0, positive_margin_changes=0, reference_discard_changes=0,
                    exact_top_tie_decisions=0, eligible_top_tie_decisions=0, win_opportunity_decisions=0,
                    wrapper_override_decisions=0,
                    small_vs_tie_decision_disagreements=0)
    first_optional = None
    for row in data['rows']:
        policy = data['policies'][row['seat']]
        checked = audit_decision(row, policy['name'], item['modes'].get(policy['name']))
        counts[row['selected_kind']] += 1
        if row['seat'] == record['focal_seat']:
            focal[row['selected_kind']] += 1
            if checked['optional']:
                detail = row['correction']
                is_discard = row['action_kinds'][detail['wrapped_reference_index']] == 'DISCARD'
                reported['optional_decisions'] += 1
                reported['reference_discard_decisions'] += int(is_discard)
                reported['changed'] += int(checked['changed'])
                reported['exact_tie_changes'] += checked['classification'] == 'exact_tie'
                reported['positive_margin_changes'] += checked['classification'] == 'positive_margin'
                reported['reference_discard_changes'] += int(is_discard and checked['changed'])
                reported['exact_top_tie_decisions'] += int(checked['exact_top_tie'])
                reported['eligible_top_tie_decisions'] += int(checked['eligible_top_tie'])
                reported['win_opportunity_decisions'] += int(detail['wrapper_active'])
                reported['wrapper_override_decisions'] += int(detail['wrapper_applied'])
                reported['small_vs_tie_decision_disagreements'] += int(checked['fractional_companion_diff'])
                if first_optional is None:
                    first_optional = row['row_index']
                diagnostics['optional_rows'] += 1
                diagnostics[checked['classification'] + '_rows'] += 1
                for key in ('changed', 'exact_top_tie', 'eligible_top_tie', 'wrapper_active', 'fractional_companion_diff'):
                    diagnostics[key + '_rows'] += int(checked.get(key, False))
            else:
                reported['forced_decisions'] += 1
    require(dict(counts) == record['chosen_action_counts'] and dict(focal) == record['focal_action_counts'],
            'Reported action counts differ')
    require(reported == record['focal_diagnostics'], 'Reported focal intervention counts differ')
    require(data['ordinal_wall_fingerprints'] == record['ordinal_wall_fingerprints']
            and [row['hand_id'] for row in data['hand_starts']] == list(range(1, len(data['ordinal_wall_fingerprints']) + 1)),
            'Explicit hand/wall ordinal coverage differs')
    require(sum(event.get('type') == 'start_kyoku' for event in data['canonical_events']) == len(data['hand_starts']),
            'Canonical events omit an explicit hand continuation')
    require(hashlib.sha256(encoded(data['canonical_events'])).hexdigest() == record['game_sha256'],
            'Canonical engine event stream hash differs')
    from riichi_ai.stage3c import terminal_result
    terminal = terminal_result(record['scores'], record['ranks'], data['canonical_events'], completed=True)
    require(terminal == data['terminal'] and terminal['utilities'] == record['utilities'],
            'Reconstructed terminal scoring/rank utility differs')
    sample = data.get('forward_sample')
    if record['block'] in SAMPLE_BLOCKS:
        require((first_optional is None and sample is None) or (sample is not None and sample['row_index'] == first_optional),
                'Fixed first-optional forward sample absent or adaptively selected')
        if sample is not None:
            sample = dict(match_id=record['match_id'], arm=record['arm'], sample=sample,
                          row=data['rows'][first_optional])
    else:
        require(sample is None, 'Unexpected forward sample outside fixed block set')
    require(record['forward_sample_rows'] == int(sample is not None), 'Reported forward sample count differs')
    require(record['forward_sample_missing'] == (record['block'] in SAMPLE_BLOCKS and first_optional is None),
            'Missing fixed forward sample coverage was misreported')
    deadline_check(item['deadline'])
    return dict(match_id=record['match_id'], arm=record['arm'], rows=record['rows'], requests=record['steps'],
                diagnostics=dict(diagnostics), sample=sample, complete=True)


def audit_all_game_decisions(root, records, execution, deadline):
    from riichi_ai.evaluation import _terminate_evaluation_pool
    modes = {item['name']: item['mode'] for item in execution['arms']}
    jobs = iter(dict(root=str(root), record=record, deadline=deadline, modes=modes) for record in records.values())
    pool = ProcessPoolExecutor(max_workers=execution['workers'], mp_context=multiprocessing.get_context('spawn'))
    result = dict(games=0, rows=0, requests=0)
    by_arm, samples = {}, []
    try:
        pending = deque()
        for _ in range(execution['workers']):
            item = next(jobs, None)
            if item:
                pending.append(pool.submit(_audit_game_decisions, item))
        while pending:
            deadline_check(deadline)
            checked = pending.popleft().result(timeout=max(.001, deadline - time.monotonic()))
            result['games'] += 1
            for key in ('rows', 'requests'):
                result[key] += checked[key]
            arm = by_arm.setdefault(checked['arm'], Counter())
            arm['games'] += 1
            arm.update(checked['diagnostics'])
            if checked['sample'] is not None:
                samples.append(checked['sample'])
            item = next(jobs, None)
            if item:
                pending.append(pool.submit(_audit_game_decisions, item))
            if result['games'] % 512 == 0:
                print(json.dumps(dict(audit='full_game_decisions', games=result['games'])), flush=True)
    except BaseException:
        _terminate_evaluation_pool(pool)
        raise
    else:
        pool.shutdown(wait=True)
    require(result['games'] == len(records), 'Full game decision audit incomplete')
    return dict(complete=True, **result, by_arm={k: dict(v) for k, v in by_arm.items()},
                all_semantic_tokens_and_simultaneous_response_links_verified=True,
                all_model_choices_recomputed_from_persisted_scores=True,
                all_terminal_scores_reconstructed=True), samples


def replay_selection(execution):
    """Exactly four per arm, independent of outcomes and report ordering."""
    names = [item['name'] for item in execution['arms']]
    require(len(names) == len(set(names)) == 8, 'Replay requires all eight fixed arms')
    selected = []
    for arm in names:
        jobs = [job for job in execution['matches'] if job['arm'] == arm]
        require(len(jobs) >= 4 and len({job['match_id'] for job in jobs}) == len(jobs),
                'Replay arm has insufficient or duplicate matches')
        selected.extend(sorted(jobs, key=lambda job: (hashlib.sha256(
            ('stage3j-audit-replay:' + job['match_id']).encode()).digest(), job['match_id']))[:4])
    return selected


def _replay_worker(item):
    from riichi_ai.stage3j_evaluation import run_one
    return run_one(item['job'], item['execution'], item['output'], deadline=item['deadline'])


def replay(plan, evaluation, output_dir, deadline, reserved):
    manifest = read(evaluation['manifest_path'])
    execution = manifest['execution_plan']
    require(evaluation['complete'] and manifest['complete'], 'Cannot replay incomplete formal game panel')
    require(sha(evaluation['manifest_path']) == evaluation['manifest_sha256'], 'Replay manifest changed')
    jobs = replay_selection(execution)
    require(len(jobs) == 32 and reserved == 32, 'Exactly 32 replay attempts require an explicit reservation')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'replay-plan.json', dict(schema='stage3j-replay-plan-v1', games=jobs,
          source_manifest_sha256=evaluation['manifest_sha256'],
          selection='First four SHA256(stage3j-audit-replay:match_id) per fixed arm'))
    original = {item['match_id']: item for item in manifest['matches']}
    from riichi_ai.evaluation import _terminate_evaluation_pool
    pool = ProcessPoolExecutor(max_workers=execution['workers'], mp_context=multiprocessing.get_context('spawn'))
    try:
        futures = [pool.submit(_replay_worker, dict(job=job, execution=execution, output=str(root), deadline=deadline)) for job in jobs]
        results = []
        for future in futures:
            deadline_check(deadline)
            result = future.result(timeout=max(.001, deadline - time.monotonic()))
            require(result['completed'] is True and result['failure'] is None, 'Replay failed; no replacement allowed')
            old = original[result['match_id']]
            for key in ('game_sha256', 'scores', 'ranks', 'utilities', 'ordinal_wall_fingerprints',
                        'focal_action_counts', 'rows', 'steps', 'policies', 'payload_sha256'):
                require(result[key] == old[key], 'Replay deterministic payload differs: ' + key)
            results.append(dict(match_id=result['match_id'], arm=result['arm'],
                                game_sha256=result['game_sha256'], rows=result['rows'], exact=True))
    except BaseException:
        _terminate_evaluation_pool(pool)
        raise
    else:
        pool.shutdown(wait=True)
    require(len(list((root / 'lifecycle').glob('*.started.json'))) == 32, 'Replay attempt count differs')
    return dict(schema='stage3j-independent-replays-v1', complete=True, replays=results, attempts=32,
                all_game_and_action_artifact_payloads_exact=True,
                limitation='Same pinned engine deterministic reproducibility, not cross-engine full rule certification')


def historical_values(value):
    """Collect explicitly named integer seed fields and recorded first walls."""
    seeds, walls = set(), set()
    if isinstance(value, dict):
        for key, item in value.items():
            if (key == 'seed' or key == 'seeds' or key.endswith('_seeds')):
                if type(item) is int:
                    seeds.add(item)
                elif isinstance(item, list):
                    seeds.update(x for x in item if type(x) is int)
            if key == 'ordinal_wall_fingerprints' and isinstance(item, list) and item:
                require(isinstance(item[0], str), 'Nonstring historical first-wall commitment')
                walls.add(item[0])
            sub_seeds, sub_walls = historical_values(item)
            seeds.update(sub_seeds)
            walls.update(sub_walls)
    elif isinstance(value, list):
        for item in value:
            sub_seeds, sub_walls = historical_values(item)
            seeds.update(sub_seeds)
            walls.update(sub_walls)
    return seeds, walls


def audit_dependencies(plan, deadline):
    dependencies = plan['dependencies']
    require(dependencies and len({item['path'] for item in dependencies}) == len(dependencies),
            'Missing or duplicate frozen dependency inputs')
    for item in dependencies:
        deadline_check(deadline)
        path = Path(item['path'])
        require(path.is_file() and not path.is_symlink() and sha(path) == item['sha256'],
                'Historical dependency bytes changed: ' + str(path))
    require(plan['training_seeds'] == list(SEEDS)
            and [item['seed'] for item in plan['selected_bundles']] == list(SEEDS),
            'Saved seed family changed')
    from riichi_ai.stage3i_model import load_head_bundle
    bundle_results = []
    for item in plan['selected_bundles']:
        deadline_check(deadline)
        heads, metadata = load_head_bundle(item['checkpoint'], expected_sha256=item['sha256'],
                                          expected_identity=plan['source_identity'], device='cpu')
        require(metadata['seed'] == item['seed'] and metadata['complete'] is True
                and metadata.get('benchmark_only') is not True
                and all(not p.requires_grad and p.grad is None for p in heads.parameters()),
                'Bundle is not a completed immutable historical head')
        bundle_results.append(dict(seed=item['seed'], sha256=item['sha256'],
                                   weights_sha256=metadata['weights_sha256'], selected_epochs=metadata['selected_epochs']))
    historical_seeds, first_walls, checked_paths = set(), set(), set()
    for source in plan['freshness']['historical_seed_sources']:
        deadline_check(deadline)
        require(sha(source['path']) == source['sha256'], 'Historical inventory snapshot changed')
        snapshot = read(source['path'])
        seeds, walls = set(), set()
        for record in snapshot['inputs']:
            deadline_check(deadline)
            path = Path(record['path'])
            require(path.is_file() and not path.is_symlink() and sha(path) == record['sha256'],
                    'Historical seed/wall evidence changed: ' + str(path))
            s, w = historical_values(read(path))
            seeds.update(s)
            walls.update(w)
            checked_paths.add(str(path))
        require(sorted(seeds) == snapshot['historical_seeds']
                and sorted(walls) == snapshot['historical_first_wall_fingerprints'],
                'Historical snapshot union differs from its hashed source inputs')
        historical_seeds.update(seeds)
        first_walls.update(walls)
    require(sorted(historical_seeds) == plan['freshness']['historical_seeds']
            and sorted(first_walls) == plan['freshness']['historical_first_wall_fingerprints'],
            'Frozen plan freshness union differs')
    formal, preflight = plan['evaluation_seeds'], plan['preflight_evaluation_seeds']
    require(formal == [202610290000 + i for i in range(256)] and preflight == [202610289000, 202610289001],
            'Fresh fixed wall block families changed')
    require(not (set(formal + preflight) & historical_seeds) and len(set(formal + preflight)) == 258,
            'Fresh seeds collide with recorded history or each other')
    return dict(complete=True, dependencies_verified=len(dependencies), selected_bundles=bundle_results,
                historical_source_files_verified=len(checked_paths), historical_seeds_checked=len(historical_seeds),
                historical_first_wall_fingerprints=len(first_walls), fresh_seed_disjointness_verified=True), first_walls


def independent_actor_fingerprint(arrays):
    """Separate byte-level implementation of the declared actor commitment."""
    fields = {'tiles': (np.dtype('float32'), (34, 23)), 'globals': (np.dtype('float32'), (45,)),
              'history': (np.dtype('int16'), (128, 10)), 'history_mask': (np.dtype('bool'), (128,)),
              'action_tokens': (np.dtype('int16'), None), 'candidate_efficiency': (np.dtype('float32'), None)}
    require(set(arrays) == set(fields), 'Forward sample contains fields outside the actor allowlist')
    count = len(arrays['action_tokens'])
    fields['action_tokens'] = (np.dtype('int16'), (count, 8))
    fields['candidate_efficiency'] = (np.dtype('float32'), (count, 4))
    result = hashlib.sha256(encoded('riichi-ai-stage3j-actor-feature-fingerprint-v1'))
    for name in sorted(fields):
        dtype, shape = fields[name]
        values = arrays[name]
        require(values.dtype == dtype and values.shape == shape and np.isfinite(values).all(),
                'Fresh forward actor sample shape/dtype/finiteness differs')
        canonical_dtype = dtype.newbyteorder('<')
        result.update(encoded([name, canonical_dtype.str, list(shape)]))
        result.update(np.asarray(values, dtype=canonical_dtype, order='C').tobytes(order='C'))
    return result.hexdigest()


def audit_forwards(plan, samples, deadline):
    """Original actor prehook exposes scorer inputs without production extractor."""
    from riichi_ai.model import NeuralPolicy, collate_encoded
    from riichi_ai.stage3c import CallPolicy, grouped_distribution
    from riichi_ai.semantics import ACTION_KINDS
    actor = NeuralPolicy.from_checkpoint(plan['reference_checkpoint'], device='cpu')
    require(actor.checkpoint_hash == plan['reference_sha256'], 'Original actor identity differs')
    call_saved = torch.load(plan['call_checkpoint'], map_location='cpu', weights_only=True)
    call = CallPolicy(actor.model, call_saved['initialization_seed'])
    call.load_state_dict(call_saved['model_state_dict'], strict=True)
    call.eval()
    bundles = {item['seed']: torch.load(item['checkpoint'], map_location='cpu', weights_only=True)
               for item in plan['selected_bundles']}
    captured = []
    hook = actor.model.action_score.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    result = dict(actor_forwards=0, action_head_forwards=0, explorer_probability_forwards=0,
                  actor_hash_reconstructions=0, expected_maximum_samples=256)
    expected_names = {f'block-{block:03d}-seat-{seat}-{arm}' for block in SAMPLE_BLOCKS for seat in range(4)
                      for arm in ['reference_wrapped', 'hash_tie', *(f'tie_seed_{s}' for s in SEEDS), *(f'small_seed_{s}' for s in SEEDS)]}
    require(len(samples) == len({item['match_id'] for item in samples})
            and {item['match_id'] for item in samples} <= expected_names, 'Forward sample membership differs')
    try:
        for item in sorted(samples, key=lambda x: x['match_id']):
            deadline_check(deadline)
            saved, row = item['sample']['encoded'], item['row']
            arrays = {}
            for key, value in saved.items():
                require(value['dtype'] in ('float32', 'int16', 'bool'), 'Unapproved actor sample dtype')
                arrays[key] = np.asarray(value['values'], dtype=value['dtype'])
                require(list(arrays[key].shape) == value['shape'], 'Persisted sample shape differs')
            fingerprint = independent_actor_fingerprint(arrays)
            batch = collate_encoded([arrays], 'cpu')
            with torch.inference_mode():
                logits, _ = actor.model(batch)
                representation = captured.pop()
                context, candidates = representation[:, 0, :256], representation[:, :, 256:]
                require(torch.equal(representation[:, :, :256], context[:, None, :].expand(-1, candidates.shape[1], -1)),
                        'Original actor scorer has nonconstant state features')
                require(np.array_equal(logits[0].numpy().astype(np.float64), np.asarray(row['legal_scores'], np.float64)),
                        'Saved fresh legal scores differ from original actor forward')
                kinds = batch['action_tokens'][0, :, 0]
                require([ACTION_KINDS[int(kind) - 1] for kind in kinds] == row['action_kinds'],
                        'Sample action kind encoding differs from semantic actions')
                result['actor_forwards'] += 1
                gate, calls = call.heads(context, candidates)
                base = grouped_distribution(logits[0].double(), gate[0].double(), calls[0].double(), kinds)['probabilities'].numpy()
                base = base / base.sum()
                wins = np.flatnonzero(np.isin(kinds.numpy(), [ACTION_KINDS.index(name) + 1 for name in ('RON', 'TSUMO')]))
                if len(wins):
                    q = np.zeros(len(base), np.float64)
                    q[wins[0]] = 1.
                else:
                    q = .95 * base + .05 * np.full(len(base), 1. / len(base), np.float64)
                    q = q / q.sum()
                require(np.array_equal(q, np.asarray(row['correction']['explorer_probabilities'], np.float64)),
                        'Saved fresh explorer probabilities differ from original grouped policy')
                result['explorer_probability_forwards'] += 1
                if item['arm'].startswith(('tie_seed_', 'small_seed_')):
                    seed = int(item['arm'].rsplit('_', 1)[1])
                    state = bundles[seed]['head_state_dicts']['action']
                    # Direct linear/ReLU arithmetic, separate from both the
                    # live extractor and ResidualHeads.forward implementation.
                    x = torch.nn.functional.linear(representation, state['layers.0.weight'], state['layers.0.bias'])
                    x = torch.relu(x)
                    residual = torch.nn.functional.linear(x, state['layers.2.weight'], state['layers.2.bias']).squeeze(-1)[0].numpy()
                    require(np.array_equal(residual.astype(np.float64), np.asarray(row['correction']['action_residuals'], np.float64)),
                            'Saved fresh action residual differs from direct immutable head arithmetic')
                    result['action_head_forwards'] += 1
                else:
                    require(all(x == 0 for x in row['correction']['action_residuals']), 'Control has unexpected learned residual')
                if item['arm'] == 'hash_tie':
                    require(row['correction']['actor_fingerprint'] == fingerprint, 'Hash control fingerprint differs from actor tensors')
                    result['actor_hash_reconstructions'] += 1
    finally:
        hook.remove()
    result.update(complete=True, missing_optional_samples=sorted(expected_names - {item['match_id'] for item in samples}),
                  interpretation='Fixed first optional focal rows only; replay additionally checks complete deterministic execution')
    return result


def statistics(values, seed=20261029031):
    values = np.asarray(values, np.float64)
    count, mean = len(values), float(np.mean(values))
    require(count > 0 and np.isfinite(values).all() and np.all(np.abs(values) <= 3), 'Invalid paired block differences')
    rng = np.random.default_rng(seed)
    samples = np.empty(5000, np.float64)
    for start in range(0, 5000, 100):
        samples[start:start + 100] = rng.choice(values, size=(100, count), replace=True).mean(axis=1)
    half = 6 * math.sqrt(math.log(40) / (2 * count))
    return dict(independent_blocks=count, mean_utility_difference=mean, mean_rank_improvement=mean,
                standard_error=float(values.std(ddof=1) / math.sqrt(count)) if count > 1 else None,
                fixed_budget_hoeffding_95=[max(-3., mean - half), min(3., mean + half)],
                bootstrap_percentile_95_exploratory=np.percentile(samples, [2.5, 97.5]).tolist(), bootstrap_replicates=5000)


def near(left, right, message):
    require(np.allclose(left, right, rtol=1e-12, atol=1e-12), message)


def audit_statistics(execution, records, saved):
    names = [item['name'] for item in execution['arms']]
    blocks = execution['blocks']
    pairs = [(name, 'reference_wrapped') for name in names[1:]]
    pairs += [(f'tie_seed_{seed}', 'hash_tie') for seed in SEEDS]
    pairs += [(f'small_seed_{seed}', f'tie_seed_{seed}') for seed in SEEDS]
    pairing = []
    require(set(saved['contrasts']) == {arm + '_vs_' + baseline for arm, baseline in pairs},
            'Descriptive contrast membership differs')
    for arm, baseline in pairs:
        values, candidate_values, reference_values = [], [], []
        for block in range(blocks):
            differences = []
            for seat in range(4):
                a = records[f'block-{block:03d}-seat-{seat}-{arm}']
                b = records[f'block-{block:03d}-seat-{seat}-{baseline}']
                common = min(len(a['ordinal_wall_fingerprints']), len(b['ordinal_wall_fingerprints']))
                require(common > 0 and a['ordinal_wall_fingerprints'][:common] == b['ordinal_wall_fingerprints'][:common],
                        'Shared seeds lack matching explicit ordinal physical walls')
                pairing.append(dict(arm=arm, baseline=baseline, block=block, focal_seat=seat,
                                    common_hands=common, paired_common_walls_match=True))
                candidate_values.append(a['utilities'][seat])
                reference_values.append(b['utilities'][seat])
                differences.append(a['utilities'][seat] - b['utilities'][seat])
            values.append(sum(differences) / 4)
        reported = saved['contrasts'][arm + '_vs_' + baseline]
        require(reported['block_utility_differences'] == values, 'Block averaging differs')
        near(reported['mean_utility'], np.mean(candidate_values), 'Candidate mean utility differs')
        near(reported['baseline_mean_utility'], np.mean(reference_values), 'Reference mean utility differs')
        near(reported['mean_rank'], 2.5 - np.mean(candidate_values), 'Candidate mean rank differs')
        near(reported['baseline_mean_rank'], 2.5 - np.mean(reference_values), 'Reference mean rank differs')
        for key, value in statistics(values).items():
            if value is None:
                require(reported['statistics'][key] is None, 'Undefined statistic differs')
            else:
                near(reported['statistics'][key], value, 'Reported descriptive statistic differs: ' + key)
    require(saved['pairing'] == pairing, 'Wall pairing coverage differs')
    require(set(saved['by_arm']) == set(names), 'Arm summary membership differs')
    for arm in names:
        rows = [record for record in records.values() if record['arm'] == arm]
        utilities = [record['utilities'][record['focal_seat']] for record in rows]
        diagnostics = Counter()
        for record in rows:
            diagnostics.update(record['focal_diagnostics'])
        expected = dict(games=len(rows), mean_utility=float(np.mean(utilities)),
                        mean_rank=2.5 - float(np.mean(utilities)), focal_diagnostics=dict(diagnostics))
        require(saved['by_arm'][arm] == expected, 'Per-arm mean or intervention summary differs')
    for seed in SEEDS:
        count = sum(records[f'block-{block:03d}-seat-{seat}-tie_seed_{seed}']['game_sha256'] ==
                    records[f'block-{block:03d}-seat-{seat}-small_seed_{seed}']['game_sha256']
                    for block in range(blocks) for seat in range(4))
        require(saved['matched_tie_small'][str(seed)] == dict(games=blocks * 4, identical_canonical_games=count),
                'Tie/fractional canonical-game identity counts differ')
    require(saved['complete'] is True and saved['games'] == len(records) and saved['blocks'] == blocks
            and saved['selected_candidate'] is None and saved['seed_selection'] is False
            and saved['policy_promotion'] is False and saved['confirmation_started'] is False,
            'Statistics changed experiment role or selected a candidate')
    return dict(complete=True, contrasts=len(pairs), physical_pair_checks=len(pairing),
                block_weighting_verified=True, descriptive_bootstrap_replicates=5000)


def audit_games(plan, evaluation, deadline, historical_walls, plan_sha256):
    require(evaluation['complete'] is True, 'Formal game panel incomplete')
    path = Path(evaluation['manifest_path'])
    require(sha(path) == evaluation['manifest_sha256'], 'Game manifest changed')
    manifest, root = read(path), path.parent
    execution = manifest['execution_plan']
    names = ['reference_wrapped', 'hash_tie', *(f'tie_seed_{seed}' for seed in SEEDS), *(f'small_seed_{seed}' for seed in SEEDS)]
    require(manifest['complete'] is True and execution['phase'] == 'formal' and execution['blocks'] == 256
            and execution['rotations'] == 4 and execution['fixed_seed_family'] == list(SEEDS)
            and execution['forward_sample_blocks'] == list(SAMPLE_BLOCKS)
            and execution['source_identity'] == plan['source_identity']
            and execution['plan_sha256'] == plan_sha256
            and execution['protocol_sha256'] == plan['protocol_sha256'], 'Formal fixed game execution identity differs')
    require(sha(root / 'execution-plan.json') == manifest['execution_plan_sha256']
            and read(root / 'execution-plan.json') == execution, 'Persisted game execution plan differs')
    require([item['name'] for item in execution['arms']] == names, 'Fixed arm membership differs')
    expected_arms = [dict(name='reference_wrapped', mode='reference_wrapped'), dict(name='hash_tie', mode='hash_tie')]
    for prefix, mode in [('tie', 'learned_tie'), ('small', 'fractional')]:
        for item in plan['selected_bundles']:
            expected_arms.append(dict(name=f'{prefix}_seed_{item["seed"]}', mode=mode,
                                     seed=item['seed'], checkpoint=item['checkpoint'], sha256=item['sha256'],
                                     strength=.25 if mode == 'fractional' else 0.))
    require(execution['arms'] == expected_arms, 'Saved bundle, choice mode or fixed correction differs')
    expected = []
    for block, seed in enumerate(plan['evaluation_seeds']):
        for focal in range(4):
            for arm in names:
                lineup = [''] * 4
                lineup[focal] = arm
                for offset, opponent in enumerate(('shanten_v2', 'defense_v2', 'reference_raw'), 1):
                    lineup[(focal + offset) % 4] = opponent
                expected.append(dict(match_id=f'block-{block:03d}-seat-{focal}-{arm}', block=block,
                                     focal_seat=focal, arm=arm, seed=seed, lineup=lineup))
    records = {record['match_id']: record for record in manifest['matches']}
    require(execution['matches'] == expected and len(records) == len(manifest['matches']) == 8192
            and set(records) == {job['match_id'] for job in expected}, 'Complete fixed block/seat/arm coverage differs')
    require(execution['planned_matches'] == evaluation['planned_games'] == evaluation['completed_games']
            == evaluation['attempted_games'] == len(records), 'Failed or missing attempts disappeared')
    require(execution['training_eligible'] is False and execution['ppo_eligible'] is False
            and execution['policy_promotion'] is False and execution['seed_selection'] is False,
            'Execution role differs')
    reference_identity = execution['authenticated_policies']['frozen']
    candidate_identity = execution['authenticated_policies']['candidates']
    arm_by_name = {item['name']: item for item in execution['arms']}
    fresh_first_walls = set()
    for job in expected:
        deadline_check(deadline)
        record = records[job['match_id']]
        name = record['match_id']
        require(all(record.get(key) == value for key, value in job.items()), 'Saved game provenance differs')
        require(record['completed'] is True and record['failure'] is None
                and record['terminal_result_verified'] is True and record['training_eligible'] is False
                and record['ppo_eligible'] is False and record['source_role'] == 'development-only',
                'Incomplete or role-unsafe game record')
        require(sorted(record['ranks']) == [1, 2, 3, 4]
                and record['utilities'] == [2.5 - rank for rank in record['ranks']], 'Complete-match rank utility differs')
        require(read(root / 'reports' / f'{name}.json') == record, 'Game report differs from manifest')
        payload = {key: value for key, value in record.items() if key not in ('payload_sha256', 'elapsed_seconds')}
        require(hashlib.sha256(encoded(payload)).hexdigest() == record['payload_sha256'], 'Deterministic game payload hash differs')
        require(read(root / 'lifecycle' / f'{name}.started.json')['plan'] == job, 'Game lifecycle start differs')
        finish = read(root / 'lifecycle' / f'{name}.finished.json')
        require(finish['completed'] is True and finish['payload_sha256'] == record['payload_sha256']
                and finish['report_sha256'] == hashlib.sha256(encoded(record)).hexdigest(), 'Game completion receipt differs')
        artifact = record['artifact']
        binary = root / artifact['path']
        require(not binary.is_symlink() and binary.resolve().is_relative_to(root.resolve())
                and binary.stat().st_size == artifact['bytes'] and sha(binary) == artifact['sha256'],
                'Auditor game log artifact changed')
        require(record['ordinal_wall_fingerprints'], 'Missing physical-wall commitment')
        first_wall = record['ordinal_wall_fingerprints'][0]
        require(first_wall not in historical_walls, 'Fresh game first physical wall repeats recorded historical evidence')
        fresh_first_walls.add(first_wall)
        require([item['name'] for item in record['policies']] == job['lineup'], 'Seat policy order differs')
        for seat, policy in enumerate(record['policies']):
            local = policy['manifest']
            require(policy['seat'] == seat and policy['policy_id'] == policy['name'] + ':' + hashlib.sha256(encoded(local)).hexdigest(),
                    'Policy provenance digest or seat differs')
            if policy['name'] in ('reference_raw', 'reference_wrapped', 'hash_tie'):
                require(all(local.get(key) == value for key, value in reference_identity.items()),
                        'Frozen reference or call-head provenance differs')
                if policy['name'] == 'hash_tie':
                    require(local['salt'] == HASH_SALT and local['mode'] == 'hash_tie', 'Nonlearned tie control identity differs')
            elif policy['name'].startswith(('tie_seed_', 'small_seed_')):
                arm = arm_by_name[policy['name']]
                metadata = candidate_identity[policy['name']]
                require(local['checkpoint_sha256'] == arm['sha256'] == metadata['checkpoint_sha256']
                        and local['seed'] == arm['seed'] == metadata['seed']
                        and local['weights_sha256'] == metadata['weights_sha256']
                        and local['mode'] == arm['mode'] and local['strength'] == arm['strength']
                        and local['source_identity'] == plan['source_identity'], 'Learned policy provenance differs')
    require(len(list((root / 'lifecycle').glob('*.started.json'))) == len(records)
            and len(list((root / 'lifecycle').glob('*.finished.json'))) == len(records),
            'Unreported game attempt or lifecycle omission')
    require(not (fresh_first_walls & historical_walls), 'Historical physical-wall reuse')
    decisions, samples = audit_all_game_decisions(root, records, execution, deadline)
    forwards = audit_forwards(plan, samples, deadline)
    stats = audit_statistics(execution, records, manifest['statistics'])
    require(evaluation['statistics'] == manifest['statistics'], 'Game report statistics differ from immutable manifest')
    return dict(complete=True, games=len(records), decisions=decisions, fresh_forwards=forwards,
                statistics=stats, distinct_first_walls=len(fresh_first_walls), historical_first_wall_disjointness_verified=True,
                all_lifecycle_and_artifact_hashes_verified=True, no_seed_selection=True)


def lease(config, mode, max_seconds):
    ledger = read(Path(config['data_root']) / 'budget-ledger.json')
    active = [item for item in ledger['operations'] if item['status'] == 'active']
    require(len(active) == 1 and active[0]['name'] == os.environ.get('RIICHI_STAGE3J_OPERATION')
            and active[0]['token'] == os.environ.get('RIICHI_STAGE3J_TOKEN'), 'Matching exclusive Stage 3J audit lease required')
    require(active[0]['name'].startswith('audit'), 'Independent audit requires an audit operation')
    receipt = read(Path(config['reports_root']) / ('operation-' + active[0]['name'] + '.json'))
    reserved = int(os.environ.get('RIICHI_STAGE3J_MATCH_ATTEMPTS', '0'))
    require(reserved == receipt['reserved_match_attempts'], 'Replay reservation differs from receipt')
    require((mode == 'replay' and reserved == 32) or (mode == 'evidence' and reserved == 0), 'Incorrect audit game reservation')
    remaining = active[0]['reserved_seconds'] - (time.time() - active[0]['started_unix'])
    require(math.isfinite(max_seconds) and max_seconds > 0 and remaining > 12, 'Invalid cooperative audit deadline')
    return time.monotonic() + min(max_seconds, remaining - 10), reserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('evidence', 'replay'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--evaluation-report', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--report', required=True)
    parser.add_argument('--max-seconds', required=True, type=float)
    args = parser.parse_args()
    from stage3j import read_config, verify_sources
    config, identity = read_config()
    deadline, reserved = lease(config, args.mode, args.max_seconds)
    torch.set_num_threads(1)
    plan = read(args.plan)
    require(plan['config_sha256'] == identity and plan['execution_admitted'] is True,
            'Audit needs the admitted fixed configuration/plan')
    require(sha(plan['protocol_path']) == plan['protocol_sha256'], 'Written audit protocol changed')
    require(plan['forward_sample_blocks'] == list(SAMPLE_BLOCKS)
            and plan['replay_selection'] == dict(salt='stage3j-audit-replay:', per_arm=4), 'Fixed audit sampling changed')
    verify_sources(plan['sources'])
    started, evaluation = time.monotonic(), read(args.evaluation_report)
    report = Path(args.report).resolve()
    require(report.is_relative_to(Path(config['reports_root'])) and not report.exists(), 'Audit report must be a fresh file in reports root')
    if args.mode == 'replay':
        require(args.output_dir is not None, 'Replay evidence needs its fresh directory')
        output = Path(args.output_dir).resolve()
        require(output.is_relative_to(Path(config['data_root'])) and output != Path(config['data_root']) and not output.exists(),
                'Replay directory must be fresh and external')
        result = replay(plan, evaluation, output, deadline, reserved)
    else:
        dependencies, historical_walls = audit_dependencies(plan, deadline)
        games = audit_games(plan, evaluation, deadline, historical_walls, sha(args.plan))
        result = dict(schema='stage3j-independent-evidence-audit-v1', complete=True, dependencies=dependencies, games=games)
    for stem in ('reference', 'call'):
        require(sha(plan[stem + '_checkpoint']) == plan[stem + '_sha256'], 'Historical frozen checkpoint changed')
    verify_sources(plan['sources'])
    deadline_check(deadline)
    result.update(elapsed_seconds=time.monotonic() - started, plan_sha256=sha(args.plan), policy_promotion=False)
    write(report, result)
    print(json.dumps(dict(complete=True, mode=args.mode, elapsed_seconds=result['elapsed_seconds'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
