"""Post-hoc saved-only fractional-correction diagnostic; never select a policy.

The declared Stage 3I calibration and game arms are immutable. This diagnostic
uses their existing 200 calibration matches solely to formulate a possible
future preregistration. It performs no model forward, training or new games.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from riichi_ai.semantics import ACTION_KINDS

GRID = (0., 2.**-12, 2.**-10, 2.**-8, 2.**-6, 2.**-4, 2.**-3, 2.**-2, 2.**-1, 1.)
ORIGINAL_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
WIN_IDS = tuple(ACTION_KINDS.index(name) + 1 for name in ('RON', 'TSUMO'))
DISCARD_ID = ACTION_KINDS.index('DISCARD') + 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def analyze_row(logits, residuals, probabilities, kinds):
    """Fixed grid choices and exact right limit using stored float constants.

    First-change infimum is min(-prior_i/(A_i-A_ref)) for eligible
    positive-slope challengers. Here reference prior is zero and prior_i is
    already clipped. Canonical ties at the breakpoint can make equality either
    changed or unchanged, so the value is expressly an infimum, not a claim
    that the choice changes at equality.
    """
    z, a, q, kinds = (np.asarray(logits, dtype=np.float64), np.asarray(residuals, dtype=np.float64),
                      np.asarray(probabilities, dtype=np.float64), np.asarray(kinds))
    require(z.ndim == 1 and len(z) >= 2 and a.shape == q.shape == kinds.shape == z.shape
        and np.isfinite(z).all() and np.isfinite(a).all() and np.isfinite(q).all()
        and np.all(q >= 0) and np.all(q <= 1) and math.isclose(float(q.sum()), 1., abs_tol=1e-12, rel_tol=0),
        'Invalid saved optional decision arrays')
    raw = int(z.argmax())
    wins = np.flatnonzero(np.isin(kinds, WIN_IDS))
    reference = int(wins[0]) if len(wins) else raw
    prior = np.maximum(z - z[raw], -8.)
    eligible = q >= .02
    eligible[raw] = True
    scores = prior[None, :] + np.asarray(GRID)[:, None] * a[None, :]
    require(np.isfinite(scores).all(), 'Nonfinite fractional scores')
    scores[:, ~eligible] = -np.inf
    choices = scores.argmax(axis=1)
    choices[0] = raw
    tied = np.flatnonzero(eligible & (prior == 0.))
    zero_plus_raw = int(tied[a[tied].argmax()])
    slopes = a - a[raw]
    challengers = eligible & (slopes > 0)
    first = float(np.min(-prior[challengers] / slopes[challengers])) if np.any(challengers) else math.inf
    if len(wins):
        choices[:] = reference
        zero_plus = reference
        first = math.inf
    else:
        zero_plus = zero_plus_raw
    return {'choices': choices, 'raw_reference': raw, 'reference': reference,
        'zero_plus': zero_plus, 'zero_plus_raw': zero_plus_raw,
        'first_change_infimum': first, 'win_wrapper': bool(len(wins)),
        'exact_raw_logit_tie': int(np.sum(z == z[raw])) > 1,
        'exact_eligible_prior_tie': len(tied) > 1,
        'eligible_top_ties': len(tied), 'eligible_candidates': int(eligible.sum()),
        'reference_discard': int(kinds[reference]) == DISCARD_ID,
        'raw_margin': float(z[raw] - np.partition(z, -2)[-2])}


def _quantiles(values):
    if not values:
        return None
    levels = (0., .01, .05, .25, .5, .75, .95, .99, 1.)
    return {str(level): float(value) for level, value in zip(levels, np.quantile(values, levels))}


def run(args):
    from stage3i import read_config, verify_sources
    config, config_sha = read_config()
    operations = read(Path(config['data_root']) / 'budget-ledger.json')['operations']
    active = [item for item in operations if item['status'] == 'active']
    require(len(active) == 1 and active[0]['name'].startswith('audit')
        and active[0]['name'] == os.environ.get('RIICHI_STAGE3I_OPERATION')
        and active[0]['token'] == os.environ.get('RIICHI_STAGE3I_TOKEN'), 'Exclusive Stage 3I audit lease required')
    remaining = active[0]['reserved_seconds'] - (time.time() - active[0]['started_unix'])
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 25 and remaining > 5,
            'Saved diagnostic requires a finite at-most-25-second cooperative deadline')
    deadline = time.monotonic() + min(args.max_seconds, remaining - 3)
    started = time.monotonic()
    sources = read(args.source_manifest)['sources']
    require('scripts/stage3i_calibration_diagnostic.py' in sources, 'Diagnostic source identity absent')
    verify_sources(sources)
    plan, calibration = read(args.plan), read(args.calibration)
    require(plan['config_sha256'] == config_sha and plan.get('execution_admitted') is True
        and calibration.get('schema') == 'riichi-ai-stage3i-calibration-v1'
        and calibration.get('complete') is True and calibration['plan_sha256'] == digest(args.plan),
        'Frozen formal calibration identity mismatch')
    cache_path = Path(plan['cache_manifest_path'])
    require(digest(cache_path) == plan['cache_manifest_sha256'] == calibration['cache_manifest_sha256'],
            'Cache manifest identity mismatch')
    cache = read(cache_path)
    require(cache['partitions'] == plan['partitions'] and len(cache['partitions']['calibration']) == 200
        and cache.get('purpose') == 'formal' and cache.get('complete') is True
        and cache['behavior_id'] == 3 and cache['target'] == 'one_hand_delta',
        'Post-hoc diagnostic requires the original fixed calibration partition')
    names = cache['partitions']['calibration']
    by_match = {item['match_id']: item for item in cache['shards']}
    seeds = plan['training_seeds']
    require([item['seed'] for item in calibration['seeds']] == seeds, 'Fixed seed family changed')
    original = {item['seed']: item for item in calibration['seeds']}
    artifacts = {(item['seed'], item['partition'], item['match_id']): item for item in calibration['artifacts']}
    require(len(artifacts) == len(calibration['artifacts']), 'Duplicated saved prediction artifacts')
    prediction_root = Path(args.predictions_dir).resolve()
    require(prediction_root.is_relative_to(Path(config['data_root'])), 'Predictions must remain in Stage 3I evidence')
    require(digest(prediction_root / 'manifest.json') == digest(args.calibration),
            'Prediction directory differs from the saved calibration report')
    summaries = {seed: {'rows': 0, 'discard_rows': 0, 'grid_changed': np.zeros(len(GRID), np.int64),
        'grid_discard_changed': np.zeros(len(GRID), np.int64), 'zero_plus_changed': 0,
        'zero_plus_discard_changed': 0, 'zero_plus_raw_changed': 0, 'win_wrapper_rows': 0,
        'exact_raw_logit_tie_rows': 0, 'exact_eligible_prior_tie_rows': 0,
        'nonwin_exact_eligible_prior_tie_rows': 0, 'first_zero': 0, 'first_positive': [],
        'first_never': 0, 'raw_margins': [], 'matches': []} for seed in seeds}
    verified_sources = []
    for name in names:
        require(time.monotonic() < deadline, 'Saved diagnostic cooperative deadline')
        record = by_match[name]
        require(record['partition'] == 'calibration', 'Unexpected cache role')
        shard_path = Path(record['path'])
        if not shard_path.is_absolute():
            shard_path = cache_path.parent / shard_path
        require(digest(shard_path) == record['sha256'], 'Saved feature shard changed')
        with np.load(shard_path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in ('candidate_offsets', 'row_indices', 'kinds', 'logits', 'explorer_probabilities')}
        offsets = arrays['candidate_offsets']
        require(offsets.dtype == np.int64 and offsets.shape == (record['rows'] + 1,)
            and offsets[0] == 0 and offsets[-1] == record['candidates'] and np.all(np.diff(offsets) > 1), 'Malformed saved offsets')
        for seed in seeds:
            artifact = artifacts[(seed, 'calibration', name)]
            prediction_path = (prediction_root / artifact['path']).resolve()
            require(prediction_path.is_relative_to(prediction_root) and artifact['source_sha256'] == record['sha256']
                and digest(prediction_path) == artifact['sha256'], 'Saved prediction identity changed')
            with np.load(prediction_path, allow_pickle=False) as saved:
                predictions = {key: saved[key].copy() for key in ('action', 'choices_by_lambda', 'raw_reference', 'wrapped_reference', 'row_indices')}
            require(predictions['action'].shape == (record['candidates'],)
                and predictions['choices_by_lambda'].shape == (len(ORIGINAL_GRID), record['rows'])
                and np.array_equal(predictions['row_indices'], arrays['row_indices']), 'Saved prediction rows or shape mismatch')
            summary = summaries[seed]
            changed = np.zeros(len(GRID), np.int64); discard_changed = changed.copy()
            zero_plus_changed = zero_plus_discard_changed = discard_rows = 0
            for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
                item = analyze_row(arrays['logits'][start:stop], predictions['action'][start:stop],
                    arrays['explorer_probabilities'][start:stop], arrays['kinds'][start:stop])
                require(item['raw_reference'] == predictions['raw_reference'][row]
                    and item['reference'] == predictions['wrapped_reference'][row]
                    and item['choices'][0] == predictions['choices_by_lambda'][0, row]
                    and item['choices'][-1] == predictions['choices_by_lambda'][1, row],
                    'Fractional diagnostic disagrees with original zero/one correction choices')
                changes = item['choices'] != item['reference']
                changed += changes
                discard_rows += item['reference_discard']
                discard_changed += changes & item['reference_discard']
                zero_change = item['zero_plus'] != item['reference']
                zero_plus_changed += zero_change
                zero_plus_discard_changed += zero_change and item['reference_discard']
                summary['zero_plus_raw_changed'] += item['zero_plus_raw'] != item['raw_reference']
                summary['win_wrapper_rows'] += item['win_wrapper']
                summary['exact_raw_logit_tie_rows'] += item['exact_raw_logit_tie']
                summary['exact_eligible_prior_tie_rows'] += item['exact_eligible_prior_tie']
                summary['nonwin_exact_eligible_prior_tie_rows'] += item['exact_eligible_prior_tie'] and not item['win_wrapper']
                first = item['first_change_infimum']
                if math.isinf(first): summary['first_never'] += 1
                elif first == 0: summary['first_zero'] += 1
                else: summary['first_positive'].append(first)
                summary['raw_margins'].append(item['raw_margin'])
            require(time.monotonic() < deadline, 'Saved diagnostic cooperative deadline')
            summary['rows'] += record['rows']; summary['discard_rows'] += discard_rows
            summary['grid_changed'] += changed; summary['grid_discard_changed'] += discard_changed
            summary['zero_plus_changed'] += zero_plus_changed; summary['zero_plus_discard_changed'] += zero_plus_discard_changed
            summary['matches'].append({'match_id': name, 'rows': record['rows'], 'discard_rows': discard_rows,
                'grid_changed': changed.tolist(), 'grid_discard_changed': discard_changed.tolist(),
                'zero_plus_changed': int(zero_plus_changed), 'zero_plus_discard_changed': int(zero_plus_discard_changed)})
            verified_sources.append({'seed': seed, 'match_id': name, 'feature_sha256': record['sha256'],
                                     'prediction_sha256': artifact['sha256']})
    results = []
    for seed in seeds:
        item = summaries[seed]
        grid = []
        for index, strength in enumerate(GRID):
            changes, discard_changes = int(item['grid_changed'][index]), int(item['grid_discard_changed'][index])
            rate = changes / item['rows']; discard_rate = discard_changes / item['discard_rows'] if item['discard_rows'] else 0.
            grid.append({'lambda': strength, 'rows': item['rows'], 'changed': changes, 'change_rate': rate,
                'reference_discard_rows': item['discard_rows'], 'reference_discard_changed': discard_changes,
                'reference_discard_change_rate': discard_rate,
                'within_original_strict_ceilings': rate <= .03 and discard_rate <= .10,
                'match_mean_change_rate': float(np.mean([m['grid_changed'][index] / m['rows'] for m in item['matches']]))})
        for index, original_index in ((0, 0), (len(GRID)-1, 1)):
            old = original[seed]['lambda_grid'][original_index]
            require(grid[index]['changed'] == old['changed'] and grid[index]['rows'] == old['rows']
                and grid[index]['reference_discard_changed'] == old['reference_discard_changed'],
                'Zero/one aggregate counts differ from frozen calibration')
        zero_rate = item['zero_plus_changed'] / item['rows']
        zero_discard_rate = item['zero_plus_discard_changed'] / item['discard_rows'] if item['discard_rows'] else 0.
        results.append({'seed': seed, 'rows': item['rows'], 'matches': len(names), 'grid': grid,
            'zero_plus': {'changed': int(item['zero_plus_changed']), 'change_rate': zero_rate,
                'reference_discard_changed': int(item['zero_plus_discard_changed']), 'reference_discard_change_rate': zero_discard_rate,
                'within_original_strict_ceilings': zero_rate <= .03 and zero_discard_rate <= .10,
                'raw_changed_before_win_wrapper': int(item['zero_plus_raw_changed']),
                'definition': 'Exact right limit as lambda tends to zero from above for fixed stored logits/residuals, with canonical ties and the declared wrapper'},
            'tie_counts': {key: int(item[key]) for key in ('win_wrapper_rows', 'exact_raw_logit_tie_rows',
                'exact_eligible_prior_tie_rows', 'nonwin_exact_eligible_prior_tie_rows')},
            'raw_reference_margin_quantiles': _quantiles(item['raw_margins']),
            'first_change_infimum': {'zero_rows': item['first_zero'], 'positive_finite_rows': len(item['first_positive']),
                'never_changes_rows': item['first_never'], 'positive_finite_quantiles': _quantiles(item['first_positive']),
                'positive_finite_at_or_below_grid': {str(value): int(np.sum(np.asarray(item['first_positive']) <= value)) for value in GRID},
                'definition': 'First nonnegative lambda infimum at which an eligible challenger can beat raw reference; winning-wrapper rows count as never; equality tie behavior is not implied'},
            'match_metrics': item['matches'], 'selected_lambda': None})
    verify_sources(sources)
    require(digest(args.plan) == calibration['plan_sha256'] and digest(cache_path) == plan['cache_manifest_sha256'],
            'Frozen plan/cache identity changed during diagnostic')
    return {'schema': 'riichi-ai-stage3i-posthoc-fractional-diagnostic-v1', 'complete': True,
        'posthoc': True, 'purpose': 'Generate hypotheses for a separately preregistered experiment only',
        'plan_sha256': digest(args.plan), 'calibration_sha256': digest(args.calibration),
        'cache_manifest_sha256': plan['cache_manifest_sha256'], 'source_manifest_sha256': digest(args.source_manifest),
        'grid': list(GRID), 'seeds': results, 'verified_artifacts': verified_sources,
        'optimizer_steps': 0, 'model_forwards': 0, 'new_games': 0, 'policy_promotion': False,
        'recalibration': False, 'selected_lambda': None, 'formal_game_arms_changed': False,
        'interpretation': 'Reused calibration outcomes-free predictions inspected after the declared grid selected zero; feasible fractional counts do not validate strength or authorize a policy choice.',
        'elapsed_seconds': time.monotonic() - started, 'numpy_version': np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--predictions-dir', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-seconds', type=float, default=25.)
    args = parser.parse_args()
    require(not Path(args.output).exists(), 'Post-hoc diagnostic report is immutable')
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(output.resolve()), 'complete': result['complete'],
        'elapsed_seconds': result['elapsed_seconds'], 'new_games': 0, 'selected_lambda': None}), flush=True)


if __name__ == '__main__':
    main()
