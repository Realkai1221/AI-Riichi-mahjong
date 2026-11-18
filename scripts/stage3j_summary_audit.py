"""Independently verify Stage 3J's four pooled primary contrasts.

This auxiliary auditor does not import the report's analysis function, fit a
model, start games, change arm membership, or select a checkpoint. Its source
must be frozen in the active controller receipt before reading the results.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

SEEDS = (20261024011, 20261024012, 20261024013)
ARMS = ('reference_wrapped', 'hash_tie', *(f'tie_seed_{seed}' for seed in SEEDS),
        *(f'small_seed_{seed}' for seed in SEEDS))
CONTRASTS = ('learned_tie_vs_reference', 'learned_tie_vs_hash',
             'small_vs_reference', 'small_vs_learned_tie')
BLOCKS = 256
REPLICATES = 10000
RNG_SEED = 20261029031


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


def near(actual, expected, label):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=1e-12, atol=1e-12), label)


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Primary summary audit cooperative deadline reached')


def lease(config):
    from stage3j import verify_sources
    active = [item for item in read(Path(config['data_root']) / 'budget-ledger.json')['operations']
              if item['status'] == 'active']
    require(len(active) == 1 and active[0]['name'] == os.environ.get('RIICHI_STAGE3J_OPERATION')
            and active[0]['token'] == os.environ.get('RIICHI_STAGE3J_TOKEN')
            and active[0]['name'].startswith('audit'), 'Matching exclusive Stage 3J audit lease required')
    receipt_path = Path(config['reports_root']) / ('operation-' + active[0]['name'] + '.json')
    receipt = read(receipt_path)
    require(receipt['reserved_match_attempts'] == 0
            and os.environ.get('RIICHI_STAGE3J_MATCH_ATTEMPTS') == '0', 'Summary audit must reserve zero games')
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    require(receipt['sources'].get(relative) == sha(__file__),
            'Independent summary audit source was not frozen in the active receipt')
    verify_sources(receipt['sources'])
    remaining = active[0]['reserved_seconds'] - (time.time() - active[0]['started_unix'])
    require(math.isfinite(remaining) and remaining > 10, 'Insufficient cooperative audit deadline')
    return time.monotonic() + remaining - 8, receipt


def reconstruct(execution, records, plan, deadline):
    """Use integer twice-utilities; average twelve learned seat/seed cells."""
    require(execution['phase'] == 'formal' and execution['blocks'] == BLOCKS
            and execution['rotations'] == 4 and execution['fixed_seed_family'] == list(SEEDS)
            and [item['name'] for item in execution['arms']] == list(ARMS),
            'Fixed formal arm/seed/block structure changed')
    require(execution['training_eligible'] is False and execution['ppo_eligible'] is False
            and execution['policy_promotion'] is False and execution['seed_selection'] is False,
            'Formal execution role changed')
    require(plan['training_seeds'] == list(SEEDS)
            and plan['evaluation_seeds'] == list(range(202610290000, 202610290256)),
            'Preregistered training or fresh wall seeds changed')
    expected = {}
    for block, seed in enumerate(plan['evaluation_seeds']):
        for seat in range(4):
            for arm in ARMS:
                lineup = [''] * 4
                lineup[seat] = arm
                for offset, opponent in enumerate(('shanten_v2', 'defense_v2', 'reference_raw'), 1):
                    lineup[(seat + offset) % 4] = opponent
                name = f'block-{block:03d}-seat-{seat}-{arm}'
                expected[name] = dict(match_id=name, block=block, focal_seat=seat, arm=arm, seed=seed, lineup=lineup)
    require(execution['matches'] == list(expected.values()) and execution['planned_matches'] == 8192,
            'Execution jobs differ from complete fixed formal panel')
    require(len(records) == len({row['match_id'] for row in records}) == len(expected)
            and {row['match_id'] for row in records} == set(expected), 'Missing or duplicate formal match records')
    cells = {}
    placements = {arm: Counter() for arm in ARMS}
    for row in records:
        deadline_check(deadline)
        require(all(row.get(key) == value for key, value in expected[row['match_id']].items()),
                'Raw record seed/arm/seat provenance changed')
        require(row['completed'] is True and row['failure'] is None and row['terminal_result_verified'] is True
                and row['training_eligible'] is False and row['ppo_eligible'] is False,
                'Incomplete, unverified or training-eligible game cannot enter estimates')
        require(len(row['ranks']) == len(row['utilities']) == len(row['scores']) == 4
                and all(type(rank) is int for rank in row['ranks']) and sorted(row['ranks']) == [1, 2, 3, 4]
                and row['utilities'] == [2.5 - rank for rank in row['ranks']],
                'All-seat complete-match ranks/utilities differ from declared objective')
        payload = {key: value for key, value in row.items() if key not in ('elapsed_seconds', 'payload_sha256')}
        require(hashlib.sha256(encoded(payload)).hexdigest() == row['payload_sha256'],
                'Immutable raw game payload hash changed')
        seat, arm = row['focal_seat'], row['arm']
        key = (row['block'], seat, arm)
        require(key not in cells, 'Duplicated arm/seat cell under different match IDs')
        rank = row['ranks'][seat]
        cells[key] = 5 - 2 * rank
        placements[arm][rank] += 1
    # Twice-utilities are exact integers. Each ordinary arm/block averages
    # four seats, whereas each learned pooled policy averages all twelve
    # seat/seed cells. Seeds are never treated as additional bootstrap units.
    arm_values = {arm: [sum(cells[block, seat, arm] for seat in range(4)) / 8.
                        for block in range(BLOCKS)] for arm in ARMS}
    tie = [sum(cells[block, seat, f'tie_seed_{seed}'] for seed in SEEDS for seat in range(4)) / 24.
           for block in range(BLOCKS)]
    small = [sum(cells[block, seat, f'small_seed_{seed}'] for seed in SEEDS for seat in range(4)) / 24.
             for block in range(BLOCKS)]
    vectors = np.asarray([[tie[block] - arm_values['reference_wrapped'][block],
                           tie[block] - arm_values['hash_tie'][block],
                           small[block] - arm_values['reference_wrapped'][block],
                           small[block] - tie[block]] for block in range(BLOCKS)], np.float64)
    require(vectors.shape == (BLOCKS, 4) and np.isfinite(vectors).all()
            and np.all(np.abs(vectors) <= 3), 'Invalid complete-block contrast matrix')
    return arm_values, placements, tie, small, vectors


def audit_summary(summary, execution, records, plan, deadline):
    require(summary['schema'] == 'stage3j-fixed-statistics-v1' and summary['complete'] is True
            and summary['independent_blocks'] == BLOCKS and summary['games'] == 8192
            and summary['policy_promotion'] is False and summary['seed_selection'] is False,
            'Primary summary experiment role/counts changed')
    require(set(summary['primary']) == set(CONTRASTS) and set(summary['by_arm']) == set(ARMS)
            and set(summary['pooled_mean_rank']) == {'learned_tie', 'small'},
            'Primary family or arm/pooled membership changed')
    inference = summary['inference']
    require(inference['bootstrap_seed'] == RNG_SEED and inference['bootstrap_replicates'] == REPLICATES
            and inference['confidence_per_contrast'] == .9875 and inference['family_size'] == 4
            and inference['joint_block_resampling'] is True, 'Fixed bootstrap/inference procedure changed')
    arm_values, placements, tie, small, vectors = reconstruct(execution, records, plan, deadline)
    # Draw one shared block-index panel for every chunk and apply it to all
    # four columns. Chunking changes memory use, not the generator stream.
    generator = np.random.default_rng(RNG_SEED)
    bootstrap = np.empty((REPLICATES, 4), np.float64)
    for start in range(0, REPLICATES, 1000):
        deadline_check(deadline)
        indices = generator.integers(0, BLOCKS, size=(1000, BLOCKS))
        for column in range(4):
            bootstrap[start:start + 1000, column] = np.sum(vectors[indices, column], axis=1) / BLOCKS
    half = 6 * math.sqrt(math.log(160.) / (2 * BLOCKS))
    required_fields = {'mean_utility_difference', 'block_standard_deviation', 'standard_error',
                       'bootstrap_9875_interval', 'simultaneous_95_hoeffding',
                       'block_utility_differences', 'independent_blocks'}
    maximum_absolute_difference = 0.
    for column, name in enumerate(CONTRASTS):
        values = vectors[:, column].tolist()
        mean = math.fsum(values) / BLOCKS
        sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (BLOCKS - 1))
        ordered = sorted(bootstrap[:, column].tolist())
        intervals = []
        for probability in (.00625, .99375):
            # NumPy's default linear quantile, independently expressed.
            position = (REPLICATES - 1) * probability
            low = math.floor(position)
            high = math.ceil(position)
            intervals.append(ordered[low] + (position - low) * (ordered[high] - ordered[low]))
        expected = dict(mean_utility_difference=mean, block_standard_deviation=sd,
                        standard_error=sd / math.sqrt(BLOCKS), bootstrap_9875_interval=intervals,
                        simultaneous_95_hoeffding=[max(-3., mean - half), min(3., mean + half)],
                        block_utility_differences=values, independent_blocks=BLOCKS)
        actual = summary['primary'][name]
        require(set(actual) == required_fields, 'Primary statistics fields changed: ' + name)
        for key, value in expected.items():
            near(actual[key], value, 'Independent pooled primary arithmetic differs: ' + name + '/' + key)
            maximum_absolute_difference = max(maximum_absolute_difference, float(np.max(
                np.abs(np.asarray(actual[key], np.float64) - np.asarray(value, np.float64)))))
    for arm in ARMS:
        actual = summary['by_arm'][arm]
        require(set(actual) == {'mean_rank', 'mean_utility', 'placement_counts', 'games'}
                and actual['games'] == 1024 and actual['placement_counts'] == [placements[arm][rank] for rank in range(1, 5)],
                'Per-arm placement counts or sample size differ: ' + arm)
        mean = math.fsum(arm_values[arm]) / BLOCKS
        near(actual['mean_utility'], mean, 'Per-arm utility differs: ' + arm)
        near(actual['mean_rank'], 2.5 - mean, 'Per-arm rank differs: ' + arm)
    near(summary['pooled_mean_rank']['learned_tie'], 2.5 - math.fsum(tie) / BLOCKS, 'Pooled learned tie rank differs')
    near(summary['pooled_mean_rank']['small'], 2.5 - math.fsum(small) / BLOCKS, 'Pooled small-correction rank differs')
    return dict(schema='stage3j-independent-primary-summary-audit-v1', complete=True,
                games_verified=len(records), independent_blocks=BLOCKS, primary_contrasts=list(CONTRASTS),
                bootstrap_replicates=REPLICATES, bootstrap_chunk_size=1000, bootstrap_seed=RNG_SEED,
                joint_block_resampling_verified=True, all_three_saved_seeds_and_four_seats_verified=True,
                all_primary_fields_verified=True, all_arm_placements_and_pooled_means_verified=True,
                maximum_primary_absolute_difference=maximum_absolute_difference,
                comparison_rtol=1e-12, comparison_atol=1e-12, policy_promotion=False,
                limitation='Reproduces development statistics; bootstrap coverage remains approximate and no policy is promoted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--evaluation-report', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    from stage3j import read_config, verify_sources
    config, identity = read_config()
    deadline, receipt = lease(config)
    started = time.monotonic()
    paths = {'plan': Path(args.plan), 'evaluation_report': Path(args.evaluation_report), 'summary': Path(args.summary)}
    for path in paths.values():
        require(path.is_file() and not path.is_symlink(), 'Audit inputs must be existing ordinary files')
    identities = {name + '_sha256': sha(path) for name, path in paths.items()}
    plan, evaluation, summary = (read(paths[name]) for name in ('plan', 'evaluation_report', 'summary'))
    require(plan['config_sha256'] == identity and plan['execution_admitted'] is True
            and sha(plan['protocol_path']) == plan['protocol_sha256'], 'Fixed admitted plan/configuration changed')
    verify_sources(plan['sources'])
    require(evaluation['complete'] is True and evaluation['planned_games'] == 8192
            and evaluation['completed_games'] == 8192 and evaluation['attempted_games'] == 8192,
            'Complete formal eight-arm panel required')
    manifest_path = Path(evaluation['manifest_path'])
    require(manifest_path.is_file() and not manifest_path.is_symlink()
            and sha(manifest_path) == evaluation['manifest_sha256'], 'Formal manifest hash changed')
    manifest = read(manifest_path)
    execution = manifest['execution_plan']
    require(manifest['complete'] is True and manifest['failure'] is None
            and execution['plan_sha256'] == identities['plan_sha256']
            and execution['protocol_sha256'] == plan['protocol_sha256']
            and execution['source_identity'] == plan['source_identity'], 'Formal manifest plan/protocol identity changed')
    require(sha(manifest_path.parent / 'execution-plan.json') == manifest['execution_plan_sha256']
            and read(manifest_path.parent / 'execution-plan.json') == execution, 'Saved execution-plan bytes changed')
    require(summary['plan_sha256'] == identities['plan_sha256']
            and summary['evaluation_report_sha256'] == identities['evaluation_report_sha256']
            and summary['evaluation_manifest_sha256'] == evaluation['manifest_sha256'], 'Summary input hash binding changed')
    report = Path(args.report).resolve()
    require(report.is_relative_to(Path(config['reports_root'])) and not report.exists(),
            'Primary audit output must be a new file in the reports root')
    result = audit_summary(summary, execution, manifest['matches'], plan, deadline)
    for name, path in paths.items():
        require(sha(path) == identities[name + '_sha256'], 'Audit input changed during computation: ' + name)
    require(sha(manifest_path) == evaluation['manifest_sha256'], 'Formal manifest changed during audit')
    verify_sources(plan['sources'])
    verify_sources(receipt['sources'])
    deadline_check(deadline)
    result.update(**identities, evaluation_manifest_sha256=evaluation['manifest_sha256'],
                  source_sha256=sha(__file__), source_receipt_name=receipt['name'],
                  elapsed_seconds=time.monotonic() - started)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(complete=True, primary_contrasts=4, independent_blocks=BLOCKS,
                         bootstrap_replicates=REPLICATES,
                         maximum_primary_absolute_difference=result['maximum_primary_absolute_difference'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
