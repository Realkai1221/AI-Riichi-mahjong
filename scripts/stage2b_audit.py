"""Summarize completed Stage 2B resource and provenance evidence; no model work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/stage2b'
IMPLEMENTATION = 'a28a19c559542dfe32713943eb45fc9f6e1bf312'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = read(ROOT / 'configs/stage2b.json')
    limits = plan['limits']
    data = Path(plan['data_root'])
    ledger = read(data / 'budget-ledger.json')
    if any(row['status'] == 'active' for row in ledger['operations']):
        raise RuntimeError('A metered operation is still active')
    runs = [read(path) for path in sorted(REPORTS.glob('run-*.json'))]
    selection = read(REPORTS / 'selection.json')
    test = read(REPORTS / 'action-test.json')
    benchmark = read(REPORTS / 'benchmark-summary.json')
    sustained = read(REPORTS / 'benchmark-sustained.json')
    evaluation = read(REPORTS / 'evaluation-summary.json')
    development = [read(row['path']) for row in evaluation['development']]
    corpus = read(data / 'dataset/manifest.json')
    corpus_count = corpus['completed_matches']
    if (data / 'aggregation/manifest.json').exists():
        corpus_count += read(data / 'aggregation/manifest.json')['completed_matches']
    training_seconds = sum(row['elapsed_seconds'] for row in ledger['operations'])
    memory = {
        'collection_process_tree_rss_bytes': corpus['sampled_peak_process_tree_rss_bytes'],
        'training_conservative_bytes': max(row['training']['sampled_peak_conservative_memory_bytes'] for row in runs),
        'development_process_tree_rss_bytes': max(row['resource_limits']['sampled_peaks']['process_tree_rss_bytes'] for row in development),
        'benchmark_process_tree_rss_bytes': benchmark['peak_process_tree_rss'],
    }
    artifact_bytes = sum(path.stat().st_size for base in (REPORTS, data) for path in base.rglob('*') if path.is_file())
    source_checks = []
    for path in sorted((ROOT / 'src/riichi_ai').glob('*.py')):
        relative = str(path.relative_to(ROOT))
        original = subprocess.check_output(['git', 'show', f'{IMPLEMENTATION}:{relative}'], cwd=ROOT)
        source_checks.append({'path': relative, 'matches_frozen_implementation': original == path.read_bytes()})
    safeguards = {
        'all_six_base_runs_complete': len(runs) >= 6 and all(row['eligible'] for row in runs),
        'training_runs_within_limit': len(runs) <= limits['maximum_full_training_runs'],
        'training_seconds_within_limit': training_seconds <= limits['total_training_related_seconds'],
        'corpus_matches_within_limit': corpus_count <= limits['maximum_new_corpus_matches'],
        'memory_samples_within_limit': max(memory.values()) <= limits['memory_gib'] * 1024**3,
        'artifact_bytes_within_limit_at_audit': artifact_bytes < limits['new_artifact_bytes'],
        'four_complete_development_campaigns': len(development) == 4 and all(row['valid'] and row['completed_games'] == 240 for row in development),
        'action_test_complete': test['complete'],
        'benchmark_valid_and_within_limit': benchmark['valid'] and benchmark['within_budget'],
        'sustained_full_requested_duration': sustained['valid'] and sustained['requested_duration_seconds'] == 1200 and sustained['elapsed_seconds'] >= 1200,
        'model_sources_unchanged': all(row['matches_frozen_implementation'] for row in source_checks),
        'final_checkpoint_unchanged': digest(selection['checkpoint']) == selection['checkpoint_sha256'],
        'development_seed_overlap_zero': read(REPORTS / 'development-plan.json')['seed_overlap_count'] == 0,
    }
    report = {
        'schema': 'riichi-ai-stage2b-resource-provenance-audit-v1',
        'valid': all(safeguards.values()), 'safeguards': safeguards,
        'implementation_commit': IMPLEMENTATION, 'model_source_checks': source_checks,
        'training_related_seconds': training_seconds, 'training_limit_seconds': 3600,
        'conservative_allowance_seconds': sum(row['elapsed_seconds'] for row in ledger['operations'] if row.get('conservative_allowance')),
        'training_operations': ledger['operations'], 'full_training_runs': len(runs),
        'new_corpus_matches': corpus_count,
        'collection_and_verification_seconds': read(REPORTS / 'collect-resources.json')['elapsed_seconds'],
        'development_elapsed_seconds': evaluation['elapsed_seconds'],
        'development_match_executions': sum(row['completed_games'] for row in development),
        'confirmation_status': evaluation['noninferiority']['status'],
        'confirmation_run': evaluation['noninferiority']['confirmation_run'],
        'benchmark_elapsed_seconds': benchmark['elapsed_seconds'],
        'sampled_memory_peaks': memory, 'artifact_bytes_at_audit': artifact_bytes,
        'artifact_size_scope': 'New bulk artifacts plus reports at this audit; final inventory includes subsequent review files and its own size check.',
        'checkpoint_sha256': selection['checkpoint_sha256'],
        'feature_alignment_rows': sum(read(REPORTS / name)['checked_rows'] for name in ('derived-stage2a-manifest.json', 'derived-stage2b-manifest.json')),
        'feature_alignment_candidates': sum(read(REPORTS / name)['checked_candidates'] for name in ('derived-stage2a-manifest.json', 'derived-stage2b-manifest.json')),
        'external_data_weights_or_paid_compute': False,
        'limitations': 'Sampled cooperative resource limits; summed RSS may double-count shared pages. Numerical implementation-test charges include explicit conservative allowances where per-invocation timings were not retained. Wall-clock benchmark time is separate from the learning budget. This is provenance and resource evidence, not a strength certificate.',
    }
    target = REPORTS / 'resource-and-provenance-audit.json'
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'valid': report['valid'], 'training_related_seconds': training_seconds,
                      'artifact_bytes_at_audit': artifact_bytes, 'memory': memory, 'path': str(target)}))
    if not report['valid']:
        raise RuntimeError('One or more completion safeguards failed')


if __name__ == '__main__':
    main()
