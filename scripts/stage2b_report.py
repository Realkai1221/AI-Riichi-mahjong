"""Build a reviewable Stage 2B results document from completed measured evidence."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/stage2b'


def read(name):
    return json.loads((REPORTS / name).read_text())


def pct(value):
    return f'{100 * value:.4f}%'


def signed(value):
    return 'unavailable' if value is None else f'{value:+.4f}'


def main():
    plan = json.loads((ROOT / 'configs/stage2b.json').read_text())
    data = Path(plan['data_root'])
    ledger = json.loads((data / 'budget-ledger.json').read_text())
    if any(item['status'] == 'active' for item in ledger['operations']):
        raise RuntimeError('Experiment work is still active; do not publish a completion report')
    runs = [json.loads(path.read_text()) for path in sorted(REPORTS.glob('run-*.json'))]
    if len(runs) < 6 or not all(run['eligible'] for run in runs):
        raise RuntimeError('A required matched comparison is incomplete')
    selection, test, evaluation, benchmark = (read(name) for name in
        ('selection.json', 'action-test.json', 'evaluation-summary.json', 'benchmark-summary.json'))
    sustained = read('benchmark-sustained.json')
    inference_times = sustained['summed_worker_inference_timing_seconds']
    full_windows = [row for row in sustained['windows'] if not row.get('partial_window')]
    if not test['complete'] or not benchmark['valid']:
        raise RuntimeError('Test or sustained runtime work is incomplete')
    checks, contingent = read('learning-checks.json'), read('contingent-plan.json')
    resource_audit = read('resource-and-provenance-audit.json')
    collection, audit = read('dataset-manifest.json'), read('dataset-audit.json')
    derivations = [read(name) for name in ('derived-stage2a-manifest.json', 'derived-stage2b-manifest.json')]
    ni = evaluation['noninferiority']
    status = ni['status']
    total = sum(item['elapsed_seconds'] for item in ledger['operations'])
    allowance = sum(item['elapsed_seconds'] for item in ledger['operations'] if item.get('conservative_allowance'))
    peak = max(run['training']['sampled_peak_conservative_memory_bytes'] for run in runs)
    lines = [
        '# Stage 2B measured results', '',
        'Completed locally on the M5 Pro MacBook with 64 GB memory. This report covers the authorized bounded Stage 2B experiment; it does not launch a later stage.', '',
        f"**Final model:** `{selection['recipe']}`, primary seed `{selection['seed']}`. **Retention outcome:** `{status}`.", '',
        '## What was built and tested', '',
        'Added exact player-computable post-discard shanten, visible-availability ukeire and improving-type count, with an explicit applicability flag. These enter a learned candidate-scoring branch; teacher labels are not actor inputs and no heuristic filter removes legal choices. The history remains 128 events. The new branch adds 512 parameters to the 989,909-parameter raw model.', '',
        'Old v1 checkpoint loading and tested choices were preserved. CPU/MPS forward and gradient checks, CPU/MPS interruption/resume, tiny-batch learning, matched shared weights and paired sampling passed. The complete implementation suite passed 233 tests and 165 subtests. These checks establish the tested implementation behavior, not independent certification of every Mahjong rule.', '',
        '## Matched learning comparison', '',
        f"All required runs used **{selection['common_steps']:,} updates**, batch size 256, the same 800 original training matches, paired shared initial tensors and paired sample streams. The common ceiling was reduced from 20,000 before formal training using measured costs and the remaining budget. Both seeds used the same fixed validation selection sample.", '',
        '| Recipe | Seed | Selected update | Full-validation discard agreement | Worse shanten | Lost ukeire at equal shanten | Optional-policy KL | Rank NLL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    order = {row['name']: i for i, row in enumerate(plan['training']['recipes_in_execution_order'])}
    runs.sort(key=lambda run: (order.get(run['recipe'], 99), run['seed']))
    for run in runs:
        m = run['full_validation']['equal_stratum_means']
        lines.append(f"| {run['recipe']} | {run['seed']} | {run['training']['best_step']:,} | {pct(m['discard_tie_agreement'])} | {pct(m['shanten_worsening_rate'])} | {pct(m['same_shanten_lost_ukeire_rate'])} | {m['policy_kl']:.6f} | {run['full_validation']['rank_nll']:.4f} |")
    lines += ['', 'Discard/error/KL columns average the teacher-focal and historical-student-focal rates equally; rows within each population are decision weighted. Rank NLL is over all validation decisions. Decisions within a match are correlated. Uniform four-rank predictions have NLL 1.3863. A rank-weight-zero head was not trained and must not be treated as a learned critic.', '',
        'Exact features make the teacher\'s discard preferences much easier to represent. This is a feature-assisted learned policy, not evidence that the raw network discovered equivalent hand reasoning. All compared models retain the original player-visible inputs and legal-action freedom.', '',
        'The two feature-assisted rank-0.1 models have validation rank NLL 1.8358 and 1.8845, worse than uniform; their Brier scores are also worse than uniform. Their approximately 41% rank accuracy accompanies mean top confidence of 68–70%. These heads are overconfident and must not be reused as calibrated critics or reward-shaping estimates. The independent training audit records per-seed baselines and calibration.', '',
        'Both rank-zero runs suffered a severe finite-value optimization collapse after their selected checkpoints at updates 5,500 and 6,000. Large losses and pre-clipping gradient norms are preserved in the run histories; passing finite-value checks did not establish stable optimization. Selection kept the earlier checkpoints under the original rule. The exact numerical mechanism remains unproven, and this outcome does not establish that a placement loss is universally necessary.', '',
        'Best-set agreement allows any teacher-tied move. In the rank-0.1 feature models, exact deterministic teacher-discard agreement is only about 68–72%, despite 100% best-set agreement. Different tie choices can change later states and whole-match results. CPU/MPS numerical closeness also does not guarantee identical choices among near ties.', '',
        'The selected checkpoint for each run maximized fixed-sample discard agreement, then minimized shanten errors, then policy KL. Recipe ranking used full validation averaged across both seeds. The final model is the predeclared primary seed of the winning recipe, frozen before test predictions or development outcomes.', '',
        f"**Conditional experiment:** `{contingent['status']}`; trigger `{contingent['trigger']}`. The observed student-minus-teacher discard-error gap was {100*contingent['student_minus_teacher_discard_error']:.4f} percentage points. Equal-stratum validation lost-ukeire rate was {pct(contingent['validation_lost_ukeire_rate'])}; fixed-training-sample rate was {pct(contingent['frozen_train_lost_ukeire_rate'])}. These thresholds allocate an experiment; they are not causal or significance tests.", '',
        '## Frozen action test', '',
        f"The new test set contains {test['metrics']['rows']:,} focal decisions from 100 matches. Model performance on it was first computed after the final checkpoint was frozen; earlier integrity/alignment checks were allowed by the protocol. Discard agreement averaged equally across the two populations was **{pct(test['metrics']['equal_stratum_means']['discard_tie_agreement'])}**.", '',
        '| Test population | Optional discard rows | Discard agreement | Worse shanten | Lost ukeire at equal shanten |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, row in test['metrics']['strata'].items():
        lines.append(f"| {name} | {row['optional_discard_rows']:,} | {pct(row['discard_tie_agreement'])} | {pct(row['shanten_worsening_rate'])} | {pct(row['same_shanten_lost_ukeire_rate'])} |")
    lines += ['', 'The two optional-discard errors both occur in historical-student states and worsen shanten by one. There were no illegal choices, equal-shanten lost-ukeire errors, red-only errors or wrong-action-kind errors on this action test. These are finite held-out observations, not guarantees for unseen play.', '', '## Full-match evaluation', '',
        'Four development campaigns compared the two highest-ranked eligible recipes, each with both seeds, against the fixed teacher/opponent panel. Each campaign used the same 30 paired seed blocks, rotating the focal seat through all four positions: 240 matches per campaign. The four campaigns share blocks and are not 120 independent blocks of evidence for the final model. Positive utility difference means a lower average numerical rank.', '',
        '| Recipe | Seed | Candidate mean rank | Baseline mean rank | Paired utility difference |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in evaluation['development']:
        lines.append(f"| {row['recipe']} | {row['seed']} | {row['arm_summaries']['candidate']['mean_focal_rank']:.4f} | {row['arm_summaries']['baseline']['mean_focal_rank']:.4f} | {row['statistics']['mean_utility_difference']:+.4f} |")
    lines += ['', 'The early rank-zero checkpoints have a larger favorable difference in this development sample. This does not change the final selection: the protocol froze the recipe and primary seed using validation before any of these match outcomes. The shared 30-block development sample does not establish one recipe\'s superior playing strength.', '']
    planning = evaluation['confirmation_plan']
    lines += ['', f"The final primary model's pilot produced planning status `{planning['status']}` and required {planning['required_blocks']} fresh confirmation blocks under the fixed 0.10-rank margin, one-sided alpha 0.05, variance floor and power rule. The maximum allowed count was 256; confirmation was {'run' if ni['confirmation_run'] else 'not run'}.", '']
    if ni['confirmation_run']:
        lines += [f"Confirmation used {ni['independent_blocks']} independent blocks. Mean utility difference was **{signed(ni['mean_utility_difference'])}**; the one-sided 95% lower bound was **{signed(ni['one_sided_lower_bound'])}**. Result: **{status}**.", '']
    lines += ['The planned confirmation method is a fixed-count normal approximation on independent paired block means. Any eventual confirmation would apply only to this synthetic panel and rule profile. The present evidence does not establish superiority, human rank, or competitiveness with Mortal, NAGA or world-leading systems.', '',
        '## Data, integrity and coverage', '',
        f"The new initial corpus contains {collection['completed_matches']} matches, {collection['hands']:,} hands and {collection['decisions']:,} focal decisions: 100 diagnostic, 100 validation and 100 test matches. Validation and test each have 50 teacher-focal and 50 historical-student-focal matches. These fresh groups did not enter base training.", '',
        f"All {audit['checked_artifacts']} corpus artifacts and every numeric row passed verification. Five full matches reproduced their actor, target and auditor artifacts. Exact feature/teacher alignment passed on {sum(row['checked_rows'] for row in derivations):,} decisions and {sum(row['checked_candidates'] for row in derivations):,} candidates across the original training/validation and fresh corpus. This is exhaustive consistency with the pinned teacher calculation, not an independent shanten oracle.", '',
        'No chi, pon or kan was executed in the new collected corpus. Offered calls are not evidence of post-call competence. Rare-event coverage is partial and several rare-rule counters remain uninstrumented. See the [collection observations](../../docs/stage2b/DATA_OBSERVATIONS.md) for exact counts and provenance.', '',
        '## Local runtime and resource use', '',
        f"The learning ledger charges **{total:.2f} seconds ({total/60:.2f} minutes)** against the 3,600-second ceiling. This comprises {total-allowance:.2f} seconds of recorded durations and a separately labeled **{allowance:.0f}-second conservative allowance** for earlier numerical implementation checks/startup whose exact wall times were not all retained. The allowance is an estimate, not a measurement; exact exhaustive pre-campaign timing cannot be reconstructed. Formal preparation, checks and all six runs were metered. The ledger includes feature preparation/alignment, loading, optimization, validation, selection and action-test work, with failed attempts retained. The complete integration suite is charged at 29.63 seconds and recovered focused training tests at 8.49 seconds. Maximum sampled conservative memory across formal training runs was **{peak/1024**3:.2f} GiB**, below 32 GiB. Collection and inference campaigns are separate.", '',
        f"The all-neural CPU worker sweep selected **{benchmark['selected_workers']} workers**, one Torch thread each. The sustained trial used a 1,200-second submission window. Including startup within that window and the final drain, it took **{sustained['elapsed_seconds']:.2f} seconds**, completed **{benchmark['sustained_completed_matches']:,} matches** and averaged **{benchmark['sustained_decisions_per_second']:.1f} audited decisions/second**. The entire sweep plus sustained run and drain took **{benchmark['elapsed_seconds']:.2f} seconds**, below the 1,800-second ceiling. Peak sampled aggregate process RSS across the benchmark was **{benchmark['peak_process_tree_rss']/1024**3:.2f} GiB**.", '',
        f"Complete roughly one-minute windows ranged from **{min(row['decisions_per_second'] for row in full_windows):.1f} to {max(row['decisions_per_second'] for row in full_windows):.1f} decisions/second**. This variation is retained in the runtime report; no temperature data was collected to establish its cause.", '',
        '| Policy phase | Share of cumulative action-choice time |',
        '|---|---:|',
        *[f"| {label} | {100*inference_times[key]/inference_times['total_choose']:.2f}% |" for key, label in (
            ('efficiency', 'Exact hand-efficiency calculations'), ('model', 'Neural forward, finite-value check and argmax'),
            ('encoding', 'Other actor encoding'), ('collation_and_transfer', 'Batch collation and transfer'),
            ('semantic_actions', 'Semantic candidate construction'))], '',
        'These phase shares use summed worker action-choice time, not total benchmark wall time. Unassigned Python/counter overhead accounts for the remainder; simulator, audit and model-load/setup costs are additional. This identifies a profiling target, not a demonstrated speedup from an untested optimization.', '',
        'These are inference-only, audited CPU measurements, including simulator and hashing costs. Completed games are credited to windows at completion; windows are not exact per-minute executed-work counts. Summed RSS conservatively counts shared pages. No temperature telemetry was measured, and no overnight reliability or RL training throughput is claimed. Benchmark trajectories do not enter the training corpus.', '',
        f"At the resource audit, new bulk artifacts and reports occupied **{resource_audit['artifact_bytes_at_audit']/1e6:.2f} MB**, below the 5 GB limit. The final inventory checks subsequent review-file additions and its own size separately. The initial corpus stayed at 300 of the allowed 600 matches; six of the allowed eight full training runs were used.", '',
        '## Decisions and recommended next review', '',
        'Keep the exact efficiency branch and the frozen selected policy as the Stage 2B candidate. The matched comparisons support its imitation benefit on the measured populations. Keep the rank head only as a documented auxiliary from this experiment; it is unsuitable as a calibrated value estimate. Preserve both unstable ablations and their early selected checkpoints.', '',
        'Teacher retention remains inconclusive because the prescribed pilot calculation requires 319 fresh blocks, above the approved 256-block cap. No confirmation was run. The smallest next study would be a separately authorized, fixed 319-block confirmation of this unchanged candidate, with the same 0.10 margin and no model tuning from its outcomes. This is a proposed budget amendment, not work performed here.', '',
        'Before longer optimization, add activation/logit-scale diagnostics and tested instability stopping criteria, then investigate the collapse with matched controls. Before general Mahjong strength claims, broaden executed-call/kan trajectories and rules coverage. Revisit rank targets, held-out calibration and stage-conditioned baselines before any critic or reward-shaping use. Reinforcement learning should be a later bounded proposal with explicit acceptance evidence.', '',
        '## Files for review and reproduction', '',
        '- [Review guide](../../docs/stage2b/REVIEW_GUIDE.md) and [accepted plan](../../docs/stage2b/PLAN.md).',
        '- [Frozen model card](../../docs/stage2b/MODEL_CARD.md).',
        '- [Suggested next review](../../docs/stage2b/NEXT_REVIEW.md); proposed studies only, none started.',
        '- [Frozen training plan](training-plan.json), [learning checks](learning-checks.json), [conditional decision](contingent-plan.json).',
        '- [Independent training audit](TRAINING_AUDIT.md): paired controls, checkpoint selection, calibration and optimization failures.',
        '- [Final selection](selection.json), [action test](action-test.json), [match summary](evaluation-summary.json), [retention result](noninferiority-result.json).',
        '- [Runtime summary](benchmark-summary.json), [sustained runtime detail](benchmark-sustained.json), [final inventory](manifest.json).',
        '- [Resource and provenance audit](resource-and-provenance-audit.json), including all charged operations.',
        '- [Budget accounting note](BUDGET_ACCOUNTING.md): recorded durations versus the explicit conservative allowance.',
        f"- Final model: [{Path(selection['checkpoint']).name}](<{selection['checkpoint']}>), SHA-256 `{selection['checkpoint_sha256']}`.",
        f"- Local budget ledger: [budget-ledger.json](<{data / 'budget-ledger.json'}>).", '',
        'The implementation was frozen at `a28a19c559542dfe32713943eb45fc9f6e1bf312`. The historical Stage 2A snapshot and its external data remain preserved. Verify with `scripts/verify_stage2a_snapshot.py` and `scripts/stage2b_manifest.py verify`. Reproduction commands refuse to overwrite completed evidence; a new campaign needs a separately identified destination and explicit budget.', '',
        'No reinforcement learning, external datasets/checkpoints, external contact, paid compute, public play or overnight work was started. Next-stage work requires a separate reviewable proposal.', '',
    ]
    (REPORTS / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'report': str(REPORTS / 'REPORT.md'), 'training_related_seconds': total,
                      'within_training_budget': total <= plan['limits']['total_training_related_seconds']}))


if __name__ == '__main__':
    main()
