# Stage 2B measured results

Completed locally on the M5 Pro MacBook with 64 GB memory. This report covers the authorized bounded Stage 2B experiment; it does not launch a later stage.

**Final model:** `features_rank01`, primary seed `202610171`. **Retention outcome:** `inconclusive_budget`.

## What was built and tested

Added exact player-computable post-discard shanten, visible-availability ukeire and improving-type count, with an explicit applicability flag. These enter a learned candidate-scoring branch; teacher labels are not actor inputs and no heuristic filter removes legal choices. The history remains 128 events. The new branch adds 512 parameters to the 989,909-parameter raw model.

Old v1 checkpoint loading and tested choices were preserved. CPU/MPS forward and gradient checks, CPU/MPS interruption/resume, tiny-batch learning, matched shared weights and paired sampling passed. The complete implementation suite passed 233 tests and 165 subtests. These checks establish the tested implementation behavior, not independent certification of every Mahjong rule.

## Matched learning comparison

All required runs used **18,500 updates**, batch size 256, the same 800 original training matches, paired shared initial tensors and paired sample streams. The common ceiling was reduced from 20,000 before formal training using measured costs and the remaining budget. Both seeds used the same fixed validation selection sample.

| Recipe | Seed | Selected update | Full-validation discard agreement | Worse shanten | Lost ukeire at equal shanten | Optional-policy KL | Rank NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| features_rank01 | 202610171 | 18,500 | 100.0000% | 0.0000% | 0.0000% | 0.000105 | 1.8358 |
| features_rank01 | 202610172 | 18,000 | 100.0000% | 0.0000% | 0.0000% | 0.000142 | 1.8845 |
| raw_rank01 | 202610171 | 18,000 | 73.1390% | 8.1192% | 18.6064% | 0.828759 | 1.1893 |
| raw_rank01 | 202610172 | 18,500 | 64.2257% | 14.6344% | 20.9960% | 0.990783 | 1.4684 |
| features_rank0 | 202610171 | 5,500 | 100.0000% | 0.0000% | 0.0000% | 0.006810 | 5.4172 |
| features_rank0 | 202610172 | 6,000 | 100.0000% | 0.0000% | 0.0000% | 0.005769 | 7.6130 |

Discard/error/KL columns average the teacher-focal and historical-student-focal rates equally; rows within each population are decision weighted. Rank NLL is over all validation decisions. Decisions within a match are correlated. Uniform four-rank predictions have NLL 1.3863. A rank-weight-zero head was not trained and must not be treated as a learned critic.

Exact features make the teacher's discard preferences much easier to represent. This is a feature-assisted learned policy, not evidence that the raw network discovered equivalent hand reasoning. All compared models retain the original player-visible inputs and legal-action freedom.

The two feature-assisted rank-0.1 models have validation rank NLL 1.8358 and 1.8845, worse than uniform; their Brier scores are also worse than uniform. Their approximately 41% rank accuracy accompanies mean top confidence of 68–70%. These heads are overconfident and must not be reused as calibrated critics or reward-shaping estimates. The independent training audit records per-seed baselines and calibration.

Both rank-zero runs suffered a severe finite-value optimization collapse after their selected checkpoints at updates 5,500 and 6,000. Large losses and pre-clipping gradient norms are preserved in the run histories; passing finite-value checks did not establish stable optimization. Selection kept the earlier checkpoints under the original rule. The exact numerical mechanism remains unproven, and this outcome does not establish that a placement loss is universally necessary.

Best-set agreement allows any teacher-tied move. In the rank-0.1 feature models, exact deterministic teacher-discard agreement is only about 68–72%, despite 100% best-set agreement. Different tie choices can change later states and whole-match results. CPU/MPS numerical closeness also does not guarantee identical choices among near ties.

The selected checkpoint for each run maximized fixed-sample discard agreement, then minimized shanten errors, then policy KL. Recipe ranking used full validation averaged across both seeds. The final model is the predeclared primary seed of the winning recipe, frozen before test predictions or development outcomes.

**Conditional experiment:** `no_trigger`; trigger `None`. The observed student-minus-teacher discard-error gap was 0.0000 percentage points. Equal-stratum validation lost-ukeire rate was 0.0000%; fixed-training-sample rate was 0.0000%. These thresholds allocate an experiment; they are not causal or significance tests.

## Frozen action test

The new test set contains 15,727 focal decisions from 100 matches. Model performance on it was first computed after the final checkpoint was frozen; earlier integrity/alignment checks were allowed by the protocol. Discard agreement averaged equally across the two populations was **99.9826%**.

| Test population | Optional discard rows | Discard agreement | Worse shanten | Lost ukeire at equal shanten |
|---|---:|---:|---:|---:|
| teacher | 4,958 | 100.0000% | 0.0000% | 0.0000% |
| student | 5,754 | 99.9652% | 0.0348% | 0.0000% |

The two optional-discard errors both occur in historical-student states and worsen shanten by one. There were no illegal choices, equal-shanten lost-ukeire errors, red-only errors or wrong-action-kind errors on this action test. These are finite held-out observations, not guarantees for unseen play.

## Full-match evaluation

Four development campaigns compared the two highest-ranked eligible recipes, each with both seeds, against the fixed teacher/opponent panel. Each campaign used the same 30 paired seed blocks, rotating the focal seat through all four positions: 240 matches per campaign. The four campaigns share blocks and are not 120 independent blocks of evidence for the final model. Positive utility difference means a lower average numerical rank.

| Recipe | Seed | Candidate mean rank | Baseline mean rank | Paired utility difference |
|---|---:|---:|---:|---:|
| features_rank01 | 202610171 | 2.5083 | 2.5583 | +0.0500 |
| features_rank01 | 202610172 | 2.5417 | 2.5583 | +0.0167 |
| features_rank0 | 202610171 | 2.4583 | 2.5583 | +0.1000 |
| features_rank0 | 202610172 | 2.4583 | 2.5583 | +0.1000 |

The early rank-zero checkpoints have a larger favorable difference in this development sample. This does not change the final selection: the protocol froze the recipe and primary seed using validation before any of these match outcomes. The shared 30-block development sample does not establish one recipe's superior playing strength.


The final primary model's pilot produced planning status `inconclusive_budget` and required 319 fresh confirmation blocks under the fixed 0.10-rank margin, one-sided alpha 0.05, variance floor and power rule. The maximum allowed count was 256; confirmation was not run.

The planned confirmation method is a fixed-count normal approximation on independent paired block means. Any eventual confirmation would apply only to this synthetic panel and rule profile. The present evidence does not establish superiority, human rank, or competitiveness with Mortal, NAGA or world-leading systems.

## Data, integrity and coverage

The new initial corpus contains 300 matches, 2,858 hands and 48,088 focal decisions: 100 diagnostic, 100 validation and 100 test matches. Validation and test each have 50 teacher-focal and 50 historical-student-focal matches. These fresh groups did not enter base training.

All 900 corpus artifacts and every numeric row passed verification. Five full matches reproduced their actor, target and auditor artifacts. Exact feature/teacher alignment passed on 600,051 decisions and 5,138,892 candidates across the original training/validation and fresh corpus. This is exhaustive consistency with the pinned teacher calculation, not an independent shanten oracle.

No chi, pon or kan was executed in the new collected corpus. Offered calls are not evidence of post-call competence. Rare-event coverage is partial and several rare-rule counters remain uninstrumented. See the [collection observations](../../docs/stage2b/DATA_OBSERVATIONS.md) for exact counts and provenance.

## Local runtime and resource use

The learning ledger charges **2844.54 seconds (47.41 minutes)** against the 3,600-second ceiling. This comprises 2544.54 seconds of recorded durations and a separately labeled **300-second conservative allowance** for earlier numerical implementation checks/startup whose exact wall times were not all retained. The allowance is an estimate, not a measurement; exact exhaustive pre-campaign timing cannot be reconstructed. Formal preparation, checks and all six runs were metered. The ledger includes feature preparation/alignment, loading, optimization, validation, selection and action-test work, with failed attempts retained. The complete integration suite is charged at 29.63 seconds and recovered focused training tests at 8.49 seconds. Maximum sampled conservative memory across formal training runs was **3.60 GiB**, below 32 GiB. Collection and inference campaigns are separate.

The all-neural CPU worker sweep selected **14 workers**, one Torch thread each. The sustained trial used a 1,200-second submission window. Including startup within that window and the final drain, it took **1201.35 seconds**, completed **14,960 matches** and averaged **7492.0 audited decisions/second**. The entire sweep plus sustained run and drain took **1345.82 seconds**, below the 1,800-second ceiling. Peak sampled aggregate process RSS across the benchmark was **4.40 GiB**.

Complete roughly one-minute windows ranged from **7058.7 to 7697.8 decisions/second**. This variation is retained in the runtime report; no temperature data was collected to establish its cause.

| Policy phase | Share of cumulative action-choice time |
|---|---:|
| Exact hand-efficiency calculations | 48.05% |
| Neural forward, finite-value check and argmax | 30.87% |
| Other actor encoding | 8.49% |
| Batch collation and transfer | 3.94% |
| Semantic candidate construction | 3.01% |

These phase shares use summed worker action-choice time, not total benchmark wall time. Unassigned Python/counter overhead accounts for the remainder; simulator, audit and model-load/setup costs are additional. This identifies a profiling target, not a demonstrated speedup from an untested optimization.

These are inference-only, audited CPU measurements, including simulator and hashing costs. Completed games are credited to windows at completion; windows are not exact per-minute executed-work counts. Summed RSS conservatively counts shared pages. No temperature telemetry was measured, and no overnight reliability or RL training throughput is claimed. Benchmark trajectories do not enter the training corpus.

At the resource audit, new bulk artifacts and reports occupied **476.25 MB**, below the 5 GB limit. The final inventory checks subsequent review-file additions and its own size separately. The initial corpus stayed at 300 of the allowed 600 matches; six of the allowed eight full training runs were used.

## Decisions and recommended next review

Keep the exact efficiency branch and the frozen selected policy as the Stage 2B candidate. The matched comparisons support its imitation benefit on the measured populations. Keep the rank head only as a documented auxiliary from this experiment; it is unsuitable as a calibrated value estimate. Preserve both unstable ablations and their early selected checkpoints.

Teacher retention remains inconclusive because the prescribed pilot calculation requires 319 fresh blocks, above the approved 256-block cap. No confirmation was run. The smallest next study would be a separately authorized, fixed 319-block confirmation of this unchanged candidate, with the same 0.10 margin and no model tuning from its outcomes. This is a proposed budget amendment, not work performed here.

Before longer optimization, add activation/logit-scale diagnostics and tested instability stopping criteria, then investigate the collapse with matched controls. Before general Mahjong strength claims, broaden executed-call/kan trajectories and rules coverage. Revisit rank targets, held-out calibration and stage-conditioned baselines before any critic or reward-shaping use. Reinforcement learning should be a later bounded proposal with explicit acceptance evidence.

## Files for review and reproduction

- [Review guide](../../docs/stage2b/REVIEW_GUIDE.md) and [accepted plan](../../docs/stage2b/PLAN.md).
- [Frozen model card](../../docs/stage2b/MODEL_CARD.md).
- [Suggested next review](../../docs/stage2b/NEXT_REVIEW.md); proposed studies only, none started.
- Frozen training plan, learning checks, conditional decision.
- Independent training audit: paired controls, checkpoint selection, calibration and optimization failures.
- Final selection, action test, [match summary](evaluation-summary.json), [retention result](noninferiority-result.json).
- [Runtime summary](benchmark-summary.json), sustained runtime detail, final inventory.
- Resource and provenance audit, including all charged operations.
- Budget accounting note: recorded durations versus the explicit conservative allowance.
- Final model: selected.pt, SHA-256 `5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f`.
- Local budget ledger: budget-ledger.json.

The implementation was frozen at `a28a19c559542dfe32713943eb45fc9f6e1bf312`. The historical Stage 2A snapshot and its external data remain preserved. Verify with `scripts/verify_stage2a_snapshot.py` and `scripts/stage2b_manifest.py verify`. Reproduction commands refuse to overwrite completed evidence; a new campaign needs a separately identified destination and explicit budget.

No reinforcement learning, external datasets/checkpoints, external contact, paid compute, public play or overnight work was started. Next-stage work requires a separate reviewable proposal.
