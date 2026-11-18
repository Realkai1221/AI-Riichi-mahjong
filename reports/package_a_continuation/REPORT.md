# Learning-rate continuation results and review

Completed 18 October 2026. **Reducing the learning rate to 0.0001 did not improve the tested continuation.** Against paired continuations at 0.0003, both seeds had higher smoothed-target KL on training and teacher validation, and lower full-validation discard agreement in both populations. All four runs completed without safety alarms. Neither the exploratory criterion nor the stronger durability criterion passed; no model is promoted.

This is a result about a particular 5,000-update continuation from the saved update-18,500 smoothed checkpoints. It does not prove that lower learning rates are generally ineffective, identify the mechanism of incomplete fitting, or establish playing strength. The earlier original rank-0.1 research reference and the fixed teacher-retention non-pass remain unchanged.

## What was run

The [accepted plan](../../docs/PACKAGE_A_CONTINUATION_PLAN.md) and [configuration](../../configs/package_a_continuation.json) fixed two seeds, two learning rates, and 5,000 additional updates per arm. Each seed's pair started with identical model weights, AdamW moments and step counters, random states, safety-monitor state and minibatch sequence. The only recipe intervention was learning rate. Every arm ended at update **23,500**.

The implementation was committed as `531a210` before formal checkpoint continuations. The frozen plan records source, checkpoint, data and row identities. Source warm-up references and safety counters were recovered from complete historical logs and independently replayed by the auditor. Neither guard thresholds nor warm-up were reset.

## Fit to the target distribution

KL below is averaged over **optional rows** and uses CPU float64 normalization of the saved MPS float32 forward logits. Lower is better. Training uses the same fixed 10,000-row sample as the diagnosis, including 9,185 optional rows. Validation has 7,293 teacher and 7,525 student optional rows. The starting and final measurements use the same indices.

| Seed | Population | Starting checkpoint | Same rate: 0.0003 | Lower rate: 0.0001 |
|---|---|---:|---:|---:|
| 171 | Teacher training sample | 0.14683 | **0.12888** | 0.13396 |
| 171 | Teacher validation | 0.16850 | **0.15597** | 0.15855 |
| 171 | Student validation | 0.23838 | **0.23473** | 0.23936 |
| 172 | Teacher training sample | 0.14344 | **0.13074** | 0.13466 |
| 172 | Teacher validation | 0.15979 | **0.15960** | 0.16369 |
| 172 | Student validation | **0.22442** | 0.22712 | 0.23464 |

Additional training improved sampled training fit at both rates, but the smaller step size did not outperform its matched control at this horizon. Seed 172 also illustrates why a training-loss improvement is insufficient: both continuations had worse student-validation KL than their starting point.

Probability outside the teacher's best action set remains much larger than the approximately 1% target:

| Seed | Population | Same rate | Lower rate |
|---|---|---:|---:|
| 171 | Teacher training sample | 6.484% | 6.914% |
| 171 | Teacher validation | 7.438% | 7.807% |
| 171 | Student validation | 11.281% | 11.722% |
| 172 | Teacher training sample | 6.437% | 6.918% |
| 172 | Teacher validation | 7.484% | 7.970% |
| 172 | Student validation | 10.736% | 11.425% |

These percentages are probability mass on optional decisions, not percentages of wrong actions or probabilities of making a call. They are descriptive averages; rows from the same match are correlated.

## Decision quality and acceptance

This table uses **optional teacher-discard rows**: 5,348 in teacher validation and 5,573 in student validation. Agreement means the chosen discard belongs to the teacher's best set. Shanten-worsening counts refer to decisions where the chosen discard increases shanten relative to the teacher; other kinds of errors are reported separately in the raw results.

| Seed / validation population | Starting agreement | Same-rate agreement | Lower-rate agreement | Starting / same / lower shanten-worsening counts |
|---|---:|---:|---:|---:|
| 171 / teacher | 97.476% | 97.569% | 97.532% | 83 / 67 / 64 |
| 171 / student | 94.222% | 94.402% | 94.079% | 170 / 158 / 148 |
| 172 / teacher | 97.962% | 97.457% | 97.345% | 46 / 70 / 76 |
| 172 / student | 95.101% | 95.030% | 94.437% | 109 / 91 / 131 |

Seed 171 shows a tradeoff: the lower rate reduces shanten-worsening counts relative to its control while making more best-set disagreement errors overall. Seed 172 shows no corresponding validation benefit. The predeclared exploratory criterion requires both KL improvement and preservation of the listed decision measures; it fails on both seeds. We did not replace those criteria with a post-hoc average score.

The stronger durability criterion uses the **original fixed validation subset and equal-stratum averaging**, a different denominator from the full-validation table:

| Arm | Additional updates | Terminal fixed agreement | Safety events | Durability pass |
|---|---:|---:|---:|---|
| Seed 171, same rate | 5,000 | 96.1920% | 0 | No |
| Seed 171, lower rate | 5,000 | 95.8898% | 0 | No |
| Seed 172, lower rate | 5,000 | 96.1019% | 0 | No |
| Seed 172, same rate | 5,000 | 96.2451% | 0 | No |

All runs stayed below the required 99.9%. Selected continuation checkpoints are saved separately at updates 19,500 / 22,000 / 21,000 / 22,500 in execution order. They do not replace the terminal evidence or determine acceptance. A guard not firing establishes only that the declared thresholds were not crossed in this continuation; it does not validate the guards as predictors of future collapse.

## Verification and resources

The main resume/regression suite passed **75 tests and 12 subtests**. A second suite for auditing and book handling passed **49 tests and 3 subtests**. Final count-convention and inventory-boundary checks passed **26 tests and 10 subtests**, giving **150 tests and 25 subtests across three targeted suites**. The older integrated suite was not rerun; these counts do not claim that every historical test ran today.

The CPU split/resume fixture matched uninterrupted weights, optimizer, random state and sample stream exactly. The actual MPS fixture used predeclared tolerances: model maximum absolute difference was 0, and the optimizer-moment maximum was approximately 1.46 × 10⁻¹¹. This is evidence for the tested fixture, not a guarantee of arbitrary cross-device bitwise reproducibility.

The final audit verifies all four attempts, their paired starting identities, 50 shared sample-stream prefixes per seed, independent safety replay, cohort arithmetic, eight endpoint array files and eight checkpoints. Its checks are bookkeeping and arithmetic checks, not independent statistical replications. It hashes model/array bytes without independently rerunning their numerical outputs.

| Resource | Measured | Ceiling |
|---|---:|---:|
| Cumulative numerical work, including tests, both book arithmetic probes and both OCR attempts | 487.465 seconds (8.12 minutes) | 900 seconds |
| Formal attempts | 4, all complete | 4 |
| Additional formal optimizer updates | 20,000 total | 4 × 5,000 |
| Highest retained sampled process-tree RSS | 2.540 GiB | 32 GiB |
| Training/endpoint conservative process plus accelerator estimate | 2.699 GiB | 32 GiB |
| Phase artifacts, approximately | 105 MB | 500 MB |
| New match campaigns / action-test access | None / none | None / none |

Memory peaks are sampled; the conservative measure can double-count unified memory. All twelve budget operations are closed: eleven completed and one OCR compilation failed because the default SDK was newer than the installed compiler. That failure consumed 8.797 seconds and remains recorded. The unchanged OCR script then completed against an already installed compatible SDK in 27.948 seconds, including compilation. Its peak process-tree RSS was 1.402 GiB. The failed compiler operation does not retain a peak-memory value; the table therefore reports retained measurements, not a measured peak for every process. There were no failed or replacement formal training attempts.

The ledger includes imports, tests, preparation, loading, training, validation, serialization, both book arithmetic probes and both OCR attempts. Source editing, PDF inspection, literature research, report writing and model-free hash checks are outside that numerical allowance. No additional model recipe was tried with the remaining time. The phase-artifact figure covers the controller's report and external model directories; the separately catalogued local book extracts add approximately 4 MB, excluding the user-supplied PDFs, which are inputs.

The prior diagnosis snapshot verification preserves its 172-file inventory against the original Git snapshot. New checkpoints and arrays are outside iCloud under `/Users/example/Library/Application Support/RiichiAI/package_a_continuation`. The current inventory records their identities.

## Book scenarios and the next review

The parallel book work adds an intake validator, strict answer/input separation, source grouping to prevent evaluation leakage, and a visually checked three-record pilot. All **five supplied PDFs (1,015 pages)** were inventoried and assessed for extraction, with representative pages visually inspected. Six additional question/explanation packets from the four Japanese books received independent transcription and meaning checks, making nine reviewed source examples in total. This is a small extraction demonstration, not a claim that every page or question has been transcribed. The PDFs, page renders and raw extraction records stay local under `Books/`; no book-derived model training or policy evaluation has run.

The [offline OCR proof](../../docs/BOOK_OCR.md) processed eight rendered pages. It recovered partly useful horizontal prose, recovered no substantive vertical explanation from the two Super Digital pages, and substituted unrelated symbols for tiles. It is suitable for assisted transcription, not automatic training-label production. The source catalog, local book integrity check and OCR summary preserve source identities and the observed limitations. The local nine-case review sheet shows the actual hands, answers, explanations and missing context in a readable format.

The conditional arithmetic report compares the project calculator with the independent `mahjong` calculator on all **31 distinct discard labels** in those three hands. They agree on shanten, improving types and unseen-copy counts. Counts assume only the supplied hand is known, with the discarded tile remaining visible; they are not known live-wall counts.

| Pilot source | Author's discard | Coarse best discard set | Shanten / improving-copy count |
|---|---|---|---|
| Seventh9, page 28, example 1 | 3m | 3m | 0 / 7 |
| Seventh9, page 43, example 1 | 6p | 6p | 1 / 16 |
| Seventh9, page 43, example 2 | 6s | 5m or 6s | 1 / 16 |

The third example shows a useful gap: immediate shanten/ukeire arithmetic ties two options, while the source prefers one for future wait quality. Agreement between calculators verifies the conditional arithmetic, not that strategic preference or a full-match optimum. The original source labels remain pending review.

The additional Uzaku count comparison checks all **12 printed candidate counts across four questions** with both calculators. Every printed count matches under the book's stated convention. Counting the original fourteen hand tiles plus the shown dora indicator gives the same results for these twelve candidates; this sample therefore does not demonstrate a numerical difference between conventions. Separate authored fixtures verify cases where counting the discarded copy and indicator does change the result. The actual book result verifies conditional arithmetic, without establishing the recommended discard's strategic superiority.

Read the all-book extraction assessment, [scenario intake guide](../../docs/BOOK_SCENARIOS.md), and [Suphx assessment](../../docs/SUPHX_ASSESSMENT.md) alongside the [next-stage review](../../docs/CONTINUATION_REVIEW.md). The failed smoothing criterion does not make perfect heuristic imitation a universal prerequisite for RL. It does mean this smaller-learning-rate intervention has not earned promotion. Keep the better earlier reference while choosing the next controlled step.
