# Package A diagnosis results

Completed 18 October 2026, Europe/London. **The smoothed models already miss their target on training positions. Numerical precision did not explain their measured errors on the fixed diagnostic states.** The next justified hypothesis is a controlled learning-rate reduction while keeping the model, data and smoothing target fixed.

This was an inference-only study: seven preserved checkpoints, a frozen 10,000-row training sample, all 16,024 validation rows, and 64 fixed positions for precision comparisons. No optimizer updates, new matches or test-split loading occurred. All seven attempts completed, using **29.920 of the allowed 600 numerical seconds**. The earlier teacher-retention non-pass remains unchanged; these measurements do not establish playing strength.

## Training fit and validation transfer

The training sample comes from the original 490,206 teacher-generated decisions. It has 9,185 optional decisions, including 6,694 optional teacher discards. Teacher validation has 7,293 optional decisions and 5,348 optional teacher discards; student validation has 7,525 and 5,573 respectively. Training-versus-teacher-validation is the closer distribution comparison. Student validation is a separate transfer population.

The following percentages measure probability assigned outside the teacher's best action set, averaged over optional decisions. The target is the actual cohort mean of the epsilon-0.01 target; it is slightly below 1% because some rows have no legal non-best action.

| Cohort | Target non-best mass | Smoothed seed 171 terminal | Smoothed seed 172 terminal |
|---|---:|---:|---:|
| Sampled teacher training | 0.9950% | 7.4481% | 7.2527% |
| Teacher validation | 0.9963% | 8.2154% | 7.9895% |
| Student validation | 0.9968% | 11.9247% | 11.3708% |

The mismatch is already substantial on training rows, so a validation-only generalization explanation is insufficient. Teacher validation adds a smaller gap; student-generated positions add a larger one. This establishes incomplete fit on the sampled training distribution, not whether the limiting cause is optimization, parameterization, or competing errors. Rows share matches; these are descriptive averages without independent-row significance claims.

Agreement with the teacher's best discard set provides a separate measure:

| Checkpoint | Training discard agreement | Teacher-validation agreement | Student-validation agreement |
|---|---:|---:|---:|
| Original rank-0.1 reference | 100.00% | 100.00% | 100.00% |
| Stopped control 171 | 96.97% | 97.05% | 95.51% |
| Stopped control 172 | 99.51% | 99.51% | 98.98% |
| Smoothed 171 selected at 3,000 | 98.37% | 98.52% | 96.34% |
| Smoothed 171 terminal at 18,500 | 98.16% | 97.48% | 94.22% |
| Smoothed 172 selected at 3,000 | 98.49% | 98.73% | 96.81% |
| Smoothed 172 terminal at 18,500 | 98.78% | 97.96% | 95.10% |

The original reference fits the original teacher target closely but is a different rank-loss recipe, not a matched smoothing control. Its near-zero non-best mass also means that it fits the deliberately softened target poorly: mean smoothed-target KL is about 10.22 on optional training rows. Its strong original-target fit must not be confused with suitability for the new probability objective.

The smoothed terminal checkpoints have the following mean KL divergences on optional rows against the smoothed target, using stable CPU float64 normalization of the same MPS float32 forward logits:

| Cohort | Seed 171 | Seed 172 |
|---|---:|---:|
| Sampled teacher training | 0.14683 | 0.14344 |
| Teacher validation | 0.16850 | 0.15979 |
| Student validation | 0.23838 | 0.22442 |

Both float32 training arithmetic and the separately normalized float64 calculations are retained in the raw reports. Float64 target copies are normalized to unit mass; original stored targets are unchanged. These KL values are row-weighted within each named cohort, not the equal-stratum average used by the older checkpoint-selection rule.

## Lower loss did not guarantee better decisions

From the selected to terminal smoothed checkpoints, aggregate validation non-best mass and smoothed-target KL decreased, but discard agreement declined and shanten errors increased:

| Measure on full validation | Seed 171 selected → terminal | Seed 172 selected → terminal |
|---|---:|---:|
| Optional-row non-best mass | 11.964% → 10.099% | 12.081% → 9.707% |
| Optional-row smoothed-target KL | 0.21925 → 0.20399 | 0.22614 → 0.19261 |
| Equal-stratum discard agreement | 97.431% → 95.849% | 97.767% → 96.532% |
| +1-shanten errors among 10,921 optional teacher discards | 76 → 253 | 53 → 155 |

This is an endpoint tension between distribution fit and the quality of the highest-scoring action. It is not a global divergence of every loss. Nor did every cohort improve: seed 171's student-validation smoothed KL rose slightly, from 0.23407 to 0.23838. A next recipe must retain decision agreement and error severity alongside loss, rather than promote a model simply because KL improves.

The approximately 10% aggregate non-best mass is **not 10% call probability**. Among the 3,465 call-available validation rows, terminal non-best mass is 1.342% / 1.574%, versus a 1% target. Actual probability on chi, pon and all kan types is 0.964% / 1.211%. On optional teacher-discard rows, non-best mass is much larger: 13.338% / 12.742%. These overlapping decision groups should not be added together. No executed-call or post-call competence was evaluated.

The candidate inputs already explicitly expose post-discard shanten and visible ukeire. Errors involving those quantities therefore do not establish that the information is missing from the actor's input. Their fitting and use remain questions for the model and optimizer.

## Numerical precision and safety indicators

All checkpoints were compared on the same 64 positions, including 60 optional decisions, using the unchanged canonical forward in CPU float32, MPS float32 and CPU float64. The latter widens the saved weights and inputs; it does not recreate a float64 training history.

| Checkpoints | CPU32 versus MPS32 action changes | CPU32 versus CPU64 action changes | MPS32 versus CPU64 action changes | Teacher-best membership changes |
|---|---:|---:|---:|---:|
| Original reference | 7 / 64 | 1 / 64 | 6 / 64 | 0 in every comparison |
| Two stopped controls and four smoothed checkpoints | 0 for each checkpoint | 0 for each checkpoint | 0 for each checkpoint | 0 |

Every changed reference action remained in the teacher's best set. Such tie-breaking variation can alter game trajectories, but this study does not establish a cause of the A1 strength result. The maximum smoothed CPU/MPS probability L1 difference per row was approximately **0.0000300**, far smaller than the measured probability-fit gap. Re-normalizing identical saved float32 logits in float64 changed probabilities by less than 1.6 × 10⁻⁷ L1 per row and never changed their raw-logit argmax.

Thus, the endpoint precision checks provide no evidence that inference rounding explains the observed smoothing errors on these states. They do not rule out numerical effects accumulated during training or untested positions. Higher-precision normalization cannot recover distinctions already lost in the forward computation.

The original reference is also a useful caution about raw scale:

| Mean on the fixed diagnostic positions | Original reference | Control 171 at stop | Control 172 at stop |
|---|---:|---:|---:|
| Absolute raw legal logit | 28,149.406 | 435.984 | 163.782 |
| Absolute common row offset | 28,593.433 | 428.971 | 162.204 |
| Absolute centered legal logit | 314.669 | 12.514 | 18.234 |
| Shared-context RMS | 0.9753 | 8.9209 | 2.6048 |

Row offsets and candidate statistics have different weighting, so their means are not an additive decomposition. The reference's large logits coexist with perfect best-set agreement on the measured training and validation cohorts. Absolute raw-logit magnitude alone cannot identify policy collapse across these checkpoints. This does not prove that the reference would trigger its own warm-up-relative guard: we do not have its matching instrumented trajectory. The current endpoints cannot estimate false-positive rates or predictive lead time, and no retrospective threshold is declared validated.

## Checks and resources

The focused inference-only suite passed **23 tests and 6 subtests**, covering cohort arithmetic, agreement/error semantics, padding, exact target mass, underflow, canonical dtype behavior, preservation of caller state/RNG, deadline cleanup and historical snapshot verification. The older integrated suite was not rerun because it includes optimizer-update tests outside this diagnosis's zero-update scope.

All seven model reports passed the final audit. It verified checkpoint/source/plan identities, row selections, recorded state-preservation assertions, cohort arithmetic and all **21 saved array files**. Its 29,790 checks are predominantly arithmetic checks across report cells, not independent statistical trials. The audit hashes arrays without independently rerunning inference; numerical correctness is supported by the focused tests and saved measurements.

| Resource | Measured use | Ceiling |
|---|---:|---:|
| Cumulative numerical elapsed time | 29.920 seconds | 600 seconds |
| Sampled process-tree RSS | 2.354 GiB | 32 GiB |
| Conservative worker memory including MPS driver allocation | 2.442 GiB | 32 GiB |
| New phase artifacts, approximately | 37 MB | 250 MB |
| Checkpoint attempts | 7, all complete | 7 |
| Optimizer updates / new matches / test-split loading | 0 / 0 / none | 0 / 0 / none |

The memory estimate can double-count shared memory, and peaks are sampled. Time includes tests, imports, dataset/model loading, preparation, all forwards, numerical summaries and serialization. Editing, document preparation and model-free archive checks are outside the numerical budget. No attempt failed or was retried. Unused time did not trigger more experiments.

The diagnostic implementation was committed as `42804c3` before the seven checkpoint evaluations. The frozen row plan records indices and source identities. The earlier Package A snapshot verified all 168 recorded files; its inventory and checkpoints remain preserved. New arrays and the live budget ledger are under `/Users/example/Library/Application Support/RiichiAI/package_a_diagnosis`, outside iCloud. The new inventory records their paths and hashes.

## Recommended next experiment

Test **one change: reduce the learning rate from 0.0003 to 0.0001 during continuation**, keeping epsilon 0.01 and all other ingredients fixed. Use each smoothed terminal checkpoint's own saved optimizer and RNG state, and compare the lower-rate arm with a same-rate continuation for the same seed and minibatch stream. This tests an optimization hypothesis; the present measurements do not establish learning rate as the cause.

The [next review proposal](../../docs/PACKAGE_A_DIAGNOSIS_REVIEW.md) specifies a bounded four-attempt comparison, guard treatment and acceptance evidence. No continuation has started. The original reference remains the preserved research model; neither smoothed model is promoted, and reinforcement learning remains unstarted.
