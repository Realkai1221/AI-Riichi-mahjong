# Package A results and next research decision

Experiments completed 17 October 2026; review finalized 18 October 2026, Europe/London. Hardware: M5 Pro with 64 GB memory. **The authorized experiments are complete, but neither scientific acceptance criterion was met.** The unchanged model did not demonstrate teacher-strength retention within the declared margin. Both smoothed training runs completed 18,500 updates without alarms, but their terminal imitation accuracy fell below the required 99.9%.

Keep the original Stage 2B model as a research reference and the new models as experimental artifacts. The next useful step is a small diagnosis of training fit, validation errors, and numerical scale before selecting another training intervention. The [review guide](../../docs/PACKAGE_A_REVIEW_GUIDE.md) gives a concrete proposed scope. No reinforcement learning, new training corpus, external acquisition, public play, or paid compute was undertaken.

## Retention confirmation

The frozen candidate completed **319 independent paired blocks and 2,552 match executions**, with zero failures or truncations. Each block contains the candidate and teacher at all four seats against the same frozen heuristic panel. Uncertainty is calculated across the 319 blocks.

| Measure | Result |
|---|---:|
| Candidate mean rank | 2.56505 |
| Teacher mean rank | 2.52665 |
| Candidate minus teacher utility | −0.03840 |
| One-sided 95% lower confidence bound | −0.10988 |
| Required lower bound | Greater than −0.10 |
| One-sided p-value | 0.07818 |
| Decision | Retention not demonstrated |

Utility remains `2.5 − rank`, so the point estimate corresponds to the candidate finishing 0.0384 places worse on average. This result does **not** establish inferiority or superiority. The realized block standard deviation was 0.77620, above the planning value of 0.71732. The earlier development estimate of +0.05 and this fresh estimate of −0.0384 come from independent samples of the same frozen model; their difference is not evidence that the model changed.

The study used its original normal-approximation criterion. It stopped at the fixed count, with no pilot pooling, candidate switching, or outcome-dependent extension. All 319 wall seeds were fixed before outcomes and checked against 17,513 earlier recorded seeds, with no overlap. See the A1 report, frozen plan, and [primary result](a1/noninferiority-result.json).

## Training stability and imitation

Four formal attempts used the same original 800-match training corpus, paired initialization and sampling, batch size 256, AdamW learning rate 0.0003, and 18,500-update horizon. The rank head was frozen and rank-loss weight was zero. The intervention allocated 1% of the training target to legal non-best actions when available, preserving the original labels for diagnostics. Forced decisions were excluded from policy loss and padding received no target mass.

“Agreement” below means selecting an action in the teacher's best discard set, averaged equally across teacher-generated and student-generated validation strata. The fixed subset contains 5,000 rows per stratum; full validation contains 16,024 rows, including 10,921 optional teacher-discard decisions. These are action-imitation measurements, not match-strength results.

| Recipe and seed | Final update | Stop reason | Fixed agreement | Full agreement | Meets durability criterion |
|---|---:|---|---:|---:|---|
| Control 202610171 | 1,400 | Raw-logit safety alarm | 96.2376% | 96.2799% | No |
| Control 202610172 | 1,900 | Raw-logit safety alarm | 99.2493% | 99.2455% | No |
| Smoothing 202610171 | 18,500 | Horizon completed, no alarms | 95.9616% | 95.8489% | No |
| Smoothing 202610172 | 18,500 | Horizon completed, no alarms | 96.6094% | 96.5316% | No |

Both smoothed seeds met the horizon/no-alarm part of the requirement and failed its **terminal agreement ≥99.9%** part. The four-attempt allocation is exhausted even though compute time remains. No replacement attempts or second intervention ran.

The selected checkpoints and terminal checkpoints are saved separately:

| Recipe and seed | Selected update | Selected full agreement | Terminal full agreement |
|---|---:|---:|---:|
| Control 202610171 | 1,000 | 91.4541% | 96.2799% |
| Control 202610172 | 1,500 | 98.6432% | 99.2455% |
| Smoothing 202610171 | 3,000 | 97.4312% | 95.8489% |
| Smoothing 202610172 | 3,000 | 97.7673% | 96.5316% |

Selection used scheduled fixed-validation checkpoints. The controls stopped between those scheduled checkpoints, explaining why their terminal observations can outperform the saved selected checkpoint. An earlier selected model cannot substitute for terminal durability.

The late fixed-validation trajectory remained below target:

| Update | Smoothing 202610171 | Smoothing 202610172 |
|---|---:|---:|
| 15,000 | 96.9452% | 96.0586% |
| 15,500 | 96.9299% | 96.2727% |
| 16,000 | 96.1930% | 96.7792% |
| 16,500 | 96.1807% | 97.1853% |
| 17,000 | 96.4288% | 96.6808% |
| 17,500 | 96.5070% | 96.8255% |
| 18,000 | 97.1159% | 96.5326% |
| 18,500 | 95.9616% | 96.6094% |

Terminal full-validation unsmoothed policy KL was **0.15932 / 0.14775** for the two smoothed seeds. Equal-stratum shanten-worsening rates were **2.3012% / 1.4080%**; there were 253 / 155 observed +1-shanten errors. Agreement was **97.4757% / 97.9619%** in teacher states and **94.2221% / 95.1014%** in student states. All four terminal evaluations selected zero illegal actions. The error concentration in student states is a reason to examine generalization, not proof of its cause.

## What the alarms establish

The guards were frozen before formal training: diagnostics every 100 updates, a 1,000-update warm-up, median warm-up references, tenfold scale thresholds, and three consecutive exceedances. The raw-logit threshold also had an absolute floor of 100. Sustained large gradients/losses and nonfinite values had separate backstops. The formal plan records the precise configuration and source identities.

Both controls stopped because the mean absolute raw legal logit crossed its threshold. Their final minibatch policy CE was **0.5925 / 0.4684**, and gradient norm was **1.5371 / 1.3573**. Neither had reached the severe policy-loss collapse observed in historical continuations. They are safety-stopped observations, not reproductions of observed collapse.

| Final fixed diagnostic mean | Control 171 | Control 172 | Smoothing 171 | Smoothing 172 |
|---|---:|---:|---:|---:|
| Absolute raw legal logit | 435.984 | 163.782 | 27.817 | 43.849 |
| Absolute common row offset | 428.971 | 162.204 | 24.606 | 42.690 |
| Absolute centered legal logit | 12.514 | 18.234 | 6.629 | 6.728 |
| Shared-context RMS | 8.921 | 2.605 | 0.859 | 0.691 |

These row/candidate statistics use different weighting and should not be converted into a formal percentage attributable to offsets. They show why raw magnitude needs careful interpretation: adding the same constant to every legal logit leaves softmax probabilities unchanged in exact arithmetic. Large offsets can matter to floating-point resolution, but these records do not establish such a failure at either control's stopping point.

The smoothed runs had smaller monitored scales and survived the full horizon under the fixed guards. That is useful evidence for this recipe on these two seeds. It does not calibrate the guards, identify the original collapse mechanism, or establish unrestricted long-run stability. Earlier alarm times than historical collapse do not by themselves validate predictive lead time. No thresholds were changed after seeing a control.

## Probability and exploration measurements

These probability summaries use **14,818 optional validation rows**. They have a different denominator from the discard-agreement metric above. Forced decisions are reported separately in the raw files. Call probability includes chi, pon, and all kan types; kan probability is a subset, not additional disjoint mass. The original reference is the frozen Stage 2B rank-0.1 model; it is not a matched rank-zero intervention control.

| Metric | Original reference | Smoothing 171 terminal | Smoothing 172 terminal |
|---|---:|---:|---:|
| Mean entropy, nats | 0.36275 | 0.63107 | 0.62907 |
| Mean entropy divided by log legal-choice count | 0.14994 | 0.27298 | 0.27435 |
| Mean probability on teacher-best actions | 99.99099% | 89.90091% | 90.29337% |
| Mean probability outside teacher best set | 0.00901% | 10.09909% | 9.70663% |
| Mean call probability when available, 3,465 rows | 0 measured | 0.96438% | 1.21118% |
| Mean kan probability when available, 156 rows | 0 measured | 0.19852% | 0.25321% |

“0 measured” means zero in the evaluated float32 softmax, not mathematically impossible under finite logits. The reference's residual entropy can include choices among tied best actions; it does not demonstrate call exploration.

The intended smoothed target assigns at most 1% total probability outside the best set on an eligible row. The roughly 10% observed on optional validation rows is therefore a **fit gap**, not an inevitable 1% concession. We have not yet separated inadequate training fit from generalization or optimization effects. More probability on calls also does not establish useful call behavior, executed-call coverage, or post-call competence. No action test was reused for tuning, and no full-match evaluation of these smoothed candidates was performed.

## Validation and provenance

The final integrated suite passed **251 tests and 165 subtests**. The final audit passed **116 protocol, identity, result and accounting checks**. It independently recalculated A1's block statistics, verified both paired sampler prefixes through the stopped controls, checked both full-horizon sample streams against their historical counterparts, and confirmed original checkpoint/configuration/source identities. All rank-head parameters remained frozen in the recorded runtime checks.

The instrumentation checks covered masking, separate smoothing targets, forced decisions, hook behavior, RNG isolation, CPU/MPS comparison, matched initialization/sampling, persisted stopping reports, and budget leases. An initial strict bitwise MPS gradient assertion failed. A plain-versus-plain repeat showed the same maximum difference as hook-versus-plain: **2.3283 × 10⁻¹⁰**. The revised check used the already established GPU tolerances (absolute 5 × 10⁻⁵, relative 5 × 10⁻⁴), with exact CPU comparison and a repeatability control. Both the failed and successful records remain preserved and charged. This is tolerance-based GPU validation, not a claim of bitwise GPU reproducibility or full-trajectory equivalence near historical collapse.

The original Stage 2A snapshot verified **3,174 files**; Stage 2B verified **2,284 files**. Their historical inventories were not regenerated. The new Package A inventory records current project sources, documentation, reports and external A2 artifacts, excluding itself and documented generated caches/verification outputs. Checksums establish file identity, not scientific success or full engine conformance.

The implementation ran with Python 3.14.8, PyTorch 2.14.1, NumPy 2.5.4 and RiichiEnv 0.4.10. A2 source/controller hashes were frozen by the probe and formal plan before all four attempts. They identify the actual working files; the pre-existing Git HEAD alone did not identify the then-uncommitted implementation.

## Resource use and failures

| Resource | A1 confirmation | A2 training and numerical checks |
|---|---:|---:|
| Charged elapsed time | 297.445 s / 4.957 min | 875.442 s / 14.591 min |
| Approved elapsed ceiling | 600 s / 10 min | 2,700 s / 45 min |
| Recorded memory peak | 1.525 GiB process-tree RSS | 3.639 GiB conservative worker estimate |
| Memory ceiling | 32 GiB | 32 GiB |
| Artifacts at final audit, approximately | 19.69 MB | 116.41 MB |
| Artifact ceiling | 1 GB | 2 GB |

A2's conservative memory figure combines worker RSS and MPS driver allocation and can double-count shared memory; its sampled process-tree peak across saved attempt markers was 2.503 GiB. Peaks are sampled, not operating-system hard caps. The generic numerical-test watchdog enforced the same memory ceiling but did not persist its sampled peak. Artifact figures describe the final audit snapshot, with small shared review/inventory files outside the two phase directories. Everything remains well below the allocated ceilings.

A1 includes a conservative five-second allowance for the initial sandbox process-enumeration failure. Its child was terminated, no saved match outcomes were produced or inspected, and the retry retained the same model and fixed seeds. A2 includes its initial failed assertion, revised checks, baseline probabilities, probe, all four formal attempts, loading/validation/finalization, and integrated tests. The copied budget ledger retains every operation; no leases remain active. Source editing, document preparation and model-free archive checks are outside the numerical budget. Unused time was not converted into extra attempts.

## Files and next decision

- [Review guide and proposed diagnosis](../../docs/PACKAGE_A_REVIEW_GUIDE.md)
- [Accepted Package A plan](../../docs/PACKAGE_A_PLAN.md) and [configuration](../../configs/package_a.json)
- A1 report, [primary result](a1/noninferiority-result.json), and all match outcomes
- Control 171, control 172, smoothing 171, and smoothing 172
- Original reference probabilities, integrated test results, audit, and inventory

Selected and terminal checkpoints, optimizer/RNG state, and live accounting are stored under `/Users/example/Library/Application Support/RiichiAI/package_a/a2`, outside iCloud. Their paths and hashes are included in the records and inventory.

I recommend the review guide's **600-second diagnosis-only follow-up**: fixed training/validation inference and numerical comparisons, with no optimizer updates or new matches. It should identify one justified next training intervention. The present authorization ends with these results and review files; Stage 3 remains unstarted.
