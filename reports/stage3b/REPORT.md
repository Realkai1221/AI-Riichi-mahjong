# Stage 3B results — 20 October 2026

**The diagnostic completed and identified why extending the current sampler is unlikely to provide useful call learning.** The original and final checkpoints assign every one of the 364 offered call/kan candidates in the fixed sample probability zero. Temperature scaling restores numerical support but leaves calls extremely rare; a 5% uniform mixture restores sampling support while leaving the direct gradient of every probed call recommendation exactly zero.

All **512 saved positions**, both checkpoints, four probability rules and **32 gradient/backend probe positions** were examined. There were **zero new matches, optimizer updates or checkpoint writes**. The closed ledger charged **21.858633 of 300 numerical seconds**, including tests, a failed first attempt, reporting recovery, a focused transfer fixture and the repaired diagnostic. The remaining allowance was not used to expand the experiment.

Start with the [review and proposed next experiment](../../docs/STAGE3B_REVIEW.md), book preparation, machine-readable summary and independent audit.

## Fixed sample and interpretation

The original plan selected positions before model outputs from Stage 3A's 9,750 optional learner requests. Selection used action availability and deterministic identity hashes, not scores, rewards or model confidence.

| Exclusive selection stratum | Positions |
|---|---:|
| Concealed kan | 53 |
| Open kan | 40 |
| Chi | 105 |
| Pon | 105 |
| Win | 105 |
| Discard/riichi | 104 |
| Total | 512 |

All available concealed/open-kan positions were included. Overlapping availability includes PASS in 339 rows, CHI in 106, PON in 148, RON in 89 and TSUMO in 16. No added-kan example was available. The sample deliberately overrepresents rare actions; its averages are not estimates of ordinary game frequencies.

The 32 fixed gradient targets comprise 6 concealed kans, 6 open kans, 5 chi, 5 pon, 5 ron, 4 discards and 1 riichi. Thus there are **22 direct call-target probes per checkpoint/device**. PASS and TSUMO have probability measurements and may receive indirect gradients, but no direct target probe. A counterfactual backward calculation does not mean that action was sampled or a learning update occurred.

## What the four rules show

The following values are for the original checkpoint. The final checkpoint gives essentially the same conclusion.

| Probability rule | Mean total call/kan probability when offered | Mean discard distribution change, total variation | Direct call targets with zero parameter gradient |
|---|---:|---:|---:|
| Unchanged | 0 | 0 | 0 / 22, but these actions cannot be sampled |
| Temperature 10 | 5.13 × 10⁻⁴⁷ | 0.1294 | 0 / 22 |
| Temperature 100 | 4.63 × 10⁻⁷ | 0.4559 | 0 / 22 |
| 5% legal-uniform mixture | 0.02343 | 0.04145 | 22 / 22 |

Total variation is half the sum of absolute changes in candidate probabilities. It describes distributional change, not a measured loss in playing strength.

**The legal mask is not suppressing the saved calls.** Every saved candidate was marked legal. Call logits instead lagged the best legal candidate by **1,016–6,803** in the initial model and up to **6,805** in the final model. Their finite log-probabilities exponentiate to zero even in CPU float64. A common offset in all logits is not the cause; the relevant quantity is the gap.

The original [teacher's priority rule](../../src/riichi_ai/semantics.py) puts wins first, then permitted riichi, discard, pass and finally unsupported actions including calls/kans. Thus the imitation targets explicitly preferred pass or discard whenever those alternatives existed. This establishes an inherited target bias and is consistent with the measured gaps; it does not prove a complete causal history of the optimizer or that every call would be strategically good.

**A finite score-function gradient is insufficient without sampling support.** Differentiating stable `log pi(call)` counterfactually produces finite, often large gradients even when `pi(call)` has rounded to zero. The current sampler never selects those calls, so this does not supply their on-policy action gradient.

Temperature 10 removes exact zeros but leaves practical probabilities negligible. At temperature 100, mean chi mass when offered is about **1.32 × 10⁻⁶**, pon **1.74 × 10⁻¹²**, open kan **2.05 × 10⁻¹⁰**, and concealed kan **8.66 × 10⁻¹¹**. Even the largest total call mass is only about **0.000040**, while discard distributions change substantially. Positive probabilities can also vanish when accumulated into a float64 cumulative distribution; the report records such order-dependent zero-width increments.

For the mixture, `q(a) = 0.95*pi(a) + 0.05/|legal|`. Its correct derivative is that of `log q(a)`. The contribution through the base policy is attenuated by `0.95*pi(a)/q(a)`, which is numerically zero for these calls. All 22 call-target parameter gradients were exactly zero on both checkpoints and both repaired backends. This concerns the direct chosen-call term; other trajectory terms could still change shared weights indirectly. Substituting `log pi` would describe a different estimator and is not an approved repair.

None of the four rules is promoted. The proposed next experiment gives call selection its own learnable probability rather than relying on an almost absent base probability.

## MPS transfer failure and repair

The first attempt reached reporting but failed because an infinite MPS parameter-gradient norm could not be serialized as strict JSON. Its ledger entry and failure observation remain intact. A reporting-only recovery repeated the same positions and grid, retaining **2,548 nonfinite metric locations** as explicit nulls with a path list. Those original MPS measurements are invalid, even where their old completion/element-finiteness flags say otherwise.

The focused transfer fixture established a local runtime problem: a combined MPS-to-CPU transfer and float64 conversion turned a known six-value tensor into zeros. Saved logits were also converted to zeros, and 40 of 41 connected parameter-gradient transfers disagreed with a staged copy. Moving the tensor to CPU in its original dtype, then converting to float64 on CPU, preserved the values. Synchronizing before the combined transfer did not repair it.

The repair declaration changes only the new diagnostic module: staged transfers and explicit detection of nonfinite derived norms. The original plan, selection, grid, checkpoint pair, failed attempt and recovery are preserved. Regression tests check known values, nonuniform probabilities, backward gradients and overflow reporting.

The repaired diagnostic completed. **All CPU row outputs, gradients and summaries exactly match the recovery.** All final MPS gradient probes are finite; weights, parameter `.grad` fields and RNG state remain unchanged. No model checkpoint was saved.

After the repair, CPU/MPS are still not numerically identical. Maximum candidate-probability differences under the unchanged rule are **0.00537048** for the original checkpoint and **0.00244109** for the final checkpoint on the 32 probe positions. These are descriptive measurements, not a post-hoc parity pass. Keep single-row CPU as the next learning reference.

Historical Stage 3A formal collection/learning used CPU, so its recorded results are unaffected. Its existing MPS categorical/replay code still contains the combined conversion at `src/riichi_ai/stage3a.py` lines 212 and 599. That historical module was left unchanged in this diagnostic. **Do not use its MPS training path until the same repair and an appropriate regression check are applied.** The prior MPS fixture did not detect this path's problem.

## Checks, resources and preservation

- Preflight: **47 tests and 13 subtests passed**. Transfer regression: **17 tests passed**, including three new checks. Independent audit regression first passed 16 tests, then **17 tests** after adding an exact-zero-support rejection check. Accounting for overlap gives **67 distinct tests and 13 subtests** across these runs.
- The independent auditor passed **531,539 data-level assertions**, including 21,208 exact zero/nonzero probability checks, reconstructing selection, masks, saved-logit probabilities, score-function gradients and norm aggregation. Model forwards and parameter gradients remain worker measurements; the auditor does not independently implement Torch. The earlier successful audit is preserved separately.
- All **9 budget operations are closed: 8 completed and 1 failed**. No reservation was exceeded. Sampled peak process-tree RSS was **596,623,360 bytes, about 0.556 GiB**, below the 32-GiB ceiling. This is sampled RSS, not a continuous measurement of all GPU-driver allocations.
- The original **287 Stage 3A artifact identities** still verify against completed Git snapshot `d412ccb6d0cd345904fe9084120839ba25341cf3`. The original reference and final training checkpoint are unchanged.
- Formal diagnostic source was initially frozen at `3a3d4e2`; the separate repair declaration binds the revised implementation. The final inventory records project, external ledger and local book artifacts. The 500-MB runtime cap covers new diagnostic reports/external data; local book metadata is separately inventoried.
- Runtime remains Python 3.14.8, Torch 2.14.1, NumPy 2.5.4, RiichiEnv 0.4.10, mahjong 2.0.0 and psutil 7.2.2. No dependencies, external corpora, paid compute or public play were added.

The numerical allowance covers tests and model/runtime operations, including failures. Source authoring, standard-library integrity/data review and manual documentation are outside that allowance; 21.86 seconds is not the total wall-clock duration of the task.

## Book preparation

The **same twelve corrected Stage 3A cases** now have strict partial-scenario records and separate question-only exports. This is conversion, not twelve new extractions. The format preserves four books, eight problem families, ten supplied draws, two unknown draws, five unresolved colour candidates, one discard-plus-riichi recommendation and two conditional annotations.

Four source PDFs, nineteen evidence images, the prior 47-count report and two cross-stage family links were authenticated. The arithmetic was not rerun. All records remain development/quarantine material; no strategic-optimality claim, full-policy adapter or book training was added. See the book review and preparation evidence.
