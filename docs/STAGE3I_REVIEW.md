# Stage 3I review: more training with the reference preserved

22 October 2026. The complete game panel and independent audit passed. The [frozen plan](STAGE3I_PLAN.md) and [measured results](../reports/stage3i/REPORT.md) are the authoritative experimental record.

## What this experiment adds

We implemented the reviewed architecture and increased training to three fixed seeds, with twelve full passes for each of the value, action-residual and state-residual heads. This produced 49,788 optimizer updates and 50,935,716 head-row presentations from the same 471,627 training decisions. Those presentations are repeated exposures, not 50 million independent positions. The original acting policy, its feature encoders and original call heads remained frozen.

All three action-residual models beat both value-only and state-only controls on the 400-match predictive-gate partition. That is more useful evidence than merely fitting a nonzero critic. It still does not establish that choosing a different move improves a counterfactual outcome: labels come from the recorded explorer continuation, and candidate features may encode state information the baseline value head missed.

The CPU proved preferable for these small detached heads on this Mac. We retain MPS and CUDA-compatible paths, but measuring the actual workload prevented assuming the GPU must be faster. Feature preparation and game simulation benefit from parallel CPU workers. The original reference was never used as a utility regression output.

## What prevented a learned policy comparison

The first positive correction strength in the frozen calibration grid was 1. It changed 3.8772%, 3.0022% and 4.0893% of calibration decisions for the three seeds. All exceed the fixed 3% overall ceiling. The second seed's miss is tiny, but rounding it down or changing the ceiling after seeing results would undermine the experiment. Larger grid values also fail, so all three correctly fall back to zero.

This is a limitation of the chosen calibration resolution, not evidence that training failed or that any positive correction is necessarily unusable. Conversely, passing the prediction gate is not a reason to bypass calibration. We kept all three fitted models and ran the predeclared reference/reset controls; no learned correction has a fresh-game strength result in Stage 3I.

The saved-state-only fractional diagnosis found that lambda 0.25 fits the original ceilings for all three seeds on the reused calibration set. Yet exact reference-score ties mean that the infinitesimal positive-correction limit already changes 2.7868–2.9685% of decisions. Smaller lambda is not a smooth safety dial. These are ties in float32 model scores, not a proof of equal strategic value. The diagnosis does not replace the original calibration, choose a promoted model, or authorize treating this stage’s control games as candidate confirmation.

## Recommended next decision

Retain this frozen-reference implementation and the three completed head bundles. All 1,880 raw/wrapped reference pairs produced identical canonical games. On the 64-block reset subset, mean rank worsened from 2.339844 to 3.367188 with raw reset and 3.312500 with wrapped reset; the win wrapper does not repair the initialization deficit. The immediate next learning-system milestone is to test whether their action signal helps in games, using a predeclared finer correction range below 1 and a new physical-wall panel. First separate exact-tie decisions from positive-margin overrides; an explicit learned tie-breaker and a fixed nonlearned tie control would make the next attribution clearer. Keep this a new experiment, since the diagnosis was post-hoc. Retain the same wrapper-matched primary control, evaluate every admitted seed, and preserve a valid zero fallback. Do not retrain or select only the apparently best seed to make the next result look better.

A larger network or paid GPU is not yet the limiting issue. The current full training programme finished quickly, while the unresolved question is whether the observational target yields useful move rankings. If a careful corrected-policy comparison fails, investigate decision-specific targets, continuation/opponent mismatch and action coverage before simply adding epochs. A later data-collection programme should explicitly separate stronger-play exploration from volume, with a fresh budget and research-data permissions.

Book scenarios remain quarantined. They can support a separately validated tactical benchmark once diagrams, legal state reconstruction, answer ambiguity and explanations are checked; no book scenario entered this experiment. The project remains far from a demonstrated world-class bot, and no venue-specific or public-play strength claim follows from this stage.
