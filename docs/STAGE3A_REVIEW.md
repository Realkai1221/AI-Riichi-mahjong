# Stage 3A review and next decision

**The complete-match RL engineering pilot passed. The next problem is exploration coverage.** All 64 matches and four updates completed within 72.374 of 600 numerical seconds, with exact behavior-probability replay and verified checkpoint resume. No playing-strength promotion follows from this pilot.

Start with the [results report](../reports/stage3a/REPORT.md) and book results. The independent audit and action-support evidence provide the underlying checks.

## What to keep

Keep the original Stage 2B checkpoint as the reference, the new complete-match collector, terminal score reconstruction, single-row replay, frozen-opponent provenance, complete-batch update gate and immutable boundary checkpoints. Keep final-rank utility `[1.5, 0.5, -0.5, -1.5]` as the reference objective. Keep the rank head outside the learner loss until a separate calibration study justifies using it.

Keep book scenarios as a separate development resource with source identities, related-family grouping and partial context. The corrected 47-count check is useful evidence for extracting shape exercises. It is not a reason to turn a teaching diagram into an invented complete game state.

## What needs attention before a larger run

The learner saw 2,235 requests offering a call or kan, but assigned every such candidate probability zero. It sampled no chi, pon or kan. Most optional choices were also highly concentrated: 61.21% had a maximum probability at least 0.999, and 27.11% had only one numerically positive candidate. A longer unchanged run does not address the observed zero support at those states.

A new sampling rule is a different behavior policy. Its exact probabilities, replay and gradients must agree. For example, if a future rule uses `q = (1-epsilon) * pi + epsilon * uniform_legal`, the learner must differentiate `log q`, not silently use the old `log pi`. That mixture can generate calls while leaving gradients through extremely saturated base logits negligible. Sampling coverage and useful gradient response both need measurement.

The four updates also have pre-clipping gradient norms of 76.6–120.7 under a clip of 1.0. This is an observation, not proof of instability or a reason to loosen the clip after seeing results. A later baseline/variance study should remain separate from changing exploration, so its effect can be interpreted.

## Recommended next scope

I recommend a **Stage 3B diagnostic capped at 300 numerical seconds, zero new matches and zero optimizer updates**, using only saved Stage 3A development states. This is a proposal for review, not work already started.

1. Freeze at most 512 saved actor rows, stratified across offered chi, pon, open/concealed kan, discard and win/pass contexts before evaluating alternatives. Preserve original arrays and behavior identities; these are already exposed development positions, not a holdout.
2. Inspect original and final checkpoint legal logits, padding masks, action-type gaps, log-probabilities, underflow and gradient response. Measure why calls have zero numerical support rather than assuming an exploration coefficient alone solves it.
3. Specify a small diagnostic grid before execution: the unchanged sampler, two declared logit-scale/temperature alternatives, and one declared legal-uniform mixture. Record the actual distribution each would define. Backward-only probes may measure per-action/per-parameter gradient response, but no optimizer step or replacement checkpoint is allowed.
4. Compare action support, probability distortion on ordinary discards, meaningful gradient response and CPU/MPS consistency. Do not select a new default from a few attractive hand examples. Use the findings to write one complete, reviewable learning experiment with fixed batches, budget and a fair initialized-policy control.
5. Keep book work alongside it: resolve red-five notation and prepare a compatible partial-scenario format. Preserve the two-revision correction history and test arithmetic before accepting more transcriptions.

The subsequent learning/evaluation experiment should freeze a candidate at a predetermined endpoint and compare it with the initialized policy under the **same sampling rule and precision**, using fresh independent seed blocks and seat rotation. Deployment argmax is a separate declared comparison. Neither the Stage 3A batch rank trend nor its pooled mean rank is a valid learning-effect estimate.

## Useful feedback for your next instructions

- Confirm whether to prioritize the proposed exploration diagnostic before extending RL; this is my recommendation from the zero-support evidence.
- Review the corrected book hands and decide whether the first book task should be partial hand-shape learning or a human-readable question-and-explanation benchmark. Neither requires inventing missing game history.
- Keep a separate strength gate: specify a fixed opponent population, action-selection rule, candidate endpoint and fresh evaluation budget before judging improvement.

The Mac handled this pilot comfortably. Nothing measured here establishes a need for paid compute or a larger model. The outstanding issue is what the learner can explore and learn from, rather than raw local throughput.
