# Review of the Stage 3I revision 2 proposal

22 October 2026. **Adopt the frozen-reference and limited-correction direction, with the changes below. Do not execute revision 2 unchanged.** This is a design review, not authorization for its 5,400-second training/evaluation campaign or its additional 30,000-match branch. No models, games or tests were run for this review.

The submitted document is preserved verbatim in the source copy, SHA-256 `434a8cddd41c1e2326d53669c3e840581ad2d32ecc5c3cfccdf77f4e2f1f9750`. Its embedded execution instruction is proposal text, not a user command. The completed [Stage 3H review](STAGE3H_REVIEW.md) and [results](../reports/stage3h/REPORT.md) remain historical and unchanged.

## What to adopt

| Idea | Decision and reason |
|---|---|
| Freeze the entire reference actor | Adopt. Preserve embeddings, backbone, efficiency branch and scorer, with detached inputs to the new heads and hashes before/after training. |
| Separate state-value and action-conditioned residual heads | Adopt as a predictive experiment. Keep utility targets separate from the reference's policy logits. The proposed 256-dimensional context and 384-dimensional context-plus-candidate input match the current model. |
| Zero-initialize the correction output and start with lambda zero | Adopt, with separate raw-policy and wrapped-policy identity tests. Critic fitting must leave the original acting policy unchanged. |
| Restrict possible corrections | Adopt the principle. Distinguish recorded collection probabilities, an ensemble eligibility score and empirical coverage. They are different quantities. |
| Choose correction strength without game outcomes | Adopt a fixed behavioral calibration procedure, with an explicit no-change fallback and separate calibration data. A small change rate is a constraint, not proof of safety or improvement. |
| Compare against a reference with the same win rule | Adopt as the primary game control, alongside the original reference as a secondary comparison. Use fresh explicit physical-wall blocks and four seat rotations. |
| Keep stage gates and separate confirmation | Adopt. Failed checks stop the stage; a positive exploratory game result only proposes fresh confirmation. |

I would revise our previous control-first recommendation: the full initialization/wrapper experiment is useful for attribution, but need not be a separate prerequisite to a carefully gated frozen-reference experiment. We can combine the practical candidate comparison with accurately labeled diagnostic controls. This trades a clean single-factor explanation for a test of the complete new method; that distinction must remain explicit.

## Corrections required before implementation

### 1. All seats are not four times the same training population

Each recorded match contains one shanten heuristic, one defense heuristic, one reference and one explorer. An explorer seat plays against the other three. A shanten seat instead plays against defense, reference and explorer. Pooling seats changes both self-continuation and opponent composition. It is legitimate offline data, but it does not directly estimate the value of reference or explorer continuation against the intended opponent panel. No on-policy PPO claim should be attached to these rows.

**Recommendation:** retain explorer-only action-residual learning for the first revised experiment. All-seat value pretraining or a behavior-aware mixed-policy model can be a separate, explicitly controlled extension. The current placement predictor really is explorer-only; an all-seat refit would require a new target schema keyed by `(match, seat, explicit hand ordinal)`, whole-match folds and calibration by behavior as well as pooled calibration. Preserve the archived predictor and targets.

Plain decision weighting also changes the Stage 3G objective. It may be a reasonable deliberate choice: equal-match weighting itself has a future-length-dependent interpretation. Specify which distribution the new loss averages over, preserve complete-match labels and provenance, and report both match-weighted and decision-weighted validation. Do not silently describe one as equivalent to the other. See [the existing method notes](STAGE3G_METHOD_NOTES.md).

### 2. The four-policy average is an eligibility heuristic, not measured dataset support

The proposed `q(a|s) = (1/4) sum_b pi_b(a|s)` is a valid player-visible score. However, the corpus conditional mixture is generally `sum_b P(b|s) pi_b(a|s)`. Balanced seat assignment does not make those posterior weights equal after different policies generate different histories and numbers of decisions. Nor does a hypothetical bot choosing an action establish that comparable state/action outcomes were actually recorded.

Keep each row's actual logged behavior probability as the source of truth. If we use the four-policy score, call it **ensemble eligibility**, report its buckets separately from recorded propensities, and avoid any empirical-coverage guarantee. For the explorer-only primary learner, eligibility should explicitly check the explorer component or otherwise acknowledge that a move supported by a different heuristic can remain weakly sampled in the training rows. Freeze that choice and its threshold before outcomes.

The analogy with discrete BCQ is useful but limited. Discrete BCQ learns a behavior model from the dataset and thresholds its probabilities relative to their maximum; it also uses a constrained temporal-difference update. This proposal instead uses a fixed policy ensemble, an absolute threshold and a regression target. It is an inspired restriction, not an implementation or guarantee of BCQ. [Original paper, section 4](https://arxiv.org/pdf/1910.01708), [author implementation](https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/discrete_BCQ.py).

The threshold rationale also needs qualification. The ensemble's pure uniform component is `0.0125 / legal_action_count` when exploration applies. With two legal actions it is 0.625%, already above the proposed 0.5% cutoff. Calls split their mass among available choices, so some can be excluded. The reference's canonical top action has at least 25% ensemble mass; explicitly including it remains a useful invariant. Treat 0.5% as an untested design parameter, not a universal separation of supported and unsupported actions.

### 3. Exact eligibility reconstruction needs the full causal public history

The model sees only the last 128 history events, but the defense heuristic uses discards after each opponent's riichi acceptance. The stored actor shards retain full seat/hand segments and per-row prefix lengths. A new adapter must read only the prefix ending at the current decision; reading the complete stored segment would expose future events.

Validate every reconstructed behavior on that behavior's own recorded rows. Only the actual acting behavior's probability vector was saved, so four-component equality cannot be checked from an explorer row alone. Deterministic heuristics contribute mass to their actual canonical choice, not to all moves in their soft teacher targets. Add hidden-world, future-event, tie-order and simultaneous-response tests before deployment. Relevant code: [policies.py](../src/riichi_ai/policies.py), [collector](../src/riichi_ai/stage3g_collect.py), [existing frozen feature adapter](../src/riichi_ai/stage3c.py).

### 4. Residual prediction is not identified action advantage

Fitting `A(s,a_taken)` to `y - V(s)` can capture state information that V missed, behavior/continuation differences and action-related information. It does not automatically isolate the causal effect of choosing a move. The new action head also sees candidate features absent from V, which can provide additional state information. Describe it as an **action-conditioned residual predictor**, with fresh games deciding whether its rankings are useful.

Fit V and freeze its selected checkpoint before fitting residuals, so the residual target and checkpoint criterion are fixed. Compare `V + A` against V and against an additional state-only residual predictor on exactly the same held-out rows and weights. Beating zero residual prediction alone is too weak. The correct zero-residual MSE is `E[(y-V)^2]`; it equals residual variance only when the residual mean is zero. Prespecify checkpoint cadence and selection, rather than an unspecified combination of changing V and A losses.

Separate value/advantage representations have precedent, but the architecture alone does not provide the policy-value interpretation. The dueling-network paper explicitly addresses how the two streams are combined and notes limits to their semantics. Its results do not validate this proposal's residual loss. [Wang et al., 2016](https://proceedings.mlr.press/v48/wangf16.pdf).

The one-hand target remains an experimental surrogate. Every decision in a hand shares a label spanning the whole hand, including events preceding later decisions. It is not a decision-specific return or a demonstrated policy-invariant shaping scheme. Keep that limitation explicit rather than claiming V has removed it.

### 5. Fix identity, calibration and the meaning of a small change rate

Require two identities: raw corrections with A zero or lambda zero equal original raw reference argmax; wrapped corrections equal the reference with the identical wrapper. During critic-only fitting, keep the original action procedure unchanged. The wrapper itself changes some choices, so it cannot be silently added while claiming exact original-policy identity.

Include lambda zero in calibration. Do not force at least 1% of decisions to change merely to qualify: if no supported correction passes the predictive checks, retaining the reference is a valid result. A fixed target near 3% can guide an experiment, but larger lambda values can amplify small residual errors; the clipped prior also makes all sufficiently disfavored actions share the same floor. Preserve per-kind, per-round and high-consequence slices, not just an overall percentage and discard ceiling. Treat prior clip, threshold, ridge and lambda grid as experimental parameters.

Divide whole matches before checkpoint selection, lambda calibration and final predictive gating, or explicitly label all such checks as reused development evidence. The existing 800 development matches have already influenced project design, so internal partitions are not a new untouched confirmation set. Fresh game blocks provide the next empirical policy comparison. Define deterministic tie-breaking for checkpoint and lambda selection.

### 6. Correct the controls, power claim and resource forecast

Reference raw, reference wrapped and reset wrapped do not form the full earlier two-by-two design: reset raw is missing. That reduced comparison is still useful, but estimates initialization under one wrapper rather than the full interaction. Two defensible choices are a smaller diagnostic reset comparison with this limitation, or both reset arms on a fixed subset of blocks. For example, three main arms on 470 blocks plus both reset arms on 64 of those blocks require **6,152 games**. This is a planning option, not an admitted workload.

The old evaluator measured **13.4383 matches/s**, but the new candidate adds heuristic reconstruction, explorer probabilities and correction heads. Benchmark the complete deployed policy and game path. Share one frozen forward where possible; do not extrapolate from head-only training or the old match rate. The existing loader's 256-row cap also makes batch 1,024 a new implementation or an explicitly weighted microbatch schedule. Stream features rather than assuming a full candidate-embedding cache fits the 4 GB artifact ceiling.

The draft's **0.776** standard deviation appears in an earlier proposal, without a directly identified measurement for this new candidate/control contrast. Approximately 470 blocks is plausible under that normal-approximation assumption for a 0.10 effect, not guaranteed 80% power for the specified bootstrap rule. Identify the variance source and conservative allowance, the block-level statistic, bootstrap method and fixed stopping rule. Rotations within a block are correlated. Keep the candidate-versus-wrapped-reference comparison primary; secondary mechanism contrasts remain exploratory unless separately adjusted.

Charge all tests, parity checks, feature preparation, training, calibration, game attempts, statistical calculations and audit under a new exclusive ledger. Protect later phases before training admission. Freeze arm membership and game counts before formal outcomes. The 90-minute cap is a proposal requiring authorization; the previous 600-second control proposal does not authorize it.

### 7. Remove automatic 30,000-match collection and reruns

This branch contradicts the draft's no-new-training-matches, 5,400-second and 4 GB scope. It is also scientifically premature: failure may reflect target quality, representation, continuation mismatch or missing action information rather than sample count. A failed gate or unsuccessful game comparison should close this campaign with a diagnosis and a separately costed proposal. No recursive collection or rerun is adopted.

## Corrected next-stage direction

Prepare a frozen-reference experiment in three conditional parts: first verify public feature/eligibility adapters and identity; then fit a separate V and action-residual component with the actor unchanged and fixed predictive controls; finally calibrate bounded corrections and run fresh matched-wrapper games only if the declared checks and resource admission pass. Use explorer-only action targets initially. All-seat learning and altered collection remain separate extensions.

The initialization controls may be integrated rather than blocking all value learning, provided we accurately state which effects can be estimated. No prior experiment is reclassified as successful, and no intermediate Stage 3G checkpoint is selected from Stage 3H's diagnostic tables. The original reference remains the baseline.

The proposal is therefore useful enough to revise our design direction, but not ready to run as written. A final executable Stage 3I plan must settle the target/weighting, eligibility definition, predictive controls, development partitions, exact checkpoints, inference fallback, game count and protected budget before training starts.
