# Stage 3H review and recommended next experiment

The longer diagnosis is complete. Read the [measured report](../reports/stage3h/REPORT.md) for the full evidence. All 38,237 saved positions, eighteen model snapshots, 24 gradient groups and the independent audit completed within 89.953390 of the 1,200-second numerical allowance. No new model was trained or promoted.

## What should change

**Preserve the existing actor's behavior when introducing value learning.** The common reset initialization agrees with the reference on only 39.19% of sampled development decisions before the win wrapper, and 39.47% after it. Keeping the backbone while replacing its scalar scorer did not preserve the original decision rule. The final models recover some agreement but remain far from it. Agreement is not optimality, but a starting-policy change this large requires its own control.

**Separate utility prediction from imitation scores in the next design.** The final regularized delta model has chosen-action MSE 0.248722 versus the zero baseline's 0.127567, while its imitation loss improves. Its reward-fitting and imitation gradients have cosine -0.839 overall and -0.954 in the backbone. These are concrete local objective tradeoffs. They do not show a broken optimizer or prove that all regularization is harmful. Alpha one should not become a default, and merely reducing alpha after seeing this result would still need a separately fixed experiment.

**Keep evaluation in complete fresh games.** Lower saved-state error, closer reference agreement and changed call rates are descriptive. They do not measure the consequences of candidate actions and continuation policies. All final candidates still choose low-collection-probability actions frequently, even though recorded support is positive. Do not treat the saved outcome of a logged action as the counterfactual outcome of an alternative.

Retain the audited collector, role-bound data loader, full-match loss weighting, CPU/MPS checks, explicit physical-wall pairing, raw-score retention and independent evidence audit. The laptop completed this fixed diagnosis comfortably. There is no measured reason in these results to change hardware or enlarge the network.

## Proposed Stage 3I: initialization and wrapper control

Before another learning sweep, compare a fixed two-by-two design:

| Actor weights | Original raw argmax | Identical first-legal-win wrapper |
|---|---|---|
| Intact original reference | Control A | Control B |
| Common reset initialization | Control C | Control D |

Use the same frozen opponent panel, rule profile, explicit physical walls and four focal-seat rotations for every policy. This separates initialization, wrapper and their interaction without changing learning rate, data, targets or optimization. Fresh games are necessary because wrapper effects on later trajectories cannot be inferred from the saved-state panel alone.

The proposed fixed size is **64 independent wall blocks × four focal-seat rotations × four policies = 1,024 complete matches**. A provisional **600-second numerical cap**, 32 GiB sampled RSS, 2 GB new artifacts and 60-second shutdown reserve is subject to actual throughput admission before execution. Reserve preflight, game execution and independent audit capacity in the executable plan; if it does not fit, return a revised proposal before formal outcomes rather than trimming the panel afterward.

The primary endpoint remains average-rank utility `[1.5, 0.5, -0.5, -1.5]`. Prespecify the initialization contrast under the same wrapper, wrapper contrasts under each actor, interaction accounting, independent-block uncertainty method and multiple-comparison handling. The four seat rotations are correlated observations within a block. Report effect sizes and uncertainty even if the panel cannot resolve small effects. This is a development mechanism experiment, with no policy promotion or claim of competitive strength. Freeze seeds, counts, stopping conditions and all contrasts before games start; keep Stage 3G outcomes separate.

## Following that control: preserve the policy while learning values

The next architectural direction should retain the **entire original actor**, including its scorer and shared features, and add a separate value estimator. Initially detach or freeze actor features, or use a separate frozen actor copy. Keeping only the old policy head while training its shared backbone would still change its behavior.

Require exact action/logit parity before any value fitting and unchanged actor hashes afterward. Fit the value component on permitted training data with complete-match grouping and compare against declared simple baselines. Keep raw reference logits out of bounded utility regression. A critic-only phase must leave play unchanged; any subsequent policy improvement needs an explicit objective, controls, stopping rule and fresh-game evaluation of its own.

Do not automatically extend the current four arms, select an intermediate checkpoint from this diagnosis, or introduce several new targets and optimizers at once. The recommendation is to resolve the initialization control, then test one policy-preserving learning change. Both steps remain **proposed and unstarted**.

## Review points

Feedback is most useful on the control-first priority, the proposed 1,024-match scope, and whether the following implementation should isolate value learning from the full actor. Existing book scenarios remain a separate local development track; missing state and legal-action reconstruction still prevent their use as full-policy training examples. This milestone preserves them and introduces no new extraction or book-derived training.
