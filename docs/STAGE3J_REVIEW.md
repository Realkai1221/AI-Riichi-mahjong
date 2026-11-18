# Stage 3J review: what changed, what we learned, and the next useful experiment

29 October 2026. The [8,192-game experiment](../reports/stage3j/REPORT.md), all evidence checks, and all 32 fixed replays are complete. The [frozen plan](STAGE3J_PLAN.md) was followed. **No learned policy has earned promotion.**

## The result that matters

The three learned tie-breakers average rank **2.504883**, compared with **2.527344** for the wrapped original reference. That looks modestly encouraging in isolation. However, the adjusted interval for the utility gain is **[−0.040039, +0.083824]**, and the nonlearned hash tie-breaker averages **2.500000**. Learned ties minus hash ties is **−0.004883**, with adjusted interval **[−0.062500, +0.051432]**. We cannot attribute an improvement to the learned residuals.

Quarter-strength correction averages **2.506185**. Its difference from learned ties is **−0.001302**, with interval **[−0.007813, +0.001953]**. Almost all its interventions are the same exact-score tie changes: only 10, 3 and 15 decisions overrode a positive reference margin in the three 1,024-game panels. The fine correction was implemented correctly and tested on fresh games, but it added little distinct behavior.

These are uncertain estimates, not proof that learning cannot help or that the policies are equivalent. There are 256 independent wall blocks despite 8,192 games. Seats, learned seeds and shared controls cannot be counted as independent observations. The intervals are approximate block-bootstrap intervals adjusted for four fixed contrasts; the conservative simultaneous bounds also include zero.

## What to keep

Keep the entire frozen reference when testing a new value or residual learner. The project can now change action choice in a narrow, auditable way without discarding the original actor. This addresses the initialization failure seen in earlier stages.

Keep the nonlearned tie control. It prevents a small apparent gain over canonical argmax from being mistaken for evidence that the new model learned useful rankings. Keep all three training seeds, the fixed four-contrast analysis, complete-game rewards, simultaneous pre-response snapshots and explicit hand ordinals.

Keep the source archives and independent arithmetic audit. This stage preserves every tested version, including the preflight auditor strengthening. All 5.215 million recorded decisions passed; 256 fixed fresh actor forwards and 192 action-head forwards reproduced, and the 32 full replays matched exactly. The primary statistical summary was independently reconstructed too. These checks support trusting this experiment's record without implying full rules certification.

## What to change or defer

Do not promote seed 20261024012 because its sample mean happens to be the lowest among the learned policies. The experiment retained all three seeds precisely to avoid that selection effect. Do not promote the hash control either: its apparent benefit is also uncertain, and it was included to explain attribution.

Do not spend the next budget simply repeating more epochs of the same observational one-hand target, enlarging the network, or raising lambda until more moves change. Stage 3I established improved prediction; Stage 3J has not established better action ranking. More of the same prediction objective may leave that distinction unresolved. Keep the target as a research baseline rather than declaring it useless from one inconclusive panel.

A larger fresh confirmation panel would narrow uncertainty about these exact policies, but it would not directly explain why their action ranking should be useful. Its value depends on whether a small gain is already worth confirming. The present comparison supplies no clear learned-versus-hash advantage to prioritize. Do not append games opportunistically to this completed fixed experiment or pool an extension without a new statistical design.

## Recommended next learning direction: action-specific tie data

I recommend a new, bounded Stage 3K pilot before another large training sweep. This is a proposal, not a started run or a reuse of the remaining Stage 3J allowance.

1. **Collect interventions prospectively.** For an initial pilot, preassign a focal seat with balanced seat coverage. Use that seat's first nonwinning exact-reference tie with at least two eligible top-score actions, identified using only the audited visible state. Uniformly randomize that one action among eligible ties and log the full probability vector. Continue with the preserved reference and fixed opponents, without further randomized interventions in that match. Retain complete matches and all seat provenance, including games with no eligible opportunity as coverage evidence. Keep these data separate from the old mixed explorer corpus and from evaluation games.
2. **Make action credit explicit.** Pair the selected visible state/action with its observed continuation outcome; never give that return to unplayed alternatives. Include full-match rank utility as the objective and any hand-level target as a separate ablation. Randomization makes action assignment known; it does not remove return variance or continuation-policy dependence. First-tie data describe that state distribution, and deploying changes at every later tie introduces a further shift that needs full-game evaluation. State-only prediction and a nonlearned tie baseline remain necessary controls.
3. **Start with a small coverage/variance pilot.** Measure how many usable tie decisions and distinct action alternatives occur, propensity balance, per-match correlation, outcome variance, serialization cost and actual learning throughput. Size a subsequent programme from those measurements rather than promise a match count or effect before evidence. No policy-strength claim should come from the pilot's training losses.
4. **Fit an action-ranking model with the actor frozen.** Separate complete matches into training, selection and untouched evaluation roles before looking at outcomes. Compare a state-only model, the current observational residual baseline and the new action-dependent target under matched data/compute. A carefully designed propensity-aware estimator or ranking objective needs its own loss/weighting and support specification; do not label mixed frozen-opponent rows as on-policy PPO.
5. **Evaluate a fixed endpoint on fresh games.** Retain reference, hash tie control and all prespecified learner seeds. Use paired walls, seat rotations and block-level uncertainty. Predeclare any selection and then use a new independent confirmation panel before promotion.

An optional alternative is branching simulator continuations from saved opportunities with common future randomness. That can reduce comparison noise, but it requires a separately audited cloning/replay path, valid hidden-state distribution and complete-match rollouts. Simulator hidden state may be used by the auditor/critic to construct labels; it must never become actor input. A single realized hidden world must not be presented as the uniquely correct move for a partially observed position. I would benchmark this route before committing to expensive counterfactual generation.

The useful ambition is to improve the training signal and demonstrated strength. This experiment's fourteen-worker local path completed 8,192 fully logged games in about 9.4 minutes, so the Mac is not yet the demonstrated bottleneck for this particular workload. That does not predict the cost of large self-play training, stronger opponents or world-class performance.

## Files to review and points for feedback

- [Results and intervention tables](../reports/stage3j/REPORT.md): the main outcome and what the policies actually changed.
- [Frozen experimental protocol](STAGE3J_PLAN.md) and machine-readable plan: exact policies, walls, limits and analysis.
- [Primary statistics](../reports/stage3j/summary.json), all descriptive comparisons, and [independent summary verification](../reports/stage3j/summary-audit.json).
- [Evidence audit](../reports/stage3j/audit.json), replays, and [closed resource accounting](../reports/stage3j/accounting.json).
- [Choice implementation](../src/riichi_ai/stage3j_model.py), [game runner](../src/riichi_ai/stage3j_evaluation.py), and [independent auditor](../scripts/stage3j_audit.py).

The most useful feedback is whether to prioritize the proposed randomized action-credit pilot or a separately designed higher-precision confirmation of the current tie policies, and what meaningful average-rank gain would justify promotion. The recommended default is the action-credit pilot. Stronger opponent diversity, calls/kan coverage and validated book scenarios remain separate requirements; no external data acquisition, public play, paid compute or book-derived training occurred here.
