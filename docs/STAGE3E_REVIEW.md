# Stage 3E review — longer learning did not change sampled evaluation play

The authorized experiment completed **448 new training matches, 28 additional updates and 256 fresh evaluation matches**. The model reached update 32 from the saved update 4 model and optimizer. Both endpoints averaged **2.5625 place**, and all 128 paired games had identical actions, scores and placements. The measured utility difference is **0**, with the predeclared 95% interval **[-1.440484, +1.440484]**. No stronger policy is demonstrated or promoted.

The [measured report](../reports/stage3e/REPORT.md) contains the numerical evidence and artifact links. The [fixed plan](STAGE3E_PLAN.md) was executed without changing the learning rate, architecture, opponents, seeds or endpoint. Total numerical work was **956.038681 of 1,200 seconds**; all 67 operations closed successfully.

## What this experiment established

The continuation is real: the original AdamW state was restored, all 28 updates changed both new heads, gradients and parameters stayed finite, and behavior probabilities replayed exactly. The original base and rank head remained unchanged. The new training population includes **68,435 learner decisions**, **710 resolved calls/kans**, **5,834 post-call decisions**, and all five call/kan families. There were no no-op updates or replaced matches.

The small probability changes nevertheless produced no sampled evaluation action changes. All **3,763** evaluation call-opportunity distributions differ between update 4 and update 32, but their mean total call-probability change is only **-0.00007770**, about **-0.00777 percentage points**. The largest absolute per-action change is **0.00017722**. All 15,318 rows without call opportunities have exactly unchanged distributions. Identical shared-randomness trajectories do not prove identical policies or equivalent expected strength.

This answers the fixed horizon question narrowly: 28 more updates under this recipe did not produce an observable change in the sampled evaluation play. It does not prove that RL cannot learn Mahjong or that every longer run would fail. It is sufficient reason to stop extending this exact loop automatically and investigate the learning mechanism.

The evaluation mean 2.5625 must not be compared with Stage 3D's 2.6328125 as an improvement. They use different fresh seeds and different endpoint comparisons. The valid comparison here is update 32 versus update 4 within the same 32 blocks.

## An important policy-design observation

During training, **two of 16 positions with both a legal win and a call selected the call**, and both calls resolved. The other 14 selected the win. Average actual win probability in these positions was about 0.95026 because the call gate reserves approximately 5% mass for calls. This is observed exploration behavior, not a theoretical possibility.

These source states deserve focused review before another unchanged training run. A legal win is not automatically the optimal action for every placement situation, so this observation does not justify silently imposing an always-win rule. It does show that the current call gate can divert probability from an already strong win action. Any proposed win-preserving distribution or strategic exception would be a separate, declared intervention with its actual probabilities recorded and differentiated.

The conditional call head also has limited direct selected-choice evidence: training contains 1,942 rows offering multiple calls, but only 83 selected a call from such a row. Many legal call opportunities therefore do not supply the same learning signal as selecting among alternative calls. The observed small changes may involve gradient cancellation, output sensitivity, sparse choice evidence or downstream frozen-base competence; the present results do not identify a single cause.

## Recommended next step — proposed, not run

Use a **bounded diagnosis on saved training data**, rather than another unchanged duration extension. A provisional 300-second numerical cap would cover a small fixed diagnostic, without new matches or optimizer updates. Freeze its selection and questions before execution, and keep every Stage 3E evaluation trajectory out of development selection.

1. Select at most 256 training call-opportunity rows with declared strata, including the available win-versus-call cases and selected multi-call decisions where possible. Preserve match grouping, behavior checkpoint and actual sampling provenance. Label this development analysis, not new confirmation data.
2. Separate gate and conditional-call signals. Examine how complete-match returns, gradient directions, clipping and restored AdamW moments relate to effective output movement on common states. Gradients from stratified selected rows are partial diagnostics, not unbiased full-match policy-gradient estimates. Any full-match direction comparison must include all eligible rows of the selected matches, with that separate bounded workload frozen before execution. Distinguish an intact but weak learning signal from an implementation error. Do not assume a learning-rate increase, a critic or a different return normalization is already the answer.
3. Review the two sampled call-instead-of-win cases using their saved player-visible inputs and legal actions. Compare them with the other win/call opportunities; keep score/placement context explicit. Do not infer optimality from eventual outcomes or expose auditor-only hidden information to an actor.
4. Produce one concrete next intervention with a fixed control and evaluation budget. A variance-reduction baseline, a change in call/win probability allocation, or a post-call training change each tests a different hypothesis. Do not combine them into an unidentifiable recipe change or promote the old overconfident rank head to a calibrated critic.

A broader and stronger opponent panel remains necessary for competitive-strength claims. This narrow experiment tests one learning mechanism against three fixed original opponents. More compute alone is not yet the evidence-based next choice.

## Books and review files

The book validator review records 20 passing tests and successful conversion of the two existing call packets. The new schema separates a call-versus-decline preference from a conditional CHI choice, preserves unknown state and physical-copy ambiguity, and rejects answer/source metadata in question exports. Both records remain ineligible for policy training or evaluation; there is no complete-state adapter.

The next source extension should represent declared melds and distinguish own-turn kan decisions from response PASS before transcribing the already located open-hand/kan cases. This can proceed as a separate data-quality task alongside a future learning diagnosis. No new book extraction or book loss was used here.

The main decisions for feedback are whether to prioritize the learning-signal diagnosis, how to treat win-versus-call exploration, and which stronger opponent should eventually join the evaluation panel. The update 32 checkpoint is preserved as an experimental endpoint, while update 4 and the original reference remain unchanged.
