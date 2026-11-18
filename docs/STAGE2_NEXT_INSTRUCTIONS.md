# Stage 2B proposed instructions for review

Drafted 17 October 2026 from the [completed Stage 2A measurements](../reports/stage2/REPORT.md). **This is a proposed next experiment, not work already executed or an automatic training schedule.** Stage 2A's engineering gate is met; teacher retention remains unmet. Preserve the original proposals, implementation, corpus, checkpoints, and unsuccessful pilots.

## Objective

Close the measurable gap between the neural policy and the transparent `shanten_v2` teacher before adding reinforcement learning. The extended model agrees with 82.47% of validation discards, but 5.01% worsen shanten and 12.49% lose visible ukeire at equal shanten. It averages 3.18 place against a teacher average of 2.57. The immediate problem is accurate, reliable discard decisions across complete matches.

Keep the existing rules, average-rank utility `[1.5, 0.5, −0.5, −1.5]`, semantic action mapping, legal information boundary, and opponent panel fixed. Maintain v1/v2 heuristic references. A higher imitation score is a diagnostic; full-match teacher retention remains the playing-strength gate.

## Proposed bounds

| Resource | Proposed Stage 2B ceiling |
|---|---|
| Hardware | Current M5 Pro; CPU collection/evaluation and MPS training |
| Cumulative training/check/loading/validation time | 3,600 seconds for this new stage; reserve stopping/checkpoint overhead |
| Memory | Approximately 32 GiB sampled conservative total, reporting RSS and MPS separately |
| Additional saved artifacts | 5 GB; bulk files outside iCloud |
| Training runs | At most eight full runs, initially 20,000 updates each; fewer if measured budget requires it |
| Newly collected matches | At most 600: 100 diagnostic, up to 300 training additions, 100 fresh validation, 100 fresh final action test |
| Development full-match evaluation | At most four 30-block campaigns, 960 match executions total |
| Final retention confirmation | At most one campaign, with count fixed by the declared variance/power rule; maximum 256 blocks / 2,048 matches |

Collection and evaluation counts are separate because paired evaluation deliberately repeats walls across arms/seats. Replays and unit fixtures must also be accounted for separately. These are ceilings, not targets to consume. No external data/weights, paid compute, public venue activity, search, or RL is included. A budget stop produces a partial/inconclusive report and does not justify silently expanding scope.

## Step 1: preserve controls and make the diagnostics useful

Before running anything, save a Stage 2B configuration with exact seed namespaces, source/dependency/encoder identities, candidate variants, loss definitions, selection order, resource accounting, and evaluation limits. Check exact seed overlap with all earlier corpus and evaluation manifests. Use a new output root; never overwrite Stage 2A artifacts.

Retain the extended Stage 2A checkpoint as the fixed historical student. Add the following metrics to each new candidate report:

1. Optional-discard best-set agreement, policy KL, and exact teacher-choice agreement reported separately.
2. Shanten-worsening rate, shanten increment histogram, equal-shanten lost-ukeire distribution, red-only errors, and legal-action-kind errors. Report per-match and per-action denominators.
3. Policy confidence/error by hand stage and decision type. Low confidence need not mean high regret; do not infer a loss-of-points score from teacher preference keys.
4. Full frozen-model train versus validation metrics on declared fixed samples, rather than comparing changing minibatch losses with final validation.
5. For any trained rank output: NLL, Brier, calibration by match stage, and public score/standing/round baselines. Group uncertainty by complete match. Rank-weight zero leaves the rank head untrained and excludes it from learned-value claims.

Make a small regression set from training/development states with severe hand-shape errors. Keep it separate from final evaluation evidence. Add focused tests for any new feature/target rather than tests that merely reproduce implementation details.

## Step 2: measure states the student actually reaches

Run 100 fresh complete diagnostic matches with the **frozen Stage 2A extended student** as the focal policy, rotating its seat against the existing heuristic panel. At each focal decision, ask the teacher for its preference set on that same permitted player view, but execute the student's chosen request. All simultaneous seat requests must still use one pre-response state.

Save executed behavior, behavior/checkpoint identity, teacher query/target, exact legal candidates, and terminal outcomes separately. The teacher's counterfactual recommendation is a supervised label, not the action that produced the transition. These records are not on-policy PPO data for the teacher. Audit the simulator/player interface exactly as before.

Compare discard-error categories on these states with teacher-generated validation states. Stratify by public game stage where feasible; the distributions differ, so an unadjusted accuracy difference is descriptive. Inspect whether the student makes calls/kan or creates states missing from the closed-teacher corpus. A stronger representation alone may help; do not assume distribution shift is the entire problem.

These 100 matches become development evidence only. Do not reuse them as the final test. If additional student-state data is later justified, label it through the same information-limited procedure.

Before Steps 3–4, also freeze and generate the reserved 100 fresh validation and 100 fresh final-test matches. For each split, use 50 matches with a teacher focal policy and 50 with the frozen Stage 2A student, against the same panel and with focal seats balanced as closely as possible. Query the teacher on focal decision states in both strata. Keep the split seed groups disjoint; report the two strata separately as well as the declared aggregate. These new validation records drive the new stage's checkpoint and recipe selection. Keep the new final-test targets sealed to performance analysis until selection is complete. Existing Stage 2A validation remains an additional exploratory diagnostic.

## Step 3: test whether the rank loss helps the policy

Run a controlled comparison on the existing 800 training matches:

- **Control:** policy loss plus rank loss weighted 0.1.
- **Variant:** the same policy loss with rank weight 0.
- Use two fixed initialization seeds for each setting: four runs total. Pair the actor initialization, sample streams, batch sizes, optimizer, update budget, and validation schedule within each seed. Keep the same network structure where practical to isolate the loss change.
- Use 20,000 updates and the Stage 2A optimizer defaults initially. If runtime requires a lower ceiling, declare the common lower budget before seeing comparative results.

Rank-loss removal is a hypothesis, not an expected win. Stage 2A's rank overfitting and policy/rank tradeoff justify the ablation; they do not prove harmful gradient interference.

Predeclare checkpoint selection using validation **discard best-set agreement**, then lower shanten-worsening rate, then lower optional-policy KL. Select the recipe using the mean result across both seeds, reporting both seeds and the spread. Do not silently select whichever seed gives the best story. Preserve the fixed Stage 2A selection rule as historical; this is a new protocol.

Declare the lower-numbered training seed as the primary seed before training. Once the recipe is selected using both seeds, its primary-seed validation-selected checkpoint is the single candidate for the sealed action test and any confirmation. Do not switch to the other seed after seeing test/match outcomes. An invalid primary run must be repaired or reported as incomplete, not silently substituted.

## Step 4: teach hand shape more directly, one change at a time

On the chosen rank-loss setting, test an auxiliary prediction of **post-discard shanten for each legal discard candidate**, using two paired seeds and the same training corpus/update budget. Compare against the corresponding two controls from Step 3. Compute the label from the permitted own hand and candidate action; verify it against the recorded teacher preference on applicable discard rows. Mask non-discard actions explicitly and define the target range from the validated scoring routine.

The shanten target is a training label. Do not feed recorded teacher keys, best-set masks, or future information into inference. If the network needs an action-conditioned post-discard hand representation, version that as a separate architectural change and report parameter/latency differences; avoid combining an architecture change and several new losses in one unidentifiable experiment.

Visible-ukeire auxiliary supervision is a follow-on if shanten errors fall but equal-shanten mistakes remain. Define exact known-copy counting and red/call semantics first. It competes with the optional data experiment below for the remaining two-run budget; do not automatically execute both. Run only the two new ukeire variants and reuse their corresponding matched controls from Steps 3–4. Four rank-loss runs plus two shanten variants plus two final variants exhaust the eight-run ceiling.

An alternative is to supply exact computed shanten/ukeire as inference features. Those are legal actor-computable inputs, but such a model is an **analytically assisted hybrid baseline**. It must be labeled and compared separately, not presented as evidence that a raw-input network learned tile efficiency. No hidden-state inputs are permitted in either version.

## Step 5: decide whether to add student-state training data

Use Step 2 to decide whether a small data-aggregation experiment is warranted. If so, collect at most 300 additional complete training matches with a frozen, identified student behavior policy and teacher queries at its decision states. Keep opponents/policy versions fixed within each declared collection round. Retain original teacher examples as a control and to reduce forgetting.

Compare teacher-only training with a declared mixture of teacher and student-state rows at matched updates and paired initialization seeds. Reuse the corresponding two teacher-only runs from Steps 3–4 as controls; the remaining two runs fund only the mixed-data variants, restarted from the matched initialization rather than given extra continuation updates. Keep match-level provenance, sampling weights, and final-outcome targets intact. If both rank and policy targets are used, the rank label must describe the actual rollout behavior, not a hypothetical teacher continuation. Do not mix rank targets from different behavior populations without reporting that change.

This is motivated by dataset aggregation methods such as [Ross, Gordon, and Bagnell's DAgger paper](https://proceedings.mlr.press/v15/ross11a.html): actions change the future observations a policy encounters. That literature motivates an experiment; it does not guarantee improvement for this partially observed multiplayer setup or our imperfect heuristic teacher.

Retain the 100 new validation and 100 new final-test matches frozen in Step 2, whether or not training aggregation is chosen. Never add them to training or recollect them under a stronger behavior policy after observing results. The original action test is historical evidence for the initial student, not an untouched test for repeated future decisions. Keep the new final action test sealed until the final candidate is frozen.

If the 100-match diagnostic does not justify aggregation, do not fill the new-data quota. Spend the remaining bounded experiment on ukeire supervision or report the next unresolved hypothesis instead.

## Step 6: judge playing strength and stop honestly

Use common development block seeds across compared frozen candidates to improve comparability; once inspected, those blocks remain development-only. Allocate the four-campaign ceiling to the two most informative recipes across both training seeds. If a historical-student campaign is preferred, it must replace one of those campaigns, with the lost replication stated. Save each result and avoid treating repeated adaptive development comparisons as independent confirmation.

After selecting one candidate, inspect its new sealed action test once and run at most one fresh retention confirmation if the declared pilot rule warrants it. Preserve the 0.10 utility margin, 95% one-sided normal approximation, variance floor/inflation, and explicit power/cap assumptions, or revise them in writing before any new confirmation outcomes. A count above the cap is inconclusive; a weak pilot stops before confirmation. No post-confirmation tuning can reuse those blocks.

Report three separate outcomes:

1. **Engineering:** new targets/data paths pass their relevant correctness, information-boundary, replay, and resource checks.
2. **Imitation:** severe discard errors fall across both training seeds and appropriate held-out state populations. The observed Stage 2A rates are comparison baselines, not universal acceptable thresholds.
3. **Strength:** teacher retention is demonstrated under the declared full-match protocol, or remains unestablished with the exact evidence and limitation stated.

Keep open-hand/kan trajectory coverage as a separate named limitation even if closed-teacher retention passes. Add a broader lawful training mixture and targeted rule trajectories before claiming general competitive Riichi coverage. A limited closed-hand teacher also imposes a ceiling: exceeding strong external agents will eventually require stronger learning targets, RL/search, or other validated improvements.

## Deliverables for the next review

Provide the configuration, new tests and schemas, all run histories including unsuccessful variants, data/source/checkpoint manifests, per-seed comparison tables, error examples with only legal information, fresh full-match evidence, and actual time/memory/storage totals. Update STATUS and DECISIONS. Explain which hypothesis was supported, which was rejected or unresolved, and the smallest useful next experiment.

Do not promote the rank head into a GRP/reward model, start PPO, collect a large external corpus, enlarge the model substantially, or rent GPUs merely because this bounded experiment ends. Each is a separate decision that should follow the measured bottleneck and remaining strength gap.
