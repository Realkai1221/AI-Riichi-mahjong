# Reviewing the diagnosis and next training proposal

**Historical proposal, now executed:** the user authorized this comparison on 18 October 2026. All four arms completed. Read the [measured continuation results](../reports/package_a_continuation/REPORT.md) and [current review](CONTINUATION_REVIEW.md); the text below preserves the proposal that preceded those outcomes.

The [diagnosis report](../reports/package_a_diagnosis/REPORT.md) shows incomplete fit on training rows, a smaller teacher-validation gap, and a larger student-transfer gap. CPU/GPU and float32/float64 inference differences do not explain the smoothed models' errors on the tested states. The useful next question is whether smaller optimizer steps can improve fitting while preserving decision quality.

## One proposed intervention

Compare continuation at learning rate **0.0001** with continuation at the existing **0.0003**. Keep epsilon 0.01, architecture, features, data, loss, batch size, weight decay and gradient clipping unchanged. Do not also lower smoothing, add a rank loss, change the corpus, or replace the safety criteria.

Start both arms for each seed from its own saved smoothed **terminal checkpoint at update 18,500**. Preserve optimizer moments, step counters and RNG/sampler state. The selected update-3,000 snapshots lack matching optimizer/RNG state and must not be presented as exact resumable continuations. Match the next minibatch sequence across arms and verify the shared starting tensors before any update.

Proposed allocation: **four formal attempts**, two seeds × two learning rates, each with **5,000 additional updates** to update 23,500. Include failed and stopped attempts; no replacement runs. Use a **900-second cumulative numerical ceiling, 32 GiB sampled memory and 500 MB of new artifacts**, including resume checks, loading, validation and finalization. Freeze the order, sampling, checkpoints and implementation before outcomes. This is a proposal for review; it has not been executed.

## Measurements and acceptance

Retain the existing fixed validation subset for scheduled checks, the full validation strata for final reporting, and the same fixed training sample for endpoint fit. Do not inspect the action test or run another retention confirmation in this comparison. Save selected and terminal checkpoints separately and preserve all four trajectories.

Keep the existing safety formulas and each seed's original smoothed-run warm-up references; apply the same references to both continuation arms. Do not reset the warm-up after observing a result. The resume implementation must preserve relevant safety state and prove that it does not silently restart the optimizer or sample stream. An unchanged-learning-rate split/resume check should be performed on a small fixture before formal continuations, within the budget.

The exploratory question is whether lower-rate continuation improves optional-row smoothed-target KL on both the fixed training sample and teacher validation relative to each paired same-rate control. For decision quality, require teacher/student best-discard agreement to be no lower, and shanten-worsening rates to be no higher, than both the paired same-rate endpoint and that seed's update-18,500 starting checkpoint. These are descriptive criteria on fixed cohorts, not statistical significance claims. Report every component and both seeds; do not replace this with a single averaged win flag. A mixed result remains mixed and may motivate a different hypothesis.

The stronger existing durability target remains separate: both lower-rate terminal checkpoints must complete the full continuation without safety alarms and reach **at least 99.9% fixed-validation best-set discard agreement**, with full-validation severity and probability results reported. A lower loss alone cannot meet this target. Even a durability pass would still require a separately planned playing-strength evaluation; it does not establish teacher retention or readiness for broad RL.

## Useful feedback

Review whether this single-factor comparison is the right next question, whether the proposed 15-minute numerical ceiling is appropriate, and whether any acceptance condition should change before execution. Alternatives such as weaker smoothing, a model constraint, more student-state data, or broader call coverage are distinct experiments. Keep their effects separable.

Current files: [measured results](../reports/package_a_diagnosis/REPORT.md), [accepted diagnosis plan](PACKAGE_A_DIAGNOSIS_PLAN.md), frozen indices and identities, audit, and budget ledger.
