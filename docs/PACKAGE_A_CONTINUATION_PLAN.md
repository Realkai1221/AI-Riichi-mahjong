# Learning-rate continuation plan

Authorized 18 October 2026 after the user accepted the [diagnosis review proposal](PACKAGE_A_DIAGNOSIS_REVIEW.md) and requested roughly one hour of project work before another review. This experiment occupies up to **900 cumulative numerical seconds** within that work session. Implementation, analysis, book-scenario preparation and documentation are separate from that numerical allowance.

## Question and fixed comparison

Does lowering the optimizer learning rate from **0.0003 to 0.0001** improve fit to the epsilon-0.01 smoothed teacher target without degrading discard quality? The previous inference-only diagnosis found incomplete fit already on training rows. It did not identify learning rate as the cause.

For each of seeds 202610171 and 202610172, start two arms from that seed's own smoothed terminal checkpoint at update **18,500**. Continue for **5,000 additional updates** to **23,500**. Preserve the exact weights, optimizer moments and step counters, random states, and sampler position. The changed learning rate is the only recipe intervention. Keep model, features, data, loss, smoothing, batch size 256, AdamW weight decay 0.0001, gradient clip 1 and original diagnostic/validation schedules unchanged.

Run sequentially in this frozen order: seed 171 same rate, seed 171 lower rate, seed 172 lower rate, seed 172 same rate. Reversing order for the second seed reduces a simple rate/order confound; two seeds still provide limited evidence. Both arms for a seed must consume the same continuation minibatch stream unless an arm stops early. Prefix hashes permit matching any common completed horizon. Include stopped and failed attempts; never replace an unsuccessful formal attempt.

The [configuration](../configs/package_a_continuation.json) fixes checkpoint hashes, source-report paths, budget, sequence and criteria. Freeze a plan with source hashes and the earlier diagnosis's exact training/validation indices before formal updates. Commit the implementation before those runs. Earlier source evidence is checked against its historical Git snapshot, rather than re-hashing changed current documents into an old inventory.

## Resume and safety checks

The old terminal checkpoints have optimizer/RNG state but do not serialize all safety-monitor counters. Reconstruct counters by replaying each saved complete loss history and diagnostic history through the unchanged monitor, then verify its references, events and flags against the original report. Carry forward the original warm-up references and absolute step numbers. Never reinitialize the monitor as a new warm-up or retrospectively adjust its thresholds.

Before the four formal attempts, run a small unchanged-rate split/resume fixture and compare its continuation with an uninterrupted reference: minibatch stream, model weights, optimizer state and safety behavior. Check that the lower-rate path changes the learning rate while preserving the shared starting state. CPU exact checks and MPS tolerance checks must be labelled separately if needed; a small fixture is not proof of arbitrary accelerator bitwise reproducibility.

Select a diagnostic checkpoint only among scheduled **continuation** validation checks; preserve it separately from the terminal checkpoint. The starting checkpoint remains available independently. Scientific acceptance uses terminal results, never whichever continuation checkpoint looks best.

## Measures and acceptance

Use the same fixed validation subset for scheduled checks and both full validation strata for terminal results. Use the diagnosis's fixed **10,000 training rows** and **16,024 validation rows** for endpoint fit. Save compact numerical arrays and explicit cohort denominators. Original-target and smoothed-target fit, non-best probability, best-set discard agreement, shanten worsening, lost ukeire and action groups remain separate measures.

For each seed, the exploratory criteria require lower-rate optional-row smoothed-target KL to be strictly lower than its paired same-rate endpoint on both sampled training and teacher validation. On full validation, teacher/student best-discard agreement must be no lower, and shanten-worsening rates no higher, than **both** the paired same-rate endpoint and that seed's starting endpoint. Training decision quality is also reported descriptively. Report each component and both seeds. These are descriptive comparisons on fixed cohorts, not independent-row significance tests or playing-strength claims.

The stronger durability criterion requires both lower-rate arms to finish all additional updates without safety alarms and reach **at least 99.9% terminal fixed-validation best-discard agreement**. A lower loss alone is insufficient. No candidate is promoted from these diagnostics; playing-strength evaluation remains separate.

## Resource and scope boundaries

- Four formal attempts maximum; all failures and stopped attempts count.
- 900 cumulative numerical seconds, 32 GiB sampled memory, 500 MB new artifacts.
- Numerical tests, imports, loading, resume reconstruction, training, evaluation and serialization are charged through a persistent ledger and subprocess watchdog. Reserve 45 seconds globally; each attempt has at most 180 seconds, including finalization. Stop rather than extend a ceiling.
- No new matches, action-test access, external acquisition, paid compute, RL or fresh retention confirmation.
- No additional recipe trial selected after seeing these four results within this authorization.

In parallel, prepare a separate book-exercise intake format. Until book titles, formats and source content are available, use only an explicitly authored example; do not invent book labels or treat unspecified context as a complete simulator state. Keep answer keys outside actor inputs, and keep evaluation exercises separated from any later training selection by source/problem family.

Deliver measured results, budget accounting, limitations, an independently reviewed recommendation and files the user can annotate. The hour is a work-session target, not a requirement to consume unused numerical budget.
