# Stage 3F value learning programme

Status on 22 October 2026: the user approved the revised direction after reviewing the external proposal. The integrated saved-data diagnosis and collector/loader engineering milestone below is now complete; see the [measured results](../reports/stage3f/REPORT.md) and [proposed first learning round](STAGE3F_REVIEW.md). This document preserves the adopted scope. No large training corpus, placement-model fitting, value-policy optimization, promotion or three-round campaign has started. The original proposal is preserved as source material; its embedded suggested instructions are not this execution contract.

The aim is to establish a practical path to learning action values from broader local experience. Earlier stages trained on 576 complete matches with 36 updates. They established real gradients and sampled call support, but did not establish a PPO plateau: PPO has not run. Retain complete-match average-rank utility `[1.5, 0.5, -0.5, -1.5]` and the original reference.

## Current engineering milestone

[The configuration](../configs/stage3f.json) caps cumulative numerical work at **7,200 seconds**, including a **300-second diagnosis sublimit**, tests, import/loading, collection, replay, backend probes and numerical analysis. Keep **120 seconds** for shutdown, **32 GiB sampled process-tree RSS**, and **2 GB new artifacts**. Bulk output is outside iCloud at `~/Library/Application Support/RiichiAI/stage3f`. Hashing, source editing and reporting are administrative work; they do not run model or engine computations. One durable numerical lease runs at a time. Development failures remain recorded and may be repaired before formal collection; failed formal matches are never replaced or silently excluded.

The current workload has zero optimizer updates or persistent model checkpoint writes. Benchmark backward passes are verification, not learning. Saved training data from Stage 3E may be used for diagnosis; every Stage 3E evaluation trajectory remains excluded from development selection. Historical evidence is verified from the completed Stage 3E commit and unchanged external/book files before shared documentation evolves.

### Saved-data diagnosis

Freeze at most 256 training call-opportunity rows, including win-versus-call and selected multiple-call strata, before model computations. Reconstruct the first complete 16-match Stage 3E batch using every eligible optional row for full-batch gradient comparisons. Report gate/conditional-head directions, cancellation, clipping and actual saved Adam/output movement without optimizer steps. Selected-row gradients are partial diagnostics, not unbiased full-match estimates. Review the two selected call-instead-of-win contexts using saved player-visible evidence without inferring strategic optimality from eventual results. A source-bound plan and bounded command preserve this distinction.

### Exploring collector

Each independent match contains exactly one `shanten_v2`, one `defense_v2`, one original-reference argmax player and one explorer, assigned to seats by a fixed seeded permutation. This balanced panel is the initial development design, not an independently sampled four-seat mixture or a claim about all opponent populations.

The explorer starts from the original base with the preserved initialized 5% Stage 3C call gate. On non-win decisions its actual distribution is 95% of that gated policy plus 5% semantic-uniform exploration. When a legal RON or TSUMO exists, use the first winning semantic action deterministically. This is a declared collection restriction, not a proof that declining a win is never strategically useful. Record conditional probabilities under the seat's assigned component; do not substitute a fixed mixture average or claim arbitrary off-policy correction is available.

Record all four seats with audited player-visible features, complete semantic probability vectors, chosen indices, actual random draws, immutable policy/checkpoint identities and independent seat RNGs. Build all simultaneous response views before selecting any response. Keep call requests distinct from resolutions, preserve full-match rewards after a seat's last action, and identify every hand including dealer continuations. Store actor arrays, targets, behavior provenance and auditor logs separately. Shared histories use only their recorded causal prefixes. Pilot records are development-only and ineligible for PPO or policy training.

The fixed ceiling is 1,200 full-match executions: 16 plans run serially and in parallel for payload parity, four disjoint 32-match worker probes at 1/4/8/14 workers, then a fixed 1,000-match development pilot if preflight, resource forecast and throughput gates permit it. Deterministic payloads must match; elapsed time and process metadata are excluded from byte parity. Worker count is chosen from measured complete-collector throughput before the main pilot. Replays and engine-based tests also count toward the execution ceiling. Instrumentation or rule failures stop formal work and preserve partial evidence.

### Bounded-memory loader and backend probe

Keep one bounded match shard resident and reconstruct visible history prefixes per row. Shuffle matches and rows reproducibly, without preloading the corpus or repeatedly decompressing arbitrary rows across all matches. Separate actor tensors from reward and behavior targets, and bind resume cursors to the exact manifest. Report compressed storage, uncompressed shard size, sampled memory, loader throughput and representative CPU/MPS forward/backward throughput. A synthetic or frozen-backbone probe must be labeled accordingly; it cannot justify an unsupported full-network training forecast. CUDA remains a portable code path, with local availability and measurement reported separately.

### Audit and completion gates

Every saved match must have valid terminal scores/ranks, legal requested actions, normalized actual distributions, exact selected-action probability, causal actor prefixes and unchanged behavior identities. Serial/parallel payload parity and focused information-boundary tests must pass. Same-engine replay checks consistency; it is not independent full-rule certification. Targeted call/kan fixtures remain required. Count uninstrumented rare paths as unknown, not zero. Sample ordinary games by a fixed rule and deliberately inspect rare call/kan evidence; record any unavailable independent oracle.

The engineering report must give completed/failed/censored counts, actual resource ledger, schema/tests, coverage by action family and post-call states, measured loader/backend behavior, and a concrete recommendation for the first learning round. Freeze the corpus count only from those measurements. Do not stop or extend a sample based on attractive match outcomes. No competitive-strength inference is attached to pilot games.

## First learning round after engineering

The learning design will compare chosen-action value regression against actual terminal placement utility with the proposed one-hand placement-prediction difference. Both use the same declared data population, architecture and matched budget. These are controlled alternatives: the hand difference is a surrogate, and its telescoping identity does not turn each hand's truncated target into a complete-match return.

Fit placement predictions using complete-match grouping and out-of-fold training potentials. Validate NLL, Brier score, classwise and expected-utility calibration by predefined match stage and behavior population. A small MLP is a baseline; tie-breaking and rule context remain explicit. Terminal utility substitution is correct label construction, not evidence of calibration. The old rank head is not assumed calibrated.

Specify frozen/trainable layers, target scale, legal masking, forced-row handling, match/row weighting, seeds and exact training steps before execution. A conservative legal-action penalty is an experimental regularizer; include an alpha-zero control if attributing benefit to that penalty. Budget every target/alpha candidate, validation pass and checkpoint. Mixed-policy Monte Carlo values describe recorded continuations and opponents; they are not automatically values under the newly greedy policy. Historical mixed data cannot be reused as ordinary on-policy PPO trajectories.

Use development games to select a frozen candidate, then fresh paired confirmation with explicit physical-wall identities and independent seed blocks. Declare the primary opponent panel and previous-champion contrast, confidence method, fixed count, planning variance assumptions, secondary descriptive comparisons and failure treatment. Any later-round fixed-sequence gatekeeping must be specified before outcomes, stop on the first failed primary gate and forbid candidate retries after confirmation. The old variance is a planning assumption, not a measured property of a new value policy.

## Deferred scale and parallel books

The external proposal's 30,000 matches per round, three rounds, 70 GB and multi-day schedule are conditional proposals. They are not unmeasured throughput promises or an instruction to begin a large run in this engineering milestone. At 4 matches/s, collection alone already exceeds a two-hour cap; use a conservative completion forecast with reserves instead of that threshold. Compressed disk size does not imply the dataset fits in RAM.

Existing supplied books and quarantined records remain preserved. Partial-state and legal-action reconstruction, not absence of copies, are the technical obstacles. Declared-meld and own-turn-kan schema work can proceed separately; no book example is added to the pilot or training by this programme.
