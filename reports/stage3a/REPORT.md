# Stage 3A results

**Stage 3A met its engineering objective.** All 64 predeclared complete matches and all four policy-gradient updates finished. Actual behavior probabilities replayed exactly, each update changed parameters and action distributions, and every saved boundary reloaded correctly. This establishes a working complete-match RL path; it does not demonstrate improved playing strength.

The experiment charged **72.374 of 600 numerical seconds**, including imports, tests, two failed preflight attempts, initialization, all training, diagnostics, checkpoint verification and both book arithmetic runs. The match/update caps were reached, so the remaining time was not used to extend training. Source preparation and manual document review are outside this numerical allowance.

## What ran

The [accepted protocol](../../docs/STAGE3A_PLAN.md) and [configuration](../../configs/stage3a.json) were fixed before formal collection. Implementation was frozen at Git commit `bfee00b`. The original Stage 2B checkpoint initialized the learner; its three opponents stayed at the original deterministic argmax policy. The learner used a versioned categorical wrapper with a fresh AdamW optimizer, learning rate 0.00001, zero weight decay, zero baseline and gradient clipping at 1.0. Model execution was CPU float32 with one thread; probability normalization and log-probability calculations used CPU float64.

Every batch contained 16 fixed matches, with four learner appearances in each seat. The learner stayed frozen within a batch. Only that learner's recorded choices entered its one update. The objective summed action log-probabilities within each match and averaged terminal rank utility across matches. No opponent action, partial match, book example, rank-head prediction or held-out action-test label entered the loss.

| Batch | Complete matches | Optional / forced learner choices | Gradient norm before clipping | Parameter change, L2 | Mean probability-vector L1 change | Charged seconds |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 2,495 / 239 | 76.5814 | 0.005605 | 0.001301 | 14.666 |
| 2 | 16 | 2,273 / 212 | 92.6085 | 0.004118 | 0.001027 | 14.026 |
| 3 | 16 | 2,499 / 220 | 120.7121 | 0.003144 | 0.000847 | 14.948 |
| 4 | 16 | 2,483 / 240 | 118.4918 | 0.002554 | 0.001035 | 14.789 |

All four updates were finite and were not numerical no-ops. All pre-update probability and selected log-probability replay errors were **zero**. The maximum difference between the reconstructed objective and differentiable objective was **3.02 × 10⁻¹⁴**. Rank-head parameter hashes stayed unchanged; unused rank predictions may change through the shared representation.

## Validation and resources

- **64 matches, 677 hands, 42,189 engine steps and 10,661 learner requests** were preserved. Of the learner requests, 9,750 were optional and 911 forced.
- **45 matches ended after the learner's last action.** Each retained its correct terminal reward. Two final outstanding-deposit awards and two pinned-engine draw-log corrections were verified.
- The integrated suite passed **152 tests and 192 subtests**. A final focused auditor check passed **19 tests and 21 subtests** after its last arithmetic assertion was added. These runs overlap; their counts should not be added as unique tests.
- The independent audit passed **472,776 data-level assertions** covering file identities, actor tensor structure, semantic action association, RNG draws, simultaneous-request links, policy provenance, terminal results, objective arithmetic and resource records. This assertion count is not a count of independent scientific trials.
- The ledger has **13 closed operations: 11 completed and two failed preflight checks**. All reservations were respected. No formal batch failed, was replaced or was retried.
- Sampled peak process-tree RSS was **663,076,864 bytes, about 0.618 GiB**, below 32 GiB. Training/report artifacts were approximately **114 MB**, below 1 GB; the final inventory also records the separate local book evidence.
- The previous continuation's **212-file historical inventory** still verifies against its original Git snapshot. Verification.

The independent audit reconstructs recorded evidence. Numerical model replay, gradients and checkpoint reloads are worker evidence backed by the numerical fixtures; the audit does not independently reimplement Torch. Sampled RSS is not a continuous system-memory maximum. Rule coverage remains the documented fixture subset.

## Findings that change the next step

**The current policy does not explore calls.** It requested 8,018 discards, 2,182 passes, 312 riichi declarations, 124 ron and 25 tsumo actions. No chi, pon or kan was selected. More decisively, every offered call/kan candidate had recorded probability exactly zero: chi appeared in 1,308 learner requests, pon in 899, open kan in 40 and concealed kan in 53. These overlap in 2,235 requests. Added kan was never offered. See the action-support audit.

Among optional requests, 5,968 of 9,750 had a maximum probability of at least 0.999; 2,643 had only one numerically positive candidate. Median entropy was **0.000299 nats**, versus mean **0.356872 nats**. The average therefore hides many nearly deterministic choices. Continuing the same sampler alone will not supply selected-call examples in those zero-support states. This is the priority for a bounded exploration and gradient diagnostic before a large RL run.

**Single-row replay matters for this checkpoint.** On 32 fixed positions, batching the same rows changed a probability by up to 0.00172584, whereas repeated single-row evaluation was exact. Single-row accumulation was chosen before the first numerical lease. The protocol did not widen tolerances to accommodate this result. Numerical notes.

**Terminal log interpretation needed repair before training.** The first preflight assumed terminal scores appeared in `end_game`; the pinned engine emits only a marker. The second exposed a draw branch whose logged deltas include previously paid riichi deposits. Explicit source-supported score reconstruction and actual-engine fixtures fixed both before collection. Raw logs, failed checks and historical files remain preserved.

Training ranks were first/second/third/fourth **17 / 12 / 20 / 15**, giving mean rank **2.515625** and utility **−0.015625**. Batch means were 2.375, 2.375, 2.5 and 2.8125. These are descriptive outcomes from different fixed seeds and four evolving sampling policies, not evidence of improvement or decline. A future comparison must use fresh paired seeds and the same action-selection rule for the initialized and trained policies. Comparing sampled training games directly with an old argmax result would confound learning with sampling.

## Book preparation

Twelve additional question/explanation pairs are available: six Uzaku 301, four Uzaku 300, one Super Digital and one Statistics example. They form eight recorded families; related examples remain together as development material.

The first arithmetic check found seven mismatches. Enlarged source images revealed transcription errors that the first independent visual review had also missed. Five records were corrected in a separate revision: two manzu candidate labels, one pinzu candidate label, and two souzu hand/count-label groups. Original packets, failed comparisons and the correction history remain intact.

The corrected revision reproduces **all 47 printed counts with both calculators** under the book's declared convention. Five candidate counts differ by one tile under the shown-hand-plus-indicator convention, all in Uzaku 301 Q009. This validates the checked shape arithmetic, not the authors' strategic optimality or expected-value estimates. Red-five identity, missing match context and compound-action representation still prevent direct conversion into full-policy training rows. No book-derived training ran.

Start with the book results and corrected review links. The original arithmetic report, correction manifest and corrected arithmetic report preserve the sequence.

## Files and reproducibility

| Artifact | Location |
|---|---|
| Review and proposed next scope | [Stage 3A review](../../docs/STAGE3A_REVIEW.md) |
| Full settings and fixed seeds | [Configuration](../../configs/stage3a.json), frozen plan, seed provenance |
| All batch results | Batch 1, batch 2, batch 3, batch 4 |
| Machine-readable summary | Summary, budget, independent audit |
| Check evidence | Integrated JUnit, final auditor JUnit |
| Final training boundary | `~/Library/Application Support/RiichiAI/stage3a/boundaries/batch-3.pt` |
| Raw actor arrays and auditor logs | `~/Library/Application Support/RiichiAI/stage3a/trajectories/` |

The final boundary SHA-256 is `dff490dc2033a057c5d869e4cf4e215187b885edfc944d415643c296a600f05a`; its model weights hash is `04bc866db87eed7f32f55a98aab52e12fbe0189343e75554fbbc864031aec4e0`. It contains four completed updates, retired batches 0–3 and optimizer/RNG state. Load it through `riichi_ai.stage3a.load_boundary` with protocol SHA `e18412e95adffbf1fb9aa7ceda6caac5f07867eefe04e9e136e772ec4710a52e`. It is a training boundary, not a promoted deployment model. The original Stage 2B reference remains unchanged.

Runtime: Python 3.14.8, Torch 2.14.1, NumPy 2.5.4, RiichiEnv 0.4.10, mahjong 2.0.0 and psutil 7.2.2. No external corpus, paid compute or public play was used.
