# Stage 3J results: fresh-game test of smaller corrections and learned ties

29 October 2026. **Completed and independently audited. No policy is promoted.**

All 8,192 formal games finished. Learned tie-breaking has a slightly better observed average rank than the reference, but the uncertainty includes no improvement, and it did not demonstrate an advantage over the fixed nonlearned tie control. Quarter-strength correction produces almost the same games as tie-breaking alone. The current evidence does not establish useful learned move ranking or a playing-strength gain.

The [frozen protocol](../../docs/STAGE3J_PLAN.md), machine-readable plan, [review and recommended next work](../../docs/STAGE3J_REVIEW.md), and [complete statistics](summary.json) distinguish the fixed experiment from interpretation.

## Complete-game outcomes

Lower average rank is better. Every individual policy played **1,024 games**, consisting of 256 shared wall blocks and four focal-seat rotations. Opponents were the fixed shanten heuristic, defense heuristic and original raw reference. These are results against this local panel, not a league rating or comparison against Mortal/NAGA/Suphx.

| Focal policy | Average rank | First / second / third / fourth |
|---|---:|---|
| Wrapped original reference | 2.527344 | 269 / 234 / 233 / 288 |
| Nonlearned hash tie-breaker | 2.500000 | 274 / 245 / 224 / 281 |
| Learned ties, seed 20261024011 | 2.506836 | 263 / 256 / 228 / 277 |
| Learned ties, seed 20261024012 | 2.496094 | 275 / 245 / 225 / 279 |
| Learned ties, seed 20261024013 | 2.511719 | 265 / 251 / 227 / 281 |
| Correction 0.25, seed 20261024011 | 2.508789 | 262 / 256 / 229 / 277 |
| Correction 0.25, seed 20261024012 | 2.497070 | 275 / 244 / 226 / 279 |
| Correction 0.25, seed 20261024013 | 2.512695 | 264 / 251 / 229 / 280 |

Across all three saved seeds, the mean ranks are **2.504883 for learned ties** and **2.506185 for quarter-strength correction**. These pooled policies average the three seed results within each wall block; they are not ensembles deployed in a single game. All three seeds remain reported, and none is selected based on these outcomes.

## Preregistered pooled comparisons

Utility is `2.5 - rank`, so a positive difference means better average rank. Each interval uses **256 independent wall blocks**, averaging correlated seats and learned seeds inside each block. We jointly resampled the blocks 10,000 times with the fixed seed. The four 98.75% percentile intervals allocate a nominal 5% family error using Bonferroni; their coverage is approximate. The separate simultaneous Hoeffding bounds are conservative finite-sample bounds under the block-sampling assumptions.

| Fixed contrast | Mean utility difference | Adjusted bootstrap interval | Simultaneous Hoeffding interval |
|---|---:|---:|---:|
| Learned ties − reference | +0.022461 | [−0.040039, +0.083824] | [−0.574907, +0.619829] |
| Learned ties − hash ties | −0.004883 | [−0.062500, +0.051432] | [−0.602251, +0.592485] |
| Correction 0.25 − reference | +0.021159 | [−0.041016, +0.082357] | [−0.576209, +0.618527] |
| Correction 0.25 − learned ties | −0.001302 | [−0.007813, +0.001953] | [−0.598670, +0.596066] |

All four intervals contain zero. This is inconclusive evidence, not proof of equality, absence of a small benefit, or superiority of the hash control. The observed reference improvement alone would be misleading without the nonlearned control: the hash control's observed mean was at least as good, with uncertainty spanning either direction.

The thirteen fixed per-seed/control contrasts, with ordinary descriptive 95% intervals, remain in evaluation.json. They do not replace this four-contrast primary family. Absolute ranks from earlier stages use different wall populations and must not be substituted as baselines.

## What the policies actually changed

Counts below are comparisons with the wrapped reference **on each policy's own visited optional-decision states**. Different policies visit different states, so these are not matched counterfactual action-quality estimates. A forced single legal action is excluded from the denominator. All observed changes in this panel occurred where the wrapped reference selected a discard.

| Policy | Optional decisions | Exact-tie changes | Positive-margin changes |
|---|---:|---:|---:|
| Wrapped reference | 148,702 | 0 | 0 |
| Hash tie-breaker | 148,605 | 4,107 | 0 |
| Learned ties, seed 20261024011 | 148,685 | 4,111 | 0 |
| Learned ties, seed 20261024012 | 147,606 | 3,866 | 0 |
| Learned ties, seed 20261024013 | 149,224 | 4,098 | 0 |
| Correction 0.25, seed 20261024011 | 148,696 | 4,108 | 10 |
| Correction 0.25, seed 20261024012 | 147,615 | 3,866 | 3 |
| Correction 0.25, seed 20261024013 | 149,239 | 4,095 | 15 |

Tie-only policies never overrode a positive raw reference-score margin. Exact ties are equality in float32 model outputs, not evidence that the moves are strategically interchangeable. The hash control uses only audited actor features and semantic actions, with a fixed public salt. Its intervention frequency is not matched to the learned policies by construction.

For the corresponding tie-only and quarter-strength policies, **1,014 / 1,024**, **1,021 / 1,024**, and **1,009 / 1,024** paired games had identical canonical event streams for seeds 011, 012 and 013 respectively. The correction's selected move differed from its own same-state tie-only companion on exactly 10, 3 and 15 decisions. This explains why their game outcomes and pooled estimates are so similar. It does not show that stronger positive-margin corrections would be safe or useful.

All arms preserved the same first-legal-win wrapper. The focal wrapper changed no raw/corrected action in the observed formal trajectories. This is a measured fact for this panel, not a universal statement about the original model or the strategic optimality of always accepting a win.

## Implemented changes and preserved learning state

- Added explicit learned exact-tie choice and a fixed nonlearned hash control, with strict float32 tie equality, canonical tie order and the original explorer eligibility mask (`q >= 0.02` or raw reference argmax).
- Added fixed lambda 0.25 correction, retaining Stage 3I's float64 clipped-prior arithmetic and matched win wrapper.
- Added fresh-game execution, complete provenance, separate tie/margin diagnostics, source archives, historical freshness checks and independent result verification.
- Reused all three selected Stage 3I head bundles and the complete original reference actor. **Zero optimizer updates, zero new training games, zero model promotions.** The experiment changes how saved predictions choose moves; it does not retrain the network.

The old one-hand target is observational. Better chosen-action prediction does not identify the value of an unchosen action. The Stage 3I fractional diagnosis motivated the fixed strength before these fresh games; its reused calibration results are not an independent validation set or a guarantee about future intervention frequency.

## Validation and resource accounting

The final frozen release passed **167 tests**, recorded in release-final.xml. A successful earlier release was strengthened with exact first-win propensity and audit-schema checks, then retested before games. Every tested source version is archived; there were no failed numerical operations or failed/replaced formal games.

The independent [evidence audit](audit.json) verified:

- All **5,215,102 decisions**, **5,161,690 response groups**, complete-match terminal score/rank reconstruction, semantic/physical action identities and per-seat policy provenance across all 8,192 formal games.
- **13,312 paired common-ordinal physical-wall checks**, including dealer continuations. All 256 new first-hand wall commitments differ from the 1,057 available historical first-wall commitments in the frozen history inventory. The 37,199 historical seed values and 11,544 source files were authenticated. This seed set conservatively includes other recorded model/RNG seeds too.
- All fixed **256 original-actor forwards**, **192 action-head forwards**, **256 explorer-probability reconstructions** and **32 hash-control fingerprints**. These are the first optional focal positions from eight fixed blocks, so this forward sample emphasizes early-game states. No expected sample was missing.
- All **32 stratified full-game replays** matched exact deterministic payloads, including canonical games and recorded action artifacts; see replays.json.
- Independent reconstruction of all thirteen descriptive contrasts and all four primary contrasts. The [separate primary summary audit](summary-audit.json) agreed within `5.56e-17` absolute difference, well inside its `1e-12` tolerance.

This is same-engine reproducibility and evidence auditing. Walls were generated by the pinned engine and then checked through explicit physical commitments. It is not independent-engine rule certification or external physical-wall injection.

Nine successful numerical operations used **609.037689 of 5,400 seconds**, including preflight, games, tests, analysis and audit. This is charged numerical runtime; implementation, documentation and hash-only packaging are separate from that clock. There were **8,288 complete-match executions**: 64 preflight + 8,192 formal + 32 replay. Formal game execution took **563.647515 seconds**, or **14.533906 matches/s**, with full audit recording. The operation's wider charged time includes setup and monitoring. Peak sampled process-tree RSS was **5,621,809,152 bytes** (5.622 GB), below 32 GiB.

The full cold-started preflight gave a conservative formal forecast of 1,247.914 seconds and 3.466 GB. All numerical work used the project environment: Python 3.14.8, PyTorch 2.14.1, NumPy 2.5.4, RiichiEnv 0.4.10, mahjong 2.0.0, pytest 9.1.1 and psutil 7.2.2. Games used fourteen CPU workers with single-thread, batch-one model inference. Existing MPS/CUDA-capable loading remains available; this stage adds no new GPU speed comparison.

The [closed accounting](accounting.json), file inventory, inventory verification, and historical preservation bind the completed evidence. Bulk game data and archived sources are stored in `/Users/example/Library/Application Support/RiichiAI/stage3j`, outside iCloud. Unused numerical allowance remains unused.

## Conclusion for the next decision

Keep the frozen-reference architecture, paired evaluation and nonlearned tie control. Retain the saved heads as research artifacts, with no deployment change. The unresolved issue is action-ranking signal rather than whether the models can produce nonzero corrections. A focused programme with prospectively randomized tie decisions and explicit action-dependent continuation targets is more informative than another epoch increase or lambda sweep. The [review](../../docs/STAGE3J_REVIEW.md) describes that proposed direction and its evidence requirements. No next-stage experiment has started, and book scenarios remain quarantined.
