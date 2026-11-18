# Stage 3I results — frozen-reference residual learning

22 October 2026. **Complete. Three longer-trained residual models passed predictive checks; none entered formal games because the frozen correction grid selected zero. No policy is promoted.**

The control games strongly support retaining the original actor: the old reset initialization lost roughly one average-rank position on the matched diagnostic subset. A post-hoc saved-state diagnosis identifies exact reference-score ties and a coarse correction grid as the next issue to test. Better prediction has not yet demonstrated better play.

## What ran

- Three fixed seeds; twelve full passes for each of V (value), A (action residual) and B (state-only residual).
- 49,788 optimizer updates and 50,935,716 head-row presentations from **471,627 distinct recorded training decisions** in 3,200 matches. Presentations count repeated exposures, not new independent data.
- Complete frozen features for 590,120 optional explorer positions across all 4,000 archived matches; every reconstructed collection probability vector matched exactly.
- Internal development split: 200 selection matches / 29,647 rows; 200 calibration / 29,712 rows; 400 predictive gate / 59,134 rows. These are reused development data.
- 4,272 fresh formal control games, 56 preflight games and 32 exact replays: **4,360 complete match executions**. No new training games.
- CPU formal training after real CPU/MPS comparison; full training took 77.786 seconds. The complete actor remained frozen, including embeddings and scorer.

## Prediction results

Lower mean squared error is better. A residual is not an identified causal action advantage. The gate requires V+A to beat both V and V+B.

| Seed suffix | V MSE | V+B MSE | V+A MSE | Reduction vs V | Selected epochs V / A / B |
|---|---:|---:|---:|---:|---|
| 011 | 0.138552 | 0.138411 | 0.132724 | 4.21% | 11 / 12 / 6 |
| 012 | 0.138485 | 0.138497 | 0.132815 | 4.09% | 12 / 5 / 11 |
| 013 | 0.138449 | 0.138434 | 0.132529 | 4.28% | 12 / 12 / 12 |

All three pass, including improvement in the descriptive match-weighted errors. This establishes useful predictive signal under this target/distribution; fresh candidate games are still required. The one-hand target includes earlier events in the same hand and follows recorded explorer continuation.

## Why the learned corrections did not play

The prespecified grid was 0, 1, 2, 4, …, 1024. Its smallest positive value already exceeded the 3% overall change ceiling for every seed. Larger values also failed. The valid outcome was lambda zero, leaving the wrapped reference unchanged.

| Seed suffix | Changed at lambda 1 | Rate | Maximum allowed count | Formal lambda |
|---|---:|---:|---:|---:|
| 011 | 1152 / 29712 | 3.8772% | 891 | 0 |
| 012 | 892 / 29712 | 3.0022% | 891 | 0 |
| 013 | 1215 / 29712 | 4.0893% | 891 | 0 |

The second seed exceeds the allowed count by one decision. We did not round it into eligibility, increase the ceiling, choose a seed from game results, or replace the formal grid after seeing the result. **There are zero fresh formal games with a learned Stage 3I correction.**

## Fresh control games

Lower mean rank is better. Each comparison uses common realized physical-tile wall commitments and all four seat rotations. Reset controls use the fixed first 64 blocks; the full reference comparison uses 470.

| Comparison | Focal mean rank | Matched control rank | Utility difference | Exploratory block-bootstrap 95% interval |
|---|---:|---:|---:|---|
| reference wrapped vs reference raw | 2.529255 | 2.529255 | +0.000000 | [+0.000000, +0.000000] |
| reset raw vs reference raw | 3.367188 | 2.339844 | -1.027344 | [-1.203125, -0.843750] |
| reset wrapped vs reference wrapped | 3.312500 | 2.339844 | -0.972656 | [-1.144531, -0.796875] |
| reset wrapped vs reset raw | 3.312500 | 3.367188 | +0.054688 | [+0.007812, +0.105469] |

All 1,880 raw/wrapped reference pairs have identical canonical game event streams, not merely identical placements. This is a result for this panel, not a universal statement that the win rule cannot affect the reference. The reference mean of 2.529255 on all 470 blocks must not be compared directly with the 2.339844 mean on the 64-block reset subset.

The reset-raw deficit is 1.027344 average-rank units; reset-wrapped deficit is 0.972656. The win wrapper helps the reset policy modestly on this subset but does not repair its deficit. These are fixed initialization/wrapper diagnostics, not a full causal explanation of every historical training failure.

Bootstrap intervals are descriptive and use 5,000 whole-block replicates. Conservative bounded intervals are retained in evaluation.json; for example the reference-wrapper difference has bound [-0.375867, +0.375867] despite the zero empirical bootstrap interval. Secondary controls are not a promotion test.

## Post-hoc fractional diagnosis — no policy selected

Only saved calibration predictions were read; no model forward, optimizer update or new game ran. This analysis does not amend the original experiment. All seeds have 1,562 calibration positions with exact top-logit ties. These are ties in stored float32 model scores, not proof of equal Mahjong move quality.

| Seed suffix | Change as lambda approaches zero from above | Change at lambda 0.25 | Change at lambda 0.5 |
|---|---:|---:|---:|
| 011 | 2.8608% | 2.8642% | 3.0055% |
| 012 | 2.7868% | 2.7868% | 2.8002% |
| 013 | 2.9685% | 2.9752% | 3.1704% |

Small fractional settings can fit the original ceilings on this reused calibration sample. However, correction size is discontinuous at zero because residuals break exact reference ties. Simply shrinking lambda does not smoothly shrink the changed-decision fraction toward zero. The next protocol should distinguish learned tie-breaking from corrections that overturn a positive reference margin. Compare a learned tie-breaker with an explicit nonlearned tie control if attributing gains to the learned signal. No fractional value is selected or promoted here.

## Verification and resources

- Release suite: **120 passed**; independent auditor and saved-state diagnostic suite: **33 passed**. The final combined suite passed **153 tests**, saved in release-final.xml. Earlier passing runs are not added as unique tests.
- Independent audit checks all 4,000 feature shards and target/probability exposure, all 108 epoch checkpoints and selected bundles, all 1,800 calibration/gate prediction artifacts, and every formal game decision.
- 64 original-actor checks and 192 selected-head checks agree exactly. All 2,692,288 game decisions and 2,661,745 response groups pass the saved-evidence checks; 2,648 paired wall checks pass. All 32 fixed replays match their complete deterministic payloads.
- Numerical ledger: **480.631934 / 5,400 seconds** across 16 completed operations, zero failed numerical operations. This is monitored numerical time, not total engineering/conversation time. Remaining allowance stays unused.
- Peak sampled process-tree RSS: **6.265 GB**, below 32 GiB. Frozen features occupy 3,580,089,774 bytes. Final artifact inventory is in manifest.json.
- Historical preservation passed for 682 Stage 3H files and all 20,000 declared original corpus artifacts. Original reference/call checkpoints, targets and proposals remain unchanged. Books remain quarantined.
- Environment: Python 3.14.8; Torch 2.14.1; NumPy 2.5.4; RiichiEnv 0.4.10; mahjong 2.0.0; pytest 9.1.1; psutil 7.2.2. Project .venv and pinned dependency snapshot used throughout.
- One preflight admission was rejected by source hashing before any numerical subprocess started; final sources were frozen and release tests rerun. No formal attempt was retried or replaced. Five source files superseded after the first successful preflight test retain their recorded hashes but lack archived old bytes; all final-release and formal execution sources are retained.
- Same-engine reproducibility and inspected rules are verified here; this is not independent full-rules certification, expert strength or world-class performance.

## Files to review

- [Review and recommended next decision](../../docs/STAGE3I_REVIEW.md)
- [Frozen executable plan](../../docs/STAGE3I_PLAN.md)
- [Compact measured summary and model paths](summary.json)
- Independent audit, replays, [accounting](accounting.json)
- Formal calibration, formal games, [post-hoc fractional diagnosis](posthoc-fractional-diagnostic.json)
- Measured workload admission, historical preservation

Bulk cache, epoch checkpoints, selected heads and complete game logs remain under `/Users/example/Library/Application Support/RiichiAI/stage3i`. The selected checkpoint for each fixed seed is recorded with a SHA-256 in `summary.json`. No original policy is replaced.
