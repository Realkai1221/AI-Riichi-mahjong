# Stage 3C measured report — 21 October 2026

**Training completed; evaluation stopped and remains incomplete.** The fixed call-learning pilot produced four valid updates from 64 complete matches. A newly added independent analysis test entered the controller's dynamic source-file set during evaluation block 8, and the watchdog stopped execution. Every originally pinned source file was unchanged. The failure was caused by our workflow coordination, not an observed game or model failure. No failed block was retried or replaced, and no policy is promoted.

Read the [review and proposed next step](../../docs/STAGE3C_REVIEW.md), [frozen plan](../../docs/STAGE3C_PLAN.md), stop evidence, and descriptive behavior data.

## Completed training

| Item | Measured result |
|---|---|
| Full matches / hands | 64 / 640 |
| Optimizer updates | Four, one per complete batch |
| Learner decisions | 10,010: 9,329 optional and 681 forced |
| Calls offered | 2,027 learner positions; 308 offer multiple call candidates |
| Selected calls | 103 requests; 100 declared and resolved; three preempted |
| Post-call states | 740 learner decisions with a meld present, across 50 matches; 736 open-hand decisions |
| Base and rank head | Byte-identical parameter identities at every saved boundary |
| Replay | Exactly zero probability-vector and selected-log-probability replay error in all batches |
| Head response | Finite, nonzero gate and conditional-head gradient and parameter changes in all four updates |
| Loss | Learner-only complete-match REINFORCE; no frozen-opponent actions in loss |

| Family | Opportunity positions | Requested | Declared | Resolved |
|---|---:|---:|---:|---:|
| CHI | 1206 | 64 | 63 | 63 |
| PON | 840 | 36 | 34 | 34 |
| DAIMINKAN | 40 | 0 | 0 | 0 |
| ANKAN | 23 | 2 | 2 | 2 |
| KAKAN | 3 | 1 | 1 | 1 |

Opportunity categories overlap. No DAIMINKAN was selected in this training sample, despite 40 opportunities; its genuine transition is covered by the preflight fixture. KAKAN has only one executed example. These are coverage counts, not strategic quality scores.

| Update | Optional learner rows | Gate gradient norm | Call-head gradient norm | Mean call mass after update on that batch's call rows | Replay error |
|---|---:|---:|---:|---:|---:|
| 1 | 2371 | 1.584445 | 0.073786 | 0.04993035 | 0 |
| 2 | 2250 | 0.466991 | 0.045623 | 0.04991145 | 0 |
| 3 | 2340 | 2.358018 | 0.062997 | 0.04986681 | 0 |
| 4 | 2368 | 0.908260 | 0.050806 | 0.04981479 | 0 |

The hidden layers correctly had zero gradient in the first update because their output weights were initialized at zero. Later hidden-layer gradients were nonzero. The mean call mass fell slightly from its initial value of 0.0500000021; comparisons across these batch means also reflect differing states, so they are not a matched-state learning curve.

Five training positions offered calls alongside RON. Their average available-win probability fell from 1.0 within the noncall group to approximately 0.95008 under the actual sampler. All five selected RON, and all 142 legal-win positions selected a win. This does not eliminate the sampler's possibility of declining a win; no hidden override was applied.

## Incomplete evaluation

Eight complete seed blocks contain 32 paired seat comparisons, 64 matches. All 32 pairs had identical complete action sequences, scores and placements. Both arms' descriptive mean rank was 2.5625. The trained heads had changed their probabilities slightly, but those changes did not alter the sampled actions in this completed prefix.

Block 8 was terminated by the source guard. Two further complete raw match records (128 and 129) were saved. Their block's final paired summary and state-after proof were not completed; they are excluded from the paired prefix statistics. Thus 130 full trajectories are preserved in total, with a stopped block and no replacement. Blocks 9–31 were not attempted. No fixed 32-block endpoint, confidence interval, strength improvement or equivalence conclusion is available.

The cause is recorded with the added file's SHA-256 in `evaluation-stop.json`. The independent analysis test was moved to `tests/test_call_behavior_analysis.py` without changing its contents, restoring the original monitored membership. This makes post-run audits possible; it does not erase the failed attempt or authorize resuming games.

## Validation and provenance

The final preflight passed 151 tests and 148 subtests, including eight genuine call/post-call cases and existing rules, visibility, efficiency, semantics and evaluation regressions. Four additional saved-data analysis tests passed. Initial preparation repeated a multiple-chi request 16 times with exact probabilities, direct gate/head gradient response and unchanged state. Its mean measured single-row time was 0.326ms. No smoke test is represented as full rule certification.

Implementation commit: `e23c78d41f89fdddac5f95e9a6b2d3d05cea0c22`. Config SHA-256: `1fb067d7b76a02eb286c9ab560678a78f8623cc6f59a75cbfa64016fe2936776`. The original reference checkpoint remains unchanged. Initial and four updated boundaries preserve model, optimizer, behavior identity and complete-batch provenance. Actor inputs remain audited player-visible features; full logs and physical-wall fingerprints are auditor-only. All simultaneous decisions use the same pre-response observations.

Environment: Python 3.14.8; Torch 2.14.1; NumPy 2.5.4; RiichiEnv 0.4.10; mahjong 2.0.0; psutil 7.2.2. CPU, one Torch thread, fp32 network and staged CPU fp64 probabilities. This run provides no new MPS/CUDA performance measurement.

## Resources, independent audit and files

The final ledger charges **181.375228 of 600 numerical seconds** across 23 closed operations, including one failed formal evaluation block. Sampled peak process-tree RSS was **0.631 GiB**. The independent saved-evidence audit passes **1,082,423 assertions**, with `valid=true`, `training_engineering_evidence_pass=true` and `experiment_complete=false`. It validates integrity and completed training while explicitly rejecting a completed evaluation claim. Final receipts are in budget.json, audit.json and the inventory. The cap remains 600 cumulative numerical seconds, 32 GiB sampled RSS and 1 GB new artifacts; failed execution and book rendering are charged. All formal work has stopped.

The final trained checkpoint is `/Users/example/Library/Application Support/RiichiAI/stage3c/boundaries/batch-3.pt`; the identical initialized control is `initial.pt` beside it. Raw complete-match evidence stays in that external Stage 3C folder; compact experiment reports are here. Book review and call-intake proposal are separate local preparation. No external corpus, paid compute, public play, book training or policy promotion occurred.
