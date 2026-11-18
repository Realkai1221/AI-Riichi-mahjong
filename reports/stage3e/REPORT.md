# Stage 3E measured results — 21 October 2026

The fixed longer-horizon experiment completed **448 training matches and 28 additional updates**, followed by **256 fresh evaluation matches**. The saved update 4 model and optimizer continued to update 32. Both fixed endpoints averaged **2.5625 place**, with identical actions, scores and placements in all 128 paired games. The measured candidate-minus-baseline utility difference is **0**; the predeclared two-sided 95% interval is **[-1.44048418698, +1.44048418698]**. No model is promoted.

Read the [review and next-step recommendation](../../docs/STAGE3E_REVIEW.md) for interpretation. The [fixed plan](../../docs/STAGE3E_PLAN.md), [configuration](../../configs/stage3e.json) and preparation record preceded collection.

## Frozen recipe and identity

Training reused the unchanged Stage 3C update function: complete-match average-rank utility `[1.5,0.5,-0.5,-1.5]`, zero return baseline, AdamW learning rate 0.0001, zero weight decay, gradient norm clip 1, no entropy/shaping term and single-row exact behavior replay. Only the call gate and conditional call head trained. The base, rank head and three original argmax opponents remained fixed. Execution used CPU, one Torch thread, fp32 network evaluation and staged CPU fp64 probabilities.

| Identity | SHA-256 or value |
| --- | --- |
| Configuration | `f50c900beecf3fe06af3e35a7a53903e4bc6ecb5a61d609ec032e78d258c52b8` |
| Frozen plan | `b86672fba38d43db9a18952e89374f98caee1c83b0e5479a3d8913d861b3e36c` |
| Frozen implementation commit | `056323b9e950d90e1195457977ca5d3bbb9a8bdb` |
| Starting update 4 checkpoint | `4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967` |
| Final update 32 checkpoint | `f9bf99a3052ad112b858aa945804222ad4f27432b358a2ac8f11cc7caa74aef9` |
| Original reference/opponents | `5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f` |
| Historical behavior protocol | `1fb067d7b76a02eb286c9ab560678a78f8623cc6f59a75cbfa64016fe2936776` |

The new campaign identity is separate from the unchanged behavior protocol. Preparation verified the restored model, optimizer, four executed steps and retired-batch history against the original update 4 manifest. Training used absolute batch indices 4–31, with match IDs 576–1023. Evaluation used IDs 1024–1279. All 1,056 new seed values are unique and disjoint from Stage 3C/3D. Original inputs remain unchanged, and no prior evaluation was pooled or retried.

The final checkpoint is stored locally at `/Users/example/Library/Application Support/RiichiAI/stage3e/boundaries/batch-31.pt`. It is an experimental endpoint, not a new champion. All 28 new continuation boundaries, including update 32, are preserved. The historical Stage 3D 1,216-file snapshot verifies unchanged at commit `7d3380f2d9c595b4da6d99871ede8ccd92380608`; see verification.

## Completed training

| Measure | Result |
| --- | ---: |
| New complete matches / hands | 448 / 4,427 |
| Additional updates / final update number | 28 / 32 |
| Learner decisions | 68,435 |
| Optional / forced learner rows | 63,419 / 5,016 |
| Call-opportunity rows | 13,934 |
| Multiple-call opportunity rows / selected calls on those rows | 1,942 / 83 |
| Requested / resolved calls and kans | 715 / 710 |
| Post-call / open-hand learner rows | 5,834 / 5,741 |
| Maximum probability / selected-log replay error | 0 / 0 |
| No-op updates | 0 |

Every update had finite nonzero gradients and parameter changes in both new heads. The base and rank head stayed unchanged. Maximum absolute parameter change per update ranged from 0.00007210 to 0.00010590; maximum per-action probability movement measured on that batch's training states ranged from 0.00000351 to 0.00005945. These changing training populations are descriptive diagnostics, not independent strength measurements.

| Call kind | Requested | Resolved |
| --- | ---: | ---: |
| CHI | 406 | 404 |
| PON | 294 | 291 |
| DAIMINKAN | 4 | 4 |
| ANKAN | 9 | 9 |
| KAKAN | 2 | 2 |

Requests and engine resolutions remain separate. All five families occurred, but sparse kan counts do not certify every rare rule. Full actor observations were built from audited player-visible data; all simultaneous responses came from the same pre-response state. Complete logs and physical-wall commitments remained auditor-only inputs. Partial/evaluation matches are ineligible for optimization, and no opponent action entered the learner loss.

There were 16 training positions offering both a win and a call: 14 selected the win and 2 selected a call, with both calls resolved. Mean actual win probability was 0.95025998, versus 1.0 before applying the call gate. This motivates a separate review of call/win probability allocation; it does not by itself establish that always taking a legal win is optimal for average placement.

## Fresh fixed evaluation

| Measure | Update 32 candidate | Update 4 control |
| --- | ---: | ---: |
| Full matches / hands | 128 / 1,252 | 128 / 1,252 |
| Mean placement | 2.5625 | 2.5625 |
| First / second / third / fourth | 30 / 31 / 32 / 35 | 30 / 31 / 32 / 35 |
| Learner decisions | 19,081 | 19,081 |
| Call opportunities / multiple-call opportunities | 3,763 / 564 | 3,763 / 564 |
| Requested / resolved calls and kans | 193 / 189 | 193 / 189 |
| Post-call / open-hand learner rows | 1,565 / 1,540 | 1,565 / 1,540 |

Both arms requested 116 CHI, 71 PON, 1 DAIMINKAN, 4 ANKAN and 1 KAKAN; 112 CHI and every other requested call resolved. Both selected RON in their two win-versus-call positions. These identical counts result from identical paired action sequences, not from exact equality of the policy distributions.

All 3,763 call-opportunity distributions changed. Mean total call-mass change was **-0.00007769921274**, approximately **-0.00777 percentage points**; maximum absolute per-action change was **0.00017722151873**. All 15,318 non-call rows had exactly unchanged distributions. No shared random draw crossed a changed action boundary in this evaluation.

The endpoint averages four seat differences within each of 32 blocks; it does not treat 256 games as independent replicates. Every block difference was 0. The fixed Hoeffding half-width is `6*sqrt(log(40)/(2*32))`. Its wide interval does not establish equivalence or rule out a useful difference. No checkpoint was selected from these results. The lower absolute rank than the earlier Stage 3D population is not a matched improvement comparison.

See fixed endpoint and descriptive behavior analysis. The latter reads saved data only, with no new forwards, games or optimizer steps, and references rather than redefines the inferential endpoint.

## Validation and resources

The release preflight passed **89 tests**. Book validation passed **20 tests**. Post-run analysis/audit tests passed **86 tests**. Eight preflight warnings concern JUnit property formatting in inherited fixtures; no case failed or skipped. These are targeted engineering checks, not full rule certification.

The independent audit passed **5,818,945 data-level assertions** over all 704 saved matches, 28 updates and the checkpoint chain. It validates score/placement reconstruction, sampling and learner provenance, exact replay/objective evidence, source membership, ordinal physical walls, resource reservations and the fixed endpoint. A separate hash-only closure revalidated **3,033 inputs** with **3,607 integrity/bookkeeping checks** after the audit lease closed, without rerunning match arithmetic. Audit prefix, final audit.

The auditor explicitly distinguishes finite zero gradients from integrity failures: a batch need not select a multi-call action. Tests cover that valid possibility while still rejecting nonfinite diagnostics, dishonest change flags, wrong replay or objective reconstruction. In this experiment both heads actually had nonzero gradients and parameter changes in all 28 updates.

| Resource | Recorded result | Cap |
| --- | ---: | ---: |
| Total numerical work | **956.038680877 s** | 1,200 s |
| Preparation before training | 3.433158665 s | 60 s |
| 28 training batches | 610.023756335 s | Included above |
| 32 evaluation blocks | 316.106749711 s | Included above |
| Book tests + conversion | 0.502844999 s | Included above |
| Closed operations | 67 completed, 0 failed | One lease at a time |
| Sampled process-tree RSS peak | 1,592,901,632 bytes, about 1.484 GiB | 32 GiB |
| Artifacts at audit closure | 848,245,516 bytes, before final reporting/inventory | 1,500,000,000 bytes |
| Unused numerical allowance | 243.961319123 s | Includes 60 s shutdown reserve |

Every training admission retained 400 seconds for evaluation, 40 for audit and 60 for shutdown; evaluation admissions retained 100 seconds. No failed work was replaced and unused allowance was not spent on extra training or seeds. The ledger includes tests, imports, collection, optimization, analysis and the full audit. Source/report editing and hash-only packaging are administrative work outside numerical execution. RSS is sampled rather than a continuous machine-wide measurement.

Environment: Apple M5 Pro / 64 GB, macOS ARM64, project `.venv` Python 3.14.8, Torch 2.14.1, NumPy 2.5.4, RiichiEnv 0.4.10, mahjong 2.0.0, psutil 7.2.2, pytest 9.1.1. Dependencies remain pinned in [requirements-macos-arm64.lock](../../requirements-macos-arm64.lock). No MPS/CUDA speed or strength claim follows from this CPU run.

## Book and review artifacts

The new strict partial-call validator converted the two already reviewed packets, preserving their question content. It separates decline-call preference from conditional CHI choice, preserves missing state and copy ambiguity, and rejects answer-only dora or source metadata in observations. The earlier source files remain unchanged. Both converted records remain quarantined, with training/evaluation/full-policy eligibility false. No new extraction, book loss or actor adapter was added. Book review.

No external corpus/weights, account registration, messages, paid compute or public play were involved. The [review](../../docs/STAGE3E_REVIEW.md) recommends a bounded training-data learning-signal and win-versus-call diagnosis before another algorithm change; this is proposed and unstarted.

The final manifest and inventory verification bind the review bundle and external/local-book evidence. Recheck with `.venv/bin/python scripts/stage3e_inventory.py --verify`. Do not rerun numerical commands into these preserved campaign directories.
