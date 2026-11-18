# Stage 3F engineering results

Completed 22 October 2026. The saved-training diagnosis, parallel exploring collector, bounded-memory loader and CPU/MPS computation probe are complete. **No policy optimization, persistent model checkpoint writes, large training corpus or promotion occurred.** These results establish an engineering route to the next learning experiment, not stronger Mahjong play.

Read the [review and proposed Stage 3G experiment](../../docs/STAGE3F_REVIEW.md), [adopted programme](../../docs/STAGE3F_PROGRAM.md), and [diagnostic review](../../docs/STAGE3F_DIAGNOSIS_REVIEW.md). The external proposal is preserved unchanged.

## Completed scope and resources

| Item | Measured result |
|---|---|
| Numerical budget | **186.193133 / 7,200 seconds**, including tests, imports, model loading, collection, backend probes and audits. Remaining 7,013.806867 seconds is unused. |
| Diagnosis sub-budget | **6.247958 / 300 seconds** across its three operations. |
| Operations | **26 closed: 25 completed, one failed development preflight preserved.** No active lease, exceeded reservation or failed formal match. |
| Formal games | **1,160 completed executions**: 32 parity executions, 128 worker-probe matches, 1,000 pilot matches. |
| Full-match audit replays | **10 completed**, making **1,170 full-match executions**. No replacements, censored games or unclosed formal starts. |
| Fixture accounting | Two test suites also ran 12 partial/rejected fixture lifecycle attempts, ten reaching engine initialization and none completing a match. Conservative total **1,182 / 1,200** admitted attempts. Their temporary files are not in the final corpus; source and JUnit records remain. |
| Peak sampled process-tree RSS | **5,615,222,784 bytes / 5.615 GB**, during the 14-worker pilot, below 32 GiB. Sampling is not an OS-enforced hard memory limit. |
| Recorded artifact peak before packaging | **335,965,726 bytes**, below the 2 GB cap. Final inventory and document bytes are recorded separately in the final manifest. |
| Historical evidence | All **3,121 Stage 3E snapshot files** verified before shared documentation changed. Original reference and initialized call checkpoint hashes remain pinned and unchanged. |

The numerical time is charged subprocess wall time; it does not include source development, document writing or standard-library hash inventory work. Completion of the fixed engineering scope, rather than exhaustion of the allowance, determines the endpoint.

The failed scheduling preflight had five failing assertions caused by a test expectation/API mismatch: it expected a failed formal phase to prohibit an arbitrary cleanup operation, while the controller intentionally prohibits later formal phases and allows audits. The tests were corrected before the plan/source freeze; the original failing receipt, XML and source version remain. The source-bound release suite subsequently passed **97 tests**. Diagnosis tests passed **47**, and the independent corpus-auditor tests passed **41**. Earlier passing suites overlap these tests and are not additional unique coverage. Eight inherited JUnit property warnings in the release suite are retained.

## Diagnosis: useful learning signal was sparse

The frozen diagnosis used saved Stage 3E training records only. It reconstructed the first complete 16-match batch, full eligible-row loss, head gradients, clipping and saved Adam movement. The largest parameter discrepancy in the Adam formula check was **3.73e-9**. No optimizer step was taken.

Gate gradients had a full-batch norm of **0.652060**, versus **2.061298** for the sum of per-match norms, a cancellation ratio of **0.3163**. Eight of sixteen match contributions opposed the full-batch gate direction. Conditional-call gradient was meaningful in only **two of sixteen matches**; the other fourteen were at most **1.09e-18**. Historical batch 13's conditional gradient norm of **1.94e-17** is numerical residue despite being technically nonzero. Only **3/28** updates clipped, so continuous clipping does not explain the small changes.

On 256 prespecified training diagnostic rows, the mean selected call-mass change from update 4 to 32 was about **-0.00007458**; the maximum per-action change was **0.00017698**. Conditional-head probability changes were further attenuated by the approximately 5% call gate. This is a narrow mechanism diagnosis, not a population signal-to-noise estimate or causal ablation.

The two PON-instead-of-RON cases occurred in East 1 and South 1 with visible context and exact sampled draws recorded. Their eventual results do not establish whether declining the win was strategically optimal. The new collector's always-take-a-legal-win rule is an explicit data-generation restriction, not a universal optimality theorem.

## Actual collection throughput

Each table contains one `shanten_v2`, one `defense_v2`, the original-reference argmax player and one explorer, with a seeded seat permutation. The explorer mixes 5% semantic-uniform exploration with 95% of the preserved initialized call-gated base on non-win decisions. Store the marginal probability of the selected action as well as the latent component, conditional probability and actual draws. Each seat has its own RNG stream. All simultaneous views are constructed before selecting any responses.

| Workers | Distinct probe matches | Complete collector seconds | Matches/s |
|---:|---:|---:|---:|
| 1 | 32 | 18.628 | 1.718 |
| 4 | 32 | 7.272 | 4.401 |
| 8 | 32 | 4.624 | 6.920 |
| 14 | 32 | 3.495 | 9.155 |
| **14, fixed pilot** | **1,000** | **86.915** | **11.505** |

The probes used disjoint fixed match plans, so these short timings are operational selection measurements, not a controlled estimate of pure parallel scaling. The main pilot amortizes worker setup more effectively. The pre-pilot admission rule selected fourteen workers and used a twofold time/storage allowance; it did not inspect match outcomes to choose or extend the sample.

Sixteen identical plans run serially and in parallel produced **identical deterministic payloads**, including actor arrays, targets, behavior records and auditor logs. Execution timing and process metadata were excluded from that identity check.

## Pilot data and useful coverage

The 1,000-match pilot contains **646,377 decisions**, **9,881 explicit hand starts** and **2,798 dealer continuations**. Continuations appear in 916 matches; the longest match has twenty hands. Hand identity comes from explicit starts and ordinals, including repeated round/dealer states.

| Measure | All four seats | Explorer only |
|---|---:|---:|
| Decisions | 646,377 | 160,721 |
| Optional semantic decisions | 591,871 | 149,235 |
| Forced semantic decisions | 54,506 | 11,486 |
| Positions offering a call/kan | 132,993 | 32,663 |
| Positions with multiple call candidates | 18,794 | 4,599 |
| Requested calls/kans | 2,333 | 2,333 |
| Resolved calls/kans | 2,316 | 2,316 |
| Selected call among multiple call candidates | 364 | 364 |
| Such calls resolved | 361 | 361 |
| Decisions after a call/kan | 19,204 | 19,204 |
| Open-hand decisions | 19,098 | 19,098 |
| History prefix exceeds the 128-event actor window | 21,823 | 5,458 |

Resolved calls are **1,431 CHI, 836 PON, 23 DAIMINKAN, 24 ANKAN and 2 KAKAN**. All 49 recorded kan declarations reconcile with replacement-draw completion; no unresolved declaration remains in the pilot. Seventeen requested CHI/PON actions were not resolved; requests are not falsely counted as accepted calls. Rare interrupted/robbed-kan paths are not certified by their absence here.

The explorer's mean actual call mass across call opportunities is **0.0733566**, including 44 win-versus-call positions where its declared win rule assigns call mass zero. All **1,802 explorer win opportunities** assigned the chosen win probability one. Its smallest selected action probability was **0.00333333**. It has full positive semantic support on **147,433 optional rows**; the other 1,802 optional rows are the deterministic win cases. Full per-behavior action families, propensity histograms and opportunities are in pilot-audit.json.

Only the explorer called. The other three behaviors are narrow closed-hand opponents in this pilot. Positive call support and larger sample counts do not establish useful action values or strong play against sophisticated calling opponents. Two KAKAN examples are especially sparse.

All data is explicitly **development-only, training-ineligible and PPO-ineligible**. Complete-match target utilities remain `[1.5, 0.5, -0.5, -1.5]`, including outcomes after a seat's last decision. Actor arrays are separate from terminal targets, behavior provenance and complete auditor logs. The loader reconstructs only the recorded causal history prefix within each seat/hand segment; it cannot expose the stored future history suffix to a row.

## Storage, loader and backend measurements

The pilot's match artifacts occupy **278,303,165 bytes** compressed. Actor and target numerical arrays total **2,476,377,299 bytes** uncompressed. The compressed number includes more than numerical arrays, so their ratio is not a pure array-compression ratio. Manifests, lifecycle records and match reports add overhead.

The loader keeps one bounded match shard resident and performs deterministic match-local row shuffling, with a manifest-bound resume cursor. It rejects malformed/oversized arrays, object dtypes, inconsistent targets and invalid history prefixes. Its numerical shard bound is 64 MiB and working-array allowance 256 MiB, separate from model/backend/Python memory.

The loader-only probe opened 28 shards and processed **16,623 rows** in **0.113737 seconds**, about **146,153 rows/s**, with **5,464,971 bytes** peak working arrays. This is a short local measurement with filesystem caching effects, not sustained cold-disk throughput.

The computation probe uses the real 990,421-parameter reference architecture, a freshly initialized scalar candidate scorer, gradients through 957,009 actor/head parameters and an unused frozen rank head. It computes terminal-utility MSE plus alpha-one legal-action conservative penalty. There is **no optimizer and no changed model weight**.

| Device | Rows / batches | Loader + transfer + forward/backward seconds | Rows/s |
|---|---:|---:|---:|
| CPU, one thread | 3,605 / 16 | 3.319376 | 1,086.05 |
| MPS | 3,605 / 16 | 1.096958 | 3,286.36 |
| CUDA | Unavailable locally | Not measured | Not measured |

On the first common batch, CPU/MPS loss matched at the recorded precision, maximum legal-value difference was **2.086e-7**, and relative gradient L2 difference was **3.683e-7**, within predeclared tolerances. All later probe losses were finite. The batch comparison does not establish all-batch gradient equivalence, Adam parity or long-run resume stability. The complete probe sampled **717,307,904 bytes RSS**; model/backend allocation is outside the loader's array bound. No hardware-utilization percentage is inferred.

The future explorer-only, match-weighted, optimizer-running workload differs from this all-seat probe. Benchmark that actual path before freezing its budget, retaining CPU fallback and portable CUDA code. The observed MPS speed is encouraging, not a promise of a threefold gain on every future workload.

## Independent checks and limitations

The separately implemented auditor checked **1,144 unique complete matches / 737,994 rows** across the pilot, serial parity corpus and all four worker probes. Checks cover artifact identity, actual marginal/component distributions, RNG draws and end states, selected-action membership in the recorded legal-action set, seat behavior identity, simultaneous-response commitments, actor/target alignment, explicit hand starts, score/rank reconstruction and requested/resolved call accounting. The duplicate parallel corpus is covered by full deterministic-payload parity.

The fixed ten pilot replays used positions `0,100,...,900`: **6,669 rows, 1% of matches and 1.03175% of pilot rows**. All complete replay payloads matched exactly. Bulk audit validates stored distributions and RNG; it does not rerun neural inference on every row. The ten complete replays rerun policies and engine evolution on their prespecified matches. There was no adaptive replay selection or replacement.

These are same-engine consistency and information-boundary checks, **not independent full-rule certification**. Existing targeted call/kan fixtures remain useful but do not certify every rare rule. The source-bound seed collision screen covered integer seed fields in project configurations and historical `reports/**/plan.json`; it was not an exhaustive scan of every historical trajectory file.

Versions: Python **3.14.8**, PyTorch **2.14.1**, NumPy **2.5.4**, RiichiEnv **0.4.10**, mahjong **2.0.0**, psutil **7.2.2**, pytest **9.1.1**, macOS ARM64. The project `.venv` was used. The formal collector/loader sources were frozen at commit `f98932dc6a85cbdc68da054462a6b73cda15af42`; explicit source hashes, not the commit label alone, bind each operation. The later independent auditor is bound separately in audit-sources.json.

The unchanged reference checkpoint SHA-256 is `5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f`; the initialized call checkpoint is `3baca74ddcb011ea7346396621c709e7499835c2cf4b9e6e201688aa52e75bf2`. Final file membership and hashes are in manifest.json, checked by inventory-verification.json. Bulk artifacts remain outside iCloud at `~/Library/Application Support/RiichiAI/stage3f`.

No new book extraction or training occurred in this milestone. Existing supplied PDFs, corrected partial scenarios and two validated partial call packets remain preserved and quarantined. Missing state, meld/own-turn-kan schema coverage and legal-action reconstruction remain the next book-track work.

## Resulting next decision

Keep the value-learning direction, parallel collector and streaming loader. Use explorer-only optional rows first to avoid silently mixing continuation policies and opponent populations. The [Stage 3G proposal](../../docs/STAGE3F_REVIEW.md) specifies a fresh 4,000-match corpus, four controlled target/regularizer arms and a separate development game panel, conditional on placement calibration and a real optimizer/backend preflight. It has not started. Large multi-round training and competitive-strength confirmation remain separate milestones.
