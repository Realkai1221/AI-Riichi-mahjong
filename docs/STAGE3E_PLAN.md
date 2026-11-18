# Stage 3E — fixed longer learning horizon

The user authorized the longer fixed experiment proposed in [the Stage 3D review](STAGE3D_REVIEW.md) on 21 October 2026. Continue the original Stage 3C update-4 model **and AdamW state** through exactly 28 additional batches of 16 complete matches, ending at update 32. Compare only that endpoint against the starting update-4 checkpoint on 32 fresh paired blocks. No best-checkpoint selection, hyperparameter search or outcome-based extension is included.

## Fixed question and recipe

Does more on-policy experience produce useful call decisions under the existing average-rank objective? The intervention is duration alone. Keep AdamW learning rate 0.0001, zero weight decay, norm clip 1, a zero return baseline, no entropy/reward-shaping term, single-row exact probability replay, the original frozen base/rank head, and the same three original argmax opponents. Reuse the Stage 3C update function unchanged. No frozen-opponent actions enter the loss.

The source checkpoint is `stage3c/boundaries/batch-3.pt`, SHA-256 `4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967`. Preparation compares its restored full state with the saved update-4 manifest, including optimizer identity, retired batches and four executed steps. The original behavior protocol `1fb067d7b76a02eb286c9ab560678a78f8623cc6f59a75cbfa64016fe2936776` remains distinct from the new campaign hash.

Training uses absolute batch indices 4–31 and match IDs 576–1023. Evaluation uses IDs 1024–1279 and compares update 32 against update 4. All 448 training matches and 256 evaluation matches, seats and seeds are fixed in [stage3e.json](../configs/stage3e.json) before collection. SHA-256 namespace `riichi-ai-stage3e-20261021` generates 1,056 unique wall/sampling seeds, checked disjoint from Stage 3C and Stage 3D. No historical evaluation is pooled.

## Resource and failure boundaries

| Limit | Fixed value |
|---|---:|
| Cumulative numerical work | 1,200 seconds |
| Preparation before the first training batch | 60 seconds |
| Shutdown reserve | 60 seconds |
| Capacity protected at each training admission | 400 evaluation +40 audit +60 shutdown =500 seconds |
| Capacity protected at each evaluation admission | 40 audit +60 shutdown =100 seconds |
| Training batch reservation | 45 seconds, worker deadline 35 seconds |
| Evaluation block reservation | 35 seconds, worker deadline 25 seconds |
| Watchdog and cooperative shutdown | Reservation minus 5 seconds; 3-second grace before forced kill |
| Sampled process-tree RSS | 32 GiB |
| New artifacts | 1,500,000,000 bytes |
| Device and precision | CPU / one Torch thread / fp32 network / staged CPU fp64 probabilities |

The budget includes imports, tests, book validation/conversion, preparation, collection, training, analysis and the leased full saved-evidence audit. The final audit closure and inventory perform hashes and bookkeeping without repeating match arithmetic. The durable ledger allows one numerical operation at a time. Admission requires the whole declared reservation to fit beyond the applicable protected capacity; reserved capacity is a safeguard, not a guarantee that future games finish within it.

A failed/incomplete training batch cannot be replaced or skipped. Preserve its partial trajectories, attempted optimizer-step evidence and any boundary already written; stop the formal chain. All 28 training batches must close successfully before evaluation begins. A failed evaluation block similarly stops the comparison, with no fixed endpoint computed from a partial prefix. No policy is promoted automatically under any outcome.

## Integrity and validation

Freeze an explicit runtime/test membership list. The continuation collector preserves old policy decisions and simultaneous-response ordering while adding cooperative cancellation and synchronized lifecycle markers. Only completed training matches and their focal-player rows are eligible; partial/evaluation matches remain ineligible. Hand identifiers include dealer continuations. Actual ordinal physical-wall commitments are compared within paired games and across seat rotations.

The final preflight passed 89 tests, covering continuation state, old/new collector equivalence, cancellation and existing explicit-wall call fixtures. Eight pytest warnings concern historical JUnit property formatting, not failed or skipped tests. All original source files remain unchanged. The historical 1,216-file Stage 3D snapshot is verified from its completed Git commit and unchanged external/local-book artifacts.

Each batch loads the exact preceding boundary, collects 16 complete matches, replays their actual probabilities exactly, performs one unchanged heads-only update, saves once, synchronizes and reloads the boundary, and compares complete manifests. Record finite gradients, component changes, base/rank immutability, completed diagnostics and state changes even on failure. These checks establish engineering properties; training rewards and numerical parameter changes do not demonstrate strength.

## Fixed endpoint and descriptive diagnostics

For each evaluation block, average the four paired-seat utility differences using `[1.5,0.5,-0.5,-1.5]`. Report the mean over all 32 successful closed blocks and the same two-sided 95% Hoeffding interval, half-width `6*sqrt(log(40)/(2*32))`, clipped to [-3,3]. Do not count the four rotations as independent blocks or change the interval after seeing results.

Separately describe training updates, gate/conditional-call probability changes, requests/resolved calls, post-call observations, win-versus-call opportunities and complete paired action divergence. Compare probabilities only on common pre-response states up to and including the first divergent response. No additional inferential tests or selection are based on these descriptive figures.

This tests additional learning from update 4. It does not establish net gate benefit against the original no-gate policy or strength against diverse expert opponents. If behavioral change remains negligible, do not extend the same loop automatically; propose a separate credit-assignment or post-call competence intervention.

## Parallel book work

Validate the two already reviewed Stage 3D partial-call packets using a new strict schema. Separate call-versus-decline preference from conditional CHI choice; preserve unknown context, physical-copy ambiguity and explanation-only information boundaries. No new extraction, arithmetic, model evaluation or book loss is included. The local records remain quarantined and are not full policy observations. Book code and independent analysis/audit code are outside the frozen numerical runtime membership and are inventoried separately.
