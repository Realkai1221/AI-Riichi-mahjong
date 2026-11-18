# Stage 3J: small corrections and learned tie-breaking on fresh games

29 October 2026. Authorized by the user's request to continue with smaller corrections and learned tie-breaking using fresh games. This is a new, fixed development experiment. Stage 3I and its original zero-correction decision remain historical evidence.

## Question and fixed policies

Do the three existing action-residual heads improve complete-game average rank when allowed to break exact reference-score ties? Does adding a small positive-margin correction help beyond tie-breaking? Is any benefit attributable to the learned ranking rather than merely changing the reference's canonical tie order?

Use the unchanged Stage 2B reference, unchanged Stage 3C initial call heads for explorer propensities, and all three completed Stage 3I selected bundles (seeds 20261024011, 20261024012, 20261024013). Preserve their original byte identities, frozen parameters and Stage 3I protocol/cache identities. No new training, optimizer steps, target fitting, checkpoint selection or model scaling occurs here. The quantity learned from observational one-hand outcome differences is not an identified causal action advantage.

Eight fixed focal policies:

1. `reference_wrapped`: original raw argmax, with the existing first-legal-win wrapper.
2. `hash_tie`: fixed nonlearned hash ordering within eligible exact raw-top-score ties, with the same wrapper.
3. Three `tie_seed_*` policies: maximize that seed's saved action residual within eligible exact raw-top-score ties; canonical first index breaks remaining ties.
4. Three `small_seed_*` policies: maximize `max(z - max(z), -8) + 0.25 * A` within the same eligibility restriction; canonical first index breaks ties. Compute correction arithmetic in float64 from actual float32 reference logits and head outputs.

Eligibility is the actual historical explorer marginal `q >= 0.02`, inclusive, or the original raw argmax. The original choice therefore always remains available. Every policy uses the same first-legal-win wrapper. Exact ties mean equality of actual float32 reference outputs, never a tolerance or rounded display value. They do not establish strategic equality. Tie-only policies cannot override a positive reference-score margin. Every choice records its eligible set, original and wrapped reference choices, exact tie set, selected reference-score gap, and classification as unchanged, exact-tie change or positive-margin change. A single legal action remains unchanged.

The hash control uses the fixed salt `riichi-ai-stage3j-hash-tie-v1`, a canonical SHA-256 fingerprint of precisely the audited `encode_view` actor tensors, and canonical semantic-action keys. It receives no match seed, full simulator state, hidden tiles, wall commitment, seat-policy provenance or auditor-only view hash. It is deterministic across replay and varies with visible state. Its intervention frequency is measured rather than forced to match learned policies. Report this limitation when interpreting learned-versus-hash comparisons.

The fixed strength 0.25 is motivated by the explicitly post-hoc Stage 3I saved-calibration diagnosis: it passed the original 3% overall and 10% reference-discard ceilings for all three seeds. Tie-only choices correspond to that diagnosis's exact right-hand limit at zero. The old calibration is reused development evidence, not a fresh holdout. Verify archived identities and the reported eligibility results before games. Do not choose a new strength or drop a seed after game outcomes. Fresh on-policy trajectories can have different change rates; the old ceilings are not guarantees or online stopping rules.

## Fresh game design

Use 256 fixed wall blocks, seeds `202610290000 + i` for `i=0..255`. Each block has all four focal-seat rotations and all eight arms: **8,192 complete formal matches**. Opponents, clockwise from the focal seat, are `shanten_v2`, `defense_v2`, and the unchanged raw reference. The primary reference is wrapper-matched. Do not add reset controls, a raw-reference focal arm, extra seeds, adaptive collection or a best-seed rerun.

The engineering preflight uses two separate blocks, seeds `202610289000` and `202610289001`, with the identical eight-arm/four-seat structure: 64 complete matches. These never enter formal estimates. Before either panel, check proposed seeds against the recorded historical seed inventories and preserve the hashed input sources and union checked. The freshness claim is relative to recorded local research history, not all possible wall sequences. Distinct seeds do not prove distinct physical walls; compare actual first-hand commitments with historical game evidence where available and check all common ordinal physical-wall commitments across arms within each block/rotation.

Save each complete canonical event stream, semantic/physical action mapping, all per-seat behavior identities, complete-match ranks/utilities, point conservation and explicit hand ordinals, including dealer continuations. Every simultaneous response is derived from snapshots committed before any response is submitted. All records are evaluation-only and ineligible for PPO/training. The objective remains `[1.5, 0.5, -0.5, -1.5]`, equivalent to `2.5 - rank`.

The pinned engine generates walls from the fixed seeds; auditors then compare explicit physical-tile commitments for common hand ordinals. This is matched within-engine evaluation, not external explicit-wall injection or independent differential rule certification. Retain full wall evidence even when game lengths or dealer sequences diverge.

## Fixed analysis and interpretation

The independent sampling unit is the wall block: **256 units**, not 8,192 independent games. First average the four focal-seat utilities within each arm/block. Average the three saved learned seeds within each block for pooled policies. Shared controls and shared walls induce dependence across policies, seats and seeds.

Four preregistered pooled contrasts, with positive utility difference meaning better average rank:

1. Learned tie-breaking minus wrapped reference.
2. Learned tie-breaking minus hash tie-breaking.
3. Quarter-strength correction minus wrapped reference.
4. Quarter-strength correction minus learned tie-breaking.

Jointly resample complete blocks 10,000 times with NumPy generator seed `20261029031`. Report means, block standard deviations, standard errors, and percentile intervals at 98.75% for each of the four contrasts (Bonferroni allocation of a nominal 5% family error). These bootstrap intervals are approximate descriptive uncertainty, not finite-sample certification. Also report simultaneous 95% Hoeffding bounds using range width 6 and family size 4: half-width `6 * sqrt(log(2*4/0.05)/(2*256))`, clipped to [-3,3]. Per-seed results and other contrasts are descriptive, with ordinary 95% block bootstrap intervals clearly distinguished from the primary family. Do not select the best seed or reinterpret outcome-selected subgroups as confirmatory tests.

Report per-arm ranks/placement counts and policy-intervention counts with explicit denominators. Distinguish changes from the reference on the candidate's visited states from paired counterfactual changes on shared states. Report exact-tie and positive-margin overrides separately; compare exact canonical-game identity between corresponding tie/small policies. Sparse interventions and 256 independent blocks may leave small effects unresolved. No statistical power claim is made without a justified effect size and variance estimate. No policy promotion, public-play strength claim, book supervision or world-class claim follows from this experiment.

## Resource limits and measured admission

Use project `.venv`, pinned dependencies and fourteen CPU workers with single-thread, batch-one reference inference. The same tiny frozen heads retain portable CPU/MPS/CUDA loading; no architecture/backend performance claim is added. A new exclusive Stage 3J ledger charges all tests, model/engine work and numerical/statistical analysis. Never borrow unspent Stage 3I time.

- Total numerical ceiling: **5,400 seconds**; preflight 600, evaluation 3,600, audit/report 900. Preserve 150 seconds for shutdown and future-phase reservations before launch.
- Maximum 9,000 attempted complete-match executions, including failed starts, preflight, tests that use real games and replay. Planned games: 64 preflight + 8,192 formal + 32 replay = **8,288**. No automatic formal retry/replacement.
- At most 32 GiB sampled process-tree RSS, 8 GB new artifacts and 10 GB free-disk reserve. Bulk data goes to local Application Support outside iCloud.
- Before formal games, forecast time and storage from the complete cold-started 64-game preflight: 1.5 times measured per-game time/bytes scaled to 8,192, plus 30 seconds for time and 500 MB for remaining artifacts. Admit only if the time forecast fits a 3,500-second cooperative formal allowance and storage fits the stage cap; otherwise stop and report. No outcome-dependent panel reduction.
- Every numerical receipt records versions, exact source hashes, deadline, sampled RSS, attempted-game reservation and outcome. Preserve original source bytes for every tested release before edits. Failed formal work stops the campaign; audit and reporting may continue within reserves.

## Independent audit and deliverables

Independently reconstruct all saved choices without calling the production choice function; verify every row/request, terminal result, hash, policy provenance and all paired common-ordinal walls. Verify saved-bundle identities and historical-stage dependencies. Save the first optional focal decision's actor tensors in blocks `[0,32,64,96,128,160,192,224]` for each seat/arm, at most 256 rows. Independently re-forward the original actor and saved heads, reconstruct explorer probabilities and hash-control fingerprints on this fixed sample. If a game lacks such a decision, report missing coverage rather than substitute a favorable sample.

Choose four replay games per arm, sorted by SHA-256 of `stage3j-audit-replay:` plus match ID, ties by match ID. Execute those 32 complete matches again and require exact deterministic payload identity. Replays are integrity checks, not new statistical observations.

Deliver the source/tests, frozen machine-readable plan, all preflight/admission receipts, complete fresh-game results, independent audit/replay reports, final source/artifact inventory, plain-English `reports/stage3j/REPORT.md` and `docs/STAGE3J_REVIEW.md`. Update implementation status and decision log factually. Preserve all historical proposals, Chinese translation, models, reports and quarantined book material.
