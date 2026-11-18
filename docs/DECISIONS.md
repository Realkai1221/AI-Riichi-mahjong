# Decision log

## Completed learning-rate comparison and added book research

Completed 18 October 2026. The user accepted the proposed four-arm continuation and requested roughly one hour of work, then added book-extraction and Microsoft Suphx research as parallel investigations. The [accepted plan](PACKAGE_A_CONTINUATION_PLAN.md) preserved the 900-second numerical ceiling. All four formal arms completed 5,000 additional updates to 23,500, with no safety events or replacement attempts. Initial model/optimizer/RNG/safety identities and all 50 continuation sample prefixes match within each seed's pair.

The lower rate 0.0001 is not adopted: both seeds have worse optional-row smoothed-target KL on training and teacher validation than their 0.0003 controls, and lower full-validation teacher/student discard agreement. Seed 171 has fewer shanten-worsening errors at the lower rate, but that isolated benefit does not satisfy the predeclared componentwise criterion. Both lower-rate fixed-validation agreements miss 99.9%. These outcomes reject the tested late continuation as a demonstrated improvement, not every use of a lower learning rate.

The stage charged 487.465 of 900 numerical seconds. Its three targeted suites passed 150 tests and 25 subtests. All eight endpoint arrays and eight checkpoints are audited; prior diagnosis bytes remain verified through their historical snapshot. No test split, new match campaign, book-derived model update, external acquisition, paid compute or RL is included. The 99.9% threshold remains specific to the smoothing experiment and must not become an invented universal prerequisite for RL research.

The book assessment and [intake format](BOOK_SCENARIOS.md) separate printed recommendations, their assumptions and later strategic review. Three partial exercises were visually checked; two shanten/ukeire calculators agree on 31 candidate labels under explicit own-hand-only visibility. One source distinguishes future wait quality where the present immediate-efficiency target ties. Preserve that distinction without claiming a complete policy state or universal optimum. Actual books and raw extraction pilots remain local under `Books/`, outside Git.

The additional book review inventories all five PDFs and preserves six richer source packets with independent transcription checks. Keep source-specific acceptance-count conventions, drawn-tile identity and dora-versus-indicator meaning explicit. The local [OCR proof](BOOK_OCR.md) completed eight pages after a retained compiler/SDK mismatch; both attempts are charged. Horizontal prose is usable as a transcription aid, but vertical prose and tile identities require another method or manual review. Do not bulk-convert these outputs into training labels.

A separate count-convention probe retains all twelve printed candidate counts across the four Uzaku packets. Both calculators reproduce all twelve; the two declared visibility conventions also coincide on this sample. Authored tests demonstrate discarded-copy and indicator effects separately. No printed number, source recommendation or quarantine status was changed to obtain agreement.

The [Suphx assessment](SUPHX_ASSESSMENT.md) reviews primary sources and separates methods that can inform a local prototype from any unavailable model artifact or costly replication. The [current review](CONTINUATION_REVIEW.md) defines the next discussion; neither an additional optimizer recipe nor RL was launched from these findings.

## Completed Package A diagnosis and proposed next intervention

Completed 18 October 2026 after the user authorized the 600-second inference-only follow-up. The [diagnosis report](../reports/package_a_diagnosis/REPORT.md) covers seven frozen checkpoints, a 10,000-row training sample, full validation and 64 precision positions. All attempts completed, charging 29.920 seconds. No optimizer updates, new matches or test-split loading occurred. The focused suite passed 23 tests and 6 subtests; the final audit verified all 21 saved arrays and report arithmetic. The earlier Package A inventory remains unchanged and verifies against its committed snapshot.

Terminal smoothing assigns 7.25–7.45% non-best probability on optional training rows, versus approximately 1% target mass. This shows incomplete training fit in addition to validation transfer gaps; it does not distinguish optimization limits from parameterization. Selected-to-terminal aggregate validation KL improves while best-discard agreement and shanten-error severity worsen. Preserve both distribution-fit and decision-quality measures.

CPU32/MPS32/CPU64 comparisons changed no control or smoothed action on the 64 fixed positions. The original reference's changed actions all remained within teacher-best ties. Precision therefore does not explain the observed smoothed endpoint errors on tested states, without ruling out trajectory-level effects. Its large raw logits also show why absolute magnitude alone is not a universal collapse test. Do not retrospectively claim calibrated alarms or alter guard thresholds from these endpoints.

Nominate one next intervention: compare learning rate 0.0001 with 0.0003, continuing each smoothed terminal checkpoint with its own optimizer/RNG state. Keep epsilon 0.01 and other ingredients fixed. The [review proposal](PACKAGE_A_DIAGNOSIS_REVIEW.md) describes four attempts under a proposed 900-second numerical ceiling; no continuation has started. Neither smoothed model is promoted, the earlier retention non-pass stands, and RL remains unstarted.

## Package A results and retained decisions

Completed 17 October 2026. The [measured report](../reports/package_a/REPORT.md) closes the authorized package. The unchanged candidate's 319-block confirmation did not demonstrate retention: utility difference −0.0384, one-sided lower bound −0.1099 against the −0.10 margin. Preserve this fixed-count non-pass without extension or pilot pooling. It does not establish inferiority.

Both epsilon-0.01 rank-zero smoothing runs completed 18,500 updates without alarms, but terminal fixed-validation agreement was 95.9616% / 96.6094%, below the predeclared 99.9% requirement. Neither is promoted. The two controls stopped at 1,400 / 1,900 on raw-logit magnitude while policy losses and gradients remained moderate. Treat the guards as exploratory safety heuristics; stopping before historical collapse does not validate predictive lead time. Keep all four attempts and their selected/terminal artifacts.

Observed optional-validation non-best probability was 10.0991% / 9.7066% despite a target of at most 1%. The [review guide](PACKAGE_A_REVIEW_GUIDE.md) proposed a separately scoped 600-second inference-only diagnosis to distinguish training fit, generalization and action-relevant numerical scale. That subsequently authorized diagnosis is now complete, as recorded above. Calls expansion, placement calibration, RL, external data and paid compute remain separate work.

Final validation passed 251 tests and 165 subtests, plus 116 final audit checks. Charge the initial failed MPS bitwise assertion and its corrected tolerance-based check to A2; preserve both records. A1 used 297.445 of 600 seconds and A2 used 875.442 of 2,700 seconds. Prior snapshots remain verifiable without rewriting their inventories.

## Authorized follow-up: Package A

The user authorized the next stage after the [revision 2 assessment](stage2b/NEXT_REVIEW_REV2_ASSESSMENT.md). Execute [Package A](PACKAGE_A_PLAN.md): the unchanged primary's 319-block confirmation under a 600-second ceiling, then two instrumented rank-zero controls and two epsilon-0.01 smoothing runs under 2,700 cumulative training-related seconds and four formal attempts. Keep separate budgets and results; all failed/stopped attempts count. A2 requires terminal-step durability on both seeds, without convergence stopping. No new test-set tuning, RL, calls expansion, external activity or corpus collection is included.

The first A1 setup failed because the sandbox denied process enumeration; preserve its empty log and plan and charge a conservative five-second allowance. Retry with monitoring access using the same fixed model/seeds. No outcomes were inspected. Completed Stage 2B project bytes remain verifiable through their immutable Git snapshot; do not regenerate its old manifest after current documents evolve.

Recorded 16 October 2026. This log translates the proposed changes and their review into implementation choices. The English research blueprint is being revised; the proposal and original Chinese translation remain preserved historical references. Execution gates are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), with current progress in [STATUS.md](STATUS.md).

Status meanings: **Accepted** is a build requirement; **Modified** keeps the useful part with stated conditions; **Deferred** is a named experiment or external decision; **Rejected** applies to a particular claim/requirement, not to the whole proposal.

## Fixed starting decisions

| Decision | Status | Current choice |
|---|---|---|
| Hardware | Accepted | M5 Pro, 18 CPU cores, 20 GPU cores, 64 GB unified memory. Measure sustained behavior instead of assuming throughput from specifications. |
| Package/environment | Accepted | Python package `src/riichi_ai`; project `.venv` with Python `3.14.8`, PyTorch `2.14.1`, NumPy `2.5.4`, RiichiEnv `0.4.10`, and pytest `9.1.1`. Preserve [requirements-macos-arm64.lock](../requirements-macos-arm64.lock); revalidate compatibility when versions change. Rust/uv were initially absent. |
| Stage 0 scope | Accepted | Setup, synthetic smoke checks, simulator API audit, and modest benchmarks. No ML training run, external data acquisition, paid compute, external contact, or public bot activity yet. |
| Initial rules | Accepted | Explicit four-player, red-five, East–South profile. A library preset is an integration starting point; exact behavior still requires validation. |
| Primary objective | Accepted | Average-rank terminal utility `[1.5, 0.5, -0.5, -1.5]`. Report rank distribution and scores as diagnostics. |
| Target venue | Deferred | None selected. A ladder/tournament objective becomes a separately declared experimental target. |
| Source documentation | Accepted | Revise the English research blueprint to reflect reviewed decisions. Preserve the original Chinese translation and human proposal as historical references; use these two documents for current implementation decisions. |

## Stage 0 findings and adapter decisions

These are preliminary results, not full rule certification or evidence of playing strength. Detailed measurements are in the [Stage 0 report](../reports/stage0/REPORT.md).

| Finding | Decision / remaining work |
|---|---|
| Final suite: 14 tests plus 17 subtests passed | Preserve the regression suite; continue rare-rule, information-boundary, and independent conformance coverage in Stage 1. |
| CPU/MPS forward and gradient parity checks passed | The tested workloads pass; recheck model-specific numerics before substantial training. |
| 100 complete random-policy games; 1,201 hands; 106,343 decision records; zero truncations; conservation checks passed for all 100 games | This is simulator/adapter smoke evidence, not strategy evaluation or complete rules validation. |
| Inactive-seat raw observations can expose `drawn_tile` | Mask it in the canonical player observation. Do not expose raw upstream serialization to policies; retain a targeted regression and expand hidden-world invariance checks. |
| `reset(seed)` does not rewind the hand counter | Use fresh engine instances for deterministic runs; record this behavior rather than assuming reset reproduces a new instance. |
| Initial explicit-wall fixtures and clone checks passed | Retain fixtures. Full arbitrary hidden-state reconstruction, native batching, and cross-engine conformance remain pending research/adapter work. |

## Main review proposals

| Proposal | Decision | Implementation consequence / adoption criterion |
|---|---|---|
| 1. Add LuckyJ and align rewards with the target | Modified | Add historical context and room-specific stable-dan reporting. Keep average-rank utility as the initial objective; report a fixed ladder utility only when that target is chosen. |
| 1. Use Phoenix 10.68 as the universal published bar | Rejected | LuckyJ's reported 10.68 and Suphx's 8.74 concern Tokujou/expert play. Phoenix and Tokujou formulas and opponent populations differ. A scalar from a local bot panel does not establish equivalent human-ladder strength. |
| 1. One table-conditioned network | Deferred | Test complete utility/rule conditioning against specialist models at matched compute. A rank-points vector alone does not encode every platform's score terms or rules; conditioning does not by itself prove PPO data are on-policy. |
| 1. LuckyJ uncertainty example | Modified | The proposed ±1.8 calculation is a hypothetical delta-method example under its stated Phoenix rates, not a verified LuckyJ interval. Do not equate games-to-promotion with a fixed-window sample size. |
| 2. Early GRP/placement-model reward experiment | Modified | Bring GRP forward into RL ablations. Keep full-match return optimization as the anchor; version the predictor, train on complete outcomes, and check calibration as policy/opponents change. |
| 2. End each hand's advantages with an unchanged-objective guarantee | Rejected | Telescoping rewards over a complete match do not establish unbiased gradients after truncating credit at hand boundaries. Treat that recipe as an approximate surrogate and measure full-match utility. |
| 2. Oversample late/close-score starts | Deferred | Useful targeted curriculum/screening experiment. Record the altered state distribution and retain natural full-match confirmation. Do not assume predictor training cost or speedup in advance. |
| 3. Privileged critic early | Modified | Test a separate history-plus-state critic soon after the baseline. Actor inputs remain legal observations; test both information boundaries and value/gradient behavior. Next-wall information is a separate ablation. |
| 3. Belief auxiliaries | Modified | Add selected targets when exact or properly masked labels exist. Calibrated marginals are not automatically a legal joint hidden-hand sampler. Measure training cost and policy improvement. |
| 3. One network automatically serves learning, search, evaluation | Rejected | These uses have different conditioning, distribution, and freezing requirements. Reuse is an experiment, not a validity argument. |
| 4. Audit search capabilities before committing to the engine | Accepted | Audit clone/snapshot, explicit walls, arbitrary-state construction, and batching in Stage 0. Record missing adapter work without building a second engine prematurely. |
| 4. Hidden-state search and expert iteration | Deferred | After a useful policy: constrained joint beliefs, observation-limited continuations, one root action per information set, and deadline/utility measurements. |
| 4. Legal hands are an exact belief posterior | Rejected | Hard constraints establish consistency, not action-conditioned probabilities. Conditional uniformity of the unseen wall can be valid under the specified information, but does not validate an arbitrary opponent-hand sampler. |
| 5. Paired deals and deployed action rule | Accepted | Use matched independent seed blocks, seat rotation, and the actual deployed deterministic/temperature rule. Preserve all within-block correlations in analysis. |
| 5. Sequential promotion testing | Modified | Start with raw bounded utility differences per independent block and a valid stopping rule. Predeclare effects/errors and manage repeated candidate tests. Do not copy chess's pentanomial/Elo likelihood into four-player Mahjong. |
| 5. AIVAT-style luck correction | Deferred | Validate actual conditional probabilities, fixed/independent evaluator provenance, unbiased sampling, and corrected-payoff bounds before integrating it with sequential tests. Compare wall-clock efficiency, not only variance. |
| 5. Guaranteed poker-like speedup | Rejected | Poker's published variance reduction is not a Mahjong or wall-clock guarantee. A deterministic policy's action correction is zero; unused softmax scores are not behavior probabilities. |
| 5. Fixed per-hand screening bank | Modified | Keep as a development diagnostic with explicit state selection and predictor dependence. Full-match results decide promotions. |
| 6. Batched tables and shared encoding | Accepted | Design for central batched inference and one semantic encoder across logs/self-play/live use. Move proven hot paths into Rust when measurements justify it. |
| 6. Train all four eligible learner seats | Modified | Admit only correct behavior-policy versions with correct per-seat transitions. More action samples are correlated and do not imply fourfold independent evidence or wall-clock acceleration. |
| 6. Assume neural inference bottlenecks, rent GPUs immediately, or dismiss compact models | Rejected | Benchmark end to end first. Architecture/inference budget and optional remote spending follow measured benefit; no paid run is authorized by Stage 0. |
| 7. Independent engines and property tests | Accepted | Progressively compare explicit physical walls, actions, legal choices, and payouts under identical rules. Save minimal failures; include independent scoring checks. |
| 7. A million games before any training | Rejected | Use staged correctness gates with targeted rare-rule fixtures and increasing coverage. Equal seeds do not establish equal walls across engines. Large campaigns follow resolved adapter differences and budget measurements. |
| 7. Pin randomness | Accepted | Record RNG algorithm/version, seed derivation, engine/dependency versions, and explicit wall fixtures. Keep reproducibility metadata distinct from the semantic rules contract. |
| 7. Separate process eliminates all license implications | Rejected | Process boundaries are not a blanket license exemption. Review code, weights, integration, distribution, and service obligations separately. |
| 8. Establish data route early | Modified | Maintain a permission/provenance ledger immediately, but continue with synthetic demonstrations while permissions remain pending. No external contact has been authorized or sent. |
| 8. Strong external teacher | Deferred | Use only after checkpoint/service/output conditions are established. No teacher access or permission is assumed. |
| 8. Broad game coverage, frozen imitation reference, fading imitation regularization | Modified | Preserve a frozen imitation reference; compare sampling/regularization choices. Match-level splits and censoring remain mandatory; decide corpus/run size from learning curves and throughput. |

## Smaller proposals and restructuring

| Proposal | Decision | Consequence |
|---|---|---|
| List LuckyJ, Mjx, libriichi and existing training recipes | Modified | Treat as historical evidence, potential independent checks, or implementation references. Availability, rules, licenses, and exact versions require inspection; names alone do not establish benchmark access. |
| Swap manzu and pinzu for augmentation | Modified | Candidate symmetry only under a verified compatible rule profile and red-five configuration. Transform every affected feature/action/target consistently and test legality/scoring. Twice as many encodings are not twice as many independent games. |
| Make suit boundaries structural in Model A | Accepted | Choose a compact suit-aware layout and validate it; padded honors/gaps are implementation options, not a fixed final architecture. |
| Current-hand history plus match state | Modified | Reasonable first encoder. Do not assert every hand fits below 256 tokens; define tokenization, overflow handling, and any omitted opponent-history information explicitly. |
| Tile-efficiency lookahead features | Modified | Start with exact cheap shanten/waits/visible ukeire where available. More expensive multi-step probabilities and EV need a stated model and measured benefit. |
| Genbutsu/suji/kabe and related safety features | Modified | Derive only from permitted information and distinguish exact safety facts from heuristics/probabilities. Benchmark and ablate rather than treating all features as necessary. |
| All-Last score requirements | Modified | Useful rule-aware feature candidate; handle payer/dealer/honba/deposits and match-end conditions. Validate calculations before model use. |
| Per-decision entropy adaptation | Deferred | First log entropy by action type and legal-choice count; compare adaptive targets after a stable PPO reference exists. |
| Remove single-option decisions | Modified | Exclude them from policy/entropy loss and report them separately. Do not blindly delete value targets, terminal rewards, or required trajectory links. |
| Value-based RL fallback | Deferred | Preserve a diagnostic path to Mortal-style value learning/offline methods or Monte Carlo targets if PPO plateaus; require matched-budget evidence. |
| Prioritized league sampling | Modified | Test difficulty-based sampling while preserving historical/reference diversity and a separate opponent holdout. Fixed percentages are initial configuration, not an optimality claim. |
| Confirm hardware and plan remote alternatives | Accepted / Deferred | Hardware is identified; benchmark locally now. Remote runs need a concrete measured workload and an explicit spending decision. |
| Short build specification and decision log | Accepted | These documents are the operational layer; the English blueprint is revised, while the proposal and historical Chinese translation are preserved. |
| Move all caveats to an appendix | Modified | Keep validity conditions beside the interface/test they constrain. Separate background discussion without hiding correctness or evidence requirements. |
| Week 0 operator/teacher/bot-account requests | Deferred | Prepare decisions and required questions if needed; do not send messages or contact anyone under current scope. No venue chosen. |
| Fixed schedule or assured “Mortal-level” target | Modified | Use stage gates, not calendar promises or unverified strength equivalence. External comparison requires an accessible, identified checkpoint and a controlled protocol. |

## Data and external-access ledger

Current state: **pending; no external training data acquired**. Synthetic local games are the available fallback. Public source-code inspection and small implementation fixtures are not an acquired human training corpus.

| Source/activity | Current status | Condition before use |
|---|---|---|
| Tenhou human records | Pending; no request sent, no data acquired | Establish intended-use/access/retention/release conditions and record provenance. Public availability does not grant unrestricted reuse. |
| Mahjong Soul records | Pending; no request sent, no data acquired | Establish a permitted export/research route under applicable terms. |
| Converted public corpora | Pending; none acquired | Audit upstream rights, completeness, semantic duplicates, rule metadata, and split feasibility. A converter's code license does not license the records. |
| Third-party teacher/opponent weights | Pending; none acquired | Verify checkpoint and generated-output/service conditions; preserve exact versions. |
| Expert contributions | Pending; none collected | Establish contribution and any source-platform permissions before collection. |
| Own synthetic demonstrations/self-play | Stage 2A: 1,000 audited heuristic matches; Stage 2B: 300 fresh focal-state diagnostic/holdout matches and six matched imitation runs; no RL | Preserve pinned rules/policies, seeds/manifests, match-level splits, and actor/target/auditor separation. Benchmark trajectories do not enter training. |
| Public bot venue | Not selected; no account/contact activity | Confirm that the chosen venue accepts the intended bot evaluation and follow its conditions. |
| Paid compute | Not started | Profile a concrete workload, set budget/shutdown conditions, and obtain the required spending authorization. |

## Evidence anchors

- [Tenhou manual](https://tenhou.net/man/#RANKING): room-specific points and stable-rank formulas. Phoenix uses `(6*p1 + 3*p2)/p4 - 2`; Tokujou uses `(5*p1 + 2*p2)/p4 - 2`.
- [Suphx, §5 and Appendix C](https://arxiv.org/html/2003.13590v2): expert-room result and explicit room-comparability limitation. [Tencent-authorized LuckyJ announcement translation](https://modern-jan.com/blog/luckyj_article_ja/): reported Tokujou result; direct original retrieval was unavailable during review.
- [Tenhou data conditions](https://tenhou.net/sc/raw/): acquisition/use conditions remain separate from software licensing.
- [AIVAT](https://arxiv.org/abs/1612.06915): unbiased variance reduction under the estimator's assumptions; reported poker gains are not a Mahjong guarantee.
- [Fishtest mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html): chess-specific paired model and GSPRT. [Bounded-mean confidence sequences](https://academic.oup.com/jrsssb/article/86/1/1/7043257) provide a relevant alternative for valid sequential utility evaluation.

## Stage 1 findings and decisions

| Finding | Decision |
|---|---|
| 90 tests plus 165 subtests pass; 768 paired campaign games complete with every-step conservation | Stage 1's documented subset gate is met. Preserve [raw evidence and review packet](../reports/stage1/REPORT.md); no full certification or learned-strength claim. |
| Independent scorer agrees on the tested cases | Pin `mahjong==2.0.0`. Distinguish arithmetic inputs, realizable hand fixtures, and rule-transition expectations; a second full engine remains deferred coverage. |
| Reading an observation consumes that seat's event cursor | Route every step-returned snapshot through one ordered adapter per seat. Reject missing initial history; never refetch snapshots casually. |
| Integer action encodings alias some physical candidates | Use immutable exact legal-action descriptors and strict resolution. Neural candidate encoding is a later explicit design; Stage 0's 46 benchmark outputs are not a gameplay contract. |
| Histories and state can repeat round/dealer identifiers through renchan | Increment an explicit hand ID at every start event and reset current-hand history there. |
| Python ignored a hidden editable-install file inside the synced environment | Move `.venv` to the local cache with a project symlink; clear the hidden flag, refresh package metadata, and preserve reproducible setup instructions. |
| Shanten beats the deliberately weak random reference decisively | Keep it as the first transparent imitation/reference candidate. This does not establish expert strength. |
| Defense vs shanten is inconclusive under the stronger panel | Keep both styles; do not promote defense as superior on average-rank evidence. Reduced fourth-place count also accompanied fewer first places in this sample. |
| Heuristic computation is material to the measured game-loop cost | Profile actual collection/encoding/model workloads before scaling; no paid GPU decision follows from the Stage 0 tensor benchmark. |
| Imitation may work without beating a limited teacher | Propose separate pipeline-readiness and strength-promotion checkpoints in [the Stage 2 review draft](STAGE2_REVIEW_DRAFT.md). This is a review recommendation, not a silent change to the existing strength gate. |

## Stage 2A adopted decisions — 17 October 2026

The user authorized continuation after the revision-2 review. [STAGE2_PLAN.md](STAGE2_PLAN.md) and [stage2_pilot.json](../configs/stage2_pilot.json) record the accepted scope. Stage 1 is preserved in commit `1a91128`; both Stage 2 proposals remain unchanged historical references.

| Proposal or finding | Adopted decision |
|---|---|
| Physical candidate multiplicity can distort imitation targets | Add semantic actions that retain red identity, drawn/held discards, and consumed-tile alternatives. Preserve v1 policies and introduce separately versioned v2 teachers. Uniform best-set targets are labels; the recorded teacher behavior remains deterministic. |
| Simultaneous requests can be preempted | Synthetic distillation uses the teacher's known request at the original information state. Such a request can remain a policy label even if preempted; it is never relabeled as an executed event or treated as an action-dependent transition that actually occurred. Unknown requests in external resolved logs remain excluded. |
| Repeated full-history JSON hashing is avoidable | Incrementally hash actual immutable safe events and current state, verify prefix continuity, and compare with complete recomputation. An adapter's state fingerprint alone is insufficient. |
| Parallel collection/evaluation needs reproducibility | Use small spawned CPU pools and deterministic seeds/order. Verify exact trajectories/results across worker counts; exclude timing from deterministic digests. |
| 1,000 total matches cannot also provide a 1,000-match training curve with holdouts | Use 800 training, 100 validation, 100 test matches and nested 100/300/800 training subsets. Keep complete matches and any shared-wall relatives together. |
| Full histories duplicated per decision waste memory | Save each seat/hand's event sequence once, with prefix references. The network uses its latest 128 events plus explicit overflow and complete current public summaries. |
| A successful argmax overfit may hide wrong soft-target probabilities | Require low policy KL, finite gradients/updates, and tie agreement on a tiny fixed batch before the bounded campaign. |
| A rank head is behavior-dependent | Train on terminal ranks under the recorded teacher mixture; report rank NLL/Brier and decision weighting. Reweighting rank probabilities is not retraining an optimal policy for a new utility. |
| Proposed sample count 185 needs assumptions | Separate a 30-block pilot and fresh fixed confirmation. Predeclare the normal approximation, 0.10 margin, nominal 95% one-sided confidence, 80% planning power, variance floor/inflation, futility rule, and 256-block cap. Report inconclusive cases honestly. |
| Readiness and strength are different claims | Report engineering readiness, teacher retention, and stronger-than-panel promotion separately. Weak imitation requires diagnosis; no automatic RL launch or world-strength claim follows. |
| Teacher does not exercise open calls/kans | Report absent action coverage and unknown rare-event causes explicitly. Broader training mixtures and optional belief/GRP auxiliaries remain later experiments. |

## Stage 2A measured findings — 17 October 2026

The [review packet](../reports/stage2/REPORT.md) and evidence manifest preserve the complete results. Package modules match implementation commit `8b14e0d`; generated untracked reports explain recorded dirty flags without being package source drift.

| Finding | Decision |
|---|---|
| 183 tests plus 165 subtests pass; 1,000-match corpus and frozen deployment are audited | Mark the bounded engineering pilot complete, retaining the documented limits of rules coverage. |
| 39.01% of optional corpus decisions have several equally preferred semantic actions | Keep soft semantic tie targets; exact deterministic tie-choice agreement is secondary. The one-seed comparison is not a universal superiority claim. |
| Four initial 2,500-update runs select their final checkpoint; 800-match train/validation KL remains close and falling | Declare one 800-match continuation to 20,000 updates within the existing resource limits. Preserve the initial selection/test/pilot and use fresh match seeds. [Extension protocol](../configs/stage2_extension.json). |
| Extended validation discard agreement reaches 82.47%, but fresh match rank is 3.18 vs teacher 2.57 | Report the accuracy diagnostic as met and teacher retention as unestablished. Apply the planned futility rule; no retention confirmation or RL is launched. |
| Validation discard errors include 5.01% worse shanten and 12.49% lost ukeire at equal shanten | Prioritize hand-shape learning and diagnostic teacher queries on student-visited states. Do not infer their exact causal contribution to match losses. |
| The rank head improves all-row NLL/Brier against a standings table but does not show early-game foresight; policy improvement accompanies worse rank validation metrics | Propose rank-weight 0 versus 0.1 at matched compute, and stronger stage-conditioned value baselines. Do not promote the head into reward shaping or claim proven harmful interference. |
| No chosen chi/pon/kan trajectories; bounded history omits events on approximately 2.4% of rows | Keep these as explicit coverage/representation limitations. Broader behavior mixtures need new evidence; legal candidate presence alone is insufficient. |
| Cumulative training-related work 563.58 s, sampled conservative peak 3.78 GiB, external artifacts about 344 MB | The Mac is sufficient for this pilot. No measured need for paid GPUs or a much larger network follows. |
| Source/checkpoint/seed audits pass; both development pilots remain weak | Preserve both frozen models and failed strength evidence. Propose [Stage 2B](STAGE2_NEXT_INSTRUCTIONS.md) before RL, with fresh holdouts and bounded controlled comparisons. |

## Stage 2B authorization and adopted protocol

On 17 October 2026 the user approved the [revision review](STAGE2B_REV2_REVIEW.md) and continuation within its proposed budget. [stage2b/PLAN.md](stage2b/PLAN.md) and [stage2b.json](../configs/stage2b.json) define the executable scope before formal outcomes.

| Decision | Adopted choice |
|---|---|
| Principal intervention | Exact player-computable shanten, visible-availability ukeire and non-exhausted improving-type count as a separate numeric candidate input. Preserve legal candidate freedom and label the model feature-assisted. |
| Attribution and compatibility | Keep history 128 and runnable v1 schema/checkpoints. Explicitly match shared initial tensors and sample streams across raw/assisted controls. |
| Training allocation | Two seeds each: assisted rank weight 0.1, raw rank weight 0.1, assisted rank weight zero. At most one conditional two-seed experiment, with priority and thresholds fixed in configuration. |
| Selection | Fixed stratified validation sample for checkpoints; full-validation equal-stratum metrics averaged across both seeds for recipes. The declared primary seed is frozen before test/match outcomes. |
| New data | 100 old-student diagnostic matches plus 100 validation and 100 test matches with fixed teacher/student strata. Only focal decisions become supervised rows; actual behavior and teacher requests remain separate. At most 300 conditional training matches. |
| Resources | 3,600 seconds training-related work including feature preparation/checks; approximately 32 GiB memory and 5 GB new artifacts. Four development campaigns and one conditional capped confirmation. Separate 1,800-second inference benchmark including a planned 1,200-second sustained run. |
| Preservation | Verify the immutable Stage 2A project inventory at Git snapshot e19a6cb and unchanged external artifacts; do not rewrite old hashes when current code changes. |
| Deferred | History expansion, calls_v1 implementation/evaluation, RL/search, external data/contact/weights, paid compute, public play and overnight runs. Long-run operation is documentation only. |

## Stage 2B measured findings — 17 October 2026

The [measured report](../reports/stage2b/REPORT.md), independent training audit and resource audit record the completed bounded experiment. No later stage starts automatically.

| Finding | Decision |
|---|---|
| Assisted rank-0.1 validation agreement 100% / 100%, versus matched raw 73.14% / 64.23% | Keep the exact actor-computable feature branch and call the model feature-assisted. Do not interpret teacher imitation as discovered raw hand reasoning or expert strength. |
| Both assisted rank-zero early checkpoints are accurate, but later continuations collapse | Preserve best and last checkpoints and full histories. Require activation/logit diagnostics and tested instability stops before longer optimization; no retrospective rescue or claim that rank loss is universally essential. |
| Assisted placement heads overfit and are overconfident | Do not promote them to calibrated critics/GRP or reward shaping. Rank/value targets and calibration need a separate controlled study. |
| Frozen test reaches 99.9826% discard agreement, with two +1-shanten errors | Keep the frozen primary selected from validation. Test and development outcomes do not retune selection. |
| Four development campaigns complete; primary mean rank 2.5083 vs teacher 2.5583 | Report the favorable descriptive +0.05 utility difference. The shared 30 blocks are not four independent confirmations. |
| Retention planning requires 319 blocks, above the approved 256 cap | No confirmation; retain `inconclusive_budget`. Propose a separate fixed 319-block study of the unchanged candidate, without automatically increasing the budget. |
| New 300-match corpus still has no executed calls/kans | Keep post-call coverage unresolved. A future behavior/data expansion needs explicit rules and trajectory validation. |
| 14-worker sustained benchmark completes 14,960 matches at 7,492 audited decisions/s | Local inference is practical for this model. Exact-efficiency calculation is the largest measured policy phase. Do not equate benchmark throughput with RL throughput or overnight stability. |
| Recorded learning-related durations 42.41 min plus an explicit estimated 5-minute allowance | Charge 47.41 min against the 60-minute cap. Preserve the timing uncertainty; meter every numerical check from the first invocation in future stages. No automatic use of remaining allowance. |
| Formal-run sampled memory 3.60 GiB; benchmark aggregate RSS 4.40 GiB; artifacts about 0.48 GB | Remain within the stated local limits. No need to infer a paid-compute requirement from this stage. |

## Stage 3A authorized engineering pilot — 18 October 2026

The user authorized the 600-second proposal and continued book preparation. [STAGE3A_PLAN.md](STAGE3A_PLAN.md) and [stage3a.json](../configs/stage3a.json) fix the experiment before formal collection. The starting point remains the original Stage 2B selected checkpoint; no smoothed model has been promoted.

| Decision | Reason and adopted scope |
|---|---|
| Full-match REINFORCE before PPO | Establish terminal returns, actual behavior probabilities and learner-only eligibility with a zero baseline and one update per complete batch. The existing overconfident rank head is frozen and excluded from the loss. |
| Four batches of 16 matches, one learner seat | A saved one-worker inference benchmark supports this bounded local attempt. Seeds, seats and counts are fixed before results; incomplete batches cannot update or be replaced. |
| CPU, one thread, single-row probability replay | Collection and gradient replay use the same forward grouping. CPU float64 softmax/log-softmax preserves a clearly identified categorical wrapper around the float32 model; no unannounced argmax-to-sampling or device change. |
| Fresh AdamW at 0.00001, clip norm 1 | Four small updates test the learning path. No hyperparameter search, baseline fitting, entropy objective or strength promotion is included. |
| Durable 600-second cumulative ledger | Imports, numerical tests, failures, collection, updates and checkpoint verification count. Keep failure evidence, a shutdown reserve, a 32-GiB sampled memory cap and a 1-GB artifact cap. |
| Independent score reconstruction | RiichiEnv 0.4.10 emits a bare end-game marker. Reconstruct scores from hand boundaries, accepted riichi payments and result events, using explicitly tested pinned-engine log conventions and terminal-pot settlement. Compare against native final scores and initial-seat-tiebroken ranks. |
| Preserve history through Git snapshots | Verify the previous continuation's 212 original files at completed snapshot `d228df07128d50cfdccdceb29f81d37119ee843b`; do not regenerate its inventory against evolving source. |
| Twelve additional book packets | Retain partial context, related-family grouping, source-specific counting conventions and red-five glyph evidence. Independent transcription review does not certify strategic optimality or make a complete player view. Optional arithmetic checks remain inside this stage's numerical budget. |

## Stage 3A measured findings — 18 October 2026

All 64 planned matches and four updates completed with exact replay and verified boundaries. Keep the new complete-match learner as an engineering baseline; do not promote it on these training outcomes. The final candidate is a resumable training boundary, while the original Stage 2B reference remains the unchanged reference.

Every offered call/kan candidate had zero numerical sampling probability. Prioritize a separately bounded action-support and gradient diagnostic before a longer unchanged run. Any new exploration rule must record and differentiate its actual distribution; a uniform mixture can sample actions without fixing saturated base-policy gradients.

The twelve-case book extraction initially missed tile glyphs in five records, producing seven arithmetic mismatches. Preserve the original review and failed comparisons, the separate source-confirmed revision, and its subsequent 47/47 agreement with both calculators. Require magnified visual review and arithmetic checks together for further shape extraction; strategic labels and full-state conversion remain separate.

## Stage 3B authorization — 20 October 2026

The user approved the proposed 300-second diagnostic and continued book preparation. [STAGE3B_PLAN.md](STAGE3B_PLAN.md) fixes its scope before outputs. Compare only the original reference and final Stage 3A boundary, using at most 512 saved actor rows and 32 fixed target-action gradient/backend probes. The four rules are unchanged softmax, temperatures 10 and 100, and a 5% legal-uniform mixture. These coarse alternatives diagnose saturation and distortion; none is automatically selected for training.

Use single-row CPU float32 network execution and float64 normalization as the reference. MPS comparisons use the same rows and CPU normalization. Differentiate each rule's actual log-probability, record zero/attenuated gradients, and verify parameter immutability. No optimizer, new matches, test-split loading, checkpoint writing or policy promotion is authorized.

Retain Stage 3A's 287 original artifact identities at Git snapshot `d412ccb6d0cd345904fe9084120839ba25341cf3`. The new snapshot verifier reads historical project bytes and unchanged external/local-book bytes. Book work remains separate development preparation: preserve unknown context, red-five ambiguity, compound/conditional recommendations and explanation provenance, with answers excluded from observation exports.

## Stage 3B measured findings — 20 October 2026

The [completed diagnostic](../reports/stage3b/REPORT.md) used 21.858633 of 300 numerical seconds, with no new matches or optimizer updates. All 364 offered call/kan candidates in the fixed sample have zero probability at both endpoints. Temperature 10 and 100 do not provide useful call frequency, and temperature 100 substantially changes discard distributions. A 5% fixed legal-uniform mixture makes calls sampleable but all 22 direct call-target parameter gradients remain zero. Adopt none of these four rules as a new learning default.

Preserve the failed first reporting attempt and the subsequent same-selection recovery. A focused fixture demonstrated corrupted values when the installed runtime combines an MPS-to-CPU transfer with float64 conversion. Change only the new diagnostic module to stage that transfer and reject nonfinite derived norms, binding the repair to the unchanged selection/grid and checkpoint pair. CPU outputs match the recovery exactly; final MPS measurements are finite but not numerically identical to CPU. Keep CPU for the proposed next learning pilot. Historical Stage 3A used CPU; its old MPS probability/replay path must be repaired and regression-tested before future MPS training.

Propose a separately authorized trainable call gate and independent call head with the base policy frozen, rather than widening an unchanged sampler. The [review](STAGE3B_REVIEW.md) specifies a 600-second engineering/learning pilot, fixed 64 training matches/four updates, 256 paired evaluation matches and an identical initialized control. It is not started or promoted. Validate post-call observations, meld accounting and efficiency features before its formal games.

Convert the same twelve corrected book cases into strict partial scenarios and question-only exports. Preserve all 47 earlier count labels and their successful audit by identity; do not claim a new arithmetic check. Keep red-five uncertainty, missing context, compound riichi action and conditional source advice explicit. No book training or full-policy adapter is added.

## Further revisit triggers

Update this log when dependency compatibility is established, a simulator capability fails the audit, a permission/venue decision changes, a measured bottleneck warrants infrastructure work, or an ablation changes the champion. Record the evidence and resulting choice; do not silently promote an experimental idea into a requirement.

## Stage 3C authorization and fixed protocol — 21 October 2026

The user requested another hour of continued work. Execute the previously proposed isolated call-learning pilot under [STAGE3C_PLAN.md](STAGE3C_PLAN.md), retaining its600 numerical seconds,64 training/256 evaluation match caps and four heads-only updates. Freeze the original base, use a5% call gate and uniform conditional call initialization, and compare the fixed endpoint with its identical untrained copy. Do not select a model from outcomes.

Require source-bound passing post-call fixtures and sampling/gradient tests before games. Add explicit evaluation-only trajectory provenance rejected by training, auditor-only ordinal physical-wall commitments, and distinguish kan declarations from completed replacement draws. Preserve historical Stage 3B through its completed snapshot `cc10de1c4134f58703e180f2a9d3ee67df375662`; its86 recorded files verify unchanged. Continue narrow book notation preparation separately without training on partial scenarios.


## Stage 3C measured training and stopped evaluation — 21 October 2026

Keep the call-learning mechanism as experimental: 64 complete matches/four updates produced 100 resolved calls, 740 post-call learner decisions and nonzero finite gradient/parameter changes in both heads. Frozen base/rank head and exact probability replay passed. Do not equate this engineering result with stronger Mahjong.

Preserve the evaluation stop at block 8. A newly authored analysis test matched a dynamic implementation glob and triggered the source watchdog. This was root workflow coordination and membership design error; all originally pinned source files were unchanged. Rename the independent analysis test outside the historical membership pattern for post-run work, retaining its original content hash and the failed receipt. Do not resume, replace or pool partial blocks into the planned 32-block endpoint. Eight complete blocks and two additional saved matches remain auditable; no fixed-endpoint interval is computed.

Future controllers should freeze explicit runtime/test paths before execution and record match-start evidence. Propose a separate fresh-seed evaluation-only recovery with the same final and initialized checkpoints. No new training or promotion is justified from the stopped comparison. Keep five red-five interpretations as evidence annotations with unknown encoder identities; add a concrete call-scenario intake proposal without representing the existing discard problems as call supervision.


## Stage 3D recovery authorization — 21 October 2026

The user authorized continued work after the Stage 3C review. Execute its separately proposed evaluation-only recovery under [STAGE3D_PLAN.md](STAGE3D_PLAN.md), preserving stopped Stage 3C evidence and its unchanged final/initial checkpoints. The campaign is 400 cumulative numerical seconds, 45 preparation seconds, 32 fresh paired blocks/256 matches, no optimizer updates or checkpoint writes.

Replace dynamic source membership with an explicit immutable runtime/test list. Ignore unrelated file additions, reject changes/deletions within the list, and pin the new campaign separately from the historical behavior protocol. Add exclusive synchronized match-start/finish records and cooperative cancellation between response steps, with incomplete trajectories ineligible for terminal utility or training. The fixed 32-block analysis is produced only after all controller receipts close. Continue a narrow two-packet book preparation task without full-policy conversion or training.


## Stage 3D measured findings — 21 October 2026

All 256 fixed fresh matches completed under explicit frozen source membership and synchronized match lifecycle records. Retain this recovery workflow. Preserve stopped Stage 3C separately; no historical evaluation was retried or pooled. The recovered comparison has zero average-rank utility difference with the predeclared wide interval [-1.440484, +1.440484], so no policy is promoted.

Keep the distinction between numerical change and useful strategic learning: 3,913 matched call-opportunity distributions changed slightly, while 127/128 full action sequences and all placements matched. Mean call mass fell by 0.00018448. Propose a single longer unchanged learning horizon before adding a critic, changing learning rates or expanding the architecture. The proposal is 28 additional 16-match batches from the saved update-4 model/optimizer, a fixed update-32 endpoint, and a separate fresh comparison. The provisional 1,200-second cap is an extrapolated proposal requiring a frozen execution plan; it is not a new run.

The final audit is computed once under the numerical ledger. After its lease closes, a separate hash-only finalization revalidates all 1,152 saved evidence inputs and the exact successful closure of that sole active lease. It preserves the 2,115,700 numeric/data assertions, adds 1,451 closure/bookkeeping checks, and avoids repeating uncharged match arithmetic. All 41 operations completed, charging 350.845432/400 seconds.

Prepare two new book call packets alongside eight locators. Do not merge a conditional choice of CHI with call-versus-PASS supervision; do not import explanation-only dora into question observations. Preserve missing state, red-identity uncertainty and the diagram-inferred later discard. Implement a strict partial-call validator before using either packet as a model input or label. No book training or policy adapter is added in this stage.


## Stage 3E authorization — 21 October 2026

The user approved the longer fixed experiment. Restore the original update-4 checkpoint and its optimizer, not a fresh optimizer or initialized call control. End at update 32 after exactly 28 new 16-match batches. Evaluate only that endpoint against update 4 on 32 fresh paired blocks. Keep the unchanged Stage 3C algorithm and new explicit campaign identity; no checkpoint selection, hyperparameter search or historical pooling.

Adopt the proposed 1,200-second cap with explicit protected capacity: training admission retains 400 seconds for evaluation, 40 for audit and 60 for shutdown; evaluation retains 100 seconds. Use new cooperative collector eligibility flags so partial/evaluation records cannot train. Preserve all prior snapshots. A strict book validator converts only the two previously reviewed packets and preserves their question exports; no book supervision is adopted.

## Stage 3E measured findings and next diagnostic proposal — 21 October 2026

The fixed continuation completed 448 new training matches, 28 updates and 256 fresh evaluation matches. Preserve update 32 as an experimental endpoint, with the original reference and update 4 unchanged. Both evaluated endpoints averaged rank 2.5625; all 128 paired action sequences, scores and placements matched. The fixed mean utility difference is 0 with the predeclared interval [-1.440484, +1.440484]. All 3,763 common evaluation call-opportunity distributions changed, with mean total call-mass change -0.0000776992 and maximum absolute per-action change 0.0001772215. No strength gain or policy equivalence is established. Do not compare absolute ranks against a different historical seed population or extend the same loop after this result.

All updates had finite nonzero gradients and parameter changes in both heads, with exact pre-update behavior replay and unchanged base/rank parameters. The audit implementation nevertheless accepts finite zero head gradients as possible coverage/no-op diagnostics rather than automatically treating them as integrity errors; absence of a selected multi-call opportunity can yield a zero conditional-head gradient. It still rejects nonfinite values, dishonest change flags and replay/objective inconsistencies. This fixes audit interpretation only; the frozen collection and update recipe were unchanged.

Training contains 16 legal win-versus-call opportunities, of which two selected and resolved a call. The call gate can divert approximately 5% mass from the original win action. Preserve these as observed development cases, not proof of a strategic error or grounds to assume an always-win rule. Multiple-call selection is also sparse: 83 selected calls from 1,942 training rows with alternative calls. The next proposed diagnosis should examine gate/conditional-head credit, per-match gradient directions, optimizer/output movement and the saved win/call contexts before choosing a single intervention. Restrict development selection to saved training data; do not mine the Stage 3E evaluation for tuning. The provisional 300-second diagnostic is unstarted and includes no new games or optimizer updates.

The final full audit passed 5,818,945 data-level assertions. After its lease closed, hash-only finalization checked 3,033 evidence inputs and added 3,607 bookkeeping/integrity checks without repeating match arithmetic. All 67 operations succeeded, charging 956.038681/1,200 seconds. Preserve the unspent allowance; no extra learning, seeds or post-hoc diagnostic is added to this experiment.

The strict book validator and migration passed 20 tests and converted exactly two prior packets, with unchanged question exports and input identities. Keep call-versus-decline preference distinct from conditional CHI choice, and preserve unknown state, copy ambiguity and answer-only information boundaries. Both records remain ineligible for policy training/evaluation. Extend the schema for declared melds and own-turn kan before processing those source cases; do not infer complete actor state from partial book diagrams.


## Stage 3F revised programme and engineering authorization — 22 October 2026

Adopt parallel exploring collection, a bounded-memory data loader and a controlled value-learning direction after the user's review approval. Preserve the external proposal as historical source; its original embedded instruction to start three rounds does not define this milestone. Follow [STAGE3F_PROGRAM.md](STAGE3F_PROGRAM.md) and its fixed configuration. The immediate integrated engineering budget is 7,200 numerical seconds with diagnosis capped at 300, 120 seconds reserved for shutdown, 32 GiB sampled RSS and 2 GB artifacts. No optimization occurs in the engineering pilot.

Use one each of the two heuristics, original argmax reference and initialized-call-gate explorer per full match, with seeded seat permutation and exact conditional behavior provenance. The explorer mixes 5% semantic-uniform exploration into the gated policy and deterministically takes a legal win when offered. This is a declared collector heuristic, not universal strategic optimality. Pilot data remains development-only. Preserve all simultaneous information states, semantic/physical distinctions, complete-match rewards and historical snapshots.

Reject the proposal's unsupported 6% hardware-utilization inference and its declaration of a PPO plateau. Existing throughput numbers measure different workloads; PPO has not run. The saved-training diagnosis now reconstructs gradients and Adam correctly but finds sparse meaningful conditional-call contributions and within-batch cancellation. A literal nonzero gradient is not sufficient evidence of useful learning signal. Benchmark actual serialization, one-shard loading and representative CPU/MPS value computations before setting a large corpus count.

For a later learning round, compare full-match Monte Carlo targets with placement-prediction hand differences using matched data/compute. Use out-of-fold placement predictions, utility calibration and complete-match grouping. Treat conservative penalties as experimental regularization, specify every alpha/control run in the budget, select on development evidence, and freeze fresh paired confirmation and round gatekeeping before outcomes. Do not apply standard on-policy PPO to the mixed historical corpus. Existing book copies remain available; incomplete state and legal-action reconstruction are their technical blockers. No book training is added.

## Stage 3F measured engineering and revised first-round design — 22 October 2026

Retain fourteen-worker collection and the one-shard streaming loader. All sixteen serial/parallel plans have identical deterministic payloads; the fixed 1,000-match pilot completes at 11.505 recorded matches/s with 646,377 rows. The MPS value-gradient probe is about three times the CPU rate with passing first-batch numerical comparison, but neither Adam updates nor long-run training were measured. Keep the twofold planning allowance and validate the actual filtered/accumulated optimizer path before the next formal experiment.

The all-row audit passes for 1,144 unique matches / 737,994 rows and the ten fixed full-match replays match. Preserve all outcomes, forty-nine completed pilot kans, the two KAKAN cases, and the limited closed-hand opponent population. Do not treat successful same-engine replay as independent rule certification. Preserve the failed scheduling preflight and corrected source-bound release test result; there were no failed formal matches or replacements. The closed numerical ledger charges 186.193133 / 7,200 seconds across 26 operations. Leave the unused allowance unused rather than inventing extra experiment arms.

Use explorer-only optional decisions for the proposed first value learner. With one of each behavior per table, a seat's own behavior also changes its opponent population; pooled all-seat Monte Carlo values would mix both continuation and opponent policies. Retain all-seat records for audit, distinguish mixed-policy returns from the greedy learner's values, and keep current pilot data development-only. Normalize losses within complete matches and then across matches; define empty-match behavior explicitly.

Propose, without starting, 4,000 fresh corpus matches split 3,200/800, four terminal/one-hand-difference by alpha-zero/one arms and two fixed passes. The hand difference uses the same out-of-fold predictor at both boundaries with exact terminal substitution and explicit hand ordinals. Budget six placement fits and predeclare proper-score and expected-utility calibration gates, sparse-stratum handling and a terminal-only fallback if calibration fails. Recognize that the conservative term here is algebraically chosen-action cross-entropy; it does not establish a Bellman/CQL guarantee. Select through a fixed common game panel, not raw losses across different targets, and reserve fresh confirmation for a subsequent fixed protocol. No Stage 3G learning or additional book extraction has started.


## Stage 3G authorization and fixed experiment — 22 October 2026

The user approved the proposed first value-learning round. Adopt [STAGE3G_PLAN.md](STAGE3G_PLAN.md) and its fixed configuration: 4,000 fresh complete games split 3,200/800 before outcomes, one reference backbone with a reset scalar scorer, two full passes and 400 scheduled updates per arm. Use only explorer optional decisions, averaging within complete matches and then across sixteen matches. Preserve all four seats for audit, never treat frozen-opponent rows as on-policy data, and retain average-rank utility.

Measure the actual CPU/MPS optimizer and save/resume path before admission. The short MPS preflight passes at 2,552 eligible rows/s; resumed weights are identical and Adam moments agree within tolerance. The final release passes 190 tests. Keep the original preflight source versions when later validation hardening adds schema and seat-behavior checks. Admission uses conservative measured time and storage forecasts rather than GPU-utilization guesses.

Fit five out-of-fold public placement predictors plus one final train-only predictor. Predetermine proper-score and utility-calibration thresholds on the 800 development games. If calibration fails, skip both hand-difference arms and complete the two terminal-return controls; do not tune the gate after outcomes. Full four-arm evaluation has 1,280 games; the declared two-arm fallback has 768. Both use the same 64 independent wall blocks and four focal-seat rotations, with explicit common-ordinal physical-wall checks. Development selection is descriptive and cannot promote a model.

Retain the 7,200-second cumulative cap with phase caps and protected future work; admit at most 32 GiB sampled RSS, 8 GB new artifacts and 5,500 match attempts including fixtures and replay. Failed formal operations stop subsequent formal work without automatic retries. Add independent saved-evidence audits and a final hash inventory; preserve the historical Stage 3F snapshot at its completed commit. Existing books remain quarantined; this milestone adds no extraction or book-derived training.


## Stage 3G measured outcome and next diagnostic proposal — 22 October 2026

Retain the audited collector, role-bound loader, complete-match value trainer and fixed evaluation workflow. The fresh corpus supplies 2,564,254 decisions; all four arms each complete 400 updates on two passes over the same 471,627 eligible explorer rows. All six placement predictors fit and pass the predetermined gate. All 1,280 paired-development games, independent saved-evidence audits and forty complete-match replays finish without failure or replacement. The closed numerical total is 1,165.485892 / 7,200 seconds. Keep the unused budget unused.

Promote none of the four policies. The original reference averages rank 2.421875; mc_a0, mc_a1, delta_a0 and delta_a1 average 3.070312, 2.960938, 2.621094 and 3.074219 respectively. The placement-change model without regularization is only the least weak observed candidate. Do not adopt the target or alpha-one penalty as defaults, infer a need for larger hardware, or use changing-batch training losses as evidence of convergence. Placement calibration is an eligibility result, not demonstrated action credit or policy improvement.

Propose a separate 300-second saved-state diagnosis, without new games or optimization. Compare the reference, common reset-head initialization and fixed training boundaries on identical hash-selected positions; examine action support and within-target errors. Decompose regression and imitation gradients on a fixed complete-match group without updating parameters. Use deployed single-state inference for action comparisons and treat original reference outputs as policy logits. Freeze scope and measure admission before execution. Counterfactual action quality and causal effects of the reset head/win rule require a later explicit-control game experiment. Neither that experiment nor the diagnosis is started here. Preserve the existing book quarantine and historical snapshots.


## Stage 3H longer diagnostic authorization — 22 October 2026

The user requested a longer version of the proposed saved-state diagnosis. Adopt [STAGE3H_PLAN.md](STAGE3H_PLAN.md) with a separate 1,200-second numerical cap, eighteen archived weight sets, a hash-selected panel of up to 128 training and 128 development matches, and four fixed sixteen-match gradient groups. Measure CPU single-row throughput and CPU/MPS gradient agreement before final admission. Permit only the predeclared 128/96/64-per-split resource fallback before formal outcomes. Preserve zero optimization/new games, independent evidence auditing, historical snapshots and book quarantine. This diagnostic does not extend the completed Stage 3G training run.


## Stage 3H measured findings and next control — 22 October 2026

The complete fixed diagnosis passed: eighteen archived weight sets on 38,237 positions from 256 matches, six model/target conditions across four complete sixteen-match gradient groups, and independent reconstruction plus 576 exact CPU re-forwards. All 75 release and 26 auditor tests pass. Ten successful operations charge 89.953390 / 1,200 numerical seconds; zero games, optimization, checkpoint writes or promotions occurred. Preserve the frozen sample and leave the unused allowance unused.

Reset initialization changes most sampled action choices: raw development reference agreement is 39.1886%, wrapped agreement 39.4699%. The wrapper changes only 0.2813% of sampled positions, but its fresh-trajectory effects remain unknown. The final delta alpha-one model trades improved chosen-action imitation loss for poorer delta prediction (MSE 0.248722 versus zero baseline 0.127567); mean MSE/CE gradient cosine is -0.838984. Treat this as a local objective tradeoff, not a causal explanation or numerical defect. Gradients and offline errors do not establish stronger play.

Adopt the review direction in [STAGE3H_REVIEW.md](STAGE3H_REVIEW.md): propose a fixed initialization/wrapper two-by-two game control before another training sweep, then preserve the entire reference actor while introducing a separate value estimator. A provisional 1,024-match/600-second control requires its own measured admission and fixed statistical design. No new control or learning phase is started. Keep reference policy logits distinct from utility targets, record that CE fits the logged explorer behavior, and do not infer unchosen-action outcomes from logged returns. Existing book quarantine and historical snapshots remain unchanged.


## Stage 3I proposal review — 22 October 2026

Review the user-supplied revision 2 as design material, not execution authorization. Preserve its verbatim source and [detailed assessment](STAGE3I_PROPOSAL_REVIEW.md). Adopt for the next design: the complete frozen reference actor, detached separate value/action-residual components, zero initial corrections, behavioral calibration independent of game outcomes, explicit eligibility restrictions and a wrapper-matched primary game control. Initialization controls may be integrated rather than requiring a separate control-first stage, with narrower claims if the full two-by-two is omitted.

Revise before an executable plan: keep first action-residual learning explorer-only; distinguish ensemble eligibility from actual data propensities and empirical coverage; reconstruct only the causal public-history prefix; separate raw and wrapped identity tests; freeze a baseline before residual fitting and add a state-only residual comparator; declare target/weighting and whole-match selection/calibration/gate partitions. One-hand potential differences remain a surrogate. Neither a small correction fraction nor improved prediction establishes action quality.

The 470-block power and 900-second evaluation forecasts remain assumptions pending a traceable variance source and full-path throughput admission. The proposed 5,400-second campaign is unstarted and unauthorized. Reject automatic 30,000-match collection/reruns; failures close the campaign and lead to a separately costed proposal. No model, numerical test or game ran during this review. Preserve all Stage 3H evidence and book quarantine.


## Stage 3I authorized programme — 22 October 2026

Proceed under the user’s subsequent continuation and ambition request with [STAGE3I_PLAN.md](STAGE3I_PLAN.md). Freeze the entire original actor, fit separate value/action-residual/state-residual heads on explorer-only archived outcomes, and retain all three fixed seeds. Exact explorer probabilities with threshold 0.02 replace the proposed four-policy ensemble eligibility; this is not an empirical-coverage guarantee. Use whole-match 200/200/400 internal development partitions, zero correction fallback and a matched-wrapper primary game control. Keep both reset controls on 64 of the 470 fresh blocks.

Retain the 5,400-second numerical ceiling, reserve later phases, and admit up to twelve full passes from measured throughput before formal outcomes. Raise the new-artifact allowance to 8 GB for exact frozen representations and the larger panel, conditional on measured size; retain the 32 GiB memory and 10 GB free-disk limits. No new training matches, automatic collection escalation, promotion or paid compute. The historical review and all earlier evidence remain preserved.


### Stage 3I closure — 22 October 2026

The authorized programme completed: three seeds × twelve passes per head, 49,788 updates, 4,272 formal control games and independent all-row/checkpoint/game auditing plus 32 exact replays. All three predictors pass the frozen gate, but every positive value in the original calibration grid breaches the 3% overall change ceiling; retain lambda zero and report no learned-policy game result. Do not round the second seed’s one-decision excess into acceptance.

The saved-only fractional diagnosis is explicitly post-hoc and leaves formal arms unchanged. Exact reference-score ties create a nonzero positive-lambda limit (~2.79–2.97% changed decisions); plan a separate finer-correction or learned-tie experiment with an appropriate tie control and fresh walls. Preserve all fixed seeds and the original actor. Reset controls lose roughly one rank position on the matched subset, while raw/wrapped reference trajectories match in all 1,880 pairs. Keep these scope limits and distinguish predictive gain from playing strength.

Final accounting: 480.631934 / 5,400 numerical seconds; 16 successful operations, no failed numerical/formal attempt; 4,360 complete-match executions including 56 preflight and 32 replay games. One stale-source admission was rejected before any lease began and repaired before the formal stages. Full [results](../reports/stage3i/REPORT.md) and [review](STAGE3I_REVIEW.md) are ready.


## Stage 3J authorized fresh-game experiment — 29 October 2026

The user approved smaller corrections and learned tie-breaking on fresh games. Adopt [STAGE3J_PLAN.md](STAGE3J_PLAN.md): preserve the original reference and all three Stage 3I residual bundles; test learned exact ties and fixed lambda 0.25 against a wrapper-matched reference and deterministic nonlearned hash tie control. No retraining, best-seed selection, adaptive panel extension or promotion occurs. The prior fractional diagnosis is explicitly reused post-hoc design evidence, not confirmation.

Freeze eight arms × 256 wall blocks × four focal seats (8,192 formal games), a separate 64-game preflight and 32 stratified replays. All four primary pooled contrasts average seats and training seeds within shared wall blocks before inference. Keep exact numerical ties distinct from strategically equivalent moves, and distinguish changed decisions on a policy's visited states from counterfactual action quality.

Use a separate 5,400-second numerical ledger with preflight/evaluation/audit caps 600/3,600/900, protected shutdown reserve 150, 32 GiB sampled RSS, 8 GB artifacts and 9,000 attempted-match ceiling. The complete preflight passes; conservative forecast admits the fixed panel at 1,247.914 seconds and 3.466 GB. All 167 final release tests pass. Preserve every tested source version, all complete-match provenance and explicit physical-wall commitments. Independent auditing, fixed replay and final analysis follow the formal games.


## Stage 3J closure and next learning decision — 29 October 2026

All 8,192 fixed fresh development games completed under [STAGE3J_PLAN.md](STAGE3J_PLAN.md). Learned tie-breaking pooled mean rank is 2.504883 versus wrapped reference 2.527344 and hash tie control 2.500000. The four preregistered adjusted intervals all include zero; there is no demonstrated learned-ranking advantage or policy promotion. Quarter-strength correction averages 2.506185 and produces only 10/3/15 positive-margin overrides across the three 1,024-game panels. Do not select the apparent best seed or tune lambda against these outcomes.

Retain the frozen-reference architecture, explicit tie control, all-seed reporting and paired-block analysis. The next proposed learning direction is fresh prospectively randomized first-tie actions with one intervention per match, fixed reference continuation and complete-match outcomes. Assign labels only to played actions; preserve propensities, whole-match separation and actor-visible inputs. A measured coverage/variance pilot should precede a larger collection/training budget. This is a proposal, not a started next phase or an extension using leftover time. More epochs or a larger model do not address the currently unproven action-ranking signal.

Final evidence passes: 167 release tests, all 5,215,102 recorded decisions, 256 fixed fresh actor forwards, 192 residual-head forwards, independent reconstruction of the four primary statistics, and 32 exact replays. Nine successful numerical operations charge 609.037689 / 5,400 seconds, with 8,288 complete-match executions including preflight/replay. No numerical/formal failures, new training, optimizer updates, promotions, external-data acquisition or paid compute. All tested source versions are archived. [Results](../reports/stage3j/REPORT.md) and [review](STAGE3J_REVIEW.md) are ready; preserve the historical Stage 3I snapshot and leave the unused allowance unused.
