# Implementation plan

This is the working build specification. The [English research blueprint](../RIICHI_AI_RESEARCH_PLAN.md) is being revised to reflect the review; the [Chinese translation](../RIICHI_AI_RESEARCH_PLAN_ZH_CN.md) and human proposal remain preserved historical references. Decisions and deferred experiments are recorded in [DECISIONS.md](DECISIONS.md); current progress is in [STATUS.md](STATUS.md).

## Stage 0 — reproducible foundation

Initial environment: M5 Pro, 18 CPU cores, 20 GPU cores, 64 GB unified memory; Python 3.14 available; Rust and uv initially absent. The project initially contained planning documents, with no application code or Git repository.

This stage covers package setup, dependency checks, synthetic smoke tests, simulator capability inspection, and modest hardware benchmarks. No learning run, external dataset acquisition, paid compute, external contact, or public bot play is part of this stage. A passing smoke test is preliminary evidence, not simulator certification.

Recorded results: Python `3.14.8`, PyTorch `2.14.1`, NumPy `2.5.4`, RiichiEnv `0.4.10`, and pytest `9.1.1` are installed; the package environment is pinned in [requirements-macos-arm64.lock](../requirements-macos-arm64.lock). The final suite passed 14 tests plus 17 subtests, including the tested CPU/MPS forward/gradient parity checks. The random-policy smoke run completed 100 full games, 1,201 hands, and 106,343 decision records, with zero truncated games and conservation checks passing for all 100 games. See the [Stage 0 report](../reports/stage0/REPORT.md) for measurements and their scope.

Two upstream behaviors require adapter safeguards: raw observations for inactive seats can expose `drawn_tile`, which the canonical observation masks; and `reset(seed)` does not rewind the hand counter, so repeatable runs use fresh engine instances. Initial explicit-wall fixtures and clone checks passed. Full arbitrary hidden-state reconstruction, native batching, broad information-boundary checks, and independent rules conformance remain pending. Carry these findings into Stage 1 regression tests and the later search audit.

| Deliverable | Acceptance evidence |
|---|---|
| Python package under `src/riichi_ai` | Import and test commands succeed in the documented project environment; setup instructions reproduce that environment. |
| Dependency specification | Installed versions and the platform-specific environment lock are recorded. Use the project `.venv`; update the lock and rerun compatibility checks after a dependency change. Rust/uv are installed only if the selected setup actually requires them. |
| Hardware/backend report | Actual CPU/GPU/memory, Python and library versions, CPU/MPS availability, and any unsupported path are reported. Backend availability alone does not count as numerical validation. |
| Full-match smoke execution | A bounded set of synthetic four-player red-five hanchans finishes with legal actions and coherent terminal results under the selected explicit mode. Record seeds, rules, failures, and game counts. |
| Reproducibility smoke check | Repeating a fixed configuration reproduces its trajectory/results within the documented implementation. Record RNG/dependency versions; do not claim cross-engine compatibility from equal seeds. |
| Simulator capability audit | Document working, missing, or unverified support for cloning/snapshots, explicit wall injection, arbitrary-state construction, legal candidates, observation filtering, and batching. Unsupported capabilities become concrete adapter tasks. |
| Modest benchmarks | Save configuration, actual elapsed time, game/decision counts, peak memory where available, and CPU/MPS timings for the same representative inference workload. Include warm-up and synchronized timing; avoid extrapolating world-class training cost. |

**Exit:** setup is reproducible; smoke tests and benchmarks have saved results; remaining engine/API limitations are explicit. Missing optional search/batching capabilities do not prevent basic correctness work. Do not label this exit “rules certified.”

## Stage 1 — rules, player interface, and evaluation reference

**Completed for the documented tested subset on 16 October 2026.** The final suite passed 90 tests and 165 subtests; 768 paired-evaluation games completed without failures or truncation. See the [Stage 1 review packet](../reports/stage1/REPORT.md), [coverage ledger](RULE_COVERAGE.md), and [evaluation contract](EVALUATION.md). Full independent-engine transition conformance remains future coverage; the current independent oracle covers scoring/readiness. Stage 2A began on 17 October 2026.

Build a versioned four-player, red-five, East–South rule profile; an engine adapter; player-visible observations; complete legal-action descriptors; and random/heuristic opponents. The deployment interface accepts player-visible events and legal candidates, never the full simulator object.

Acceptance criteria:

- Fixtures cover scoring, furiten, call priority, riichi/kan timing, red tiles, ties, continuations, and match end conditions. Each expected result has a documented basis.
- Action encoding round-trips every supported legal candidate, including pass and permitted win-declining choices. Simultaneous requests are distinguished from resolved actions.
- Tile conservation and score/deposit accounting hold; failures produce minimal reproducible records.
- Information-set invariance holds across legally consistent hidden worlds. Seat memories remain separate, with explicit reset behavior.
- Snapshot/clone operations preserve state and continuation behavior where supported. Independent engine checks use the same explicit physical wall, actions, and rules—not merely the same numeric seed.
- A progressive conformance suite combines hand-crafted fixtures, property checks, and independent scoring/transition comparisons. Track coverage and unresolved discrepancies; do not require a million games before any learning experiment.
- The evaluation harness freezes opponents/configuration and compares candidates using matched independent seed blocks with seat rotation. Save raw placements, scores, utility, action failures, and latency.

**Exit:** the validated fixture subset has no unexplained discrepancies, observations/actions pass their invariants, and raw paired evaluation is reproducible. Any untested rule or unsupported adapter remains visible in the report. Extend coverage during subsequent stages.

## Stage 2 — data pipeline and trained baseline

The user authorized continuation after reviewing revision 2 on 17 October 2026. [STAGE2_PLAN.md](STAGE2_PLAN.md) is the accepted bounded Stage 2A specification and [stage2_pilot.json](../configs/stage2_pilot.json) records its limits and seed namespaces. The original drafts remain historical proposals. Engineering readiness, teacher-strength retention, and stronger-than-panel promotion are now distinct milestones; this prevents imitation infrastructure from being mistaken for improved playing strength.

**Measured on 17 October 2026:** Stage 2A's engineering exit is met. The integrated suite passes 183 tests plus 165 subtests; the 1,000-match corpus and frozen neural checkpoints are audited. A declared extension reaches 82.47% validation discard agreement, but its fresh match pilot averages 3.18 place versus the teacher's 2.57 and triggers the prespecified futility rule. Teacher retention and stronger-than-panel promotion remain unmet. [Complete results](../reports/stage2/REPORT.md). The subsequently authorized Stage 2B experiment is now complete; [its results](../reports/stage2b/REPORT.md) supersede the earlier proposal as current evidence. The subsequent Stage 3A engineering pilot is recorded below.

Use the data route permitted by the ledger in [DECISIONS.md](DECISIONS.md). Until third-party access is established, generate synthetic demonstrations locally. Record source/policy/rule versions and split complete matches before creating seat views or augmentations.

Acceptance criteria:

- Raw, canonical, player-view, and training-record schemas are versioned. Semantic deduplication, split assignment, and rejection/censoring reports are reproducible.
- Unknown choices cannot be inferred from resolved logs. For synthetic request-level distillation, a known teacher request remains a valid policy label even if another simultaneous request preempts its execution; save requested and resolved actions separately. Never label a preempted action as executed or use an action-dependent transition target that assumes it executed. Valid state/rank targets retain independent masks.
- Heuristic opponents and the first compact neural baseline have frozen configurations. Compare CPU/MPS numerical behavior before a substantial run.
- A tiny learning check succeeds before scaling. The run budget is derived from measured throughput and memory, not a promised corpus size or strength level.
- Policy learning and value/rank initialization are explicit. Evaluate full-match outcome calibration and performance on held-out seeds.
- A baseline checkpoint, configuration, data manifest, optimizer/resume state, and evaluation report can be reproduced. Preserve the imitation checkpoint for later league and regression use.

**Stage 2A engineering exit:** an audited synthetic dataset, tested trainer/resume path, frozen checkpoint, and honest fresh-match evaluation exist, with correctness regressions resolved. Report any teacher-retention failure or statistical uncertainty separately. This permits proposing the next controlled experiment, not silently launching RL or claiming competitive strength.

**Strength promotion:** a trained policy beats the declared heuristic panel under the prespecified evaluation criterion, with no correctness regressions. This remains outstanding until demonstrated, potentially during Stage 3. Diagnose a weak or inconclusive pilot before enlarging the run.

## Stage 3 — reinforcement learning and controlled ablations

Stage 2B is complete under [its accepted plan](stage2b/PLAN.md) and [configuration](../configs/stage2b.json). Six matched runs and the untouched action test support the exact-efficiency branch: selected-model discard agreement is 99.9826%. Its original confirmation exceeded the 256-block cap. The subsequent Package A confirmation did not demonstrate retention, and its smoothed rank-zero runs missed the 99.9% terminal imitation criterion. The [checkpoint diagnosis](../reports/package_a_diagnosis/REPORT.md) established incomplete training fit. The now-completed [paired learning-rate continuation](../reports/package_a_continuation/REPORT.md) did not support reducing the late learning rate to 0.0001: both seeds had worse target fit and validation discard agreement than matched 0.0003 controls. No smoothed checkpoint is promoted. Placement calibration and call coverage remain unresolved. Stage 3A is now complete under [its fixed engineering plan](STAGE3A_PLAN.md); see the preceding [continuation review](CONTINUATION_REVIEW.md), book assessment and [Suphx assessment](SUPHX_ASSESSMENT.md) for the next scoped choices. Perfect heuristic imitation is not a universal prerequisite for a separately authorized RL engineering pilot.

**Measured on 18 October 2026:** [Stage 3A](../reports/stage3a/REPORT.md) completed all 64 full matches and four finite, non-no-op updates with exact probability replay and verified checkpoint boundaries, using 72.374 of 600 numerical seconds. Its engineering exit is met; competitive strength remains unestablished. All offered calls/kans had zero recorded probability, so the [current review](STAGE3A_REVIEW.md) proposes a bounded action-support and gradient diagnostic before scaling. No later stage starts automatically.

**Authorized on 20 October 2026:** [Stage 3B](STAGE3B_PLAN.md) executes that diagnostic within 300 cumulative numerical seconds, using at most 512 saved actor rows and 32 fixed backward/backend probes. The grid is the original sampler, temperatures 10 and 100, and a 5% legal-uniform mixture. No new matches, optimizer updates, checkpoint writes or policy promotion are included. Book preparation converts the twelve corrected examples into an explicit partial-scenario format. Preserve Stage 3A through its completed snapshot rather than changing its historical inventory.

**Measured on 20 October 2026:** [Stage 3B results](../reports/stage3b/REPORT.md) complete all fixed comparisons in 21.858633 numerical seconds. Neither temperature alternative supplies useful call frequency; the uniform mixture's direct call gradients are zero. The diagnostic's MPS transfer repair is tested and preserves CPU results exactly; the old Stage 3A MPS path still requires that repair before use. [Stage 3C](STAGE3B_REVIEW.md) is a proposed 600-second isolated call-gate/head learning pilot, with 64 training and 256 paired evaluation matches maximum and an identical initialized-policy control. That historical proposal is now authorized by the 21 October continuation request; the fixed [Stage 3C plan](STAGE3C_PLAN.md) governs execution.

Keep full-match PPO with terminal utility `[1.5, 0.5, -0.5, -1.5]` as the planned learning reference. The completed Stage 3A is a smaller engineering substage before PPO: full-match Monte Carlo policy gradients with a zero baseline, at most four complete batches and four updates, to verify returns and behavior-policy provenance. It uses a newly identified categorical sampling wrapper; the present `NeuralPolicy.choose` uses deterministic argmax. Freeze the policy within each predeclared batch, require every planned match to complete, and use that batch for exactly one update. Do not learn from only the faster completed matches when the rest hit a deadline. The user authorized the 600-second scope on 18 October 2026; [STAGE3A_PLAN.md](STAGE3A_PLAN.md) is the executable protocol. Begin with one learner seat against frozen opponents, then admit other learner seats only when policy versions, old probabilities, and per-seat trajectories are correct.

Acceptance criteria:

- Per-player transitions include intervening events and terminal reward even when an opponent ends the match after the learner's final action. True termination and rollout truncation have different bootstrap handling.
- Collect old/new action probabilities under identical legal masks. Log policy lag, KL, clipping, entropy by decision type, value diagnostics, and actual opponent composition.
- Compare early GRP-based reward/credit assignment and privileged history-plus-state critics against the full-match reference at matched compute. Separate actor inputs from privileged inputs.
- Treat hand-boundary truncation as a surrogate requiring evidence; a telescoping reward identity alone is not a proof of an unbiased truncated policy gradient.
- Retain historical/fixed opponents and a separate holdout. Four eligible learner seats provide additional decisions, not four independent matches or a guaranteed fourfold speedup.
- Add raw bounded-utility sequential evaluation at the independent seed-block level. Freeze the candidate, opponent panel, effect threshold, stopping rule, and error budget for each test.
- Use multiple training seeds for serious comparisons; preserve failed and inconclusive runs. Diagnose a plateau before switching to an offline/Q-learning/Monte Carlo fallback.

**Exit:** a frozen RL candidate improves over the frozen baseline on fresh full-match evaluation, meets the declared correctness/latency criteria, and the gain survives the chosen repeatability checks. GRP, privileged critics, conditioning, and league changes become defaults only after their ablations justify it.

## Stage 4 — belief modeling and selective search

Use the cloning/state-construction capabilities audited earlier. Search is an additional experiment after a useful policy and evaluator exist.

Acceptance criteria:

- Joint hidden-state samples satisfy tile conservation, observations, and rule constraints. Distinguish hard-constraint validity from a calibrated posterior conditioned on action history.
- Counterfactual continuation policies receive only their own information. One root action is selected for the real information set; no per-world privileged root optimization.
- Measure root-only value evaluation, then bounded hand/match continuations, against the same policy without search. Document leaf-value/GRP approximation and opponent mismatch.
- Enforce a deployment time budget with policy fallback. Report full-match utility gained per additional millisecond and memory cost.
- Test distillation only after search produces a measured gain. Reevaluate the deployed student independently.

**Exit:** search or its distilled policy improves held-out full-match utility within the deployment budget. Otherwise retain the policy-only champion.

## Stage 5 — independent confirmation and release

Freeze the rules, objective, code/model/data manifests, opponent panel, deployment action rule, and statistical protocol. Release reproducible local-play artifacts and a model card with tested conditions and unavailable comparisons.

Acceptance criteria:

- Confirm the result on fresh seed blocks and the prespecified opponent mixtures; report uncertainty, per-seat/per-opponent behavior, latency, and training-seed variability.
- Any AIVAT-style estimator is independently validated against raw results, uses actual known probabilities and an appropriately frozen evaluator, and saves correction provenance. It is not required for release.
- Stable dan is labeled with its room formula and evaluated opponent population. Local bot results are not presented as directly equivalent to historical human-ladder records.
- Public venues, external bot checkpoints, and third-party data are used only after their relevant access/usage conditions are resolved. No venue has been selected.

## Progress reporting

At each stage, report: completed artifacts and commands; measured results; failures/unknowns; acceptance criteria met or unmet; and the next bounded task. Keep “implemented,” “smoke-tested,” “validated on the fixture corpus,” and “competitive strength demonstrated” distinct. A failed exit gate extends the stage rather than becoming an implied success.


**Stage 3C checkpoint, 21 October 2026:** Four updates on 64 complete training matches establish real call execution and trainable head response with frozen base and exact replay. Evaluation is incomplete: eight paired blocks completed before a new analysis-test file changed the controller's dynamic source membership during block 8. Original runtime/test bytes were unchanged. Preserve the failed attempt and stop under the no-retry rule. [STAGE3C_REVIEW.md](STAGE3C_REVIEW.md) proposes a separate evaluation-only recovery after fixing file membership; no strength promotion is supported.


**Stage 3D authorized, 21 October 2026:** Execute the separate [evaluation-only recovery](STAGE3D_PLAN.md) within 400 cumulative numerical seconds, 45 preparation seconds and 256 fresh paired full matches. Keep the Stage 3C final/initialized checkpoints unchanged. Fix explicit source membership and durable match lifecycle before formal games; do not retry or pool the stopped Stage 3C blocks.


**Stage 3D measured, 21 October 2026:** The fixed recovery completed 32 fresh paired blocks / 256 full matches with no retries or failures and no training. Both checkpoints averaged rank 2.6328125; mean utility difference 0, predeclared 95% interval [-1.440484, +1.440484]. All 128 paired placements matched; 127 action sequences matched. The independent audit and final hash-only closure pass, and 350.845432 of 400 numerical seconds are charged. Keep the original reference and both experimental checkpoints unchanged. The [review](STAGE3D_REVIEW.md) proposes one longer fixed learning horizon, ending at update 32, plus a strict partial-call book validator; neither has begun.


**Stage 3E authorized, 21 October 2026:** Execute the [fixed longer learning horizon](STAGE3E_PLAN.md): 28 additional 16-match batches from the saved update 4 model and optimizer, ending at update 32, followed by 32 fresh paired evaluation blocks. Keep architecture, learning rate, reward and opponents fixed. Cap numerical work at 1,200 seconds; protect 400 seconds for evaluation, 40 for audit and 60 for shutdown during training admission. Continue strict partial-call book validation without training on books.

**Stage 3E measured, 21 October 2026:** All 448 training matches, 28 additional updates and 256 fresh evaluation matches completed within 956.038681 of 1,200 numerical seconds. Both endpoints averaged rank 2.5625; all 128 paired action sequences and placements matched, yielding utility difference 0 with the fixed 95% interval [-1.440484, +1.440484]. All updates changed both heads, and evaluation call probabilities changed slightly, without demonstrated stronger play. Targeted tests, full evidence audit and hash-only closure passed. The two partial book packets now pass strict structural validation and remain quarantined. Preserve the original and experimental checkpoints. The [current review](STAGE3E_REVIEW.md) proposes a 300-second saved-training-data diagnosis of learning signals and win-versus-call behavior before another intervention; this is unstarted and adds no authorized run to the completed campaign.


**Stage 3F engineering authorized, 22 October 2026:** The user accepted the revised value-learning direction. Execute [STAGE3F_PROGRAM.md](STAGE3F_PROGRAM.md): an integrated training-only diagnosis, parallel four-seat exploring collector and bounded-memory loader/backend pilot. Cap this milestone at 7,200 cumulative numerical seconds, including a 300-second diagnostic sublimit, 120-second shutdown reserve, 32 GiB sampled RSS and 2 GB artifacts. Use a fixed development-only population and benchmark before proposing the first large corpus/learning budget. Preserve complete-match targets as the future value-regression control, cross-fit placement-derived targets, and keep PPO off historical mixed-policy data. No three-round or large-corpus training has begun.

**Stage 3F engineering completed, 22 October 2026:** The saved-data diagnosis, 16-plan serial/parallel parity, four 32-match worker probes, fixed 1,000-match pilot, streaming/backend measurements and independent audit/replays completed. The pilot supplies 646,377 decisions at 11.505 matches/s; the MPS gradient probe reaches 3,286 rows/s without optimization. All 1,144 unique matches / 737,994 rows pass the bulk audit, and all ten prespecified replays match. The closed ledger charges 186.193133 / 7,200 seconds, with one preserved pre-formal test failure and no failed formal game. Read the [results](../reports/stage3f/REPORT.md).

**Stage 3G proposal at Stage 3F completion:** Follow [the review](STAGE3F_REVIEW.md) toward a fresh 4,000-match corpus, explorer-only optional-row values, complete-match weighting, terminal-versus-hand-difference targets and alpha-zero/one controls. Budget five out-of-fold placement fits plus a final predictor, calibration gates, actual MPS Adam/resume checks, four fixed two-pass arms and a common 1,280-game development panel. Freeze the executable counts, thresholds and resource reservations before work; keep competitive confirmation separate. All current pilot and book records remain training-ineligible.


**Stage 3G authorized and admitted, 22 October 2026:** Execute the [fixed protocol](STAGE3G_PLAN.md) within 7,200 cumulative numerical seconds, 32 GiB sampled RSS, 8 GB artifacts and 5,500 match attempts. Actual CPU/MPS Adam and tolerance-verified resume checks passed; measured forecasts admit the full fixed programme. Freeze 4,000 fresh corpus matches (3,200 training/800 development), six whole-match-separated placement fits, two-pass explorer-only optional-row fitting, and the 64-block common development panel. A failed calibration gate skips both placement-difference arms; a failed formal operation stops later formal work. Keep all source identities, lifecycle evidence and the unchanged historical snapshots. Confirmation and policy promotion remain separate future work.


**Stage 3G completed, 22 October 2026:** The fixed 4,000-match corpus, all six placement fits, all four 400-update learning arms and all 1,280 development games completed under [the protocol](STAGE3G_PLAN.md). All calibration gates passed. The best candidate averaged rank 2.621094 versus the original reference's 2.421875; all four observed differences are negative, so no candidate is selected or promoted. All-row, checkpoint/target/result and forty fixed-replay audits passed. Twenty successful numerical operations charge 1,165.485892 / 7,200 seconds. Preserve these results and checkpoints; do not extend the same run after inspecting outcomes. [STAGE3G_REVIEW.md](STAGE3G_REVIEW.md) proposes an unstarted 300-second saved-state diagnosis before another learning intervention.


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


## Stage 3I — frozen-reference residual learning

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
