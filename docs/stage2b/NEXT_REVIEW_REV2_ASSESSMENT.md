# Assessment of the Stage 2B next review revision 2

The revision is useful and improves the direction of the next study. Keep its emphasis on stability, independent match outcomes, measurable exploration and profiling. Revise several causal claims, the stopping rules and the four-run allocation before turning it into an executable protocol.

This assessment recommends a revised Package A for a later authorized run. The current request is a review: no training, inference campaign, new corpus or external contact has started. The original revision 2 proposal, [first proposal](NEXT_REVIEW.md) and completed Stage 2B evidence remain separate. The checked source is commit `60ceff6b7fd8d4d40f96d5ceec54a2cac720ba3c`; the copied proposal has SHA-256 `44ed599e46fe8176e1a0abb98734d6a57b369c0a9006304d963a162b74a2f213`.

## Recommended disposition

| Suggestion | Decision | Change needed |
|---|---|---|
| Bundle fixed retention and stability diagnosis into one review package | Adopt as proposed scope | Keep separate budgets/results. Run confirmation against the unchanged deployment first, then instrument training. They need not succeed together. |
| Fixed 319-block confirmation | Keep | Same frozen primary, panel, utility, 0.10 margin and one-sided 0.05 alpha; fresh fixed seeds; ten-minute safety ceiling. |
| Earlier stability instrumentation | Adopt | Measure actual activation/logit scales and gradient spikes. Frozen-head loss is a warning proxy, not a direct norm measurement. |
| Label smoothing as a candidate intervention | Adopt for testing | One precise legal-action smoothing definition, separate training targets and original labels, both seeds. It is not a proven fix. |
| Two interventions within four runs, with control reruns if needed | Replace | Use two contemporary controls and two runs of one intervention. A second intervention plus controls would require six runs. |
| Convergence stopping | Adopt the principle, rewrite the rule | Distinguish target reached, plateau and deterioration. The example rule can stop because KL worsens. |
| Measure policy entropy and non-teacher probability before RL | Adopt | Low KL to a tie-uniform teacher is not the same as low policy entropy. Measure distributions by decision type. |
| Separate placement model with more matches | Adopt as a later study | Count independent matches, not seats; begin with a small collector pilot and learning curve. Match the intended opponent population. |
| Compile/vectorize exact-efficiency work | Profile first | Preserve exact outputs and actor visibility; speedup is a measured outcome, not a prerequisite promised in advance. |
| Time-aware evaluation sizing | Adopt with fixed counts | Keep variance safeguards and candidate-specific planning. Time is a safety ceiling, not an outcome-based stopping rule. |
| External human data and stronger yardsticks | Keep as future options | Establish exact data/checkpoint/interface availability first. No contact or acquisition follows from this review. |
| Running-guide status and review-summary fixes | Valid corrections | Apply in the next documentation revision while preserving the completed experiment's archived bytes and inventory. |

## What the saved evidence actually says

### The stability signal is useful but not a proven cause

The quoted untrained placement-loss medians reproduce. For seeds 171 and 172, respectively, the median is 3.6956 and 4.7433 at steps 4,501–5,000, then 1,029.0178 and 15.7542 at steps 6,501–7,000. This is a useful reason to instrument the shared representation earlier.

A frozen random head responds to representation direction/alignment as well as magnitude. Its loss alone does not establish that the context norm was already growing. The very large late losses support investigating scale, but actual context norms and activation ranges were not recorded. Likewise, the proposed explanation involving separability, adaptive optimization and weak decay is a hypothesis, not the identified cause.

Cross-entropy can drive increasing margins in separable settings; the established result for unregularized linear predictors under gradient descent provides motivation, not a theorem about this nonlinear AdamW network. [Soudry and colleagues](https://jmlr.org/papers/v19/18-188.html). Label smoothing has empirical support as a way to reduce overconfidence, but that does not establish a remedy for this collapse. [Müller and colleagues](https://arxiv.org/abs/1906.02629).

The statement that gradient norms gave no warning is too strong. Their medians stayed modest, but individual spikes preceded severe policy loss:

| Seed | First gradient norm above 10 | First above 50 | First policy cross-entropy above 10 |
|---|---:|---:|---:|
| 171 | 5,828 | 9,337, norm 629.21 | 9,380 |
| 172 | 7,337 | 10,671, norm 439.14 | 10,888 |

Record upper quantiles and maxima as well as medians. A lone spike is not automatically a correct abort threshold; stable controls are needed to assess false alarms.

Log actual context L2/RMS norms, per-module parameter and update norms, raw legal logits, centered legal logits and gaps between best and non-best candidates. Track mean and extreme values on both sampled batches and a fixed diagnostic batch. A frozen probe can supplement these measurements. **Exclude padded candidates:** the implementation masks them with the float32 minimum, approximately −3.4e38, so a naive maximum absolute logit would report an enormous value even in healthy runs. Centered gaps are invariant to a harmless common offset; raw offsets are still relevant to floating-point resolution. Centering after precision has already been lost cannot recover candidate differences.

The proposed ten-times thresholds need a defined reference batch, warm-up, minimum scale, sampling cadence and persistence. They cannot be validated retrospectively from missing context/logit measurements. Replaying the other example, 50 consecutive losses above ten times the prior median, fires only after severe policy loss: steps 9,427/10,933 for an expanding previous-history median, or 9,427/10,951 for a trailing-500 median. That condition alone does not deliver the promised advance warning. Log alarm traces, define success prospectively and preserve stopped controls as failures observed by the study.

### The convergence rule needs a precise rewrite

I replayed the natural interpretation of the suggested rule on the saved rank-0.1 validations: all three recent discard agreements at least 99.9%, and `(first_KL − last_KL) / first_KL < 0.10`.

| Seed | First stop | Updates saved versus 18,500 | Reason the KL condition passes |
|---|---:|---:|---|
| 171 | 12,000 | 35.1% | KL worsens by 25.5% across the window |
| 172 | 8,500 | 54.1% | KL worsens by 40.1% across the window |

The combined update saving is 44.6%, but these are deterioration triggers, not evidence of smooth convergence. Requiring KL to improve monotonically makes neither run trigger this example rule. The checkpoint selector also does not literally always select the last step: seed 172 selected step 18,000 rather than 18,500.

Use separate prospective conditions for reaching an acceptable imitation target, lack of material improvement and instability. An absolute unsmoothed-KL tolerance alongside agreement and error severity is worth testing; a threshold chosen after seeing these histories must be labeled retrospective. Do not change the Stage 2B selection or reuse its test for tuning. **Do not early-stop the stability intervention merely because imitation looks good:** it must survive the full intended horizon, beyond the known collapse region.

### Low teacher KL does not mean a deterministic sampling distribution

The teacher target spreads mass uniformly among equally preferred actions. Matching that distribution can leave several actions probable. On the selected primary model's full validation, saved policy cross-entropy minus KL is **0.362164 nats**, the mean target entropy under the same row weights. This is not a direct measurement of policy entropy, but it demonstrates that the target itself is not uniformly deterministic across decisions. The deployed wrapper's deterministic argmax is a separate choice from the sampling distribution PPO would use.

The concern about insufficient non-teacher exploration is nevertheless sound. For one state, if `m` is total model probability on the teacher best set, then `KL(q || p) = −log(m) + KL(q || p conditioned on that set)`. Low KL constrains mass outside that set. An overall average does not establish the probability of calls in every rare call-eligible state.

Measure entropy, normalized entropy where applicable, teacher-best-set mass, non-best mass and legal-call/kan mass, stratified by action availability, legal-choice count and teacher/student population. Forced decisions should be reported separately. A tempered or smoothed policy must use the same distribution for sampling and recorded old probabilities; merely changing sampling while retaining original log probabilities invalidates PPO ratios. A strong forward KL anchor to a teacher that rejects calls can also suppress the intended exploration, so its direction, weight and schedule need explicit treatment.

Temperature, an earlier checkpoint or label smoothing are options to compare, not proofs of RL readiness. A tiny probability of arbitrary legal calls is not a substitute for valid, useful post-call examples and the calls contract. A separate placement model and a compiled efficiency loop should not become mandatory blockers for every minimal terminal-utility RL experiment.

## Revised Package A for a later authorization

### A1 Frozen retention confirmation

Use the unchanged Stage 2B primary checkpoint, 319 fresh independent blocks and 2,552 match executions. Keep the original hypotheses, panel, rules, seat rotation, margin and alpha. Freeze the complete seed list before outcomes; inspect operational failures/progress, not interim strength to alter the test. Do not pool the pilot into confirmation or switch the candidate after observing it.

The saved rate is 9.09061 executions/second, giving a **4.68-minute planning estimate**. A ten-minute ceiling, 32 GiB sampled memory and 1 GB new artifacts is reasonable proposed scope. A resource stop produces an incomplete study. Run this on the unchanged implementation before instrumentation, or use a separately frozen checkout. A later changed candidate does not inherit its confirmation.

### A2 One intervention with current controls

Use a **45-minute cumulative training-related ceiling**, **four full-run attempts including stopped/failed runs**, 32 GiB sampled memory and 2 GB additional artifacts:

1. Instrument and test safe scale measurements, unchanged forward/loss behavior, RNG isolation, masking and stop reporting. Charge all probes, CPU/MPS checks, loading, derivation and validation from their first invocation.
2. Time the instrumented code and freeze the protocol, thresholds, sampling, seeds and common 18,500-update durability horizon. If the complete comparison cannot fit, reduce scope before starting; do not quietly shorten the horizon below the failure region or leave only one intervention seed.
3. Run two instrumented original rank-zero controls and one prespecified intervention on the same two seeds. A 1,000-step tolerance check alone does not establish historical-control equivalence near the 9,000–11,000-update failure region. Keep old histories as reference evidence; defer a second intervention.
4. Require the intervention to complete the horizon with no scale alarm and at least 99.9% fixed-validation discard agreement on the **terminal 18,500-update checkpoint of each seed**, then report full validation, unsmoothed KL, error severity and probability mass. Show the late-validation trajectory and report selected and terminal checkpoints separately; an earlier good checkpoint cannot establish durability. A safety-stopped control is an observed unstable outcome. Report all attempts, including inconclusive results; do not add rescue runs from unused time.

One reasonable first intervention is ε=0.01 smoothing over legal non-best **semantic** candidates. Let `B` be the teacher best set, `N` the legal non-best set and `q` the original target. When `N` is nonempty, use `(1−ε)q(a)` on `B` and `ε/|N|` on `N`; otherwise keep `q`. Keep forced decisions unchanged/excluded from policy loss, and never allocate mass to padding or illegal actions.

The current training validator requires original target mass to stay within the best mask. Add a separate smoothed **training target** rather than changing the stored teacher labels or disabling that invariant. Compute diagnostic KL and agreement against the original targets. Smoothing discourages infinite relative separation, but it does not bound a common logit offset or hidden-layer rescaling; those measurements must remain acceptance criteria. Bounded output scores and head-only weight decay likewise do not guarantee bounded internal activations.

Four runs at the historical rank-zero costs would take **25.29 minutes**. A 20% instrumentation allowance plus eight minutes for checks/finalization gives **38.34 minutes**, making 45 minutes plausible rather than guaranteed. This is a planning estimate, not a booked runtime or automatic extension.

A1 and A2 may share one authorization while retaining independent hypotheses, immutable identities and separate budgets. Execute sequentially on this Mac so timing measurements and source identity remain interpretable. The package ends with review; it does not launch RL, a large outcome corpus, compilation work or external contact.

## Later studies worth retaining

### More independent outcomes for placement prediction

The underlying diagnosis is persuasive: many decision rows repeat a small number of final outcomes. Correct the independence language to **800 match clusters yielding 3,200 correlated seat labels**. Four ranks within a game form a permutation. A 30,000-match corpus would provide 120,000 seat labels from 30,000 match clusters, a 37.5-fold increase in matches, not 120,000 independent trials.

Split, weight and estimate uncertainty at match level, including any shared-wall relatives. Limit sampled rows so longer matches do not silently dominate. First collect a small measured pilot and establish a learning curve before committing to 30,000 matches. The 40.15-minute extrapolation from the benchmark is arithmetically valid, but a collector with hand-start observations has different serialization/storage costs. The benchmark's compact records do not contain the proposed hand-start training states.

Define pre-deal versus post-deal timing. A score/round model is a useful coarse GRP baseline. Preserve absolute scores in relative-seat order, initial-seat/tie priority, round/dealer, honba, sticks, rules, per-seat behavior versions and explicit continuation-safe hand IDs. Score differences alone lose bankruptcy/end-threshold information. If predictions are intended to condition on a dealt hand, include audited own-hand/dora information or explicitly accept a coarser conditional predictor.

All-four-seats candidate self-play is a different return population from one learner against a frozen mixed panel. Collect the relevant population or measure transfer, and account for policy drift. A separate calibrated placement model can support a later GRP ablation; it is not automatically a decision-state critic or a mandatory prerequisite for the terminal-utility reference.

### Optimize the measured bottleneck after profiling

The native shanten calculation is already compiled. The surrounding Python candidate/improving-tile loops, conversions, cache calls and boundary overhead are plausible targets. Preserve existing candidate deduplication and caching, profile the exact workload, then consider a native or batched path.

Efficiency computation consumes 48.0513% of measured policy-choice time. Eliminating the entire phase would give an ideal **1.925× policy-choice ceiling**; making that phase twice as fast yields about **1.316×**, and five times as fast about **1.624×**. Whole-match gains are smaller. Those are Amdahl estimates, not promised optimization results.

Require equality across the 5,138,892 existing aligned candidates, plus targeted open-hand, red/drawn, exhausted-tile and kan fixtures. Existing closed-hand data alone cannot validate unseen post-call states. Preserve the player-visible boundary; an upstream all-seat convenience encoder needs its own visibility audit. Re-benchmark with the exact frozen policy and documented cache/startup conditions. Profile now when separately scoped, but make a substantial rewrite conditional on a measured need rather than a universal RL gate.

### Keep power-based counts and time safety ceilings

The proposed approximately 1,030 blocks for a +0.05 difference is a valid raw-SD illustration, but it omits the current 1.25 variance multiplier:

| Illustration | Blocks | Executions | Time at the saved rate |
|---|---:|---:|---:|
| Existing retention confirmation | 319 | 2,552 | 4.68 min |
| Two-sided superiority, 80% power, raw SD | 1,034 | 8,272 | 15.17 min |
| Same superiority target, 1.25× SD safeguard | 1,616 | 12,928 | 23.70 min |

For the last two rows the normal planning formula is `ceil(((z_0.975 + z_0.8) × SD / 0.05)^2)`. A new opponent/candidate comparison may have different variance, and multiple candidates or endpoints need an explicit selection/testing plan. Preserve a fixed computed count plus a wall-time safety cap. If the required count cannot be completed, report incomplete/inconclusive; do not treat an arbitrary time stop as a completed fixed-count test.

### Human data and stronger comparisons

Both are strategically useful directions, but their necessity should follow the next scientific question. Suphx reports human-log supervised initialization followed by RL, and NAGA's original technical article reports training on expert Tenhou records. [Suphx paper](https://arxiv.org/html/2003.13590v2), [Dwango's NAGA article](https://dmv.nico/ja/articles/mahjong_ai_naga/). The public [Mortal introduction](https://mortal.ekyu.moe/) establishes a deep-RL system but does not by itself verify the proposal's blanket claim about every system's training initialization; use checkpoint-specific provenance instead.

A stronger yardstick requires an identified available model/version, rule and action adapter, runtime measurement and permitted use. Asking an author is one possible route, not automatically the first necessary step. This review does not authorize messages, data acquisition or accounts. Do not assume an access request guarantees free data or weights.

## Documentation corrections and preserved evidence

`RUNNING.md` has a stale implementation-in-progress line; the correct status is that Stage 2B is complete with retention inconclusive. A revised review guide should lead with the feature gain, frozen test result, retention uncertainty and instability finding, followed by the distinct costs of A1/A2.

This assessment is saved separately so the original completed Stage 2B files and hashes remain intact. Apply those small text fixes in the next versioned project update, retaining the completed snapshot as the evidence reference rather than regenerating its historical manifest against changed text.

The descriptive checks contain the proposal/source hashes, reproduced medians, plateau-rule replay, entropy identity and resource/power arithmetic. These checks read saved data only. They do not measure new policy entropy, prove the collapse mechanism, validate a prospective stop rule or run either package.
