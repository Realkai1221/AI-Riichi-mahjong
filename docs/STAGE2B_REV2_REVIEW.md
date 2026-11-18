# Review of the revised Stage 2B proposal

Reviewed 17 October 2026 against the saved Stage 2A measurements and current implementation. The supplied revision 2 is preserved verbatim; [the first draft](STAGE2_NEXT_INSTRUCTIONS.md), trained checkpoints, and Stage 2A evidence are unchanged. This review proposes changes to the next experiment; it does not start implementation, training, self-play benchmarking, or external contact.

**The most useful change is to make exact, player-computable tile-efficiency features the next main candidate.** That directly addresses the observed errors. I recommend adopting that direction, keeping the history window fixed, adding matched raw-input controls, and narrowing the extra work so the experiment still has a clear result.

## Recommendations at a glance

| Proposal | Recommendation | Reason or correction |
|---|---|---|
| Exact shanten, ukeire, and improving-type features | Adopt as the principal candidate | Targets 7,213 of 7,227 observed validation discard errors; keep the model labeled feature-assisted |
| Features before the rank-loss ablation | Adopt | Test the most directly relevant intervention first; rank loss still needs a controlled comparison |
| Historical raw-input student as the only reference | Modify | Preserve it, but add two matched raw runs if we want to attribute improvements to the features |
| History window 128 to 160 in the same change | Defer to a separate comparison | It changes normalized inputs on almost every nonempty-history row, not only overflow rows |
| Below 95% agreement suggests a bug | Modify | Exact feature and rule reconstruction are correctness checks; neural accuracy is a learning diagnostic |
| Remaining-hand encoder | Keep as a conditional experiment | Persistent errors despite exact features first require diagnosis of scaling, scoring, and optimization |
| Student-state data aggregation | Keep conditionally | Useful if coverage is limiting; define the trigger and keep real behavior separate from teacher labels |
| Neural self-play throughput | Adopt with a bounded protocol | Measure the new feature cost and complete game loop; fit all worker trials and sustained work within 30 minutes |
| Long-run operating routine | Adopt as documentation | Useful preparation; no overnight run or persistent machine-setting changes are implied |
| Call-capable heuristic | Retain as a separate coverage task | Valuable, but its yaku predicate and post-call legal-discard calculation need specification |
| Explicit Stage 3 proposal trigger | Adopt | Retention plus a usable runtime can justify a bounded RL proposal; neither proves expert play |
| Human records and stronger opponents | Retain as a parallel planning item | Identify an exact source/checkpoint and intended use before deciding what permission or integration is needed |

## Why the feature proposal is useful

The policy diagnostic contains 41,225 optional teacher-discard decisions. The extended model makes 2,065 choices with worse shanten, 5,148 with worse ukeire at equal shanten, and 14 differing only in red retention. Shanten and ukeire therefore explain **99.81% of these recorded preference errors**. This does not mean they explain 99.81% of the full-match loss: error frequency is not causal placement regret.

The features remain legal inference inputs because they are computed from the actor's own hand, public tiles, and legal candidate action. Using them still produces a trained neural policy. It changes the research claim from learning tile efficiency entirely from raw counts to learning decisions with an exact tile-efficiency calculation available. Keep the raw model for the former question and the assisted model for the practical strength objective.

This is consistent with established practice: Suphx describes more than 100 engineered look-ahead features, including simplified winning-hand/value calculations. Its design supports considering computed features, but does not establish the value of our particular three fields or justify copying its compute budget. [Suphx, section 2.2](https://arxiv.org/html/2003.13590v2).

Keep the features as inputs while leaving all legal actions selectable. Later strategic training must still be able to prefer a less efficient hand when defense or placement warrants it.

## Exact feature contract to adopt

Use a separate numeric candidate array, for example `candidate_efficiency[N,4]`, containing shanten, ukeire, improving-type count, and applicability. Keep the existing semantic action tokens and their red/drawn/consumed-tile distinctions. Declare scaling, dtype, schema identity, and how the numeric branch enters the candidate scorer. Merely appending numeric values to the existing categorical token fields would not provide this interface.

The current safe arrays appear sufficient for the original corpus:

```text
own_counts     = 4 * tiles[:, 0]
visible_counts = 4 * tiles[:, 4]
discard_type   = action_tokens[candidate, 1] - 1
```

The counts use exactly representable quarter increments. Validate exact integer values and allowed ranges before conversion. Subtract the discard from the own-hand counts, but **leave visible counts unchanged**: the copy is already known and remains visible in the river. Compute once per distinct discard type, then reuse the result for equivalent red/normal and drawn/held variants. The new efficiency values are identical across those variants; their existing action descriptors remain distinct.

A read-only feasibility check on the first three training matches found **1,726 decision rows and 14,051 discard candidates**, with exact agreement between reconstructed shanten/ukeire and the saved teacher keys. Probe and provenance. This is a small alignment check, not the required exhaustive feature audit or an independent proof of the shanten algorithm.

Before training, require exact equality on every applicable training and validation candidate, plus direct-view fixtures, information-boundary invariance, and candidate-order/masking checks. Precompute numeric features once for stored training data, save separate derived artifacts, and bind them to the original manifest and converter version. Inference must compute the same values from `PlayerView`. Measure preprocessing and inference cost; do not hide feature calculation outside the resource ledger.

Three wording corrections matter:

1. `DiscardScore.improving_types` is a tuple. The new scalar is its length; that length is not already present in the stored teacher preference keys and needs its own check.
2. The helper excludes types with no unseen copies. At tenpai it counts non-exhausted shape-completing types, not all structural waits or necessarily legal ron/tsumo waits. It does not account for yaku or furiten. Ukeire is a count of potentially available improving copies, not actual wall contents or a winning probability.
3. RIICHI is a separate declaration action in this engine. It receives the explicit not-applicable feature; the subsequent legal DISCARD receives ordinary efficiency features. Do not merge the two decisions.

Build a deterministic diagnostic that reconstructs the teacher's best set from the computed values, red flag, and applicable action-priority rules. It should match exactly where the rule is defined. Use this to distinguish conversion/target bugs from a neural model that has not learned the ordering. Agreement below 95% is a reason to inspect learning curves, normalization, ranking margins, and loss competition; it is not proof of an implementation error. Near-perfect agreement likewise does not replace full-match evaluation.

These conclusions follow from [policies.py](../src/riichi_ai/policies.py), [features.py](../src/riichi_ai/features.py), [semantics.py](../src/riichi_ai/semantics.py), and the [heuristic contract](HEURISTICS.md).

## Keep history and compatibility separate

Keep 128 events for the first feature-assisted comparison. The current encoder divides history length and omitted-event counts by `MAX_HISTORY`; raising it to 160 changes normalized global inputs on nearly every row with a nonempty history. It also changes position-embedding dimensions and the dataset's history-prefix reconstruction. Thus the revision understates the extent of the change when it describes only the approximately 2.4% of overflow rows.

The longest observed history of 147 events is a corpus observation, not a universal upper bound. Retain explicit overflow reporting even in a later 160-event experiment. Recover earlier events from saved safe history streams using validated prefix boundaries; padding a truncated tensor does not restore them.

Introduce version-specific encoder/model loading. The current loaders accept only the current global schema, so replacing a constant would make historical checkpoints incompatible. Preserve a runnable v1 path and test old-model inference after adding v2. A new feature tensor requires a derived-data/model schema, not rewriting the original corpus. [Model contract](MODEL.md), [model implementation](../src/riichi_ai/model.py), [dataset implementation](../src/riichi_ai/dataset.py).

## A clearer use of the eight-run budget

For this project's scientific as well as practical goals, I recommend the following allocation. All runs initially target the same update budget, data, optimizer, validation schedule, and two declared seeds; reduce the common budget before comparisons if measured runtime requires it.

| Execution order | Variant | Runs | What it answers |
|---|---|---:|---|
| First | Feature-assisted, rank weight 0.1, history 128 | 2 | Does the main candidate learn and run reliably? |
| Second | Raw-input, rank weight 0.1, history 128 | 2 | What changes when the exact features are added under matched conditions? |
| Third | Feature-assisted, rank weight 0, history 128 | 2 | Does the rank objective help or hinder this candidate? |
| Conditional | One remaining-hand **or** student-data experiment | At most 2 | Which diagnosed limitation is worth the remaining budget? |

Reuse the appropriate earlier paired controls for the conditional comparison; do not start another four-run control/variant study. If no condition justifies it, leave the last two runs unspent. The historical Stage 2A checkpoint remains useful, but it has a different selection history and only one seed, so it cannot replace a clean two-seed control for a feature-effect claim.

Match shared initial weights explicitly where shapes permit. Simply giving different architectures the same random seed does not guarantee identical shared weights after their initialization code consumes different numbers of random draws. Preserve the row-sampling stream and report new parameters, preprocessing, and inference cost. This design isolates the feature package; it does not prove that each of its three fields is independently necessary.

For the two-percentage-point triggers, define the denominator as optional teacher-discard decisions, report teacher-focal and frozen-student-focal strata separately, and specify row/match weighting and aggregation over training seeds before training. Equal match counts do not imply equal decision weights. A clear primary recipe score is the equal-weight mean of the two strata's discard agreement, then the corresponding shanten-error rate and KL, averaged over both seeds. Report per-match variability as a diagnostic.

Treat these thresholds as rules for spending the remaining experiment budget, not significance tests or causal findings. Student-state error differences can also reflect game stage, hand difficulty, or action mix. A fixed holdout generated by the old student is not a complete test of the new candidate's own state distribution. Preserve the new holdout; inspect new candidate behavior through separately identified development rollouts if needed.

When exact ukeire is already supplied, remaining ukeire errors do not specifically establish a need for a remaining-hand encoder. First check whether the model can learn the provided ranking on a tiny batch, then examine frozen training/validation error and score margins. A remaining-hand branch is worth testing if that diagnosis supports an architectural limitation; label it a separate schema change. Teacher queries on student states remain a sensible alternative when coverage is the stronger hypothesis. [DAgger paper](https://proceedings.mlr.press/v15/ross11a.html).

Resolve selection after all eligible experiments: select the recipe by the predeclared two-seed validation score, then freeze that recipe's lower-numbered primary-seed checkpoint for the single action-test inspection and possible retention confirmation. Do not select one recipe before the conditional experiment and leave the final eligibility ambiguous. Keep the original held-out test historical and the new action test sealed until this choice.

## Throughput and long-run preparation

Adopt a short all-neural self-play benchmark with 1 and 4 workers first; attempt 8 and 14 only if memory and throughput justify them. Use one CPU inference thread per worker initially and record the configuration. A practical aggregate 30-minute envelope is short trials plus a 20-minute sustained run and setup/drain reserve. The sustained run and worker sweeps must share the same ceiling; the separate one-hour training ceiling still applies to training-related work.

The old 0.456 ms average includes forced bypasses and the old encoder. Multiplying it by roughly 650 decisions gives approximately 0.30 seconds of policy-call work per match under those assumptions. It excludes the new shanten/ukeire calculation and other game-loop costs, and the all-neural policy can change match length and action mix. It is not a measured end-to-end self-play or PPO training rate.

Record complete-match throughput, optional network calls, forced choices, feature/encoding/model/engine/audit time, warm/cold cache statistics, process-tree memory, and throughput over time. Temperature or thermal-pressure telemetry can be included when available; throughput drift alone should not be attributed exclusively to heat. No learning in this benchmark means the result still excludes optimizer and rollout-buffer work.

Keep compact source/checkpoint/seed manifests, completed/failed/truncated match counts and per-match summaries, plus sampled or failure replay evidence. “Summary metrics only” should mean no bulk training-trajectory corpus, not loss of reproducibility. Account for benchmark executions separately from the 600 saved-data matches and the paired evaluation campaigns.

A documented long-run routine should cover scoped sleep prevention, checkpoint cadence, actual resource-stop behavior, crash recovery, and log/disk retention. Defining that routine does not run it or change persistent system settings.

## Calls are useful coverage work but need a narrower specification

The call-capable heuristic would exercise important missing states. I would preserve it as a separate follow-up coverage task rather than automatically spend one of the four neural-comparison campaigns on it. That keeps the main Stage 2B result focused and allows two finalist recipes to be compared across both seeds. If a call-heuristic campaign is chosen within Stage 2B, it must replace a named campaign, with the reduced neural replication stated.

For the first implementation, define the yaku predicate conservatively: an exposed value-honor triplet already exists, or the proposed pon creates one. This locks in a qualifying yaku for an eventual completed hand; it does not guarantee completion. A concealed honor pair, prospective tanyao/honitsu/toitoi, or dora alone is not the same guarantee. Seat wind follows the current dealer-relative position, not the absolute initial-seat identity; test double-wind cases and rotations.

Evaluate a call together with its best **legal mandatory discard**, respecting kuikae. Compare that result with the current standing hand under PASS; evaluating only the immediate consumed-tile hand or an unrestricted discard can mis-score the call. Use the actor's permitted hand/public information, never hidden simulator state. If another response preempts the call, do not apply its planned continuation. When a call resolves, choose from the actual next player view's legal discards.

Test exact red/copy consumption, chi direction, yaku preservation, call priority, legal continuation, and visible-copy deduplication against real explicit-wall transitions. Record requested, executed, and preempted calls separately. Excluding kan is reasonable for this first heuristic, but does not establish kan coverage. The current shape-only shanten helper is not a yaku evaluator.

## What passing Stage 2B would justify

Adopt the explicit trigger for proposing Stage 3: a frozen candidate passes the declared teacher-retention confirmation, the information/data checks remain valid, and measured runtime supports a bounded experiment. Use the existing maximum of the variance floor and inflated pilot SD, the 0.10 margin, fresh confirmation blocks, and the explicit power/cap/futility rules. Do not reinterpret the recipe comparison or the action test as confirmation.

This can justify a small PPO readiness experiment with complete-match rewards and correct learner-seat provenance. It does not establish expert strength, validate the existing rank head as a critic, or remove open-hand/kan coverage gaps. Privileged critics, GRP, and other reward changes remain distinct later comparisons.

A stronger external reference and permitted human data remain useful parallel planning topics. Suphx and the original NAGA developer report explicitly describe training from human Tenhou games. Mortal exposes a documented offline/online training framework, but the particular released checkpoint's original data lineage should be verified before asserting that all three followed the same initial procedure. [Suphx, section 3](https://arxiv.org/html/2003.13590v2), [NAGA developer report](https://dmv.nico/ja/articles/mahjong_ai_naga/), [Mortal documentation](https://mortal.ekyu.moe/).

Identify the exact bot version/checkpoint, interface, rules, intended use, and published conditions before assuming author contact is necessary or sufficient. Optional human play/replay is useful qualitative feedback after a small interface exists; it is not a replacement for the fixed-panel measurements. No contact, corpus acquisition, account registration, or paid compute follows from this review.

## Recommended next instruction

Prepare Stage 2B around the feature-assisted model with history 128, exact actor-only feature validation, preserved v1 loading, matched raw controls, rank-weight ablation, and at most one conditional two-seed experiment. Keep the existing data, training, storage, memory, and statistical ceilings. Add the bounded throughput protocol and long-run documentation. Keep calls_v1 as a separately specified coverage task. Freeze the complete experiment and selection rules before training, then report both successful and unsuccessful results.

The original Stage 2A manifest deliberately hashes the documents as they existed at the completed snapshot. This review and the proposal copy are new files, so historical verification remains intact. Future implementation changes should verify the old evidence against its saved Git snapshot rather than rewrite its historical hashes.
