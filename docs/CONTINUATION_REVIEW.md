# Review: continuation results, book data and the next build

The [completed experiment](../reports/package_a_continuation/REPORT.md) does not support the tested learning-rate reduction. All four continuations finished, but lowering the rate worsened target fit and validation discard agreement relative to matched controls on both seeds. Keep every outcome and the earlier original rank-0.1 reference. Do not promote a smoothed checkpoint or extend the same comparison until it appears to pass.

The useful next direction is **a small full-match RL engineering pilot, alongside better book-scenario evidence**. These are distinct tasks with distinct success measures. The failed 99.9% smoothing criterion is not a universal condition that must be met before any RL infrastructure can be built.

## What to keep from this session

| Work | Decision | Reason |
|---|---|---|
| Exact supervised continuation and state recovery | Keep | Paired model, optimizer, random state, sample stream and original safety state are verified. This machinery is useful for later controlled studies. |
| Late learning-rate reduction from 0.0003 to 0.0001 | Do not adopt as an improvement | Both seeds miss the declared comparison. A smaller step size is not a demonstrated remedy at this starting point and horizon. |
| Original rank-0.1 reference | Preserve as a possible initialization | It remains the better imitation reference, with its earlier strength uncertainty unchanged. A future RL experiment receives its own optimizer and identity. |
| Book questions and explanations | Develop as a separate dataset | They can encode distinctions our immediate-efficiency teacher omits. Source assumptions and transcription quality must remain visible. |
| Suphx research | Use as design guidance | Whole-match learning, separate placement prediction and careful privileged learning are relevant. A verified Microsoft checkpoint/code release was not found in the checked official sources. |
| Suphx-scale replication or runtime adaptation | Defer | Neither follows from our local throughput evidence or the existence of the paper. See the [research assessment](SUPHX_ASSESSMENT.md). |

## Proposed Stage 3A: complete-match RL engineering

This is **proposed and unstarted**. The full rationale and primary sources are in the [Suphx assessment](SUPHX_ASSESSMENT.md).

Start from the original reference with the existing model size, audited actor inputs and utility `[1.5, 0.5, -0.5, -1.5]`. Use one learner seat against three frozen opponents, with learner-seat rotation. Keep actual match completion and per-seat behavior-policy identity explicit.

Proposed limits are **600 cumulative numerical seconds, 32 GiB sampled memory, 1 GB of new artifacts, 64 complete matches maximum, and four optimizer updates maximum**. Plan four batches of at most 16 matches. Before collection, fix each batch's match count, seeds, learner seats and opponents. The caps include checks, loading, collection, training and finalization; they are not a promise that the machine reaches every maximum.

Hold the learner weights fixed throughout a whole batch. Every planned match in that batch must finish and pass validation before exactly one optimizer update; retire those trajectories after that update. Do not reuse them for later updates or mix behavior-policy versions. Reserve completion time. If any planned match is truncated, failed or missing, preserve the entire batch as incomplete and make no update from it. Updating only its faster completed matches would select data by completion time.

This is a **REINFORCE engineering substage before the planned full-match PPO reference**, not a replacement for that later algorithm. Use a fixed zero baseline and undiscounted final utility. For a batch of `N` complete matches, minimize `L = -(1/N) * sum_match[final_utility * sum_eligible_action(log_probability)]`, evaluated at the same weights that generated the batch. Average by matches, not decisions. Freeze optimizer, learning rate, clipping, sampling and stopping details in the implementation protocol before collection. Keep the current overconfident rank head frozen and unused as a critic. Learned critics, entropy adjustments, potential shaping and more learner seats remain separate changes.

The existing `NeuralPolicy.choose` uses deterministic argmax. This pilot therefore needs a **new categorical softmax behavior wrapper**, with its own versioned sampling rule, checkpoint identity and random state; the historical argmax policy's scores are not its action probabilities. Sample from the actual legal semantic-action distribution and record the probability used for that draw. Single-option actions have probability one and contribute zero to the policy loss while retaining their trajectory links. An emergency fallback or overridden action invalidates the batch for this first learning pilot; never record it as if it were sampled from the actor.

Verify on the declared fixtures and collected batches that:

- Returned rewards agree with an independent reconstruction of complete match results and focal-seat identities.
- Policy inputs contain only legal player-visible information; changing hidden worlds does not alter the actor's visible-state output.
- Simultaneous requests use a shared pre-response state, and only the learner's own behavior-policy actions enter its loss.
- Stored behavior probabilities reproduce before the update, with legal masks and policy versions intact.
- Opponent-ended matches still deliver final rewards after the learner's last action; truncation is not relabeled as termination.
- Optimizer steps, gradients, weights, interruption/resume and resource accounting behave correctly.

Record policy entropy, actual parameter change and action-distribution change. The starting logits are sharp, so the new wrapper's unmodified softmax may explore mostly within teacher-best ties. A finite but effectively zero update must be reported as a no-op; finiteness alone is not evidence of learning. This tiny experiment cannot establish strength improvement, and it must not silently add temperature or epsilon to make exploration look healthier.

An engineering pass requires at least one fully valid, prespecified batch and an executed optimizer update, all applicable fixture checks passing, and closed resource accounting. Completing fewer than the four planned batches must remain explicit; no valid batch/update means incomplete, not passed. A no-op can be a diagnosed numerical outcome but cannot be described as successful learning. These checks provide evidence for the tested cases, not full rule certification or a universal information-boundary proof.

If engineering checks pass, the next task is a separately planned learning and fresh-match evaluation campaign. A failed or incomplete pilot is preserved and diagnosed, not replaced by extra undocumented runs.

## Parallel book-data track

The source assessment covers all five supplied PDFs and identifies where questions, context and explanations can be extracted. The [intake guide](BOOK_SCENARIOS.md) distinguishes partial exercises from complete policy states. Actual source pages and transcriptions remain local under `Books/`.

For the next bounded batch, target **10–20 independently checked question/answer pairs**, with at least one from each useful exercise format. Preserve book hash, edition, page/problem identifiers, the original recommendation and a faithful explanation. Retain all shown context: drawn tile, dora versus indicator, round, seat, turn and any threat/score assumptions. Keep unsupported formats as raw source packets instead of discarding information to fit the current schema.

Use three separate evidence labels:

1. **Transcription verified:** the hand, context, answer and explanation match the source. This does not prove strategic optimality.
2. **Conditional arithmetic checked:** shanten, acceptance counts or scoring agree under an explicitly named convention. The book and simulator may use different conventions.
3. **Strategic recommendation:** an expert-source label under its stated objective and assumptions, with disagreements and confidence recorded. It can be useful imitation supervision without claiming a mathematically proven global optimum.

Do not require every expert recommendation to have an exact game-theoretic proof. Do require enough context to define what is being taught. A partial-hand task can support a separate supervised auxiliary task or diagnostic benchmark; it cannot automatically become a full-policy training row with invented rivers, scores or legal actions.

Before any book-derived optimization, freeze source/problem-family groups and an untouched evaluation allocation. Translations, paraphrases, suit permutations and changed-context variants of the same question must remain grouped. The two Uzaku collections may share concepts or related problems; distinct file names do not establish independence. A set used to choose features, learning rates or checkpoints is development data.

The first arithmetic pilot already illustrates why this matters: two calculators tie 5m and 6s at the same shanten/acceptance count, while the author prefers 6s for future wait quality. That is evidence for a richer learning target, not proof that every book answer should override the current policy.

## Feedback that would improve the next instructions

The most consequential choices are whether to prioritize the small RL engineering pilot or the reviewed book batch first, whether the proposed 600-second numerical ceiling is appropriate, and whether book exercises should initially target hand efficiency, push/fold decisions, or a mixed curriculum. The source assessment explains which formats we can currently preserve accurately.

Keep the current average-rank objective unless a different venue or utility is explicitly selected. Keep full-match strength evaluation separate from book-answer agreement and engineering checks. The next protocol should make its one scientific question and measurable exit condition explicit before work begins.
