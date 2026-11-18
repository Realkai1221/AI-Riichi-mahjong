# Reviewing Package A

Start with the [measured results](../reports/package_a/REPORT.md), then the A1 confirmation report if you want the statistical detail. The [accepted plan](PACKAGE_A_PLAN.md) and [configuration](../configs/package_a.json) define the criteria fixed before these outcomes.

Package A answers two narrow questions: whether the existing candidate retained the heuristic teacher's strength within a 0.10 average-rank margin, and whether one smoothing intervention met the declared training-durability criterion. Completing the experiments does not imply passing those criteria.

## Decisions to review

| Decision | Reason |
|---|---|
| Close the 319-block confirmation at its fixed count. | The non-inferiority criterion was not met. Adding blocks after inspecting the result would change the statistical procedure. Preserve the estimate and uncertainty. |
| Keep the original Stage 2B checkpoint as a research reference. | Its identity and earlier action test remain intact. Retention is unestablished, so it should not be described as a confirmed teacher-strength replacement. |
| Keep the smoothed models as experimental artifacts. | Finishing without a safety alarm is only one part of acceptance. Both seeds must also meet the frozen terminal agreement threshold. |
| Diagnose the probability mismatch before another training recipe. | The 1% smoothing target does not imply that the learned model assigns only 1% to non-best actions on validation. Training fit, generalization, and optimization need to be distinguished. |
| Reassess safety alarms in a separately declared study. | The controls stopped on raw-logit growth while loss and gradients remained moderate. A common shift of all legal logits does not change softmax probabilities in exact arithmetic. The present alarms are useful records, not calibrated collapse predictions. |

## Diagnosis proposed after Package A — now complete

The diagnosis below was subsequently authorized and completed on 18 October 2026. Read the [measured results](../reports/package_a_diagnosis/REPORT.md) and [current next-experiment proposal](PACKAGE_A_DIAGNOSIS_REVIEW.md). The remaining paragraphs preserve the rationale that led to that study.

The proposed diagnosis used preserved checkpoints and existing training/validation splits, with a ceiling of **600 cumulative numerical seconds, 32 GiB sampled memory, and 250 MB of new reports**, and **zero optimizer updates, zero new matches, and no action-test access**. All seven attempts completed within those limits.

First freeze the checkpoint list and row indices. Compare the original rank-0.1 reference, the two stopped controls, and both selected and terminal smoothed checkpoints. Measure fit to the original and smoothed targets on a fixed training sample and on the full validation set. Report teacher/student strata, optional discards, call opportunities, forced decisions, legal-choice counts, shanten errors, and lost ukeire separately. Do not compare percentages with different denominators as if they were the same measure.

Then record per-row logits on those frozen states and compare raw logits, their shared row offsets, centered logits, action gaps, context norms, and CPU/MPS probabilities. The current reports retain diagnostic summaries, not full logit arrays. Normalizing identical float32 logits in float64 can reveal softmax underflow or normalization differences; it cannot recover distinctions already rounded away or change the raw-logit argmax. A separate small CPU float64 forward comparison using the same frozen weights and features can test upstream rounding and action-order sensitivity against CPU/MPS float32. Neither check reproduces an entire high-precision training trajectory. Compare the guard statistics with the known-stable rank-0.1 reference before proposing new thresholds.

The deliverable should distinguish poor training fit from a validation gap and determine whether offsets materially affect predictions on the tested states. Endpoint comparisons cannot establish an alarm's prospective predictive value or false-positive rate; that needs suitable trajectories or a later fixed study. Preserve uncertainty about future training instability. The diagnosis should nominate **one** next intervention with a fresh protocol and acceptance criteria. A lower smoothing coefficient, a learning-rate change, and a representation constraint are different interventions; choose among them after this diagnosis rather than combining them into an unexplained improvement.

Do not use another confirmation of the unchanged model as a way to obtain a pass. A materially changed candidate can receive a separately planned evaluation later. Broader call coverage, placement-model calibration, and independent rule conformance remain separate workstreams before serious RL claims.

## Useful instructions for the next review

Feedback is most useful when it specifies the scientific question, the local compute ceiling, and what evidence would change the decision. For example: prioritize finding the cause of the smoothing fit gap; test whether the raw-logit guard has useful predictive value; or propose a separate call-coverage experiment with its own budget.

The integrity audit separates valid execution from successful scientific outcomes. The file inventory records source, reports, and local experimental artifacts at Package A completion. Verify its project bytes against snapshot `c7ae395babcbc56d056a37995ad23952b9d9d4ba` with `scripts/verify_package_a_snapshot.py`; do not regenerate historical hashes after current files evolve. Earlier Stage 2A and Stage 2B evidence also remains checked against its original Git snapshots.
