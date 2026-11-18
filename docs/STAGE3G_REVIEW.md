# Stage 3G review and recommended diagnosis

**Stage 3G is complete. Keep the original reference; promote none of the four new policies.** All fixed training runs, all 1,280 development games and the independent audits completed. The best new policy averaged rank 2.621094, versus 2.421875 for the reference, where lower is better. Read the [measured results](../reports/stage3g/REPORT.md) and [method notes](STAGE3G_METHOD_NOTES.md).

## What worked

The Mac completed the full experiment within 1,165.485892 of its 7,200 numerical seconds. Four arms each completed 400 real Adam updates on two passes over the same 3,200 training matches. Their initialization and example ordering match, checkpoints are preserved, and the independent audit confirms the intended training exposure. All 4,000 fresh corpus games completed; all forty preselected replays matched. This provides a usable platform for controlled learning experiments.

The separate public placement predictor passed its fixed development gate. It improves point estimates of NLL, Brier and expected-utility error over the public-score baseline. Whole-match cross-fitting prevents each training game's answer from fitting its own target predictor. Keep this predictor as an experimental component; successful calibration did not establish useful action credit or stronger play.

The safeguards produced a useful negative finding without silently switching candidates or extending a promising-looking run. Every final model reached the same endpoint. No policy beat the reference descriptively, so the frozen rule selects none.

## What the result changes

| Recipe | Mean rank ↓ | Decision |
|---|---:|---|
| Original reference | 2.421875 | Retain as comparison policy |
| Full-match reward, no imitation penalty | 3.070312 | Do not promote |
| Full-match reward, alpha-one penalty | 2.960938 | Do not promote |
| One-hand placement change, no penalty | 2.621094 | Best of the four; below reference |
| One-hand placement change, alpha-one penalty | 3.074219 | Do not promote |

The placement-change target and alpha-one penalty are not adopted as defaults. The penalty helped the full-match recipe descriptively but substantially worsened the placement-change recipe. One initialization and this development sample cannot establish a reproducible interaction or identify its cause.

Do not respond with a larger model, a longer unchanged run or purchased compute. The experiment demonstrates functioning optimization, while leaving the connection between fitted values and useful actions unresolved. Possible mechanisms include loss of the reference's preferences when resetting the scorer, poorly supported greedy actions, values dominated by state rather than action differences, and the relative strength of the imitation term. These are hypotheses to test, not explanations established by the rank table.

Almost every full-match update was clipped: 399/400 without the penalty and 400/400 with it. The corresponding placement-change counts were 18/400 and 140/400. Clipping frequency is a diagnostic clue, not proof that clipping caused weaker play. First and last batch losses use different matches and cannot establish convergence. A common fixed-state comparison is the next useful measurement.

## Recommended Stage 3H, proposed and unstarted

Use **300 cumulative numerical seconds** for saved-data diagnosis, with no new games, optimizer steps, retraining, model promotion or book-derived training. Freeze the exact source/input identities and selection before inference. Benchmark the prescribed inference path before admitting the full diagnostic; the following allocation is a proposal, not a measured runtime guarantee.

| Work | Proposed seconds |
|---|---:|
| Identity checks, tests and inference-path timing | 30 |
| Fixed saved-state comparisons | 140 |
| Separate loss-component gradients | 80 |
| Report and independent checks | 30 |
| Protected shutdown | 20 |
| **Total** | **300** |

Select 32 training and 32 development matches by a declared hash ordering of split and match ID, ignoring outcomes. Use every optional explorer decision in those complete matches, retain separate split summaries, and report both match-weighted and row-weighted descriptions. Do not add interesting examples after observing the result.

Compare eighteen fixed weight sets: the original reference, the common saved reset-head initialization, and updates 100/200/300/400 from each of the four arms. For action comparisons, use the deployed single-state CPU inference path; batch/backend changes near ties must not be mistaken for learning-induced decisions. Use reference outputs only as policy logits/action preferences, not as predicted utility.

Measure agreement with the reference, initialization and logged behavior; action-kind transitions; the effects of the legal-win rule; value margins; and the behavior probability assigned to each proposed action. Use fixed probability buckets of zero, (0, .01], (.01, .05], (.05, .2] and (.2, 1]. Report call opportunities separately from overall discards, alongside the number of games supporting each rare category. Compare chosen-action errors within each target definition against a zero-prediction baseline; raw errors across the two target definitions cannot identify the stronger policy.

On the first sixteen selected training matches, reconstruct regression and imitation gradients separately at the common initialization and each final model, using the verified training batch path. Record component norms, their alignment and the fixed clipping factor without taking an Adam step. Check model hashes before and after. This tests whether regularization overwhelms or conflicts with regression on that declared group; it is not a population estimate of gradient signal-to-noise.

If the measured admission fails or the fixed diagnostic hits its deadline, preserve the incomplete result. Do not shrink the sample after inspecting outcomes. The next learning intervention should follow the diagnostic evidence and change one factor at a time.

## Interpretation boundaries

The labels describe explorer continuation; greedy extraction changes both actions and future states. Equal-match weighting uses a decision-count denominator partly determined by future events, making this a match-reweighted regression. Alpha one adds chosen-action cross-entropy rather than a guaranteed conservative Bellman update. A one-hand potential difference is a different surrogate from the full decision return. These distinctions explain what needs testing without claiming any one caused the observed weakness.

The placement gate passed on point estimates, but its NLL/Brier difference bootstrap intervals include zero. Early-stage maximum supported-bin utility bias is 0.29321, close to the 0.30 threshold. The game panel uses 64 independent shared-wall blocks; its 256 games per policy are not 256 independent comparisons. Candidate-versus-reference results include a procedural legal-win-rule difference. The three fixed opponents made no calls in the audited collection corpus, and rare kan paths remain sparse. No result establishes expert-level or world-class performance.

Saved outcomes cannot label an unchosen action as optimal, measure causal regret, or show which discard caused a loss. Measuring the playing-strength effects of resetting the head or adding the win rule requires a later, separately fixed fresh-game comparison with explicit controls: original reference, reference with the same win rule, reset-head initialization with that rule, and a prespecified trained model. Beating the reset-head control alone would still not establish improvement over the original reference.

Existing book copies and reviewed partial scenarios remain preserved and quarantined. Their missing state and action reconstruction remain separate work. Stage 3H has not started; this review is ready for feedback before the next experiment.
