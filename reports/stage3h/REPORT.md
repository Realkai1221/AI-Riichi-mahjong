# Stage 3H results: longer saved-position diagnosis

Completed 22 October 2026. **The full fixed diagnosis and independent audit passed.** Resetting the action scorer substantially changed the starting policy, and the final regularized placement-change learner shows a strong conflict between its prediction and imitation objectives. These findings identify concrete issues for the next experiment. They do not demonstrate improved play or fully explain Stage 3G's weaker game results.

All eighteen archived weight sets were evaluated on **38,237 optional explorer decisions from 256 saved matches**, producing **688,266 model decisions**. Six model/target conditions received separate MSE and imitation-gradient checks on four disjoint sixteen-match groups. No new games, optimizer steps, checkpoint writes or policy promotions occurred. The original reference remains unchanged.

The closed ledger charges **89.953390 / 1,200 numerical seconds** across ten successful operations. The entire predeclared workload completed; the unused allowance is not extended into an outcome-driven training run. This budget covers numerical subprocess time, not the surrounding implementation and review time.

## Fixed sample and interpretation

The [protocol](../../docs/STAGE3H_PLAN.md), execution plan and admission preserve selection rules, input identities and resource forecasts. IDs were ordered by a fixed hash before diagnostic outcomes. The largest predeclared panel fit, so no fallback was needed.

| Split/workload | Saved matches | Eligible positions |
|---|---:|---:|
| Training inference | 128 | 19,418 |
| Development inference | 128 | 18,819 |
| Gradient sample, within the training panel | 64 | 9,784 |
| Separate existing preflight corpus | 16 | 1,024 selected positions |

The eighteen models are the original reference, the common reset initialization, and updates 100/200/300/400 of all four Stage 3G arms. `mc` fits complete-match utility; `delta` fits the archived one-hand placement-potential difference. `a0` uses MSE only; `a1` adds chosen-action cross-entropy with weight one. That imitation term fits the logged explorer behavior, not the original reference probability distribution. All candidate policies use the declared first-legal-win wrapper; the original reference uses its original argmax.

Unless stated otherwise, tables below use **development means of within-match means**, giving each contributing match equal weight. The machine-readable inference report also contains row-weighted results, training results, complete action-kind transition matrices and all seven opportunity slices. Original reference scores are policy logits; they are not evaluated as utility predictions.

## 1. The reset initialization did not preserve the reference policy

| Model | Agreement with reference | Agreement on discard opportunities | Chosen action had collection probability in `(0, 1%]` |
|---|---:|---:|---:|
| Original reference | 100.00% | 100.00% | 0.00% |
| Reset initialization | 39.47% | 41.48% | 23.71% |
| Final MC, alpha 0 | 57.80% | 50.40% | 19.64% |
| Final MC, alpha 1 | 64.33% | 54.02% | 21.34% |
| Final delta, alpha 0 | 62.47% | 55.68% | 15.85% |
| Final delta, alpha 1 | 63.57% | 53.03% | 22.41% |

Before applying the win wrapper, initialization agreement is **39.1886%**; after it, **39.4699%**. The wrapper changes **0.2813%** of all sampled development decisions. Thus most of the observed initial action change is already present in the reset scorer's raw ordering. Training-side initialized agreement is similarly low at **40.2235%**.

This is an observable behavior change, not proof that every departure is bad. The reference is a project baseline, not an optimal-action oracle. Stage 3G did not play the reset initial policy as a separate game-control arm, so its contribution to the final rank deficit remains unmeasured.

Every selected action in this sample has positive recorded behavior probability. The low-probability column indicates weak collection support, not literally unseen or illegal actions, and does not supply counterfactual outcomes. It is a warning against treating logged chosen-action prediction error as sufficient evidence for greedy policy improvement.

## 2. Better prediction and imitation do not imply better action selection

| Model | MC chosen-action MSE | Delta chosen-action MSE | Chosen-action cross-entropy | Mean legal score spread |
|---|---:|---:|---:|---:|
| Predict zero | 1.453125 | 0.127567 | — | — |
| Reset initialization | 1.416109 | 0.130156 | 2.012628 | 0.020283 |
| Final MC, alpha 0 | 0.947918 | 0.617223 | 1.991317 | 0.064427 |
| Final MC, alpha 1 | 1.047994 | 0.727570 | 1.676727 | 1.032518 |
| Final delta, alpha 0 | 1.409814 | 0.119480 | 1.993717 | 0.055049 |
| Final delta, alpha 1 | 1.512034 | 0.248722 | 1.627995 | 1.176229 |

Compare each learner primarily with its own declared target. MC and delta have different scales and meanings; their raw MSEs cannot rank targets. Zero is a declared simple baseline, not a fitted constant or strong predictive baseline.

The unregularized delta model modestly beats its zero predictor. Adding alpha one reduces cross-entropy but worsens delta MSE beyond the zero baseline. MC alpha zero also fits its own target better than MC alpha one, while agreeing less with the reference. The regularized models' much larger score spreads reflect their changed scoring objective; they are not calibrated confidence or evidence of strategic quality.

The same positions allow a genuine fixed-sample learning-trajectory comparison:

| Arm | Metric | Update 100 | Update 200 | Update 300 | Update 400 |
|---|---|---:|---:|---:|---:|
| MC alpha 0 | Own-target MSE | 0.977243 | 0.958414 | 0.955682 | 0.947918 |
| MC alpha 0 | Reference agreement | 66.40% | 59.37% | 59.67% | 57.80% |
| MC alpha 1 | Own-target MSE | 1.018551 | 1.003543 | 1.026274 | 1.047994 |
| MC alpha 1 | Reference agreement | 68.03% | 64.83% | 63.27% | 64.33% |
| Delta alpha 0 | Own-target MSE | 0.121062 | 0.120992 | 0.120385 | 0.119480 |
| Delta alpha 0 | Reference agreement | 60.83% | 63.91% | 62.28% | 62.47% |
| Delta alpha 1 | Own-target MSE | 0.186767 | 0.200895 | 0.232954 | 0.248722 |
| Delta alpha 1 | Reference agreement | 68.33% | 63.56% | 64.11% | 63.57% |

No intermediate checkpoint is selected from this table. These saved states follow the original collection policy. A newly deployed greedy policy changes subsequent actions, states and opponents' responses, which these error measures do not evaluate.

## 3. Call and win slices narrow the interpretation

Development support includes 14,609 discard-opportunity rows, 4,172 call-opportunity rows, 416 riichi-opportunity rows, 194 kan-opportunity rows and 214 win-opportunity rows. Slices overlap. All matches support discard/call slices; only 80 support kan and 102 support win, so slice averages have different match denominators.

The reference chooses no CHI or PON in this sample. The reset initialization chooses CHI on 10.2213% and PON on 4.7751% of all positions, in match-weighted terms. Final MC alpha zero chooses them on 3.3041% and 0.2371%; final delta alpha zero on 1.7188% and 1.2380%. Both final alpha-one models choose neither. This describes differing behavior; it does not label calling or declining as inherently correct.

For the final MC alpha-zero, MC alpha-one and delta alpha-zero models, raw and wrapped decisions match everywhere in this development panel. The final delta alpha-one wrapper changes 0.0143% of all positions. The wrapper is therefore a small direct contributor here, while its effect on fresh trajectories remains a separate question.

## 4. Separate gradients reveal an objective tradeoff

The gradient report covers four fixed groups, each retaining complete-match loss weighting. The table describes the **mean of the four group gradients**, before any hypothetical clipping. No optimizer or Adam step was performed.

| Condition | MSE gradient norm | Imitation gradient norm | Cosine between them | MSE cancellation ratio |
|---|---:|---:|---:|---:|
| Initialization, MC target | 3.005621 | 0.592199 | -0.039509 | 0.569142 |
| Initialization, delta target | 0.761041 | 0.592199 | 0.056344 | 0.975911 |
| Final MC alpha 0 | 3.139255 | 0.575341 | -0.075653 | 0.465411 |
| Final MC alpha 1 | 3.417872 | 0.923505 | -0.375091 | 0.468636 |
| Final delta alpha 0 | 0.256351 | 0.537167 | 0.042553 | 0.911816 |
| Final delta alpha 1 | 1.586261 | 1.281288 | -0.838984 | 0.979279 |

The cancellation ratio is the norm of the four-group mean divided by the mean of their norms. A lower value indicates more disagreement across these four groups. MC gradients exhibit more cancellation than delta gradients here. This is a small descriptive sample, not a population signal-to-noise estimate.

At the final delta alpha-one parameters, the objective gradients strongly oppose one another, including cosine **-0.954274** in the shared backbone and **-0.977847** in the efficiency branch. Their combined full-actor norm remains **0.864595**: this is not a vanished gradient. Initialization cosines are near zero, so the conflict is not universal across models or training times. The evidence supports a local tradeoff between fitting utility-scale targets and using the same scores for imitation. It does not establish a numerical bug or by itself explain the game-result deficit.

Norm-one clipping would scale the mean final MC alpha-zero gradient by **0.318547** and final MC alpha-one combined gradient by **0.313623**. These are diagnostic gradients at saved parameters, not a reconstruction of the historical Adam trajectory. The report retains per-group and per-module values rather than inferring historical training behavior from these means.

## Validation, resources and reproducibility

The 75-test release suite and 26-test independent auditor suite pass. The earlier 43 controller tests were also run separately; they are included in the 75, not additional distinct coverage. The CPU/MPS preflight compares the same initial sixteen-match group: relative gradient L2 differences are **2.53e-7** for MSE and **1.37e-6** for imitation, below the fixed .01 threshold. Objective checks and unchanged-weight hashes also pass.

The independent audit verifies 565 input identities, reconstructs all 252 split/model/opportunity summaries from original shards and saved legal scores, and checks raw gradient-vector arithmetic for all 24 groups. It independently repeats **576 CPU forwards across sixteen matches**, with **zero maximum score difference**. Gradient objectives are checked against saved CPU scores; the full gradients are not independently differentiated again. CPU/MPS differentiation parity covers one preflight group. Neither audit is independent rule certification.

| Numerical phase | Charged seconds |
|---|---:|
| Preparation, tests and preflights | 14.299742 |
| Formal inference | 36.265742 |
| Formal gradient diagnosis | 31.519899 |
| Auditor tests and evidence audit | 7.868007 |
| Total | 89.953390 |

All ten leases closed successfully, without failures or replacements. Peak sampled process-tree RSS is **6.440 GB**, below 32 GiB. The full preflight forecast was intentionally conservative: 218.546 seconds for inference and 96.597 for gradients. No model scale, sample size or objective was changed after formal outcomes.

Runtime: project `.venv`, Python 3.14.8, Torch 2.14.1, NumPy 2.5.4, RiichiEnv 0.4.10, mahjong 2.0.0, pytest 9.1.1 and psutil 7.2.2. Versions and source hashes are recorded in the operation receipts and ledger. Actor inputs remain the audited player-visible features; outcomes, provenance and hand-target mappings enter only diagnosis/loss calculations. Frozen-opponent actions are excluded.

The historical Stage 3G snapshot is preserved at commit `420a4163f7b8ef210b800115133aea09ee1df29e`, with 37,758 recorded files verified. Its reference mean rank 2.421875 and all four weaker candidate ranks remain the latest game evidence. Stage 3H adds no games and no new strength estimate. Existing books and partial scenarios remain unchanged and quarantined.

## Files to review and next decision

Start with [the interpretation and next-stage proposal](../../docs/STAGE3H_REVIEW.md). The measured tables above are supported by inference.json, gradients.json, audit.json, plan.json and admission.json. The final manifest and verification bind the completed files. Full legal scores and raw gradient vectors remain in the local `RiichiAI/stage3h` directory outside iCloud.

The recommendation is a fixed initialization/win-wrapper game control before another training sweep, followed by a design that preserves the complete reference actor while fitting a separate value estimator. This recommendation is proposed, not executed. Simply fitting a separate critic would not itself improve play.
