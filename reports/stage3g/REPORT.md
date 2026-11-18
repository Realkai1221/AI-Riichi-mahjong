# Stage 3G results

22 October 2026. **All four fixed learners and the entire development panel completed, but none improved on the original reference.** The best observed candidate, `delta_a0`, averaged rank **2.621094**, compared with **2.421875** for the reference. Lower average rank is better. The frozen selection rule therefore selects no candidate, and no policy is promoted. The independent data, checkpoint/target/result audits and all forty prespecified replays passed.

## Playing outcomes

Each policy played 256 complete games: 64 independent shared-wall blocks with four focal-seat rotations. All candidates faced the same three fixed opponents. The complete panel contains 1,280 games. The unit of uncertainty is the block, not a decision or seat game.

| Fixed candidate | Average rank ↓ | Utility difference vs reference ↑ | Descriptive bootstrap 95% interval | Fixed-budget Hoeffding 95% interval |
|---|---:|---:|---|---|
| Full-match reward, alpha 0 (`mc_a0`) | 3.070312 | -0.648438 | [-0.855469, -0.437500] | [-1.667014, +0.370139] |
| Full-match reward, alpha 1 (`mc_a1`) | 2.960938 | -0.539062 | [-0.730469, -0.351562] | [-1.557639, +0.479514] |
| One-hand placement change, alpha 0 (`delta_a0`) | 2.621094 | -0.199219 | [-0.386719, -0.015625] | [-1.217795, +0.819357] |
| One-hand placement change, alpha 1 (`delta_a1`) | 3.074219 | -0.652344 | [-0.863281, -0.449219] | [-1.670920, +0.366232] |

The utility is `2.5 - rank`, equivalent to `[1.5, .5, -.5, -1.5]`. All observed differences are negative. The exploratory bootstrap intervals are also negative; the deliberately conservative Hoeffding intervals remain wide and include zero. These are development results with one training initialization, not independent confirmation or a multiplicity-adjusted strength claim. The alpha-one penalty helped the terminal recipe descriptively but hurt the hand-difference recipe; this does not establish either contrast as a reproducible causal advantage.

The candidates use a deterministic legal-win rule, while the reference retains its original argmax policy. The comparison therefore measures the complete resulting policies, including that procedural difference. Common-ordinal physical-wall commitments are checked explicitly; equal numeric seeds alone are not the evidence for pairing.

## Data and learning completed

The fresh corpus contains **4,000 complete matches and 2,564,254 all-seat decisions**, collected in **353.071 seconds** at **11.329 matches/s** on fourteen workers. The fixed split is 3,200 training matches and 800 development matches. Only the training explorer's optional decisions enter fitting: **471,627 distinct rows**, presented twice. Other seats remain audit data. Historical pilots, books and development games do not train these value models.

Every arm uses the same reference backbone, identically reset action scorer, match/row order, sixteen-complete-match updates and two epochs. The initial model tensor hash is `a2814b42d302cff83da42e055c3373f1eeaec7e2e8f32240b43b2aacc580d622`. All arms reach 400 scheduled and actual Adam updates; each saves initial, update-100/200/300 and final boundaries. Each changes 41 parameter tensors while preserving the unused rank-head weights. No formal retry, early checkpoint selection or hyperparameter change occurred.

| Arm | Adam updates | Eligible row presentations | Clipped updates | Parameter movement L2 | Learner seconds |
|---|---:|---:|---:|---:|---:|
| `mc_a0` | 400 | 943,254 | 399 | 2.367210 | 143.045 |
| `mc_a1` | 400 | 943,254 | 400 | 3.232935 | 138.198 |
| `delta_a0` | 400 | 943,254 | 18 | 2.572428 | 150.449 |
| `delta_a1` | 400 | 943,254 | 140 | 4.047140 | 142.359 |

Gradient clipping is a declared part of the optimizer. Frequent clipping is recorded here without equating it with an integrity failure or proof that the learning rate is wrong. Batch loss values vary with match composition; first-versus-last batch loss is not a matched measure of convergence. Full development value-loss validation has not been run.

## Placement gate

All six predictors fitted: five whole-match cross-fitting folds plus one final training-only model. The corpus supplies **39,072 explicit hand starts**, including dealer continuations. The 800 development matches supply 7,844 hand states for the gate.

| Prediction measure, lower is better | Learned predictor | Public-score baseline |
|---|---:|---:|
| NLL | 1.136127 | 1.151516 |
| Brier | 0.615851 | 0.621737 |
| Expected-utility squared error | 0.842403 | 0.894875 |

All predeclared support and utility-calibration thresholds passed, so no arm was skipped. Overall weighted absolute supported-bin utility bias is 0.02510. Early, middle and late values are 0.05942, 0.02773 and 0.05184. The early-stage maximum supported-bin bias is **0.29321**, close to the fixed 0.30 limit; 98.61% of early-stage weight is supported. The other strata have full supported weight. The bootstrap NLL/Brier difference intervals include zero, so eligibility does not prove predictor superiority.

Training potentials come from a predictor that excluded the entire match. A hand target is the next-hand potential minus the current one, with true final utility substituted at actual match termination. This is a one-hand surrogate, not a full decision return. The development outcome demonstrates that passing the placement gate was insufficient to produce a stronger policy in this experiment.

## Verification and resources

All **2,564,254 corpus decisions** pass the independent record audit. It checks stored probabilities/RNG, role and seat provenance, labels, explicit hand starts and kan resolution. The audit records 39,072 hand starts, 10,787 dealer continuations, 9,387 resolved explorer calls and 77,925 post-call rows across the full corpus. Completed kans include 106 ANKAN, 92 DAIMINKAN and 31 KAKAN; two further KAKAN declarations ended before completion. These whole-corpus counts include the development split and are not all training examples. The other three behaviors made no calls in this corpus. The bounded history overflow flag appears in 21,915 explorer rows; the actor still uses the declared capped safe-history representation.

All **40 fixed full-match replays / 25,891 rows** match their original deterministic payloads. The separate learning audit verifies all twenty value checkpoints, both-pass exposure and row ordering, all 39,072 hand-difference targets, six predictor memberships, and predictor inference on 96 predetermined games. It reconstructs all 1,280 development outcomes and the selection/interval arithmetic, checking 9,115 common hand-ordinal physical-wall pairs across 1,024 candidate/reference game pairs. These are same-engine consistency checks, not independent rule certification. The audit does not replay every training gradient or the GRP optimizer/bootstrap history, and does not reconstruct every evaluation action from its logs.

The closed ledger charges **1,165.485892 of 7,200 numerical seconds** (about 19.42 minutes) across **20 successful operations**, with no failed or replaced formal attempt. There are **5,336 complete match executions**: 4,000 corpus, sixteen preflight, 1,280 development and forty replays. Eight separate partial engine fixtures bring conservative attempt accounting to 5,344 / 5,500. The unused 6,034.514108 seconds remain unused.

| Phase | Charged seconds | Cap |
|---|---:|---:|
| Preparation and tests | 17.715660 | 600 |
| Collection | 353.341000 | 900 |
| Placement predictor and calibration | 11.010916 | 600 |
| Four learning arms | 578.665928 | 4,000 |
| Development games | 96.070057 | 700 |
| Audits and replay | 108.682331 | 280 |

Peak sampled process-tree RSS was **10.237 GB**, below 32 GiB. Peak artifacts during numerical work were **1.638 GB**, below 8 GB. The final hash inventory includes the later review documents and records the final byte total. Sampling does not measure every transient memory peak. The source-bound release suite passed **190 tests**; the independent evidence-auditor suite passed **26 tests**. CPU/MPS optimizer and checkpoint-resume checks passed on sixteen separate preflight matches. Reloaded MPS model weights matched exactly, while optimizer state agreed within tolerance rather than bitwise.

The actual optimizer preflight measured 2,552 eligible rows/s on MPS and admitted the fixed experiment with conservative reservations. Formal training ran on MPS; collection, placement fitting and development games ran locally on CPU. The code retains CPU and device-parameterized CUDA paths; this milestone did not benchmark CUDA. No rented compute, third-party game scraping or public bot play occurred.

Pinned environment: Python 3.14.8, PyTorch 2.14.1, NumPy 2.5.4, RiichiEnv 0.4.10, mahjong 2.0.0, psutil 7.2.2 and pytest 9.1.1. The historical Stage 3F 9,493-file snapshot verified before shared documentation changed. Source identities and original preflight versions are preserved; the formal runtime snapshot is local commit `2b21e82`.

## Interpretation and review files

Keep the functioning data/training/audit infrastructure, the controlled comparisons and the public placement-prediction result. Keep the original reference as the comparison policy. The present four learned policies do not justify promotion, simply extending their training, or buying more hardware.

Review [recommendations](../../docs/STAGE3G_REVIEW.md), [method and interpretation notes](../../docs/STAGE3G_METHOD_NOTES.md), and the [fixed protocol](../../docs/STAGE3G_PLAN.md). Machine-readable evidence includes development outcomes, placement calibration, checkpoint identities, collection, resource admission, numerical ledger, corpus audit, learning/result audit, replay audit, and final inventory. Bulk actor arrays, auditor logs and checkpoints remain outside iCloud under `~/Library/Application Support/RiichiAI/stage3g`.

The existing book records remain preserved and quarantined. This milestone adds no book extraction or book-derived training. The next proposed work should diagnose the policy changes on a fixed saved-state sample before another learning intervention; it is not part of this completed four-arm comparison.
