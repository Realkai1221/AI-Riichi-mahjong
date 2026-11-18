# Stage 3G: fixed first value-learning experiment

Authorized 22 October 2026 following [the Stage 3F review](STAGE3F_REVIEW.md). This is a single-initialization mechanism experiment on this Mac, with development selection and no strength confirmation or promotion. Preserve all historical evidence and the Stage 3F development-only data. The historical 9,493-file Stage 3F snapshot verified before shared documents changed.

## Fixed scope and resource contract

Use the project `.venv` and [configuration](../configs/stage3g.json). Charge every numerical subprocess, including tests/imports, to one exclusive durable lease. Keep bulk artifacts outside iCloud under `~/Library/Application Support/RiichiAI/stage3g`.

| Phase | Cumulative numerical cap, seconds |
|---|---:|
| Preflight, preparation and tests | 600 |
| Fresh formal collection | 900 |
| Placement predictor and calibration | 600 |
| All value-learning arms | 4,000 |
| Development game panel | 700 |
| Final numerical audits and replay | 280 |
| Protected shutdown reserve | 120 |
| **Total** | **7,200** |

Keep sampled process-tree RSS below 32 GiB, artifacts below 8 GB, 10 GB free disk, and at most 5,500 match attempts including preflight, fixtures and replays. The larger storage cap covers a fourfold corpus, optimizer checkpoints and full development-game evidence; it does not authorize a larger model or external compute. Record timeouts, killed starts and partial records. Never replace failed formal matches, silently retry failed learning arms or extend samples based on outcomes. A failed formal operation stops subsequent formal work; inspection and audit remain allowed. A completed calibration check returning false is a declared branch, not an infrastructure failure.

Before formal collection, admit the whole fixed programme from measured optimizer throughput and the preceding sustained collector pilot with a twofold allowance and startup reserves. Keep the complete per-phase reservations visible. If it does not fit, stop before collecting the formal sample and report the issue; do not shrink an arm after seeing results. Freeze an explicit runtime/test source membership after the release preflight. Preserve earlier source versions used by numerical preflights when hardening an implementation before the formal freeze.

## Data population and separation

Collect exactly **4,000 new complete matches**: indices 0–3,199 training, 3,200–3,999 development, fixed before outcomes. Use one each of shanten_v2, defense_v2, original-reference argmax and the Stage 3F explorer, with seeded seat permutations and independent seat RNGs. The explorer uses the original reference plus initialized 5% call gate; on non-win decisions it mixes 5% semantic-uniform exploration with 95% gated policy, and takes the first semantic legal win deterministically. This is a collection restriction, not universal optimality.

Save all four seats, exact marginal/component probabilities and random draws, safe actor arrays, terminal targets, explicit hand ordinals and separate complete auditor logs. Build all simultaneous views before any decisions. Only **training-split explorer decisions with more than one semantic candidate** may enter formal value fitting. Frozen opponents, development records, old Stage 3F pilot records, partial games and all book cases are excluded. No record is on-policy PPO data.

Sixteen fresh, separately seeded preflight matches precede this corpus. They permit transient optimizer tests only; neither their models nor records may become formal candidates or corpus training examples. The new adapter verifies Stage 3G report identity and each numeric behavior ID against the declared seat assignment before filtering. Actor/target numeric formats reuse the validated compact Stage 3F arrays without broadening the actor information boundary.

Fresh evaluation seeds and 40 corpus replay positions `0,100,...,3900` are fixed in the seed plan before collection. The historical seed screen covers integer seed fields in project configurations and historical plan files, not every historic trajectory. Equal seeds alone are not a wall-equivalence claim; paired development games must additionally have matching explicit physical-wall commitments at every common hand ordinal.

## Value models and training

All four arms share the same original reference backbone and identically reset scalar action scorer, initialization seed 20261023011. Train all actor/scorer parameters; freeze and leave the old rank head unused. Keep the existing actor-only feature schema and architecture. Candidate inference uses legal-action value argmax, except for the declared first-legal-win rule. The original-reference baseline retains its original argmax behavior.

| Arm | Chosen-action target | Penalty alpha |
|---|---|---:|
| mc_a0 | Complete-match utility | 0 |
| mc_a1 | Complete-match utility | 1 |
| delta_a0 | One-hand cross-fitted placement-potential difference | 0 |
| delta_a1 | One-hand cross-fitted placement-potential difference | 1 |

Use `mean_match mean_optional[(Q_taken-y)^2 + alpha*(logsumexp_legal(Q)-Q_taken)]`, preserving utility units `[1.5, .5, -.5, -1.5]`. This penalty is algebraically chosen-action cross-entropy here. It is an imitation-like regularizer, without an automatic conservative Bellman guarantee. Terminal Monte Carlo targets reflect the recorded explorer continuation and fixed opponents, not automatically the new greedy policy's value. The one-hand difference is an explicitly different surrogate, not a full decision return.

Use Adam learning rate 0.0001, betas (0.9,0.999), epsilon 1e-8, no weight decay, and gradient L2 clipping at 1. Accumulate sixteen whole-match means before each update, with tensor batches capped at 256. Empty matches contribute zero numerator with the fixed group denominator; an entirely empty group is recorded without an optimizer step. Both epochs use the same predeclared match/row shuffle seed 20261023012 across arms. Run **two complete passes, 400 scheduled updates per arm**; no early checkpoint selection or hyperparameter sweep. Save initial, update-100/200/300 and final-400 boundaries with model, optimizer, cursor, initialization, data, target and RNG identities.

Stop on nonfinite values/gradients/parameters, absolute legal value above 100 or per-row objective above 10,000. These are fixed integrity alarms, not a tuned strength criterion. Retain stopped evidence without automatic resumption. Resume implementation is tested at a preflight boundary; it does not authorize retrying a failed formal arm.

## Placement prediction and predetermined gate

Use a separate public-hand-start predictor with 31 explicitly listed inputs: relative scores and differences, round/hand, honba, riichi deposits, dealer relation, initial seat/tie priority and explicit hand ordinal. No private tiles, complete engine logs or future outcomes enter those features. Predict the explorer's final rank with a 32/32 tanh MLP and four-way softmax, fixed 80 epochs, CPU Adam learning rate 0.003, minibatches 512 and per-game hand-state weighting. Fit five balanced whole-match folds plus one final train-only predictor: **six fits**. Initialization, fold assignment and shuffle are fixed before outcomes.

Training hand potentials use the model that never fitted any hand of that match. Development potentials use the final predictor. The label for hand k is `V_fold(start[k+1]) - V_fold(start[k])`, with actual utility replacing the next potential only at true match termination. Use the same predictor for both boundaries and explicit hand starts, including dealer continuations. Serialize every target, predictor identity and corpus hash separately from actor data.

Compare on the fixed 800-game development split against an unfitted public-score Plackett–Luce rank baseline, enumerating all 24 seat permutations with score scale 10,000. Both game-weighted NLL and Brier score must be no worse than that baseline. Check expected-utility calibration overall and in early (East 1–2), middle (East 3 through South 2), and late (South 3 onward) strata. Each needs at least 100 distinct games. Fixed predicted-utility bins have edges `[-1.5,-.9,-.3,.3,.9,1.5]`; a bin needs 50 distinct games to be supported. Supported bins must cover at least 90% of weight. Weighted absolute supported-bin bias must be at most 0.15, and maximum supported-bin bias at most 0.30, in every stratum. Report sparse-bin mass; do not silently ignore it. Include classwise calibration and 1,000 game-cluster bootstrap replicates as descriptive uncertainty.

If any gate fails, **skip both delta arms** and continue only mc_a0/mc_a1 under this predeclared fallback. Report the factorial comparison as incomplete. Do not relax thresholds, refit until passing or replace the target. A software failure or deadline in fitting is a failed formal operation and stops the campaign rather than invoking this calibration fallback.

## Development selection, distinct from confirmation

Freeze 64 independent shared-wall blocks and four focal-seat rotations. Each passing arm and one shared original-reference baseline faces the same three fixed opponents: shanten_v2, defense_v2 and reference, rotated around the focal seat. With four arms, this is **1,280 games**; if calibration disables the delta arms, it is **768 games**. Fix all checkpoint identities and panel assignments before the first game. Save all completed and interrupted attempts, response commitments, action-policy identities, scores and ordinal physical-wall commitments. No development game can train.

Choose the highest observed mean utility, with fixed tie order mc_a0, mc_a1, delta_a0, delta_a1. If no candidate exceeds the original reference descriptively, select none. Compare candidates through this common game outcome; raw losses for different targets are not comparable selection metrics. Report paired block differences and descriptive uncertainty without calling them confirmatory evidence. No policy promotion occurs in Stage 3G. Any later confirmation needs new blocks, a separately frozen confidence/power rule and a separately authorized budget.

## Preflight evidence and completion

The fresh-data optimizer preflight passes on CPU/MPS. MPS measured about **2,552 optional rows/s or 15.639 match presentations/s**, including two complete 16-match updates and a saved boundary. The repeated second MPS update after reload produced identical weights; optimizer moments agreed within declared tolerance, not bitwise. The CPU/MPS two-update maximum weight difference was 5.96e-8. These short measurements justify a provisional time forecast; they do not establish long-run stability. The final behavior-ID/schema adapter hardening only adds validation and does not change the optimizer computation; its revised source is separately tested before formal collection.

Completion requires the fixed corpus, all gate-eligible arms at their scheduled endpoints, the appropriate entire development panel, source/record/target/checkpoint audits and prespecified same-engine replay. Distinguish complete, skipped-by-gate, failed and unstarted phases. Keep rule coverage, weak opponent population and sparse kans explicit. Package a factual results report, review recommendations, closed resource ledger and file inventory. Supplied books remain preserved; this experiment adds no new book extraction or training.
