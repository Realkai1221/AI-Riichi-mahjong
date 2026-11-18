# Stage 3B review and proposed next experiment

**Stage 3B is complete. The next useful step is to make call selection learnable without flattening the entire discard policy.** The fixed diagnostic used 21.859 of 300 numerical seconds, with no new matches or optimizer updates. [Detailed results](../reports/stage3b/REPORT.md) and book preparation are ready for review.

## Decisions supported by the results

Keep the saved-state diagnostic, legal-mask audit, complete-match training infrastructure and original checkpoint reference. Keep CPU single-row execution for the next experiment.

Do not adopt temperature 10 or 100 as a general repair. They leave calls far too rare in this sample; temperature 100 also substantially changes ordinary discard probabilities. Do not adopt the tested 5% uniform mixture as an on-policy learning solution: it gives calls positive probability, but their direct `log q` gradients remain exactly zero.

Keep the MPS diagnostic's staged transfer repair. The discovered combined device/dtype conversion corrupted values on this installed build. The original Stage 3A MPS collector/replay path needs the same repair before future use; its historical CPU results remain valid.

Book preparation should continue as an explicit partial task. The new records preserve the twelve existing corrected examples, questions and explanations separately. Red-five identities and missing game context remain unresolved; neither should be guessed to create training rows.

## Proposed Stage 3C: a small call-selection learning pilot

This is a **proposal for review, not an experiment already started**. It changes the behavior policy and adds small trainable components, so the original checkpoint stays preserved.

Partition legal candidates into calls/kans `C` and other actions `N`. Keep the original frozen model's distribution, renormalized within `N`. Add an independent call gate `g(s)` and conditional call distribution `rho(a|s,C)`:

```
q(a|s) = g(s) * rho(a|s,C)              when a is in C
q(a|s) = (1-g(s)) * pi(a|s,N)           when a is in N
```

Compute `pi(a|s,N)` with stable softmax/log-sum-exp restricted directly to the original logits in `N`, not by dividing rounded full-policy probabilities. When only one group is available, give that group all the mass. Wins, pass, riichi and discards remain explicitly in `N`; calls include chi, pon and all kan forms. The gate also reduces win probability when a call competes with a win, so that effect must be measured and reported rather than hidden behind a safety override.

Initialize the gate to **5% total call mass** wherever both groups exist, and the call head to uniform over legal call candidates. Unlike the tested fixed mixture, a selected call then has a direct gradient through a trainable gate/head. This is a mathematical motivation; usefulness and playing strength remain untested.

Use detached, normalized player-visible representations from the frozen original network. A small gate and candidate-scoring head may each use one 64-unit hidden layer, with zero-initialized final weights and the gate bias set to the 5% log-odds. Fix the precise architecture, initialization seed and feature interfaces before any formal rollout. Freeze the base network, including its rank head, to isolate the intervention. Benchmark the added heads and test sampling/gradient identities before starting collection.

Zero final-layer initialization intentionally gives the hidden layers zero gradient on the first update. If there is only one legal call, its conditional probability is one and the call-head gradient correctly vanishes; the gate must respond when both groups are available. Test the conditional head on positions with multiple legal calls. The requirement is a response in the appropriate components, not a nonzero gradient for every parameter.

### Proposed fixed execution envelope

| Item | Proposed limit or rule |
|---|---|
| Numerical budget | 600 cumulative seconds, including checks, failures, preparation, training and evaluation; 32 GiB sampled process-tree RSS and 1 GB new artifacts |
| Device and precision | CPU, one thread, single-row float32 network execution and float64 probability normalization |
| Preparation/check allowance | At most 90 numerical seconds; stop before formal games if actual log-probability replay or direct gate/head gradients fail |
| Training | Four fixed batches of 16 complete matches, one learner seat against three frozen original argmax opponents; 64 matches and exactly four updates maximum |
| Loss | Learner-only full-match REINFORCE with `[1.5, 0.5, -0.5, -1.5]`, zero baseline, no entropy bonus, no book targets |
| Optimizer | Fresh AdamW on new heads only, learning rate 0.0001, weight decay 0, norm clip 1; one update per complete batch |
| Candidate | The endpoint after batch four, fixed in advance; no best-batch selection |
| Comparison | Trained heads versus their **identically initialized, untrained copy**, using the same gate/head probability rule and the same frozen base |
| Evaluation | 32 fresh paired seed blocks, four learner-seat rotations and both arms: 256 full matches maximum, with frozen parameters throughout |
| Total match cap | 320, including training and evaluation; no substituted games or extending a disappointing endpoint |
| Shutdown reserve | Keep at least 45 numerical seconds; do not begin a batch/block that cannot finish within its declared reservation |

Generate and save disjoint training/evaluation wall and sampling seeds before the first game. Do not reuse Stage 3A diagnostic positions as a test set. Use the existing paired evaluation contract for seat rotation and common seed handling.

Before formal games, require transition fixtures for the call/kan types and subsequent player-visible observations, hand/meld accounting, exact-efficiency features and legal choices. The original training corpus's lack of executed calls makes this a substantive coverage gate. Require exact sampling-rule provenance and replay, finite gradients, unchanged base weights, complete terminal rewards, and no simulator legality/accounting failures. Track requested and actually resolved calls separately, because simultaneous requests may preempt one another; also record subsequent open-hand states, call-family frequencies, probability mass lost from wins, and gate/head gradient response. A legal call appearing among candidates is not evidence of post-call coverage.

Evaluate paired terminal utility at the independent seed-block level, retaining all raw placements and scores. Let each block's difference be the average of its four matched seat differences, bounded in `[-3,3]`. Report the fixed-sample mean difference and a two-sided 95% Hoeffding interval with half-width `6 * sqrt(log(40)/(2*32))`, clipped to `[-3,3]`. Its half-width is about 1.44 utility units, so this 32-block pilot cannot establish a small gain. A favorable mean or engineering pass does not justify competitive-strength promotion.

If any required batch or paired block is incomplete, retain the failure and label the corresponding outcome incomplete. Do not replace it with faster games. Do not use frozen-opponent actions as learner data. No model, sampler or head initialization is tuned from evaluation outcomes.

### What would justify moving forward

First require the engineering result: calls actually occur, correct nonzero gradients reach the new heads, and complete-match updates change those heads while preserving the base. Then inspect the controlled outcome against the identical initialized policy. Only a separately sized confirmation could establish a reliable strength gain.

This pilot still faces a narrow opponent population and sparse terminal credit. The frozen base has little or no demonstrated open-hand training experience, so poor play after a call could limit improvement or drive the gate back toward zero. It does not yet add a calibrated value baseline, broad open-hand expert data or a league. Those are later experiments, not ingredients to change simultaneously if this small comparison disappoints. Four updates chiefly test the mechanism; the 320-match cap is conditional on measured throughput and the cumulative budget.

## Useful review points

- Whether the next priority should be this isolated trainable call mechanism; the current measurements support it over a longer unchanged run.
- Whether 600 numerical seconds and the fixed 64-training/256-evaluation match caps fit the next iteration.
- For the book track, corrections to red-five conventions, conditional recommendations or missing supplied context, identified by case and source page.

No Stage 3C code, training, new games or evaluation has started.
