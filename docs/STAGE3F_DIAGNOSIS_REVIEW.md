# Stage 3F saved-training diagnosis

The saved Stage 3E update reconstructs correctly. The diagnosis found **sparse conditional-call learning signal, substantial within-batch gate-gradient cancellation and small output changes despite real parameter movement**. It did not find a broken optimizer or replay mismatch. These observations support broader controlled data collection and a value-learning comparison; they do not identify one proven cure or establish playing strength.

This is training-development analysis. No new matches, optimizer steps or evaluation rows were used. All model, optimizer and input files remained unchanged. The frozen selection plan, measured result and three operation receipts preserve the evidence. The 47 targeted tests passed. Tests, preparation and execution consumed **6.248 charged seconds** of the 300-second diagnosis allowance; the diagnostic itself reported 4.112 seconds. Numerical execution used CPU, one thread, Python 3.14.8, Torch 2.14.1 and NumPy 2.5.4.

## What was reconstructed

The complete first Stage 3E batch contains 16 matches, 2,328 optional learner rows and 196 forced rows. Every row's behavior probabilities and selected log probability reproduced exactly. All optional rows contributed to the complete-match objective; 511 supplied a trainable gradient. The reconstructed loss, **13.420581422353646**, agrees with the recorded **13.42058142235365** within floating-point summation precision.

Both head gradients reproduce the recorded values. The total norm was **0.6523874**, below the clip threshold of 1. Saved Adam moments agree with the reconstructed clipped gradients, and the largest difference between formula-predicted and saved next parameters was **3.73 × 10⁻⁹**. No optimizer step was executed to obtain this result.

| First-batch component | Total gradient L2 | Sum of individual match-gradient L2 norms | Ratio: total norm / sum of norms |
|---|---:|---:|---:|
| Call gate | 0.652060 | 2.061298 | 0.3163 |
| Conditional call choice | 0.020679 | 0.026496 | 0.7804 |

The gate contributions point in conflicting directions: eight of 16 have a negative cosine with the total gradient. This is direct evidence of cancellation within this batch, **not** an estimate of population variance or proof that a learned baseline will help.

The conditional-call gradient has material contributions from only the first two matches. Contributions from the other 14 are at most **1.09 × 10⁻¹⁸**, numerical residuals rather than useful evidence of learning. Cosines involving these near-zero vectors are unstable and should not be interpreted. Across the 28 recorded updates, batch 13's conditional-head norm was similarly only **1.94 × 10⁻¹⁷**. Earlier statements that both head gradients were “nonzero” remain literally true, but do not imply meaningful conditional-choice signal in every batch.

Only **three of 28 updates** exceeded the norm-1 clipping threshold. This rules out continuous clipping as the explanation for every small update; it does not measure the effect of removing clipping.

## Parameter movement and behavior

Selection was frozen before model computation: all 16 win-versus-call rows, all 83 selected multiple-call rows and 157 SHA256-selected remaining call rows, from 13,934 training call opportunities. The 256-row set is deliberately enriched for rare events and is not a representative policy-gradient estimate.

On these identical saved states, update 4 to update 32 produced:

- Gate parameter L2 movement **0.04345** and conditional-head movement **0.07126**.
- Mean call-probability change **−0.00007458**, approximately **−0.00746 percentage points**; call probability decreased on all 256 selected states.
- Largest absolute action-probability change **0.00017698**, approximately **0.01770 percentage points**.
- Largest conditional-call probability change **0.00110387**, approximately **0.11039 percentage points**, before multiplication by the roughly 5% call gate.

The architecture helps explain why visible parameter changes need not produce large action changes: conditional-call differences are multiplied by a small call probability. Moreover, the final scalar bias shared by every call candidate cancels under the conditional softmax; movement in that parameter cannot improve action ranking. These are properties of the implemented distribution, not additional experiments.

The measured results therefore support separating coverage, credit assignment and output sensitivity in the next experiment. They do not establish that a higher learning rate, an unvalidated critic, or a different normalization is superior.

## The two calls selected instead of an available win

Both cases offered PON, RON and PASS, and both selected PON. The exported context contains the actor's own encoded hand, current public scores, explicit hand ordinal, legal actions and sampling provenance. Hidden hands and walls are excluded.

| Saved training position | Public situation | PON probability | RON probability | Sampled uniform value |
|---|---|---:|---:|---:|
| Match 756, row 15 | East 1, honba 0, hand ordinal 1; learner seat 0 is dealer. Scores by absolute seat: 25,000 / 24,000 / 25,000 / 25,000; one riichi stick. PON offered on 6p. | 0.0497281 | 0.9502719 | 0.0482180 |
| Match 789, row 90 | South 1, honba 1, hand ordinal 6; learner seat 1, dealer seat 0. Scores: 58,800 / 15,400 / 9,700 / 14,100; two riichi sticks. PON offered on 2s. | 0.0498114 | 0.9501886 | 0.0158971 |

Neither is a final-round situation. Each had one existing open meld; both PON requests resolved. The selected actions follow the recorded sampler probabilities. The eventual utilities were +1.5 and −0.5 respectively, but those outcomes do not establish whether either PON was optimal or what a RON would have achieved.

A win-preserving exploration heuristic is consequently reasonable for the collector pilot, provided it is explicit and its actual behavior probabilities are recorded. It is a collection design choice, not a certified rule that declining a win is always wrong.

## Implication for the programme

Proceed with a measured, broader exploration collector and bounded-memory loader. The future learning comparison should retain actual complete-match placement utility as a control and isolate any placement-prediction target. The diagnosis supports that programme without justifying another unchanged continuation, a blanket learning-rate increase, or promotion of an existing checkpoint. The first-batch mechanism check and enriched state sample are narrow diagnostics, not a strength evaluation or a causal ablation.
