# Stage 3G: how to interpret the learning experiment

22 October 2026. This note explains the experiment's scientific scope; it does not report development-game outcomes. The [frozen plan](STAGE3G_PLAN.md) specifies the execution rules, and [STATUS](STATUS.md) records progress. This stage compares four fixed learning recipes with one initialization and one opponent population. It can demonstrate functioning learning and identify a candidate for independent testing; it cannot establish general superiority or world-class strength.

## What the learner actually optimizes

Only optional explorer decisions from the 3,200 training matches enter value fitting. All other seats remain available for audit. The actor sees audited player-visible features and legal actions; final outcomes, behavior identifiers and placement targets are labels or provenance, never actor inputs. The separate placement predictor does not replace the frozen, unused rank head in the actor network.

For a group of sixteen complete matches, the implemented objective is:

```text
L = (1/16) × sum over matches m:
      (1/N_m) × sum over eligible decisions t in m:
        [(Q(x_t, a_t) - y_t)^2
         + alpha × (logsumexp over legal a of Q(x_t, a) - Q(x_t, a_t))]
```

`N_m` is that match's number of optional explorer decisions. A match with none contributes zero under the fixed denominator of sixteen; an entirely empty group causes no optimizer step. Tensor chunks preserve the complete-match denominator. Both passes use the same fixed match and row orders across arms. These are offline value-regression updates; neither PPO nor a bootstrapped Bellman update is implemented.

The terminal-target arms use the explorer's actual final rank utility `[1.5, 0.5, -0.5, -1.5]`. All eligible decisions from a match share that terminal outcome. Many rows therefore share substantial uncertainty; their count is not a count of independent learning outcomes.

## Equal-match weighting changes the statistical target

Averaging within each match prevents long games from dominating simply by supplying more decisions. That is the declared design choice. It also gives each eligible row weight proportional to `1/N_m`, where `N_m` depends partly on events after the decision being fitted.

Consequently, even the alpha-zero terminal arm should be described as a **match-reweighted return regression**. In an unrestricted population regression, its conditional optimum has the form `E[wY | x,a] / E[w | x,a]`, with `w = 1/N_m` under the unweighted recorded-row distribution. It need not equal the ordinary decision-visitation quantity `E[Y | x,a]` when match length and outcome are related. The finite neural network adds approximation error. This is an interpretation caveat, not evidence of an implementation error or a reason to change the frozen objective mid-experiment.

A later controlled comparison could isolate equal-match versus decision weighting. It would need a declared exposure/update budget and fresh evaluation, since changing the weighting also changes gradient magnitudes and sampling emphasis.

## Recorded actions, coverage and policy change

Monte Carlo outcomes describe continuation by the recorded explorer against the three fixed opponents. The deployed candidate instead chooses the highest predicted legal value, subject to its win rule. That changes both immediate actions and future states visited. Approximate regression on the existing corpus does not guarantee reliable values for the newly preferred actions or resulting states.

The exploration mixture gives non-win alternatives sampling support, but a positive theoretical probability is not the same as useful observed coverage. Rare calls and kans may still have little evidence. At legal-win decisions the collector deterministically takes its declared win, so unchosen alternatives do not acquire outcome labels through exploration there. Exact behavior probabilities make these restrictions auditable; retaining those probabilities does not by itself correct policy shift.

A reduction in training loss, substantial parameter movement, or changed action frequencies establishes different facts from improved play. Fresh complete games are necessary to test the extracted policy. The current fixed opponent pool also limits conclusions about defense against strong calling strategies.

## What alpha zero and alpha one mean

The penalty is exactly the negative log probability of the chosen action under a softmax of the legal values. It is therefore an imitation-like regularizer on relative action scores. It also sends gradients to legal actions that were not selected in that row, but those gradients are not counterfactual outcome observations.

Alpha one trades outcome fit against this regularizer. Its scores are not automatically calibrated expected utilities, and the CQL-shaped expression supplies no automatic conservative Bellman-value guarantee in this Monte Carlo setup. Alpha zero is the control for that tradeoff.

The same alpha can have different relative influence across target distributions. If hand-delta labels are concentrated more tightly than terminal outcomes, the penalty competes with a differently scaled regression problem despite both targets using utility units. A target-by-alpha interaction would support a comparison of these four recipes; it would not establish a universally best target or penalty strength. Raw losses across different targets cannot select the strongest playing policy.

## One-hand placement change is a different objective

The delta target for hand `k` is `V(start[k+1]) - V(start[k])`, with actual terminal utility replacing the next potential only when the match truly ends. Explicit hand ordinals preserve dealer continuations. Whole-match cross-fitting prevents a training match's own outcomes from fitting the predictor that supplies its potentials; both boundaries use that same held-out predictor.

Each eligible decision in a hand receives that hand's single delta. The learner does not sum future deltas into a full return. Repetition across decisions and the within-match averaging also prevent interpreting this loss as a simple telescoping sum over hands. It is a deliberately different surrogate, not a demonstrated policy-invariant reward transformation.

The saved placement report passed the predetermined gate: development NLL was **1.13613** versus **1.15152** for the fixed public-score baseline, and Brier score was **0.61585** versus **0.62174**. All four learning arms are therefore eligible. Passing the gate is an eligibility decision, not proof of predictor superiority: the reported bootstrap intervals for the NLL and Brier differences include zero. Calibration of public placement estimates also does not establish accurate action values or beneficial strategic credit assignment.

## What the development panel can establish

All four candidates share the deterministic first-legal-win rule. The original reference baseline retains its original argmax behavior. Thus candidate-versus-baseline results compare the entire resulting policies, including that rule difference. Comparisons among the four candidates hold the rule fixed. An advantage over the original reference alone cannot isolate the contribution of learned parameters from the wrapper or other changes to the action scorer.

There are **64 shared-wall blocks**, each with four focal-seat rotations. The rotations within a block share randomness, and candidates share the baseline. The block is the uncertainty unit; the 256 games per policy are not 256 independent paired comparisons. Explicit physical-wall commitments support pairing, without constituting cross-engine rule certification.

The best observed arm is selected using the frozen tie rule and only if its mean exceeds the reference. Selecting among four arms favors positive noise as well as real advantage. The development intervals remain descriptive; they do not become independent confirmation after selection. One initialization also leaves variability across training seeds unmeasured.

## Conditional next steps

- **If a candidate is selected:** freeze its identity and test it on fresh paired blocks with a predetermined precision or power calculation and stopping rule. Keep the original-reference comparison primary. A secondary reference with the same win rule would help separate wrapper effects.
- **If none is selected:** first inspect a bounded, fixed set of identical visible states for action changes, observed coverage, value errors and rare-action behavior. Treat that as diagnosis rather than a new strength test. Do not infer that a larger model, longer run or more expensive hardware is the necessary remedy.
- **If a target or regularizer looks promising:** replicate the relevant contrast with a separately fixed initialization and evaluation sample before generalizing. Use one-factor follow-ups to distinguish target choice, regularization and match weighting.
- **Before broader strength claims:** add stronger and more varied opponent policies, deliberate rare-state coverage, and independent rule/behavior checks. The current corpus and book quarantine do not supply those guarantees.

These are proposals for a later bounded stage. They do not change the current sample, training exposure, calibration thresholds or development selection rule.
