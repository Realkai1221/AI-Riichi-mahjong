# Suphx assessment for this project

Reviewed 18 October 2026. Research and design recommendations only: no external game corpus, model weights or repository was downloaded, and no Suphx-derived training run was started.

**We should build on Suphx's research ideas while retaining our audited implementation. I did not find a Microsoft-published repository or pretrained checkpoint that we can verify and use as a starting model.** That is a bounded finding from the official project/publication pages and targeted searches, not proof that no release exists anywhere. The official pages currently point to the paper, news and game/review material, without a publisher-linked implementation or weight download. [Microsoft project](https://www.microsoft.com/en-us/research/project/suphx-mastering-mahjong-with-deep-reinforcement-learning/), [Microsoft publication](https://www.microsoft.com/en-us/research/publication/suphx-mastering-mahjong-with-deep-reinforcement-learning/).

## What the publication actually establishes

Five deep CNNs handle discard, riichi, chi, pon and kan, with tile channels and look-ahead features. Figures 4–5 show 256 channels and 50 repetitions. Supervised training precedes distributed policy-gradient RL with adaptive entropy; the reported RL ablations update only discards. Training examples: discard 15M; riichi 5M; chi/pon 10M each; kan 4M. Tenhou supplied expert logs.

A GRU predicts final rewards; differences between successive predictions supply hand rewards. Oracle training progressively removes hidden features. Runtime adaptation samples hidden deals, updates a temporary policy, and resets it next hand.

Each 1.5M-game RL ablation used 44 GPUs—four Titan XP learners plus forty K80 rollout GPUs—for two days. One million-game evaluation used twenty K80 GPUs for two days. Runtime-adaptation experiments used 100K training trajectories and 10K test trajectories per fixed initial hand. The online agent trained on approximately 2.5M games; its 5,000-plus Tenhou games were evaluation. Runtime adaptation was excluded online because of latency. Thus its published online achievement does not validate real-time adaptation. [Suphx paper, §§2–5, especially §5 footnote 8](https://arxiv.org/pdf/2003.13590).

Microsoft's explanation also emphasizes that final placement can make a locally unattractive hand decision strategically correct. Its public account describes hidden-information guidance, whole-game prediction and exploration management as complementary contributions. These are useful research directions, not a guarantee that adding any single component improves our current policy. [Microsoft Research technical article](https://www.microsoft.com/en-us/research/articles/mahjong-ai-suphx/).

The historical performance claim is about its particular evaluation setting. It is not a current world ranking, and it is not directly comparable with our heuristic-imitation agreement, local opponent panel or average-rank utility. Keep the room/utility distinctions already recorded in [DECISIONS.md](DECISIONS.md).

## What we should adopt, change or defer

The table below is our project-specific recommendation, not a claim that these modifications reproduce Suphx.

| Component | Decision | Concrete project application | Gate before training or promotion |
|---|---|---|---|
| Audited Mahjong structure | Adopt now as a design principle | Retain explicit legal candidates and exact shape/efficiency calculations. Add value, wait quality or threat features only in separately versioned ablations. | Independent examples, actor-visibility audit and measurable benefit at matched compute. |
| Supervised starting policy | Retain | Keep our frozen imitation reference and use accepted book exercises as separate diagnostic evidence. Do not replace a verified pipeline with an unverified third-party one. | Full-match retention and appropriate scenario coverage remain separate tests. |
| Final-placement predictor | Adopt as a future isolated study | Train a separate predictor against complete-match rank targets. Start from simple standings/stage baselines, then test a small history model. | Calibration and predictive benefit on held-out matches, especially early hands and policy/opponent shift. |
| Privileged information | Modify | Prefer a separate critic that receives actor history plus a declared privileged state. Keep the actor's input contract unchanged. | Hidden-state permutation tests, correct return semantics, and a controlled full-match RL comparison. |
| Exploration management | Test later | Measure legal-action entropy, normalized entropy, non-best probability and strategic error severity separately. | Compare one exploration mechanism at a time after the base learner is reliable. |
| Action-specific learning | Keep as an ablation option | Use per-action-family reporting and targeted call scenarios. A shared backbone with separate heads is a possible experiment; five full networks are not required. | Genuine post-call data and legal-response coverage; action probabilities must be calibrated across choices. |
| Temporary policy adaptation | Defer | Begin with a fixed-policy offline rollout evaluator before allowing online gradient updates. | A valid hidden-state sampler, fresh rollout evaluation, bounded latency and evidence of gain. |
| Distributed training scale | Defer | Keep a portable learner interface and per-seat behavior-policy provenance. Benchmark one local inference/rollout pipeline first. | Measured throughput bottleneck and an explicit later compute decision. |
| Venue-specific reward or win-declining rules | Do not copy by default | Preserve utility `[1.5, 0.5, -0.5, -1.5]` and the project's configured rules. | A separately declared venue/rule experiment. |

## Why this does not resolve today's imitation problem

The [completed diagnosis](../reports/package_a_diagnosis/REPORT.md) found incomplete fit on the training sample as well as validation difficulty. The learning-rate comparison has now completed: the lower rate missed the prespecified KL comparisons on both seeds and did not improve full-validation discard agreement against either paired control. A new reward predictor, oracle or larger network would change the question and must receive a separate experiment identity. The 99.9% threshold belonged to that smoothing study; it is not a mathematical prerequisite for a small RL engineering pilot. Retain the original rank-0.1 reference rather than promoting one of the new smoothed checkpoints.

Our current teacher is an efficiency/risk heuristic, not a supplied corpus of expert strategic decisions. High agreement with that teacher tests whether we copied its behavior; it does not establish optimal Mahjong. Conversely, disagreement with a book can reflect missing assumptions, a different objective, a genuine teacher limitation or a transcription error. Keep those explanations separate before deciding that the learner is wrong.

The present corpus also lacks executed chi, pon and kan coverage. An extra call head does not supply missing experience. We need reviewed scenarios and eventually complete matches containing these actions, with simulator behavior, player-visible observations and training provenance audited together.

Useful near-term work is therefore a small scenario audit: classify each accepted example by objective, compare the teacher's recommendation with the supplied answer, measure the learned policy separately, and preserve unresolved cases. An efficiency-only exercise should not be relabeled as a full-match strategic optimum. See [book scenario intake](BOOK_SCENARIOS.md).

## A safe placement-prediction experiment

Our existing rank head is not ready to supply calibrated value estimates; its prior validation result is recorded in [STATUS.md](STATUS.md). I recommend a separate module with independent weights rather than attaching a new role to that head.

Use complete local matches with their original seat identities and final ranks. Split by match, not decision; keep all related book variants and any repeated wall family in the same partition. Inputs should state exactly what is known at each prediction time: scores, seat/dealer, round, honba, riichi deposits and the sequence of completed-hand events. A prediction recorded before a hand cannot consume that hand's result. Hand-end information becomes available only after its settlement.

Predict the four placement probabilities and compute expected utility as their dot product with our declared utility vector. Compare a standings table, a small feed-forward model and a small recurrent model at matched data/compute. Report NLL, Brier score, calibration by match phase and opponent population, and a simple current-standing baseline. A value estimate that only improves late-game prediction is useful in a narrower way than early strategic foresight.

Keep this predictor frozen during an initial RL ablation. Record its version in every trajectory and reward transform. Retain actual final-match outcomes in storage so we can recompute and audit targets instead of committing permanently to one learned reward definition.

For any shaping experiment, our starting formula should be explicitly declared, for example

`r'_t = r_t + gamma * Phi(I_{t+1}) - Phi(I_t)`

with consistent discounting, a fixed potential during the run and zero potential at true terminal states. The classical potential-shaping result supplies the relevant objective-preservation conditions; it does not promise that approximate value learning or truncated optimization is unbiased. [Ng, Harada and Russell, 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

For this project, preserve the complete-match reference and explicitly test termination versus truncation. Dealer continuations are distinct hands even when round/dealer indices repeat. A telescoping sum over a full match does not justify cutting the policy-gradient credit assignment at every hand boundary. Those constraints already appear in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md); this assessment does not weaken them.

## Privileged critics without actor leakage

Literal hidden-feature actor training is incompatible with our current actor-only feature contract. A separate history-plus-state critic is the first admissible experiment here. The critic's training data can contain declared privileged fields; the policy serialization, normalization statistics and inference function must not receive them.

Using a state-only critic with a history-dependent actor is not automatically sound under partial observability. The history/state distinction and gradient consequences require care. [Baisero and Amato, *Unbiased Asymmetric Reinforcement Learning under Partial Observability*, AAMAS 2022](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf).

Our proposed controls are:

1. Keep actor and critic input schemas, encoders and saved artifacts separate. Avoid an accidentally shared privileged representation.
2. Construct paired simulator states with identical player-visible information but different opponent hands/walls. The actor's logits and legal decisions must remain identical; the critic may change.
3. Preserve the actor's history/memory in critic examples. Compare visible-history and history-plus-state critics on the same complete-match trajectories.
4. Begin without future wall order. If we later test it, treat that as its own explicit ablation and report its incremental value and sensitivity.
5. Check value prediction, advantage scales and policy behavior as well as end-match utility. Lower critic loss alone is insufficient.

An oracle's hindsight action can be impossible to infer from the acting player's information. Distilling its single hidden-world choice as if it were a universally correct visible-state label can introduce contradictory targets. Any oracle-target experiment would need conditional averaging or another clearly stated target definition, and a comparison with the visible-information baseline. It is not a replacement for the current book-label review.

## Runtime adaptation: useful research, substantial prerequisites

For us, the first implementable precursor is a fixed-policy simulator study, not a changing policy in public play. Given a reviewed information state `I`, define an explicit distribution `q(hidden_state | I)` and evaluate actions under that model. Everything below is a proposed local experimental contract.

A valid sample must respect physical tile multiplicities, red-tile identity, all known private/public tiles, melds, dora information, remaining wall counts and the history of legal events. It must not import the actual simulator's unseen wall or opponents' true hands as privileged answers. At mid-hand, uniform allocation of remaining tiles is generally only a crude prior: observed discards, calls and riichi also carry information about opponent behavior. Call it an approximation rather than a calibrated posterior.

Monte Carlo averages can be unbiased for the declared sampling model and fixed rollout policy when sampling and simulation are correct. They are not automatically unbiased estimates of real opponents' conditional value. More rollouts reduce sampling noise; they do not remove an incorrect belief model or opponent model.

Each rollout policy must receive only that seat's legal information. The focal policy must not choose different actions for otherwise identical information states merely because the planner knows different sampled hidden states. Audit this explicitly to avoid a planner that relies on information unavailable during play.

If gradients are introduced later, adapt a temporary clone and reset at the declared boundary. Keep model-selection rollouts and evaluation rollouts independent. Evaluate any improvement on fresh root scenarios and fresh conditional deals; reusing the optimization samples to claim success creates optimistic bias. Freeze the opponent policies and retain behavior probabilities and checkpoint identities for all relevant decisions. Frozen-opponent actions cannot become on-policy PPO examples just because they occurred in the same rollout.

Measure rollout count, wall-clock latency, policy drift, uncertainty and complete-match utility. Impose a deadline with a tested fallback to the frozen policy. A tiny Mac pilot would establish whether this machinery is correct and promising; it would not replicate the published experiment.

## Starting from somebody else's code

One public repository found during this search, [HongyiHao-SXIT/MahJong_RLAI](https://github.com/HongyiHao-SXIT/MahJong_RLAI), describes itself as a personal Suphx reproduction. Its README advertises supervised/RL training, specifies CUDA, and asks users to provide trained model files. It also lists rule choices and a custom engine. This is a third-party implementation claim, not a Microsoft release or independently verified Suphx-strength checkpoint.

I would not switch our base to it on the basis of that README. A potential code reuse review would first pin a commit, inspect license and asset provenance, audit observations and simultaneous responses, compare rules against our explicit contract, and test MPS/CPU portability. Any useful component could then be introduced in isolation. No part of that repository has been executed or imported in this assessment.

## What fits the M5 Pro / 64 GB plan

Our present machine is suitable for small supervised experiments, scenario validation, predictor prototypes, correctness fixtures and bounded RL infrastructure pilots. This is an engineering judgment based on our existing local workflow, not a measured Suphx reproduction claim. The latest continuation's compact fixed-data training rate cannot be converted directly into neural self-play throughput.

Before scaling, measure complete matches per second with the actual candidate actor, legal observation construction and data recording enabled. Also measure learner throughput while rollout workers run: unified memory and shared compute can make separate peak benchmarks misleading. Record sustained memory, thermal behavior, queue depth, inference batch size and learner utilization.

Keep model size unchanged for the first future RL correctness pilot. Use a small fixed opponent panel and complete matches, then evaluate frozen checkpoints on fresh explicit walls with seat rotation. Only after this pipeline works should we decide whether throughput warrants CUDA training elsewhere. The historical GPU inventory is not a defensible conversion into M5 training days or a current rental quote.

## A concrete next Mac pilot, proposed and unstarted

A bounded Stage 3 engineering pilot can proceed without another open-ended imitation sweep. It should start from the preserved original rank-0.1 reference, with a new, explicitly declared optimizer for RL; this is a new experiment, not an exact continuation of a supervised optimizer. The present smoothed candidates remain unpromoted.

Proposed ceiling: **600 cumulative numerical seconds, 32 GiB sampled memory, 1 GB of new artifacts, at most 64 complete matches and four optimizer updates.** Freeze the complete protocol before collection. These are work caps, not a claim that this Mac has been benchmarked to reach all of them. Any unfinished collection is recorded as incomplete, with no replacement campaign.

Use one learner seat against three frozen baseline opponents, rotating the learner seat. Plan four batches of up to 16 matches and fix their match counts, seeds, learner seats and opponent identities before collection. Freeze the learner for each complete batch. All planned matches must finish and validate before exactly one update; retire the batch immediately afterward, without further optimizer passes or stale reuse. A failed, missing or truncated match invalidates the whole batch for learning. Do not update on only the faster completers. Reserve time to finish active matches, while respecting the cumulative deadline; an incomplete batch is preserved without an update.

This is a **REINFORCE engineering substage before the planned full-match PPO reference**, not a substitute for that later algorithm. Use a fixed zero baseline, undiscounted final utility, no learned critic and no new entropy/smoothing schedule. Minimize `L = -(1/N) * sum_match[final_utility * sum_eligible_action(log_probability)]` at the weights that collected the `N` complete matches. Match averaging preserves this complete-match objective; replacing it with decision averaging would change the weighting. Keep the uncalibrated rank head frozen and unused as a critic. Fix the update learning rate, clipping and stopping rules prospectively; do not choose them from outcomes.

The existing `NeuralPolicy.choose` is deterministic argmax. A **new categorical softmax behavior wrapper** must therefore define the actual legal semantic-action distribution, random state and behavior-policy identity. Never reuse unused softmax scores as the historical argmax policy's probabilities. Single-option actions have probability one and zero policy-loss contribution; retain their trajectory links. An emergency fallback or action override invalidates the batch for this initial learning pilot instead of being mislabeled as an on-policy sample.

The starting logits are very sharp: the new wrapper's unmodified softmax may explore almost exclusively within the teacher's best set. This tiny pilot tests trajectory handling, accounting and numerical behavior; it must not promise policy improvement. Any temperature or epsilon change needs a separate declared intervention. Every actor input must still pass through the canonical audited player-visible encoder.

The collector must record the acting policy and checkpoint versions, actual sampled-action probability, legal mask, actor-visible history and match/hand/seat identities. Verify stored probabilities before the update. Simultaneous responses use the same pre-response state. Only eligible learner actions enter its policy loss; frozen-opponent actions never do. Preserve final utility even when an opponent finishes the match after the learner's last action. A wall-time cutoff is truncation, not a terminal outcome.

**Pilot pass requires at least one complete, prespecified, valid batch and an executed optimizer update.** On the declared fixtures and collected batches, independently reconstructed returns must agree, observations must pass the information-boundary checks, sampled actions must be legal, behavior-policy provenance must be complete, gradients and weights must be finite, and checkpoint interruption/resume and resource accounting must pass. Report how many of the four planned batches completed; no valid batch/update means incomplete. Record entropy, parameter change and action-distribution change. A finite but effectively zero update is a diagnosed no-op, not evidence of successful learning. These checks cover the tested cases; they do not certify all rules or prove a universal information boundary. This is an engineering gate with no playing-strength acceptance claim and no deployment.

If that gate passes, propose a separate, fresh, complete-match comparison at fixed compute and opponent mixture. Only then add one critic/reward component. Keep the book-scenario audit parallel to this work: it can reveal strategic blind spots without requiring endless imitation tuning before any RL infrastructure exists.

## Recommended next sequence and unresolved questions

1. Finish the current review packet and preserve all four learning-rate outcomes without promoting a new model.
2. Resolve the first supplied book scenarios: transcription, assumptions, objective, acceptable answer set and calculator/teacher disagreement.
3. Freeze the bounded RL engineering pilot above; retain the original reference and explicit limits.
4. If the engineering gate passes, plan a fresh full-match learning/evaluation study. Prepare placement-predictor calibration separately.
5. Add one critic, reward or exploration mechanism per ablation. Temporary adaptation follows a working belief/rollout evaluator.

Still unresolved: a verified Microsoft source/checkpoint release; reusable rights and exact preprocessing for expert logs; full feature/optimizer/schedule details sufficient for a faithful reproduction; the quality and call coverage of our eventual training population; an empirically calibrated hidden-state model; sustained neural-rollout throughput on this Mac; and a fresh strength evaluation showing improvement beyond the current heuristic reference.

