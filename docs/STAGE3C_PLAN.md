# Stage 3C — frozen-base call-learning pilot

Authorized by the user's 21 October 2026 request to continue for another hour, following the [Stage 3B proposal](STAGE3B_REVIEW.md). The review checkpoint is approximately 16:31 UTC. This is the work window; numerical execution has a separate, stricter cumulative 600-second cap.

## Fixed question and method

Can a small trainable call gate and conditional call head learn from complete-match rank utility while the original policy stays frozen? Compare the fixed four-update endpoint with an identical initialized, untrained copy. This isolates learning from the effect of introducing a 5% call probability.

The frozen original Stage 2B checkpoint has SHA-256 `5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f`. Its noncall logits are normalized directly within legal noncalls using stable log-softmax. CHI, PON, DAIMINKAN, ANKAN and KAKAN form the call group; the other legal actions form the noncall group. Both groups use the gate when present, and a sole available group gets all mass.

Two new heads each have one 64-unit ReLU hidden layer. Non-affine LayerNorm normalizes detached context and candidate representations from the audited visible-input base. The gate consumes context; the conditional head consumes context plus candidate representation. Final weights start at zero, call bias at zero and gate bias at log(0.05/0.95). Seed 310211 fixes initialization. The base and rank head are frozen. Single-row CPU fp32 network execution and staged CPU fp64 probability transforms are fixed; the portable implementation does not authorize an MPS/CUDA campaign here.

Fresh AdamW trains the heads only, learning rate 0.0001, no weight decay, norm clip 1. One full-match REINFORCE update averages the sum of learner log-probabilities times terminal utility over 16 matches. Utility is `[1.5, 0.5, -0.5, -1.5]`; baseline zero, discount one, no entropy or book targets. Forced actions contribute no gradient; frozen-opponent actions are ineligible. Exact replay of collected probability vectors and selected log-probabilities is required before updating. Evaluation records are explicitly ineligible for training.

## Frozen execution envelope

The [configuration](../configs/stage3c.json) contains all seeds and match identifiers before games. SHA-256-derived namespace `riichi-ai-stage3c-20261021` separates training walls, training sampling, evaluation walls and evaluation sampling. There are 64 unique training walls, 32 fresh evaluation walls, and disjoint sampling seeds. Each training batch has four matches at each learner seat. Each evaluation block shares one wall seed across four learner-seat rotations and both arms, with a shared sampler seed within each pair.

| Limit | Fixed value |
|---|---|
| Numerical time | 600 cumulative seconds, including checks, rendering, failed operations, preparation, training and evaluation |
| Preparation before formal games | At most 90 actual numerical seconds |
| Memory / new artifacts | 32 GiB sampled process-tree RSS / 1,000,000,000 bytes |
| Training | Four sequential batches ×16 full matches; one update each; no replacement games |
| Evaluation | 32 sequential paired blocks ×4 seats ×2 arms =256 full matches |
| Endpoint | Batch four; no checkpoint selection from outcomes |
| Maximum full matches | 320 total |
| Reservations | Preparation30s, training batch100s, evaluation block40s; cumulative ledger keeps45s shutdown reserve |
| Precision / threads | CPU fp32 network, CPU fp64 probabilities, one Torch thread |

The controller records durable attempts before workers run. It refuses repeat batch/block identifiers and stops the chain after any failure. A watchdog checks time, sampled parent/child RSS, source hashes and artifact size. Workers stop cooperatively before the watchdog deadline. No model tuning, replacement matches, external acquisition, paid compute, test-split loading or policy promotion is included.

## Required preflight and evidence

Mandatory fixtures cover genuine CHI/PON/all three KAN transitions, subsequent legal actions, exact shanten/ukeire against the independent `mahjong` library, visible meld accounting, hidden draw filtering, consecutive concealed kans, preemption and robbed added kan. Preflight also checks stable group normalization, initial 5% mass, multi-call gradients, singleton/no-group behavior, win-mass reduction, exact base logits, detached inputs, checkpoint round trips and complete-match loss/provenance.

A successful current-source preflight receipt is required before preparation and formal collection. Initial preparation repeats a genuine multi-call request 16 times, benchmarks it, checks exact probabilities and direct gate/head gradients, and confirms no parameter/optimizer change. Hidden-layer gradients are correctly zero at zero-final-weight initialization. Singleton calls have no conditional-head gradient.

Save complete raw trajectories, safe encoded actor rows, terminal scores/ranks/reward, behavior-policy identity, sampling seeds/RNG states, requested actions and simultaneous pre-response snapshots. Auditor-only physical-wall fingerprints verify common ordinal shuffles within each evaluation pair. Full logs never enter the policy. Track requested, declared and resolved calls separately: a kan counts as resolved only when followed by the caller's replacement draw before hand-ending wins, so a robbed kan is not falsely counted. Track open/post-call states, each call family, gradient response and win probability surrendered when calls compete with wins.

## Fixed analysis

For each valid evaluation block, average the four paired seat utility differences to obtain D in [-3,3]. Report the mean of all32 blocks and fixed two-sided95% Hoeffding interval, half-width `6*sqrt(log(40)/(2*32))`, clipped to [-3,3]. No optional stopping or extension is allowed. Incomplete formal results are labeled incomplete and cannot produce this endpoint comparison. The wide interval is intentional: this small engineering pilot cannot establish a small competitive gain.

Success first requires real executed calls, gradients reaching the appropriate new heads, finite updates and unchanged base weights. It does not certify the complete rules engine, a calibrated critic, an adequate opponent league or playing strength. The base was not trained broadly on open hands; that remains a likely limitation.

## Parallel book preparation

Preserve the twelve prior cases and all historical source packets. Inspect a narrow set of Uzaku front-matter pages and prior magnified glyphs to resolve or document red-five uncertainty. Store new evidence separately, keeping answers/explanations outside observation exports and every partial case out of the full-policy trainer. These existing discard/riichi problems do not provide call-policy supervision.
