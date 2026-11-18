# Suggested next review after Stage 2B

This is a proposal for the user's review, not authorization to start another experiment. The completed Stage 2B evidence and limits remain unchanged.

## Keep the demonstrated improvement

Keep the four exact candidate-efficiency inputs and the frozen `features_rank01` primary checkpoint. Both feature-assisted seeds reach 100% full-validation teacher best-set discard agreement, versus 73.14% and 64.23% for the matched raw controls. The selected model reaches 99.9826% equal-population discard agreement on the untouched test, with two shanten errors in 10,712 optional discard decisions. This is strong imitation of the pinned heuristic's preferences on these populations. It is not expert-level Mahjong evidence.

The added feature branch has 512 parameters. It supplies exact, player-computable hand-efficiency calculations to a learned scorer; it does not expose hidden tiles, future outcomes or teacher labels. Future reports should continue calling this a feature-assisted model.

## First decision: resolve teacher retention without changing the model

The primary candidate's 30-block pilot averages 2.5083 place versus the teacher's 2.5583. Its mean utility difference is +0.05, but the observed block variability produces a required confirmation count of 319 under the predeclared planning formula. Stage 2B allowed at most 256, so no confirmation ran. This is an inconclusive strength gate, not a pass or evidence of inferiority.

The smallest follow-up would be one separately authorized **319-block, fixed-count confirmation**, equivalent to 2,552 full match executions across candidate/baseline arms and four focal seats. Keep the same frozen candidate, opponents, rules, utility, 0.10 non-inferiority margin and one-sided 0.05 alpha. Freeze the complete seed list and revised budget before outcomes; do not inspect partial results to change the count, candidate or margin. The confirmation would use a fresh, disjoint namespace, and any failure would retain replay evidence.

The four development campaigns took 105.60 seconds for 960 executions. A simple workload ratio suggests several minutes for confirmation on this machine; that is a planning estimate, not a runtime guarantee. A proposed separate ceiling of 10 minutes, 32 GiB sampled memory and 1 GB additional artifacts would be a reasonable bounded request. No training is needed for this particular study. If the time limit prevents completion, the result must remain incomplete rather than claiming the planned test passed.

The fixed-count normal approximation and synthetic-panel scope remain limitations. A successful confirmation would support retention within the 0.10 margin against this teacher and panel; it would not establish superiority, human rank or competitiveness with Mortal/NAGA.

## Before longer training: investigate the observed instability

Both zero-placement-loss runs learned accurate policies early, then suffered severe finite-value optimization collapse. Their selected checkpoints are at updates 5,500 and 6,000; their final checkpoints are poor. Preserve both versions and all learning histories.

A follow-up should first instrument activation and candidate-logit ranges, pre-clipping gradient norms, update norms and per-action losses. Confirm the measurements on CPU/MPS, and establish tested stopping/reporting behavior for runaway finite values. Then compare a small, predefined stabilization intervention against the original setup with matched seeds, initial tensors, samples and budget. Lower learning rate, logit parameterization or normalization may be candidates; the present evidence does not identify a remedy or prove a backend bug. Do not silently restart failed runs and report only their successful replacements.

## Do not promote the placement head into a critic yet

The selected recipe's placement heads are overconfident: validation NLL 1.836/1.885 and Brier 0.815/0.829 are worse than uniform prediction, despite about 41% rank accuracy. Mean top confidence is approximately 68–70%. The action-selected checkpoint is therefore not automatically a good value checkpoint.

Any rank/value experiment should compare uniform and current-standing baselines, report match-phase results and calibration, and distinguish teacher-behavior returns from the returns of the current policy. Consider freezing the policy representation while evaluating a separately trained head, but treat that as a controlled experiment. Do not use the current head for reward shaping or claim validated foresight.

## Broaden Mahjong coverage before broad strength claims

The new corpus contains no executed chi, pon or kan. Offered call candidates do not teach the states that follow calls. A broader policy needs a separately reviewed behavior mixture, legal/yaku-aware call decisions, post-call examples, kan transitions and corresponding rule fixtures. Data provenance, per-seat behavior identities, full-match splitting and actor/target/auditor separation must remain intact.

After the retention question and these correctness/stability issues are addressed, propose a small RL study with terminal match utility, one learner seat against frozen opponents, explicit trajectory semantics, fresh evaluation and a separate compute cap. Do not bundle call-policy expansion, new rank targets, architecture changes and RL into one experiment whose improvement cannot be attributed.

## Useful contents of the next instruction

Specify one scientific question, the unchanged baseline/checkpoint, the permitted local time and storage, and the acceptance evidence. For example: authorize only the fixed confirmation above, or authorize only a bounded stability diagnosis. Those are distinct studies. Neither has started as part of Stage 2B.
