# Stage 3A complete match learning pilot

Authorized on 18 October 2026. This is the accepted implementation of the 600-second proposal in the preceding continuation review. The purpose is to establish a correct, resumable full-match reinforcement-learning path. Four updates cannot establish improved playing strength.

## Fixed experiment

The starting checkpoint is the original Stage 2B reference, SHA-256 `5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f`. No smoothed continuation checkpoint is promoted. One learner seat faces three copies of that frozen reference using their existing deterministic argmax action rule. The learner uses a separately identified categorical sampling wrapper. All actors consume audited player-visible features only.

There are four sequential batches, each containing exactly 16 complete matches. Within each batch, seats rotate 0, 1, 2, 3 four times. All 64 wall seeds and all 64 independent sampling seeds are fixed in [the configuration](../configs/stage3a.json) before collection. The seed namespace is SHA-256 of `riichi-ai-stage3a-20261018:{wall|sampling}:{index}`, first eight bytes interpreted as an unsigned big-endian integer. No seed is selected or replaced using results. The learner is frozen for the whole batch. A failed, truncated, fallback or otherwise invalid planned match prevents its batch from updating and stops the formal chain. Completed earlier batches remain evidence.

The run uses CPU float32 model execution, one Torch thread and a CPU float64 softmax output transform. No temperature, epsilon mixture or entropy bonus is introduced. The exact categorical probabilities, random draws, semantic action keys, physical action tokens, actor encoding hashes and behavior-policy versions are saved. Forced actions retain probability 1 and log-probability 0. Frozen-opponent requests are retained for audit and excluded from the learning loss. Simultaneous requests are all built from one pre-response state; an ordinary losing response under engine priority remains a sampled request, not a policy override.

The existing one-worker benchmark completed 62 matches in 35.176 seconds (1.763 matches/second) against neural actors. It supports trying batches of 16 on CPU, but does not promise learning throughput: trajectory persistence, gradient replay and checkpoint verification add work. No benchmark is rerun to choose favorable settings.

## Reward and optimizer

The undiscounted utility is `[1.5, 0.5, -0.5, -1.5]` for final first through fourth place. Final scores and initial-seat tie priority independently reconstruct ranks and are checked against the engine ranks and its terminal event. The learner receives that terminal utility even when opponents finish after its last action. There is no hand-level reward, rank-head critic, learned baseline or reward shaping.

For a complete batch of N matches, the minimized objective is

`L = -(1/N) * sum_match(final_learner_utility * sum_eligible_action(log pi(action | player_view)))`.

The denominator is matches, not actions; longer matches therefore retain their full score-function sum. Every planned match must validate before gradient computation. All stored behavior distributions must replay within the prespecified absolute tolerance `1e-6` and relative tolerance `1e-5`, and selected-action log-probability absolute tolerance `1e-4`. Optional decisions are processed one at a time with gradient accumulation and the same forward shape as collection, without changing the match denominator. This is one score-function update at the collection policy, not multi-epoch replay.

A fresh AdamW optimizer uses learning rate `1e-5`, weight decay 0, betas `(0.9, 0.999)`, epsilon `1e-8`, and global gradient norm clipping at 1.0. The rank-head parameters are frozen. Exactly one optimizer step is allowed per complete batch; that batch is then permanently retired. Initial and completed-update boundaries save model, optimizer, policy identity and random-state information, and undergo immediate reload verification. Reports distinguish executed optimizer steps from successfully committed boundaries if finalization fails.

## Resource accounting

The cumulative numerical ceiling is **600 seconds**, including imports, numerical fixtures, model loading, failed checks, collection, updates, diagnostics, checkpoint writing/reloading and shutdown. A durable file-locked ledger reserves every numerical subprocess before it starts. Source editing, static review, byte/hash inspection and manual PDF transcription are outside this numerical budget. An unreconciled active lease remains fully charged.

Each formal batch has at most 100 seconds, preparation at most 30 seconds, and each numerical check at most 90 seconds, always subject to the remaining cumulative allowance and a 45-second reserve. The worker stops cooperative computation 20 seconds before its allowance ends; the parent begins forced shutdown 10 seconds before the allowance ends. No silent retries follow a failed formal batch. Resource observations, including failures, are retained.

Sampled process-tree resident memory is capped at **32 GiB** and new training/report artifacts at **1 GB**. The memory measurement is sampled and does not prove an instantaneous peak bound. Model artifacts are outside iCloud in `~/Library/Application Support/RiichiAI/stage3a`; compact review evidence is under `reports/stage3a`. No new external data, cloud compute, held-out action-test loading or separate strength campaign is included.

## Acceptance and review

Engineering acceptance requires the mandatory fixture suite, at least one complete planned batch followed by its executed finite update, correct terminal utilities and learner-only provenance, an auditable boundary checkpoint and a closed budget ledger below its caps. Log pre-clipping gradient norm, loss reconstruction, parameter change, distribution change and optional-action entropy. A finite numerical no-op is reported as a no-op, not learning success. Later batches are attempted only while the formal chain and resources remain valid.

The independent evidence audit checks artifacts, complete match ordering, scores/ranks/utilities, sampler draws, actor encodings, policy versions, requested action links, optimizer-step counts, retired batches, source identities and resource records. Its integrity result is distinct from engineering acceptance and from playing strength. Training placements are descriptive, and cannot support promotion over the frozen reference.

The book track prepares 12 additional source-attributed question/explanation packets with visual evidence. Partial teaching diagrams remain separate from complete player views. Red-five identity, source counting conventions and missing match context remain explicit. If numerical budget remains after the main pilot, a separately leased arithmetic check may compare all 47 printed counts in the ten new Uzaku examples with both existing calculators under the source and shown-visible-copy conventions. All comparisons, including mismatches, are retained. These examples are development material; no book training or holdout claim is included in this pilot.

After review, the next candidate stage is a separately bounded extension that addresses the observed entropy, variance and coverage. PPO, a calibrated baseline, privileged critics, multiple eligible learner seats and league expansion remain distinct future experiments.
