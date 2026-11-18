# Full-match evaluation contract

The executable is `python -m riichi_ai evaluate`; implementation is [evaluation.py](../src/riichi_ai/evaluation.py). This is a local development reference. Stage 1 used heuristic policies. Stage 2 adds separately versioned semantic teachers, checkpoint policies, and small CPU process pools. No public ladder or sequential promotion decision is included.

## One full match

`run_match(seed, policy_names, policy_seeds=..., max_steps=..., failure_dir=...)` constructs a fresh pinned environment, four independent adapters, and four policy instances. It returns a JSON-compatible record. The runner alone owns full simulator state.

For each simultaneous decision batch, it first builds all immutable player views from the observations returned by the previous step. It obtains every policy choice, resolves each exact legal candidate against that same original snapshot, and submits the complete action dictionary once. It never refetches an observation after its event cursor has advanced. Submitted choices and resulting public events are distinct: a pon request can be preempted by a ron.

After every step, physical-tile allocation and scores plus riichi deposits must balance. The execution limit counts simulator step calls, rather than emitted events. Only genuine match termination produces rank utilities. A failure or cap exit has `utilities=null` and cannot count as a completed game.

The record includes per-seat policy/version/configuration/RNG provenance, rule hash, seed, steps, decisions, hands, terminal scores/ranks/utilities, conservation outcome, action/view/log hashes, ordinal-wall fingerprints, stage timings, and per-seat latency summaries. Deterministic hashes exclude timings. Full logs and wall fingerprints are auditor material and do not enter the player view.

Failures and truncations save a separate record containing the submitted exact action tokens, last policy requests, seed/configuration, and full auditor log. This supports reproducing the run and locating the first failure; automatic trace minimization is not implemented. A future deployment adapter will need its own failure/reconnection protocol.

## Pairing and statistical unit

The candidate replaces one focal seat against three named opponents. The baseline then occupies that same seat against the same opponent lineup. Opponents are ordered clockwise from the focal seat. Repeat this for all four focal seats: **eight full matches form one block**.

Root seed plus block index derives a 64-bit environment seed using SHA-256. Separate labeled hashes derive policy RNG seeds; paired arms receive the same per-seat seeds. Every match uses a fresh engine with the same initialization sequence. The pinned engine derives each hand's shuffle from its episode seed and ordinal hand counter. The auditor additionally hashes each dealt hand's remaining physical wall, first draw, and relative-dealer allocations; all common ordinal-hand fingerprints must agree across paired arms.

This couples chance without requiring matching game states. Dealer, round, score, honba, actions, and match length may diverge after different decisions. Extra hands in a longer match are valid. Equal numeric seeds are not a cross-engine wall specification. The saved complete-wall fixtures are the appropriate basis for future independent-engine checks.

For each focal seat, subtract baseline terminal utility from candidate utility. Average those four differences within each block. Only the block means enter uncertainty calculations; seats, decisions, hands, and the eight games are not independent observations.

Terminal utility is `2.5 - rank`, yielding `[1.5, 0.5, -0.5, -1.5]`. Mean utility difference therefore equals mean rank improvement. Each block difference lies in `[-3, 3]`.

The report provides sample standard error, an exploratory 5,000-resample percentile interval over blocks, and a conservative fixed-budget 95% Hoeffding interval using range width six. The latter assumes independent sampled seed blocks and a fixed candidate/panel/budget. It is not an anytime-valid interval. Neither is a replacement for predeclared sequential promotion testing, correction for repeated candidate selection, or fresh confirmation. A one-block run omits the standard error/bootstrap interval.

Any failed/truncated game, wall mismatch, or changed source hash invalidates aggregate inference for the run. Raw records remain available. Completed-arm averages in an invalid run are descriptive only; the report's `statistics` field is null. No incomplete block is silently dropped.

## Reproducibility and timing

Reports capture Python/dependency versions, all package source hashes, resolved rules, and individual policy manifests. Sources are checked again after the run. Seed reproducibility is scoped to this recorded stack. Tests verify exact repeated match digests, identical-policy zero differences, arm-swap antisymmetry, rejection of wall mismatches, and incomplete-run handling.

Measured wall-clock time includes history/view hashing, audits, policy execution, and simulator work. Stage 2's v2 report separates incremental safe-view hashing, wall commitments, setup, finalization, and residual loop overhead. The actual full safe history and hand ID remain committed; cached hashes are tested against complete recomputation, including history mutation and truncation. Summed worker times are work totals and can exceed parallel campaign wall time.

Candidate runs precede baseline runs, and the bounded shanten cache is shared across matches within a worker. Latencies therefore depend on cache state and execution order; they are useful engineering diagnostics, not unbiased policy-speed comparisons. A later performance study should warm both arms consistently, counterbalance order, and capture sustained thermal behavior.

`--workers` uses spawned independent CPU processes with deterministic seeds and output ordering. Unit checks compare one- and two-worker trajectory digests and block statistics exactly. Neural checkpoints run on CPU in evaluation workers; training uses one CPU/MPS learner. The checkpoint cache keys on file content, not just timestamps. Each match receives a fresh policy wrapper with separate counters around immutable evaluation weights. Source and frozen policy provenance are checked before and after each campaign and against recorded games. Start a fresh process after edits; hashing current source files cannot identify stale code already imported into a long-lived process.

## Stage 2A retention experiment

The prespecified protocol is in [STAGE2_PLAN.md](STAGE2_PLAN.md). A separate 30-block development pilot estimates variability and screens clearly weak candidates. A frozen confirmation count is chosen using a normal approximation, 0.10 non-inferiority margin, one-sided alpha 0.05, assumed true gap zero, and 80% target power. Planning SD is the larger of 0.546017 (Stage 1) and 1.25 times the pilot SD. Confirmation uses fresh seeds and 185–256 independent blocks; insufficient budget/variance evidence is reported as inconclusive.

The confirmation lower bound is `mean_delta - z(0.95) * sample_sd / sqrt(block_count)`. Passing requires that bound to exceed `-0.10`. This is an approximate fixed-sample test, not a finite-sample distribution-free guarantee. It does not show superiority, expert skill, or generalization to other rules/opponents. Failing to establish non-inferiority does not itself prove inferiority. The existing Hoeffding interval remains a separate conservative diagnostic. Validation chooses the checkpoint; test actions and confirmation outcomes never tune it.

The Stage 1 seeds are development material. Keep them out of future validation/test datasets. Changing a heuristic's behavior requires a version change and fresh evaluation; do not overwrite historical reports with a new implementation under the same label.
