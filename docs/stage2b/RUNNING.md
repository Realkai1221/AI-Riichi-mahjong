# Running and reviewing Stage 2B

Status: Stage 2B is complete; measured results are in [the report](../../reports/stage2b/REPORT.md). Its original retention result was inconclusive under the then-approved budget. The separately authorized [Package A](../PACKAGE_A_PLAN.md) is also complete; its [results](../../reports/package_a/REPORT.md) and [review guide](../PACKAGE_A_REVIEW_GUIDE.md) are the current follow-up evidence. Preserve the original Stage 2B runs and use its snapshot verifier for historical integrity.

The user authorized the bounded Stage 2B experiment on 17 October 2026. The executable protocol is `configs/stage2b.json`; the original revised proposal remains preserved. This stage is imitation learning with an exact public-information feature branch, followed by inference-only matches. It does not start reinforcement learning or establish world-class strength.

## Before a formal campaign

1. Preserve the historical archive: run `scripts/verify_stage2a_snapshot.py`. That verifier checks Stage 2A project bytes at commit `e19a6cb` and external artifacts at their recorded paths. The old manifest should not be rewritten to match new source files.
2. Run the integration suite and record Python, Torch, engine, rule-profile and source identities. Commit the implementation and configuration before collecting formal data. Do not edit source code during a formal operation.
3. Keep bulk data and checkpoints outside iCloud at the configured local Application Support path. Reports and review documents stay in the project. Verify available disk space and power; use a normal, ventilated working position. The benchmark measures current conditions, not a claim of thermal instrumentation.
4. Freeze disjoint match assignments and teacher/behavior identities before collection. A teacher target is a label queried at the focal player's actual state; it is not necessarily the action that caused the subsequent trajectory.
5. Charge feature preparation, alignment checks and all numerical learning/checkpoint work to the persistent training ledger. Collection and inference campaigns have separately reported costs. Failures remain counted.

## Reading the comparisons

The required recipes are exact features with rank weight 0.1, raw inputs with rank weight 0.1, and exact features without rank loss. Each has two declared initialization seeds. Common initial tensors and sample streams are explicitly paired. The raw control is a fresh run under the same protocol; historical Stage 2A numbers are background, not a matched control.

Checkpoints use fixed stratified validation decisions. Recipe ranking uses full validation and averages both seeds and both behavior populations. The final checkpoint uses the predeclared primary seed. It is copied and hashed before test predictions or development match results. A better result from the other seed does not replace it after the fact.

Only one conditional recipe is allowed. Its trigger, architecture or data mixture must be recorded before that recipe runs. If two comparable runs cannot fit the remaining budget, record the trigger and defer the experiment. A triggered branch is a diagnostic allocation, not evidence that its hypothesized cause is correct.

## Match evaluation

Run `scripts/stage2b_runtime.py evaluate` after selection. The top two complete recipes each contribute both seeds: four campaigns, each with thirty independent paired blocks and eight matches per block. Each candidate is compared against the same fixed heuristic opponent panel on the same ordinal shuffled walls, with all four focal seats rotated. Four recipes/seeds sharing blocks are correlated comparisons; there are not 120 independent blocks of evidence for the final model.

The final primary model's pilot alone determines whether a new confirmation campaign is justified. A pilot utility difference at or below −0.10 stops for futility. If the required sample exceeds 256 blocks, confirmation is inconclusive within budget. A confirmation run uses fresh blocks and a fixed saved sample count. Its normal approximation concerns this local opponent panel; it does not certify human rank or dominance over other bots.

## Sustained CPU inference benchmark

Run `scripts/stage2b_runtime.py benchmark` once. All four seats use the frozen final model. Each worker uses one Torch CPU thread; the worker counts are 1, 4, 8 and 14, stopping escalation when the preceding increase fails to improve observed throughput or resources fail. Short trials include process startup and drain, so their ordering and warm caches are limitations.

The best measured worker count receives a 1,200-second sustained submission window. Setup, the worker sweep and match drain share an aggregate 1,800-second deadline. A hard deadline or resource failure is reported as an incomplete run. The report includes actual elapsed time, completed decisions, completed matches, failures, minute windows, summed process memory and separated inference costs. Summed process RSS counts shared pages conservatively; it is not unique physical memory. Model-phase timing includes output checks and action selection.

The benchmark retains compact per-match seeds, outcomes, hashes and timing/counter evidence. Failure records retain the trace and auditor log needed for investigation. None of these benchmark matches becomes training data.

## Longer runs after this stage

Longer or overnight work is a separate authorization. A proposed later run should name its objective, model/data hashes, maximum wall time, memory/disk limits, checkpoint cadence, interruption/resume procedure and comparison gate. On macOS, a bounded foreground command can be wrapped with `caffeinate -i` to inhibit idle sleep only while that command lives; do not disable sleep globally or assume this prevents lid-close sleep. Keep the laptop on power with its lid open. The same wrapper can protect this authorized short campaign without starting a persistent service.

Save progress and optimizer/sample-stream checkpoints at the declared cadence (1,000 updates for this stage), while logging elapsed time, memory, disk usage and throughput. Use actual macOS thermal-pressure or temperature telemetry only if a supported read-only source is available; otherwise report that temperature was not measured. A falling throughput window is evidence of a slowdown, not by itself a temperature measurement or proof of thermal throttling. Do not claim the short run establishes overnight reliability.

Before a longer campaign, demonstrate crash/resume on a small fixture, then verify model, optimizer and sampled-index continuation against an uninterrupted reference. Resume must reproduce the configuration, optimizer and sample stream; source or data changes require a new experiment identity. Use bounded chunks with saved progress, and do not silently extend a completed budget.

Calls, longer histories, reinforcement learning, external datasets and paid compute remain separate proposed stages. A calls experiment first needs a locked contract for guaranteed yaku, legal mandatory discards, kuikae, relative seat wind and whether a submitted call was preempted. Exposure to legal calls is not evidence of training on post-call states.
