# Stage 1 review packet

Completed 16 October 2026 on the M5 Pro MacBook Pro with 64 GB memory.

**Stage 1 is complete for its documented tested subset.** We now have a reproducible game runner, targeted rule checks, a player information boundary, three reference policies, and a paired full-match evaluator. The final suite passed **90 tests and 165 subtests**. Both evaluation campaigns were valid: **768 full matches, 8,070 hands, and 625,266 decisions**, with no failures or truncations. All stepped states passed physical-tile and point/deposit conservation; all common paired ordinal walls matched.

No Mahjong neural policy has been trained. This is a tested research foundation, not a claim of complete rules certification or competitive/world-leading strength. Stage 2 has not started.

## What to review first

1. Read the results and remaining limits below.
2. Review [the proposed Stage 2 instructions](../../docs/STAGE2_REVIEW_DRAFT.md), especially the suggested separation between pipeline readiness and playing-strength promotion.
3. Change any objective, time/memory/disk limit, data preference, or acceptance criterion. Those changes can become the next task specification before training.

The current defaults remain four-player red-five East–South play and average placement over complete matches. No external game corpus or pretrained weights were acquired, no operator was contacted, and no paid compute or public bot play was started.

## What was built

| Component | Current behavior | Review file |
|---|---|---|
| Rule contract | Explicit mode, all nine exposed engine rule flags, whole-match settings, version, and stable hash | [rules.py](../../src/riichi_ai/rules.py), [profile snapshot](../../tests/fixtures/stage1/profile.json) |
| Rule evidence | Independent scoring/readiness checks and coherent complete-wall transition fixtures | [coverage ledger](../../docs/RULE_COVERAGE.md), [rule tests](../../tests/test_rules.py) |
| Player boundary | Frozen own-hand/public-state/history views; exact physical legal-action candidates; strict resolver; separate seat memory | [interface specification](../../docs/PLAYER_INTERFACE.md), [player.py](../../src/riichi_ai/player.py) |
| Reference opponents | Seeded random, shanten/visible-ukeire, and conservative genbutsu defense | [heuristic specification](../../docs/HEURISTICS.md), [policies.py](../../src/riichi_ai/policies.py) |
| Match evaluator | Simultaneous choices from one state; every-step audits; seat rotation; paired chance; full outcomes, provenance, timing, and failure records | [evaluation contract](../../docs/EVALUATION.md), [evaluation.py](../../src/riichi_ai/evaluation.py) |
| Reproducible setup | Added independent `mahjong==2.0.0` scorer; updated dependency snapshot; repaired editable imports | [README](../../README.md), [dependency snapshot](../../requirements-macos-arm64.lock) |

The original proposal and historical Chinese translation remain unchanged. The English implementation documents remain the working specification.

## Test evidence and scope

The final integrated run passed **90 tests + 165 subtests in 12.08 seconds**, with no skipped or expected-failure cases. Machine-readable test evidence is saved. This includes the earlier clone/fresh-instance regressions and tested CPU/MPS numerical checks.

The rule subset includes:

- 104 independent payment-arithmetic cases across han/fu, dealer/nondealer, ron/tsumo, and honba combinations. Some arithmetic inputs do not describe a realizable winning hand; separate actual-hand tests cover five closed hands, two open hands, and a no-yaku case.
- Same-discard ron/pon/chi priority, double-ron payment/deposit allocation, triple-ron abort, and permanent/temporary/riichi furiten examples.
- Riichi eligibility and acceptance timing, canceled deposits on a winning declaration discard, ankan/daiminkan/kakan dora timing, and selected allowed/forbidden post-riichi kans.
- Exhaustive-draw payment, dealer continuation/advance, South-four ending or extension, West sudden death/cap, bankruptcy at negative rather than zero points, final deposits, and basic tie priority.
- Fifteen saved complete 136-tile walls. Transition fixtures take legal actions from coherent walls rather than independently overwriting hidden hands and status flags.

The independent [MahjongRepository calculator](https://github.com/MahjongRepository/mahjong) checks selected scoring/readiness results. Transition expectations use the [Tenhou rules](https://tenhou.net/man/#RULE); a second complete game engine has not yet replayed these trajectories. No unexplained discrepancy was found in this subset, and no upstream behavior was silently patched.

The player tests exercise every current four-player action kind, physical-copy/red-five distinctions, alternate legally consistent hidden worlds with a common public prefix, inactive-seat private draws, immutable response snapshots, and hand-memory reset through dealer continuation. These are bounded fixture families, not a universal information-flow proof.

## Paired full-match results

Each independent block contains candidate and baseline games in all four focal seats: eight games. Both arms use the same ordinal wall stream and per-seat policy RNG seeds. Candidate and baseline actions can produce different dealers, scores, hand counts, and match lengths; the evaluator does not force those states to agree. Inference uses the four-seat block mean, not individual games or decisions as independent samples.

Utility is `2.5 - final rank`, so a positive utility difference is the same amount of average-rank improvement. Both run budgets were fixed before execution.

| Comparison | Opponent panel | Independent blocks / games | Candidate average rank | Baseline average rank | Rank improvement |
|---|---|---:|---:|---:|---:|
| Shanten vs random | Three random opponents | 64 / 512 | 1.0000 | 2.5117 | +1.5117 |
| Defense vs shanten | Shanten, defense, shanten, clockwise from focal seat | 32 / 256 | 2.4609 | 2.4766 | +0.0156 |

**Shanten versus random:** shanten finished first in all 256 candidate games. The deliberately weak random policy can pass wins and make poor calls/discards; beating it is an integration/reference result, not a meaningful claim against strong Mahjong players. The exploratory block-bootstrap 95% interval for improvement is `[1.4258, 1.5977]`; the conservative fixed-budget Hoeffding interval is `[0.4931, 2.5303]`. Raw report.

**Defense versus shanten:** the result is inconclusive. The exploratory interval is `[-0.1641, 0.2031]`; the conservative interval is `[-1.4249, 1.4561]`. Defense's first-through-fourth counts were `[25, 37, 48, 18]`; shanten's were `[37, 24, 36, 31]`. Fewer fourth places came with fewer first places in this sample. Keep both as different reference styles; there is no supported reason to crown defense the stronger average-rank policy. Raw report.

These are development comparisons without a promotion threshold, optional-stopping rule, or independent final confirmation. The Stage 1 seed blocks are now development material and must not enter future held-out datasets. The bounded intervals assume independent sampled seed blocks; the shared games within each block remain correlated.

## Measured local execution

| Workload | Time | Decisions per second | Full matches per hour, short-run equivalent |
|---|---:|---:|---:|
| Shanten/random campaign | 126.73 s | 3,546 | 14,544 |
| Defense/shanten campaign | 101.55 s | 1,732 | 9,075 |

These CPU game-loop measurements include Python history hashing, observation construction, conservation checks, and heuristic work. They are not neural training or learned self-play throughput estimates, and do not establish sustained overnight thermal performance. On the stronger panel, heuristic choice computation alone took 57.20 seconds; optimization should follow measured workloads rather than assuming neural inference is the bottleneck.

The bounded shanten cache is shared across matches, and candidate arms execute first. Per-policy latency therefore depends on cache/order; these runs are not unbiased comparative speed benchmarks. Detailed timings and each match's latency summaries are retained in the raw reports. The earlier Stage 0 MPS benchmark remains a synthetic tensor measurement.

## Integration findings worth keeping

1. **Observation reads consume event cursors.** The runner must process every step-returned snapshot through the appropriate seat adapter. Refetching an observation can lose its history. The adapter rejects a missing initial hand boundary.
2. **A hand counter must survive renchan.** Every `start_kyoku` increments `hand_id`, even when dealer and round identifiers repeat. Histories reset at that boundary, not merely when the dealer changes.
3. **Raw observations are too permissive.** The adapter masks opponents' private draws, strips unknown event payloads, and supplies only immutable approved fields. Exact legal candidates preserve choices that an integer mask can alias.
4. **A repeated seed needs a fresh engine.** The pinned engine's reset retains a shuffle counter. The runner always creates a new instance and verifies common paired wall commitments.
5. **The synced environment caused intermittent imports.** Python skipped an editable-install `.pth` file carrying the macOS hidden flag. The environment now lives at `/Users/example/Library/Caches/riichi-ai/venv-py314`, with the project's `.venv` pointing to it; the flag was cleared and package metadata refreshed. The environment is reproducible from pinned dependencies if the cache is purged.

## Remaining limits

The [coverage ledger](../../docs/RULE_COVERAGE.md) is the detailed gap list. Material items include full independent-engine transition replay; chankan, pao, combined yakuman, several abortive draws, nagashi mangan, last-tile/kan-chain interactions, and broader terminal-tie/furiten/riichi-kan families. Allocation checks do not independently validate every meld or dora-reference metadata field. Automatic failure shrinking and general hidden-state reconstruction are not implemented.

The heuristics do not provide expert call/kan strategy, hand-value optimization, nuanced push/fold decisions, or placement-aware play. No neural encoder, demonstration dataset, optimizer run, league, search system, or public-play adapter exists yet. Those limits remain explicit inputs to the next stage.

## Proposed next stage and review decision

Proceed next with a small, versioned synthetic-data pipeline and a bounded imitation pilot, after your review. The [draft instructions](../../docs/STAGE2_REVIEW_DRAFT.md) propose at most 1,000 pilot matches, complete-match/shared-wall-block splits, a compact legal-candidate network, tiny-batch learning and model-specific CPU/MPS checks, and a one-hour local training cap. Those are proposed bounds, not completed work or authorization for paid training.

My main recommended plan refinement is to separate “the learning pipeline and reproducible checkpoint work” from “the policy is demonstrably stronger.” A learner can correctly imitate a weak teacher without surpassing it. The current Stage 2 strength gate has not been silently relaxed; the draft makes that choice reviewable.

## Reproduce and audit

Run from the project directory using the existing environment:

```bash
.venv/bin/python -m pytest -q --junitxml=reports/stage1/tests.xml
.venv/bin/python -m riichi_ai evaluate --candidate shanten --baseline random --blocks 64 --seed 2026101601 --output reports/reproduced-shanten.json
.venv/bin/python -m riichi_ai evaluate --candidate defense --baseline shanten --opponents shanten defense shanten --blocks 32 --seed 2026101602 --output reports/reproduced-defense.json
```

Reproduction writes new reports rather than replacing the historical comparison files. Expect timing variation; deterministic trajectories/results should match under the recorded stack. manifest.json records evidence hashes, versions, source hashes, fixture/test counts, and stage status.
