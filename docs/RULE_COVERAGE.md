# Rule coverage — Stage 1 tested subset

The Stage 1 fixture suite has no unexplained differences in its tested subset. This is **not complete rules certification**. The independent MahjongRepository calculator checks scoring and readiness; a second engine has not yet replayed the same transition trajectories.

Run the suite from the repository root with `PYTHONPATH=src .venv/bin/python -m pytest tests/test_rules.py -q`. Dependency pins are RiichiEnv `0.4.10` and `mahjong==2.0.0`; use the project environment and [lock file](../requirements-macos-arm64.lock). See [STATUS.md](STATUS.md) for the combined-stage result.

## Contract and fixture provenance

The [resolved profile snapshot](../tests/fixtures/stage1/profile.json) records the whole-match contract, all nine exposed `GameRule` flags, detailed preset, mode name and numeric mode, schema/profile version, and engine version. Its SHA-256 is:

`c3f043568af2dcefc1be8de65fd241311e53347e56401f3ac09f5edcde3dbe98`

The hash covers the sorted compact JSON returned by `resolved_rule_profile()`. RNG/seed provenance belongs to the run manifest, separate from semantic rules. A listed contract field is not automatically a tested behavior.

[cases.json](../tests/fixtures/stage1/cases.json) contains locally authored hands, intended outcomes, and primary references. [walls.json](../tests/fixtures/stage1/walls.json) saves complete 136-tile permutations for the initial dealer-zero fixtures. `wall_for_hands()` reproduces these arrays and can build coherent seat-rotated variants. Tests never assemble a game by independently overwriting hands, discards, or furiten flags: they start a complete wall plus valid round/score metadata, then take legal actions. The fixtures are synthetic and contain no acquired human game data.

The rule source is the [Tenhou four-player manual](https://tenhou.net/man/#RULE). Expected scoring/readiness values are also compared with [MahjongRepository/mahjong](https://github.com/MahjongRepository/mahjong), an independently implemented calculator. Its options explicitly match open tanyao, red fives, single yakuman variants, and no kiriage where applicable. Installed RiichiEnv code/stubs establish API behavior, not independent rule truth.

## Exercised behavior

| Area | Concrete coverage and evidence |
|---|---|
| Numeric payment calculation | Independent comparison across 13 han/fu combinations, dealer/nondealer, ron/tsumo, and zero/two honba: 104 arithmetic cases covering rounding and limit boundaries. Some contexts, such as 2 han/25 fu tsumo or 3 han/20 fu ron, are not realizable winning hands; actual-hand evaluation is tested separately. |
| Hand evaluation | Five closed-hand cases, two open-hand cases, and a no-yaku case. Checks include chiitoitsu, sequence hands, red-five increments, open tanyao/yakuhai, and the requirement for a yaku despite dora. |
| Simultaneous claims | Genuine same-discard chi/pon/ron choices. Ron preempts calls; pon preempts chi. Double ron has exact separately checked payouts, with honba/deposits assigned to the nearest winner. Triple ron aborts. |
| Furiten | Own-discard furiten blocks ron but permits tsumo; temporary furiten clears after the player's next turn; declining ron after accepted riichi remains furiten. |
| Riichi | 900 versus 1,000-point eligibility; declaration precedes acceptance; a winning declaration discard incurs no deposit. |
| Kan | Ankan dora is immediate; daiminkan/kakan dora is delayed. Post-riichi ankan is checked for one unchanged-wait and one changed-wait case. |
| Call restrictions | Chi kuikae excludes both same-tile discard and the tested sequence swap. Physical red/non-red copies are preserved. |
| Continuations and draw payment | Dealer win and nine-terminals/triple-ron aborts continue. Independently established tenpai/noten hands exhaust naturally; payments and dealer advance/repeat are checked. |
| Match end | South-four extension; dealer win/tenpai stopping; below-target continuation; West sudden death and West-four cap; negative versus exactly zero points; final outstanding sticks awarded to top. |
| Ties | Rank calculation for equal scores follows initial seat order. This is not a comprehensive terminal-tie/deposit interaction test. |
| Allocation | Every step of targeted fixtures and three complete random-policy matches passes the physical-tile audit; score plus outstanding deposits is preserved in targeted trajectories. |

## Physical-tile audit contract

`audit_physical_tiles(env)` reads auditor-only state and raises `ValueError` on inconsistent allocation. It counts the remaining live/dead wall, concealed hands, all melds, and uncalled physical discards. RiichiEnv retains a called tile in its donor's river history: the audit subtracts exactly one historical alias from the documented donor for each open meld. It does not simply deduplicate all tiles, which could hide corruption. Dora indicators are references into the wall, not extra physical tiles. The function scans current bounded state and does not repeatedly serialize the growing game log.

The test harness checks legal actions before each selected move, obtains all simultaneous responses from one response state, and checks conservation after every step. This verifies complete allocations along these trajectories; it does not prove all possible transition or scoring paths correct. The allocation audit does not independently validate dora-reference consistency or the meld's `called_tile` metadata; it infers the donor alias from physical tiles and the source river. Those metadata checks remain separate follow-up coverage.

## Open coverage and acceptance limits

These cases remain explicit follow-up work, with no skipped/expected-failure placeholders hiding discrepancies:

- Independent full-transition replay against an additional engine, using identical physical walls/actions and aligned rules.
- Chankan, kokushi/ankan restrictions, pao, combined yakuman, four-kans/four-riichi/four-winds aborts, nagashi mangan, haitei/houtei, kan-chain timing, riichi final-draw eligibility, and exhaustive special-hand variants.
- Broader multi-wait/post-riichi kan families, exact temporary-furiten timing before versus after the player's discard, called-discard furiten, and declined-win interactions with simultaneous calls.
- Terminal ties with outstanding deposits, dealer-containing double ron, extension dealer-priority cases, and arbitrary initial-dealer seat identity across an entire match.
- Large property-generated coverage and minimal automatic shrinking. The current complete-wall fixtures and three random games complement targeted tests; they do not establish a rare-bug rate.

No upstream behavior was silently patched. Raw-observation masking and fresh-instance reset requirements remain documented adapter safeguards from Stage 0. Any future discrepancy must retain its explicit wall/action trace, identify whether the rule profile differs, and be resolved or visibly exclude the affected path before the corresponding gate is called passed.
