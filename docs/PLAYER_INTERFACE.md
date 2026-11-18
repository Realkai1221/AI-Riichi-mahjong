# Stage 1 player boundary

This interface targets **RiichiEnv 0.4.10**, four-player red-five hanchan, under the project's pinned rule profile. It provides immutable player inputs and exact legal-action candidates. It is not a learned encoder, a hidden-state sampler, a sandbox for untrusted policy code, or a certificate of complete rules/information correctness.

## Runner and policy responsibilities

The runner owns `RiichiEnv`, native `Observation` objects, and four `PlayerAdapter` instances. A policy receives only a `PlayerView` and returns one of its `ActionSpec` values. The runner calls `adapter.resolve(spec, current_observation)` to recover an engine-provided legal native action. It must not pass the environment, raw observations, serialized engine state, full MJAI logs, or the adapters themselves into policies.

```python
adapters = [PlayerAdapter(seat) for seat in range(4)]
env, observations = create_env(seed)
while not env.done():
    views = {seat: adapters[seat].build(observation)
             for seat, observation in sorted(observations.items())}
    actions = {}
    for seat, view in views.items():
        if view.legal_actions:
            chosen = policies[seat].choose(view)  # policy API is runner-defined
            actions[seat] = adapters[seat].resolve(chosen, observations[seat])
    observations = env.step(actions)
```

Call `reset()` on every adapter at each new match, or construct new adapters. Reproducible simulator restarts use a fresh environment; the pinned engine's `reset(seed=...)` retains its internal hand-shuffle counter.

## Immutable player inputs

`PlayerView` is a frozen dataclass with slots. All nested collections are tuples; event payloads recursively freeze lists and mappings. Seats are absolute `0..3`.

| Field | Meaning |
|---|---|
| `seat`, `hand` | Observer seat and its sorted concealed physical tile IDs; open/concealed-kan tiles live in meld descriptors. |
| `legal_actions` | Complete tuple of distinct exact physical candidates supplied for this seat. |
| `discards`, `melds` | Four immutable per-seat collections of public discards and meld descriptors. |
| `scores`, `riichi_declared` | Four public point totals and riichi flags. |
| `drawn_tile` | Private current draw only when its unique physical ID belongs to this observer's hand; otherwise `None`. |
| `dora_indicators` | Public physical indicator IDs. |
| `round_wind`, `oya`, `honba`, `riichi_sticks` | Public match/hand metadata. |
| `hand_id` | One-based monotonically increasing observed hand count within this match. Every `start_kyoku`, including dealer continuation and abortive redeal, increments it. It is not `kyoku_index` or dealer seat. |
| `history` | Sanitized current-hand events in order, starting with `start_kyoku`. It clears at the next hand; earlier immutable views remain unchanged. |

`MeldView` contains `kind`, physical `tiles`, `opened`, `from_player`, and `called_tile`. `is_closed_hand(view)` checks `opened` on the observer's melds; a concealed kan preserves a closed hand.

Physical tile IDs are `0..135`; `tile_type(id)` returns `id // 4` in `0..33`. The red-five IDs are `16`, `52`, and `88`. These IDs follow this pinned engine and are not a portable cross-engine action vocabulary.

`visible_tile_counts(view)` returns 34 counts for the union of the observer's concealed hand, public rivers, melds, and dora indicators. It deduplicates physical IDs: RiichiEnv keeps a called discard in both the donor's river and the recipient's meld. It does not include opponents' concealed hands, hidden wall tiles, or ura indicators.

## Action contract

`ActionSpec(kind, tile, consume_tiles, token)` is frozen. `kind` uses the engine enum's uppercase spelling: `DISCARD`, `CHI`, `PON`, `DAIMINKAN`, `ANKAN`, `KAKAN`, `RIICHI`, `RON`, `TSUMO`, `KYUSHU_KYUHAI`, and `PASS` in the current four-player game. `tile` can be `None`, including the separate riichi-declaration action. A following discard is another decision.

`consume_tiles` preserves exact physical IDs in sorted order. The stable token is canonical JSON of `[kind, tile, consume_tiles]`. Native duplicate candidates with the same physical semantics collapse to one token; distinct physical candidates do not. The adapter does not use `encode()`, a fixed 46-output assumption, or an integer action mask to reconstruct moves. Those encodings can alias physical tile copies and other choices.

`resolve()` validates the seat and matches the whole specification against the provided observation's current legal candidates. A forged token/field combination or a choice absent from that set raises `ValueError`. It returns an existing native candidate and never manufactures or silently substitutes a move. The runner is responsible for supplying the current observation; stable tokens are not temporal authorization tokens.

## Event handling and information boundary

In the installed release, obtaining an observation through `reset`, `step`, or `get_observation(s)` advances that seat's event cursor. Calling `new_events()` repeatedly on the resulting object returns its same batch. An additional environment read may return no events. Therefore **route every returned observation for a seat to its one adapter, in order**. Do not discard a step-returned observation and refetch it. Initial construction without a `start_kyoku` fails rather than inventing history.

Immediate repeated builds of an identical observation or serialized copy are idempotent. Reusing a previously consumed older snapshot raises an error. The adapter cannot reconstruct a missing event batch or independently certify that an external caller has delivered a complete stream.

`EventView` exposes `kind` (MJAI lowercase), `actor`, `target`, `tile`, `tile_type`, `consumed`, `tsumogiri`, and immutable `data`. Event tiles are MJAI strings, not physical IDs. `mjai_tile_type()` handles suited tiles, red-five spellings such as `5mr`, honors, and unknown draws. This lets a policy interpret public riichi timing and discards without receiving raw JSON.

The per-event allowlist:

- Keeps the observer's initial hand, public round/deposit/score metadata, public discards/calls/dora, riichi declaration/acceptance, and public result deltas/scores.
- Hides every opponent's initial hand and private draw even if an input event unexpectedly contains an unmasked value.
- Omits wall layouts, salts, seeds, unknown extra fields, detailed hand-reveal payloads, and other unreviewed scoring/result fields. Unknown event types fail explicitly.

The raw engine also exposes the current player's `drawn_tile` on inactive players' observations. The adapter filters it by unique physical membership in the observer's own hand. Neither raw `hands` arrays nor raw observation caches/history are exposed to policies.

## Current checks and limits

Tests cover immutable snapshots, action round trips from genuine deal/call fixtures for all current four-player action kinds, red/normal-five and physical-copy distinctions, chi alternatives, a pon-to-kakan continuation, separate riichi declaration/discard, response snapshots, called-discard counting, event sanitization, missing-history rejection, repeated builds, adapter reset, and hand IDs across an abortive dealer continuation.

Privacy fixtures construct complete alternate 136-tile walls. They vary concealed opponent allocations and unseen live-wall order while keeping the observer's hand/draw and public action prefix fixed. Another family varies an opponent's private drawn tile while preserving current tile allocations and a subsequent common public hand-discard. The observer's immutable state and history agree in the tested worlds.

These are limited fixture families, not a general information-flow proof. Later work must expand adversarial and rule-boundary coverage, particularly riichi/kan timing, simultaneous responses, result disclosures, stale/missing event streams, and every deployment adapter. The implementation uses the pinned engine's candidate legality; it does not independently certify that legality.
