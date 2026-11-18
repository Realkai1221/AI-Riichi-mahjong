# Stage 1 heuristic reference policies

These policies provide transparent opponents and a possible synthetic demonstration source. They are not trained agents or evidence of competitive strength. All decisions consume the immutable `PlayerView`; no policy receives an engine, raw observation, opponent concealed hand, future wall, or seed describing hidden game state.

The implementation is [policies.py](../src/riichi_ai/policies.py). Its public interface is `make_policy(name, seed).choose(view)`. `manifest()` and `policy_manifest(name, seed)` record the configuration under version `heuristics-v1`. Changing behavior requires a new version and fresh evaluation. The shanten function comes from the pinned RiichiEnv `0.4.10` dependency; it is not an independent rules check.

| Name | Default decision rule |
|---|---|
| `random` | Uniformly sample canonically sorted legal actions using a private seeded Python RNG. It can pass ron/tsumo when another action is legal; this deliberately weak reference has no automatic win preference. |
| `shanten` (`shanten_ukeire`) | Take a legal win, declare legal riichi, then select a discard minimizing shanten and maximizing visible-availability ukeire. Retain red fives on exact shape/ukeire ties, then use the stable action token. Pass optional calls; do not model kan. |
| `defense` (`conservative`) | Take a legal win. Against an opponent's riichi, avoid declaring a new riichi and first minimize the number of riichi opponents for whom the discarded type is not known genbutsu; use the same offensive rule for ties. Otherwise behave like `shanten`. |

Heuristic tie-breaking is deterministic; their supplied seeds are recorded for a uniform factory interface but do not alter those choices. Random-policy reproducibility additionally depends on the recorded Python RNG implementation/version and identical decision ordering. A forced single action is returned directly. If an unsupported action is the only available kind and passing is impossible, the policy returns the first stable legal candidate; this fallback is not strategic evaluation of that action.

## Offensive calculation

For each discard type, calculate the remaining concealed hand's shanten. Standard hands, seven pairs, and thirteen orphans are handled by the engine function. The pinned API accepts physical tile IDs and infers the exposed-meld count from concealed-hand length; it has no `num_melds` argument. The default heuristics preserve a closed hand by passing calls and declining optional kan, but the helper also has fixtures for shorter concealed hands.

For every tile type, count physical tiles visible in the player's own hand, public rivers, public melds, and revealed dora indicators. Deduplicate physical IDs: a called discard can appear in both the river and the meld. A just-discarded tile remains visible. Ukeire is the sum of `4 - visible_count` over types whose addition lowers shanten. This is a count of possible unseen improving copies, not their actual wall availability, draw probability, winning probability, or expected score. Opponent hands, the dead wall, furiten, future calls, and future policies affect those other quantities.

Shanten results use a bounded 131,072-entry LRU keyed by 34-type counts. Red identity does not affect shanten and is retained separately for the discard tie-break. `shanten_cache_info()` exposes cache accounting for profiling. Cache size and warm/cold state should be recorded when comparing throughput.

The baseline does not optimize dora value, wait quality beyond visible counts, dama versus riichi, placement, score thresholds, call value, kan value, or dynamic push/fold expected value. `allow_riichi=False` is an explicit constructor option, not the default factory configuration. Riichi declaration and the following discard are separate legal decisions in the pinned engine.

## Defensive calculation

For each riichi opponent, genbutsu includes types in that opponent's own current-hand river and public discards after that opponent's `reach_accepted` event. Their own river invokes discard furiten; passing a winning tile after riichi creates persistent riichi furiten. The adapter's history resets at every new hand, including dealer continuations, so prior-hand discards do not become safety evidence.

This only establishes that the specified opponent cannot ron that tile under the current rule conditions. It does not establish safety against another player, stop a tsumo, or guarantee a favorable match outcome. The implementation makes no suji/kabe probability or universal safety claim. With several riichi opponents it counts opponents lacking genbutsu evidence equally; with no available genbutsu against any threat, it falls back to offensive ordering. It can fold excessively, ignore an expensive silent threat, or abandon a valuable hand. Those limitations make it a useful distinct reference, not an optimized defense agent.

## Verification and use

The focused tests cover standard/special-hand shanten, visible-copy ukeire, called-discard deduplication, red-five ties, separate riichi declaration, wins before defense, pass-calls behavior, per-opponent and post-riichi genbutsu, deterministic legal choices, and seeded random behavior. Native adapter smoke games check that each policy completes a full match using legal round-tripping actions. These checks validate the implemented reference behavior; broader simulator conformance and independent strength evaluation are separate Stage 1 gates.

Freeze manifests and seeds in evaluation. Compare candidates in the same opponent/seat configurations and retain complete-match placement utility as the outcome. Four identical copies averaging rank 2.5 is a symmetry property, not evidence of improvement. Synthetic teaching data should only be generated after the relevant fixture/observation gates pass, with this policy version and its limitations attached to the data manifest.
