# Stage 2B feature-assisted model

This is an additive implementation for a controlled imitation experiment. It
does not establish playing strength. The Stage 2A raw-input path and its saved
checkpoints remain supported; no historical artifact is rewritten.

## Versioned interfaces

| Component | Raw reference | Feature-assisted candidate |
|---|---|---|
| Feature schema | `actor-features-v1` | `actor-features-v2` |
| Model schema | `suit-history-policy-v1` | `suit-history-efficiency-policy-v2` |
| Legal actions | `semantic-actions-v1` | The same complete semantic action set |
| Current-hand history | 128 events | 128 events, identical encoding/scaling |
| Default parameters | 989,909 | 990,421 |

`features.FEATURE_SCHEMA` and `model.MODEL_SCHEMA` retain their v1 values.
`encode_view(view, actions=None)` and `PolicyNetwork()` remain v1 by default.
Use `encode_view(..., feature_schema=FEATURE_SCHEMA_V2)` and
`PolicyNetwork(efficiency_model_config(...))` to opt in explicitly.
`feature_manifest(feature_schema)` describes either encoder.

V2 adds one float32 array, `candidate_efficiency[N,4]`, alongside the unchanged
state arrays and integer `action_tokens[N,8]`. Candidate rows follow the same
complete legal semantic ordering. `collate_encoded` accepts a uniformly v1 or
v2 batch, pads candidates, and rejects mixed versions. A model rejects the
other version's batch. Saved model and feature schemas must agree; runtime
checkpoint loading selects the matching supported path.

## Exact public-information calculation

The new `efficiency.py` accepts only actor tile channels and semantic candidate
tokens. It decodes own counts as `4 * tiles[:,0]` and deduplicated visible
counts as `4 * tiles[:,4]`, requiring finite integral counts in `[0,4]` and
`own <= visible`. For each legal DISCARD type, it invokes the same pinned
`policies._discard_score` calculation used by the teacher. A discard changes
the remaining concealed hand but leaves visible counts unchanged: the tile
moves to a public river and does not become unseen.

| Column | Exact value | Float32 divisor |
|---|---|---|
| 0 | Post-discard shanten | 8 |
| 1 | Visible ukeire: unseen copies of improving types | 136 |
| 2 | Number of non-exhausted improving types | 34 |
| 3 | 1 for DISCARD, 0 otherwise | 1 |

Non-discard rows are all zeros. RIICHI declaration is separate from its
following discard; its own row is not applicable, while the following legal
discard is scored normally. Red/normal and drawn/held semantic variants of a
type share the efficiency values but retain their separate action descriptors.
The helper computes one score per distinct discard type per call.

`exact_efficiency_from_encoded(tiles, action_tokens)` returns the unscaled
int16 values for exact audits. `efficiency_from_encoded(...)` returns the
normalized float32 network inputs and is re-exported from `features.py`.
The exact audit compares integers, avoiding attempts to invert rounded
float32 division by 136 or 34. Recorded preference keys and targets may be
used as audit references, but are never arguments to these functions.

The third `DiscardScore` field is an improving-type tuple; its length is the
new scalar. Types with all four copies already visible are excluded. These
features describe hand shape and possible unseen improving copies, not
hidden-wall availability, yaku/furiten-aware winning waits, danger, or
calibrated score/rank value. Short concealed-hand calculations retain the
pinned engine's existing meld-count inference by concealed hand length.

## Small additive model branch

V2 keeps every v1 parameter name and shape. A new bias-free
`efficiency_encoder: Linear(4, candidate_width)` projects the numeric inputs
and adds its result to the existing semantic candidate embedding immediately
before the shared action scorer. This adds 512 parameters at the default
candidate width of 128. Non-applicable zero rows contribute zero through this
branch. There is no teacher-mask filter, forced strategic rule, or removal of
legal choices; only existing candidate-padding positions are masked.

The state/history context and rank head are unchanged. In particular, the
rank head does not directly consume the new candidate efficiency vectors.
Setting the new branch's weights to zero reproduces a matched raw model's
policy and rank outputs exactly in the tested CPU fixture.

`matched_initialization(seed, raw_config=None, assisted_config=None)` returns
raw and assisted CPU state dictionaries plus a JSON manifest. It initializes
the raw reference from the declared seed, initializes the assisted model from
a separately derived CPU stream, and copies all common tensors exactly.
The manifest records both configurations, seeds, common/full weight hashes,
and added tensor names. Caller CPU RNG state is restored and accelerator RNGs
are not accessed. Reuse the exact assisted state for rank-weight 0 and 0.1;
matching only a seed is not a substitute for checking starting tensor hashes.

## Runtime and cost accounting

`NeuralPolicy` dispatches encoding from its model configuration. One semantic
candidate returns before feature computation, collation, or model execution.
`inference_stats()` retains the existing three integer counters: decisions,
forced decisions, and network calls.

Separate `inference_timing()` cumulative wall seconds report
`semantic_actions`, `encoding`, `efficiency`, `collation_and_transfer`,
`model`, and `total_choose`. Encoding includes its validation/reconstruction
of the semantic candidate set. The model phase includes forward execution,
finite-logit validation, and argmax resolution. Accelerator transfer phases
synchronize and argmax completion synchronizes model work. These are observed
wall timings, not operation counts. They are absent from policy manifests and
deterministic game identities; the evaluator may attach them as report metadata.

Report cache state with feature costs: the underlying shanten LRU is shared
within a process and bounded at 131,072 entries. Feature cost depends on
distinct discard types and cache warmness. Training-data feature derivation
and storage are separate from the online policy calculation; neither is a
claim about sustained self-play throughput.

## Verification and limits

Focused tests cover exact known-wait arithmetic, exhausted waits, red/drawn
distinctions, complete hidden-world/copy permutations, a shared public action
prefix, suit relabeling, separate RIICHI/discard decisions, actual post-chi
short hands with kuikae restrictions, non-applicable rows, and invalid count
decoding. Model tests cover version/padding masks, candidate permutation,
numeric gradients, matched common weights and RNG restoration, checkpoint
dispatch, forced bypass, unchanged v1 arrays, and a pre-v2 golden v1 weight
hash from commit `18364394c8595c4f0312cde787ebea719baad63c` under the pinned
CPU Torch dependency.

These are limited fixtures. The exhaustive actor-to-feature corpus audit,
backend comparison, paired training runs, resource accounting and fresh
full-match evaluation have their own reports. Reusing the teacher's shanten
helper checks reconstruction consistency; it is not an independent rules
oracle or evidence of expert Mahjong strength.
