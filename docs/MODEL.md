# Stage 2 actor encoder and compact network

The encoder and model are implementation components, not a trained strength result. Their inputs come exclusively from the immutable [player interface](PLAYER_INTERFACE.md). Training objectives, data provenance, run budgets, and promotion results belong to the training/evaluation reports.

## Versioned interfaces

- Feature schema: `actor-features-v1` in [features.py](../src/riichi_ai/features.py).
- Action schema: `semantic-actions-v1`; grouped legal candidates preserve action kind, tile type, red identity, drawn versus held discard, and consumed tile semantics. Interchangeable physical copies are resolved at the engine boundary.
- Model schema: `suit-history-policy-v1` in [model.py](../src/riichi_ai/model.py).
- Default parameter count: **989,909**, in FP32. Counts change when configurable widths change.

`encode_view(view, actions=None)` returns the following NumPy arrays. Supplying `actions` changes candidate row order only; the complete legal semantic set must be present exactly once. Empty legal sets can be encoded for inspection, but cannot be collated as policy decisions.

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `tiles` | `34 × 23` | float32 | Own tiles, draw, publicly visible tiles, dora, rivers, and melds |
| `globals` | `45` | float32 | Public match/seat context and explicit history coverage |
| `history` | `128 × 10` | int16 | Most recent current-hand event rows, right-padded with zeros |
| `history_mask` | `128` | bool | True for retained events |
| `action_tokens` | `N × 8` | int16 | Semantic descriptors of the N legal candidates |

`collate_encoded(samples, device=None)` produces PyTorch tensors, converts categorical tokens to integer indices, pads the candidate dimension to the batch maximum, and adds `candidate_mask`. It does not impose a fixed maximum legal-action count. `PolicyNetwork(config=None).forward(batch)` returns `(policy_logits[B,N], rank_logits[B,4])`. Padding logits are the finite minimum for the current floating dtype, so padded probabilities underflow to zero without `0 × -inf` in soft-target losses. Each collated decision must have at least one candidate. Single-candidate rows remain available for rank learning; the trainer decides which rows contribute policy loss.

## Actor features

Tile channels are:

| Channels | Values |
|---|---|
| 0–3 | Own count/4, own red-five count, current own-draw type flag, own-draw red flag |
| 4–6 | Distinct visible physical copies/4, public indicator count/4, derived dora multiplicity/5 |
| 7–10 | River counts/4, ordered self then clockwise opponents |
| 11–14 | Meld tile counts/4 in the same relative seat order |
| 15–18 | Meld red-five counts in that order |
| 19–22 | River red-five counts in that order |

Only the visible-count channel deduplicates physical aliases across rivers/melds. Other channels intentionally describe distinct public zones: a called discard can remain in the river and also appear in its meld. Arbitrary physical copy identity is discarded after counting; the three red-five IDs are interpreted only as their public semantic red flags. Dora successor arithmetic respects numbered suits, wind cycling, and dragon cycling.

Global channels are:

| Indices | Values |
|---|---|
| 0–3 | Scores/50,000, self then clockwise opponents |
| 4–7 | Riichi flags |
| 8–11 | River lengths/30 |
| 12–15 | Meld counts/4 |
| 16–19 | Open meld counts/4 |
| 20–23 | Concealed kan counts/4 |
| 24–27 | Absolute actor seat one-hot, preserving starting-seat tie context |
| 28–31 | Dealer relative to actor, one-hot |
| 32–35 | Round wind, one-hot |
| 36–40 | Honba/10, riichi sticks/10, hand ID/20, concealed hand length/14, closed-hand flag |
| 41–43 | Full current-hand history length/128, overflow flag, omitted event count/128 |
| 44 | Number of observed draw events/70; this is not asserted to be the exact remaining live-wall count |

History columns encode event kind, relative actor, relative target, tile type plus one, red flag, ternary tsumogiri, and up to four consumed tiles encoded by type/red identity. Zero represents absent/padded categorical information. `encode_history_events(events, seat)` returns stable rows independent of the current prefix length, allowing the data pipeline to store a seat/hand stream once and reconstruct prefixes without repeating full histories.

The encoder deliberately retains only the **last 128 events**, with a visible overflow flag/count and complete current river/meld/count summaries. It does not silently claim the entire action history fits this window. Training audits must report the truncation rate and amount. Event payload dictionaries are not traversed: historical score payloads, initial-hand snapshots, and detailed settlement data are not additional features. Current scores and own hand have explicit channels. The history network also compresses retained events through local convolutions and pooling, so this is a bounded information representation rather than a lossless history model. Increase capacity or history length only through a new schema/configuration and a controlled comparison.

Teacher preference keys, tie-set membership, final ranks, hidden hands, wall order, and other auditor labels have no encoder input path. Keeping a label artifact separate does not replace information-set invariance tests; both controls are required. An auxiliary label may influence gradients in later training without becoming an inference feature.

## Architecture

Each numbered suit goes through the same two-layer MLP over its nine positions. The three outputs retain their suit order, so the network can distinguish suits while sharing the tile-pattern transformation. Honors use a separate two-layer MLP. No convolution crosses a numbered-suit boundary or connects honors as though they were sequences.

Current public context has a small MLP. History uses categorical embeddings, learned retained-window positions, two short one-dimensional convolutions, and masked mean/final-event pooling. These branches feed a 512-unit hidden layer and a 256-unit context vector. Every legal semantic candidate uses the same descriptor encoder and scoring network; permuting candidate rows permutes their logits without changing the rank prediction. There is no candidate-index embedding, dropout, batch normalization, or recurrent state shared between players.

The four rank logits describe final-rank probabilities for the acting seat. During supervised initialization they predict outcomes under the dataset's behavior policies and opponent mixture. Their expectation can be recomputed for different **rank-only** utility vectors; that does not make the actor optimal for another objective or provide score-based utility.

`model_config()` returns the complete configuration including model, feature, and semantic-action schemas. Widths are configurable; schema changes and unknown configuration keys fail explicitly.

## Deployment and verification

`NeuralPolicy(model=..., device="cpu")` wraps an existing model. `NeuralPolicy.from_checkpoint(path, device="cpu")` requires `model_config` and `model_state_dict`, validates schema versions, loads with `weights_only=True`, and records file/weight hashes and source provenance. The file hash covers the same bytes loaded into the model. The caller must freeze shared weights during evaluation; sharing a model does not authorize sharing future recurrent player memories.

Inference uses evaluation mode and `inference_mode()`. Stable semantic argmax chooses a legal representative. A single semantic candidate bypasses the network. `inference_stats()` reports decisions, forced decisions, and network calls separately; counters are excluded from the policy's static manifest. No deadline or search behavior is implemented here.

Tests cover alternate legally consistent hidden worlds, arbitrary normal-copy relabeling, red/drawn action distinctions, explicit overflow, stable prefix encoding, candidate permutation, legal padding masks, finite gradients, cross-sample independence, stable argmax, forced-action bypass, and checkpoint/schema round trips. CPU/MPS numerical comparison and measured inference/training speed are separate runtime checks run by the trainer; passing these unit tests alone does not establish backend parity or playing strength.
