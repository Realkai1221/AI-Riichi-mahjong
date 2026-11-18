# Semantic actions and v2 teachers

`riichi_ai.semantics` defines `semantic-actions-v1` for the pinned four-player
RiichiEnv adapter. It groups a single player's current legal physical candidates,
not arbitrary actions across different states. It does not replace legality
checks: the runner still resolves an existing `ActionSpec` against the current
native observation.

`semantic_actions(view)` returns an immutable tuple of frozen `SemanticAction`
values. Empty legal sets return an empty tuple; unknown action kinds fail.

| Field | Contract |
|---|---|
| `kind` | Uppercase action kind, from exported `ACTION_KINDS`. |
| `tile_type` | `0..33`, or `None` when the native action has no tile. |
| `red` | Whether the action's tile is red; false when there is no tile. |
| `tsumogiri` | For discards, whether this is the current drawn physical tile; otherwise `None`. |
| `consumed` | Sorted tuple of `(tile_type, red)` pairs, preserving multiplicity. |
| `key` | Canonical JSON containing the semantic version and the preceding semantic fields; no physical IDs. |
| `representative` | One legal physical `ActionSpec`, chosen by numeric tile ID then numeric consumed IDs. Runner-only resolution metadata. |
| `physical_tokens` | All distinct physical candidate tokens in this class, in representative order. Runner/audit metadata. |

Model inputs may use the semantic fields. They must not encode the representative
IDs or token strings. Network consumers should preserve drawn-versus-held flags,
red identity, and consumed-tile choices; tile type alone is insufficient.

Same-type held normal copies are interchangeable here. A drawn discard and a held
discard remain different because their public tsumogiri status differs. Red and
normal discards, calls consuming different red combinations, and different chi
sequences remain separate. Within one decision, a called tile's source and target
are fixed by the observation; the descriptor is not authorization for another
decision. No state is synthesized when resolving the representative.

Semantic classes sort by the exported numeric `ACTION_KINDS` order, then tile type
(`None` first), red flag, drawn before held, and consumed descriptors. Type ordering
precedes drawn preference, so a drawn high-numbered type does not outrank a tied
lower-numbered type merely because it was drawn. No arbitrary physical copy ID
affects semantic ordering or target mass.

## Teacher targets and deterministic behavior

`teacher_decision(view, defense=False, *, allow_riichi=True)` returns frozen
`TeacherDecision(actions, preference_keys, best_mask, chosen_index, soft_targets)`.
Every field aligns with `actions`, the same ordering as `semantic_actions(view)`.

Each lower-is-better lexicographic preference key is:

```
(branch, unsafe_riichi_opponent_count, shanten, -visible_ukeire, red_discard)
```

Branch priorities are legal win `0`, permitted riichi `1`, discard `2`, pass `3`,
and unsupported fallback `4`. Inapplicable fields are zero. The defense teacher
uses the v1 public genbutsu calculation and suppresses new riichi under threat.
Red retention follows safety, shanten, and ukeire. All legal candidates receive
an explicit key, including actions the teacher does not choose. Fallback keys
are an explicit lack of strategic call/kan evaluation, not a value estimate.

`best_mask` marks the exact minimum preference key. `soft_targets` is uniform
over those **semantic classes**, regardless of the number of physical candidates
in each class. `chosen_index` selects the first best class in semantic order;
the engine executes its representative. The actual teacher is deterministic:
the soft distribution is an imitation target, not the teacher's behavior
probability. Store these as separate quantities in trajectories.

For example, three held copies of a tile and its drawn fourth copy produce two
semantic discard classes. If both are optimal, their target masses are `0.5`
each, not `0.75` and `0.25`. The deterministic teacher chooses the drawn class.
Soft-label cross-entropy therefore has a positive target entropy; zero loss is
not the appropriate fit criterion.

The additive policy factories `shanten_v2` and `defense_v2` report
`heuristics-v2`. Existing `random`, `shanten`, `shanten_ukeire`, `defense`, and
`conservative` names retain their v1 behavior and manifests. V2 changes tie
resolution, not the tactical preference function. Equal heuristic preferences
do not prove equal true value; no near-zero match-result difference is promised.

## Validation and limits

Tests use complete-wall engine fixtures for every supported action family,
including riichi's separate declaration, chi alternatives, consumed red copies,
and a genuine pon-to-kakan continuation. They verify physical round trips,
duplicate target mass, red/drawn distinctions, immutable results, and semantic
invariance under legal normal-copy permutations of complete walls. V1 and V2
choices are also compared on the same real decision views to check that changed
choices remain inside the intended pre-tie optimum.

This is a bounded fixture family, not a general proof of strategic equivalence
or complete simulator conformance. New action fields, source-choice mechanics,
or supported game modes require review and a semantic-version change.
