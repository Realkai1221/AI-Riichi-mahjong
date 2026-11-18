# Synthetic data contract

Stage 2A uses locally generated four-player red-five hanchans. No external game logs or pretrained weights are included. [dataset.py](../src/riichi_ai/dataset.py) generates the corpus; [dataset_validation.py](../src/riichi_ai/dataset_validation.py) audits its saved artifacts. The accepted resource limits and seeds are in [stage2_pilot.json](../configs/stage2_pilot.json).

## Assignment and provenance

`plan_matches(1000, root_seed)` assigns 800 training, 100 validation, and 100 test matches before any player views are extracted. Each match has a unique derived 64-bit seed, match ID, and independent-group ID. All four seats and all decisions remain together. Training curves use the first 100, 300, or 800 training IDs. There is no random row-level split. The separate small engineering dataset is not added to the learning corpus.

Recorded Stage 1 development roots and exact environment seeds are rejected. Distinct Stage 2 namespaces cover engineering checks, collection, teacher comparison, throughput measurement, student pilot, and any confirmation. This is reproducibility within the pinned engine; equal seeds do not imply cross-engine equivalent walls.

Each dataset saves a split-plan hash, source/runtime hashes, Git revision/status, rule profile/hash, teacher version and configuration, per-seat behavior provenance, and every artifact's SHA-256. Existing nonempty output directories are refused. A failed/truncated match has an audit record but no training shard or invented terminal rank. The aggregate manifest lists rejected and unwritten matches and is invalid if collection is incomplete, sources change, a duplicate game appears, or resource checks stop the run.

## Three separate artifact families

| Directory | Contents | Training access |
|---|---|---|
| `actor/` | Numeric player-visible tile/global features, ordered semantic candidates, and safe history tokens with prefix references | Passed to the network |
| `targets/` | Uniform best-set probabilities, best mask, actual teacher choice, full preference keys, final ranks, and row/seat/hand alignment metadata | Only selected labels enter losses; teacher preferences are never model features |
| `auditor/` | Compressed canonical engine events, exact submitted physical actions, semantic requests, provenance, terminal results, and failure evidence | Default actor/target loader never opens these files |

The NPZ arrays contain no pickled objects. ZIP timestamps and gzip timestamps are fixed so equivalent serial/parallel generation produces identical artifact bytes. Arrays are hash-checked before loading and treated as read-only. Loader metadata is checked against the saved split plan rather than trusting an editable list of split IDs.

At each simulator step, all returned observations are consumed by separate seat adapters before any simultaneous choices are submitted. Never refetch an observation casually: the engine's observation API advances an event cursor. Terminal rank labels are attached only after genuine whole-match termination.

## Requests, semantic ties, and forced rows

The teacher records a known requested action at its original information state. If another simultaneous request preempts it, it is still a valid synthetic request-level policy label. It is not relabeled as an executed event or assigned an action-dependent transition target that assumes execution. Unknown requests cannot be reconstructed from resolved external logs.

Semantic candidates group interchangeable physical copies while preserving red tiles, drawn versus held discards, and consumed-tile alternatives. Best sets use the complete teacher priority before its deterministic final tie-break. Uniform mass over that best set is an imitation target; the actual deterministic behavior probability is one for the submitted request. [SEMANTIC_ACTIONS.md](SEMANTIC_ACTIONS.md) defines the versioned ordering and priorities.

Single-semantic-option rows remain for trajectories and rank training but are excluded from policy loss. Reports distinguish physical and semantic forced decisions, tie-set sizes, selected action kinds, legal candidate kinds, and per-seat counts.

## History and memory

Each seat/hand's safe event sequence is stored once. A row references only the prefix available at that decision. The loader reconstructs its latest 128 events and mask; globals record overflow explicitly, and tile features retain complete current public summaries. Replay checks compare these reconstructions with direct encoding at each original decision, preventing later events in the stored sequence from entering earlier actor inputs.

`load_split` can preload compact per-match arrays, avoiding repeated decompression during shuffled minibatches. Its default byte limit is 8 GB per loaded split, with a bounded match-cache alternative. These are array-budget checks; the trainer separately measures process and accelerator memory. Small spawned worker pools use bounded ordered submission, including on Python versions before 3.14.

Collection checks worker RSS and sampled aggregate process-tree RSS where permitted. A sandbox may prohibit aggregate enumeration; unavailable measurements are reported explicitly. These checks and the 5 GB artifact budget are cooperative limits, not operating-system hard caps. The pilot runner accounts for local bulk artifacts and Stage 2 reports and reserves space for audit manifests.

## Verification and coverage limits

The verifier checks saved hashes, plan/split consistency, array dimensions and finite values, legal candidate/target alignment, normalized probabilities, valid terminal rank labels, and complete match metadata. Selected first/middle/last matches are replayed from their seeds and recorded requests. Every replayed actor encoding, teacher target, final rank, and canonical event stream must agree. Tests also remove auditor artifacts and confirm that actor-loader results remain identical.

The first corpus uses `shanten_v2` in all four seats. This closed-hand teacher usually passes optional calls and kans. Therefore absent call/kan trajectories are a limitation of the behavior mixture, not validation of those rules. Observable settlements and declarations are counted with denominators; unavailable causes such as individual rare yaku remain `not_instrumented`, not fabricated zeros. Optional belief labels and their separate auditor-side reader are deferred.

This corpus supports a bounded imitation experiment. Its size, audit success, or low training loss does not establish expert strength; fresh complete-match evaluation provides the relevant outcome evidence.
