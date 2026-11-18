# Stage 2A review: the first trained Riichi agent

Measured on 17 October 2026, on the M5 Pro MacBook with 64 GB memory. Implementation: `8b14e0d842d55d4c0230fad8bb6f7bd5ce91d1c1`. Stage 1 remains preserved in `1a91128`.

**The bounded engineering pilot is complete. We have an audited dataset, a working neural trainer, reproducible checkpoints, and an agent that plays complete legal matches. The agent has not retained the heuristic teacher's playing strength.** The final model reaches **82.47% optional-discard agreement on validation**, but averages **3.18 place versus the teacher's 2.57** in its fresh development match pilot. This is a useful first learned baseline, not an expert or competitive-strength result.

The recommended next step is [Stage 2B: diagnose and close the imitation gap](../../docs/STAGE2_NEXT_INSTRUCTIONS.md). No reinforcement learning, external data acquisition, or paid compute was started.

## What is complete

| Gate | Evidence | Result |
|---|---|---|
| Correctness regression | 183 tests plus 165 subtests passed; no skips or expected failures | Passed for the documented tested scope |
| Dataset integrity | 1,000 complete matches; 611,926 decisions; every saved artifact and numeric row checked; three complete matches replayed | Passed |
| Learning machinery | Nontrivial 32-example soft-target fit; CPU/MPS forward/backward checks; CPU exact and MPS tolerance-based resume | Passed |
| Frozen deployment | Recorded checkpoint hashes, legal masked argmax, fresh paired full-match evaluation, no failed/truncated evaluation games | Passed |
| Proposed 80% discard diagnostic | Extended model: 82.47% on full validation | Met on validation, not a fresh action test |
| Teacher-strength retention | Final development utility difference −0.6167; prespecified futility rule stops before confirmation | Not established |
| Stronger-than-panel / world-class play | No qualifying evidence | Outstanding |

Passing the engineering gate does not waive the strength gate. The tests and replay checks do not certify every Mahjong rule or constitute independent full-engine conformance.

## What we built

The [data pipeline](../../docs/DATA_PIPELINE.md) separates numeric actor inputs, teacher/outcome targets, and auditor-only canonical records. The actor receives its own hand and permitted public information. Teacher preference scores, hidden hands, wall order, and future outcomes do not enter its inputs. Requested actions remain distinct from executed events when simultaneous responses are preempted.

[Semantic actions](../../docs/SEMANTIC_ACTIONS.md) group interchangeable physical tile copies while retaining red identity, drawn-versus-held discards, and consumed-tile distinctions. The teacher records a deterministic behavior choice and a separate uniform target over its equally preferred semantic choices. These are different concepts: target probabilities do not describe a stochastic behavior policy.

The [989,909-parameter network](../../docs/MODEL.md) shares a numbered-suit encoder, handles honors separately, encodes public match context and the latest 128 current-hand events, and scores each legal semantic candidate. A separate output predicts four final-rank probabilities. Full current public summaries and explicit history-overflow fields accompany the bounded event window. This representation is not a lossless record of all previous play.

The [trainer](../../docs/TRAINING.md) uses soft policy cross-entropy on optional decisions and rank cross-entropy on all decisions, with rank weight 0.1. It saves optimizer, sampler, RNG, validation selection, implementation, and data identities for resuming. Evaluation uses the actual deterministic deployment rule. Parallel evaluation preserves paired seeds, four-seat rotation, complete-match outcomes, and policy provenance.

## Data: quantity, integrity, and coverage

The formal corpus contains **1,000 matches, 9,769 hands, and 611,926 decision rows**. All four seats use the versioned `shanten_v2` teacher. It is a transparent tile-efficiency heuristic, not a pretrained expert. Splits were assigned at match level before extracting observations:

| Split | Complete matches | Decision rows | Use |
|---|---:|---:|---|
| Training | 800 | 490,206 | Nested 100/300/800 learning curves; extended run |
| Validation | 100 | 61,757 | Checkpoint selection and development diagnostics |
| Test | 100 | 59,963 | Initial frozen model only; not re-evaluated for the extended model |

There were no rejected, failed, truncated, or duplicate corpus matches. A separate ten-match engineering dataset was used for early checks; it is excluded from the 1,000-match corpus. Stage 1 seeds were excluded. See dataset manifest, full integrity audit, and embedding/dtype audit.

The verifier checked **3,000 artifact hashes/sizes and every numeric row**. It replayed matches 000000, 000499, and 000999 from their saved provenance, comparing 1,593 decisions, features, targets, events, terminal outcomes, and conservation. This verifies internal consistency with the pinned engine; it is not an independent-engine proof. A second scan checked 16,000 arrays, 5,232,879 semantic candidate rows, and 3,759,623 deduplicated history-event rows against their exact dtypes and embedding bounds.

Coverage is deliberately narrow:

- 50,559 decisions have one semantic choice and are excluded from policy loss, while remaining available for rank learning. Physical and semantic forced counts differ because equivalent physical copies can collapse to one action.
- **39.01% of optional decisions have multiple equally preferred teacher actions.** This supports keeping semantic tie targets.
- Chosen actions: 461,299 discards, 125,106 passes, 16,846 riichi declarations, 6,576 ron requests, and 2,099 tsumo requests.
- The teacher chooses no chi, pon, or kan. Some such actions appear among legal candidates, but that does not train play after taking them. Kakan is absent entirely.
- The corpus includes 112 double-ron hands. There are 8,675 winning settlements and 1,206 drawn hands; multiple winners explain why settlements plus drawn hands exceed hand count.
- Draws comprise 998 exhaustive, 25 four-winds, and 183 four-riichi draws. Causes such as haitei, houtei, rinshan, chankan, nagashi, pao, and wall-length riichi refusal are not instrumented here and remain **unknown**, not zero.

The 128-event history window overflows on 12,060/490,206 training rows (2.46%), 1,522/61,757 validation rows (2.46%), and 1,371/59,963 test rows (2.29%). The maximum observed full history is 147 events, so at most 19 are omitted in this corpus. All stored lengths/overflow flags agree. These counts do not establish a universal maximum hand length. History-window coverage, seed checks, storage, and source identity are consolidated in the resource and provenance audit.

## Initial fixed-update experiment

The original [protocol](../../configs/stage2_pilot.json) used the same initialization seed, batch size 256, AdamW learning rate 0.0003, weight decay 0.0001, gradient clipping 1.0, and 2,500 updates per run. Checkpoint selection used highest best-set agreement, then lowest soft-target KL, on a fixed 10,000-row validation sample. No test outcomes selected the checkpoint.

“Best-set agreement” means the selected action is one of the teacher's equally preferred actions. “Exact agreement” instead requires the teacher's arbitrary deterministic tie choice. KL measures the whole target distribution; lower is better.

| Training run | Validation best-set agreement | Validation discard agreement | Policy KL ↓ | Rank NLL ↓ | Run time |
|---|---:|---:|---:|---:|---:|
| Soft targets, 100 matches | 77.77% | 69.66% | 0.8298 | 1.2697 | 48.5 s |
| Soft targets, 300 matches | 78.37% | 70.47% | 0.8051 | 1.1643 | 49.3 s |
| Soft targets, 800 matches | 77.98% | 69.94% | 0.8016 | 1.1659 | 49.9 s |
| One-hot targets, 800 matches | 77.09% | 68.73% | 0.9443 | 1.1633 | 49.9 s |

The fixed rule selected the **300-match soft-target model**. On its full held-out test it achieved **78.73% overall best-set agreement, 70.88% discard agreement, and policy KL 0.7993**. Random legal selection would achieve 24.67% overall and 17.90% discard best-set agreement on these same states. There were no illegal predictions. See learning curves, selection, and initial held-out results.

These are one-seed, equal-update comparisons. The datasets receive different average sampled exposures: approximately 10.94 for 100 matches, 3.52 for 300, and 1.31 for 800. The small 300-versus-800 accuracy difference does not show that more data hurts. One-hot training improves exact tie-choice reproduction, but its worse best-set KL/agreement gives no reason to replace the soft targets in this pilot.

All four best checkpoints occurred at their final update. The 800-match run's recent training policy KL and validation KL were still close and falling. This made additional optimization a reasonable diagnosis before buying compute or collecting a large corpus.

## One declared optimization extension

After preserving the initial results, we wrote [a separate extension protocol](../../configs/stage2_extension.json): continue the 800-match soft-target run from 2,500 to **20,000 total updates**, changing neither data, architecture, objective, optimizer, learning rate, nor sampler/RNG continuation. The extension had a 600-second additional-run ceiling inside the original one-hour cumulative budget. It completed in 345.3 seconds including validation and checkpoint work.

Selection still used the existing validation rule. The initial action-test results remain attached only to the initial model. **The extended checkpoint was not evaluated on the action test.** Its performance table below is full validation, which has been used for development and selection:

| Metric | Initial selected model, full validation | Extended model, full validation |
|---|---:|---:|
| Best-set agreement, all optional actions | 78.72% | **87.18%** |
| Best-set agreement, optional discards | 70.90% | **82.47%** |
| Policy KL ↓ | 0.8037 | **0.6078** |
| Rank NLL ↓ | **1.1658** | 1.1943 |
| Rank Brier score ↓ | **0.6371** | 0.6434 |

The policy improved while the auxiliary rank output did not. The final 250 training minibatches have mean policy KL 0.5607 versus full-validation KL 0.6078. This is only an optimization diagnostic: changing minibatches and changing weights are not a frozen-model train/validation comparison.

## Full-match playing strength

Each pilot used **30 independent paired blocks**, with candidate and teacher each occupying all four focal seats. That is 240 complete matches per pilot, 120 per arm. Opponents were the frozen `shanten_v2`, `defense_v2`, `shanten_v2` panel. Positive utility difference means better average rank for the student; utility is `[1.5, 0.5, −0.5, −1.5]`.

| Pilot | Student average place ↓ | Teacher average place ↓ | Student utility advantage ↑ | Student 1st / 2nd / 3rd / 4th |
|---|---:|---:|---:|---|
| Initial frozen model | 3.5083 | 2.5583 | −0.9500 | 1 / 11 / 34 / 74 |
| Extended frozen model, fresh seeds | 3.1833 | 2.5667 | −0.6167 | 9 / 21 / 29 / 61 |

All matches completed, with zero failures/truncations and unchanged frozen policies/package sources. The initial and extended pilots used different seed blocks. Their point estimates suggest progress, but they are not a paired head-to-head comparison between the two students.

The extended pilot's exploratory block-bootstrap interval is **[−0.8667, −0.3667]**; its conservative fixed-budget Hoeffding interval is **[−2.1044, +0.8711]**. They use different assumptions and neither is the prespecified retention confirmation. The raw pilot, not a promotion decision, is in extended-student-pilot.json.

Both pilots met the predeclared futility condition: mean utility difference at or below −0.10. Therefore **no confirmation campaign ran**, and teacher non-inferiority was not demonstrated. Independently, the extended pilot's inflated variance estimate would require 482 confirmation blocks under the planning assumptions, above the 256-block cap. The unused confirmation seeds remain unused. See initial plan and extended plan.

## Why 82% discard agreement is insufficient

The policy error diagnostic re-ran the extended checkpoint on CPU using only the full validation actor inputs. CPU and MPS produced the same aggregate discard agreement; row-by-row action identity was not checked. Teacher preference keys were inspected only after predictions had been produced.

Among **41,225 optional teacher-discard decisions**, 33,998 choices match the best set and 7,227 do not:

| First differing teacher preference | Error count | Fraction of all optional discards | Interpretation |
|---|---:|---:|---|
| Post-discard shanten | 2,065 | **5.01%** | Every error leaves shanten one higher than the teacher's choice |
| Visible ukeire at equal shanten | 5,148 | **12.49%** | Loses an average of 3.70 improving tile copies; median 2, maximum 58 |
| Red-tile retention after equal earlier criteria | 14 | 0.034% | Avoidable under the teacher's preference ordering |

There were no first differences in action branch or danger flags in this discard subset. This dataset's teacher is `shanten_v2`; zero danger-key differences do not establish learned defense. The percentages describe states the teacher visited, not all states the student encounters in deployment. Ukeire counts potentially available improving copies by subtracting own/publicly known copies from four. These are not observed remaining-wall counts, exact winning probabilities, or points lost.

The errors show a concrete hand-shape learning gap. They do **not** prove how much of the match deficit each error category causes, or that distribution shift is the only explanation. Better discard representation, auxiliary-loss interference, and student-visited state coverage are testable next hypotheses.

## What the rank predictor actually learned

The rank diagnostic evaluates **only the initial frozen model** on its already-reported test split. A simple comparison predictor was fitted on the same 300 training matches: a smoothed 4×4 table mapping current public standing to terminal rank. It handles initial-seat tie priority. No neural weights changed.

| Test subset and predictor | Rank NLL ↓ | Brier ↓ | Top-rank accuracy | Top-confidence ECE ↓ |
|---|---:|---:|---:|---:|
| All 59,963 decisions: initial neural model | 1.1982 | 0.6532 | 43.09% | 1.99% |
| Same decisions: public-standing table | 1.2476 | 0.6758 | 45.69% | 1.96% |
| First decision per match/seat: neural model | 1.4087 | 0.7608 | 23.00% | 8.74% |
| Same 400 first decisions: uniform distribution | 1.3863 | 0.7500 | 25.00% | 0.00% |

The neural output improves NLL/Brier over the simple standings table across all decision rows, but has lower top-rank accuracy and similar ECE. It does not demonstrate early-game foresight: first-decision predictions perform worse than uniform descriptively. The standings table was fitted across all stages, so it is a weak early-game baseline; a round/score-conditioned comparison is a better next check. ECE here uses ten fixed top-confidence bins and is not full multiclass calibration.

All these observations remain correlated within 100 matches. Do not attach row-independent confidence claims to them. The rank head is behavior-dependent and is **not yet justified as a reward-shaping or search value model**.

## Runtime, correctness, and reproducibility

- The integrated suite passed **183 tests and 165 subtests in 16.65 seconds**. JUnit evidence. This includes the preserved Stage 1 regressions.
- The model-specific CPU/MPS fixture passed with maximum absolute difference **2.38×10⁻⁷** across recorded outputs/loss/gradients, within the declared tolerances. Parity.
- The CPU eight-update resume fixture was exact. MPS resume maximum parameter difference was **9.31×10⁻¹⁰**, within the fixed tolerance. These are same-stack fixtures, not cross-platform equivalence. Resume.
- The 32-example nontrivial soft-target fit reached **100% best-set agreement and KL 0.01427** after 500 updates. Exact deterministic-choice agreement was only 68.75%, as expected when learning tie distributions. Tiny fit.
- Formal corpus generation took **133.98 seconds** with four workers; the full integrity audit took 17.95 seconds. Sampling recorded approximately 0.87 GiB aggregate process RSS during collection.
- The one-versus-four-worker heuristic benchmark reproduced exact trajectories/results and measured **22.69 versus 7.13 seconds**, a 3.18× speedup. It is a short workload, not a sustained thermal or universal scaling claim. Throughput.
- Cumulative learning/check/loading/validation work was **563.58 seconds (9.39 minutes)** against the 3,600-second ceiling. This is not the total development-session duration. Budget.
- Maximum recorded training memory was approximately **3.78 GiB** under the conservative RSS-plus-MPS-driver sum. Unified memory may be double-counted; sampling can miss short peaks. The 32 GiB target was not approached.
- External Stage 2 files occupy **343,782,644 logical bytes (343.8 MB)**, including the corpus, engineering artifacts, resumable runs, and both selected checkpoints. Small project reports are additional and the combined artifacts remain far below the 5 GB ceiling. Logical file sizes are not APFS physical allocation measurements.
- In the match pilots, CPU neural choice calls average **0.422 ms initially and 0.456 ms for the extended model**, weighted over all neural choices, including forced bypasses and encoding/model overhead. These are not isolated forward-pass latency or a deployment deadline guarantee.
- The 192 benchmark executions and 480 student-pilot executions all complete without failures/truncations, separately from 1,000 formal corpus matches and ten engineering matches. Paired/repeated wall executions are not independent samples; strength uncertainty uses blocks.

A small v2-versus-v1 heuristic comparison was also preserved: eight blocks give v2 a −0.4063 mean-rank advantage with exploratory interval [−0.9375, +0.2188]. This does not establish v2 as stronger. v2 exists to version semantic target/tie behavior; v1 and its historical evidence remain available. Teacher comparison.

Package modules used for the runs match implementation commit `8b14e0d`. Some evaluation manifests report `git_dirty: true` because generated evidence was untracked at run time; that flag is preserved. Per-module hashes and the final audit distinguish those report files from package source drift. The actual external dataset manifest and its prettified workspace copy have different byte hashes but identical parsed content.

## Files to review and checkpoints to preserve

Start with this report, then the [proposed next instructions](../../docs/STAGE2_NEXT_INSTRUCTIONS.md), [current status](../../docs/STATUS.md), [model contract](../../docs/MODEL.md), and [training contract](../../docs/TRAINING.md). The evidence manifest hashes the saved project evidence and binds it to external artifacts.

Bulk artifacts are outside the iCloud project:

- Audited corpus — manifest SHA-256 `55d58a7a8de5c29dc5137eca80926829be786b201f2372bcb0f2ef3208b211d7`.
- Initial frozen model — SHA-256 `a80bd9ae91dd72451279b73ebfd8efefbe26660f984ee4719149c6b170f201a8`.
- Extended frozen model — SHA-256 `09d42bf7c2b881f57d76f4ed38b33fa04070a90c7294d183338e5d0a30d1537d`.

The [original runner](../../scripts/stage2.py), [extension runner](../../scripts/stage2_extend.py), and diagnostic scripts preserve the executed procedures. Reproduction needs a new configured output root: do not overwrite these completed runs or blindly rerun evaluation into the historical report names. The external bulk directory is local storage and is not included in the Git snapshot; preserve it separately when moving machines.

To verify the saved inventory without generating games or training, run `.venv/bin/python scripts/stage2_manifest.py verify` from the project directory. This checks file hashes, not model quality or independent rules correctness.

The decision for review is to keep the audited infrastructure and semantic targets, retain both frozen students as baselines, and run a bounded Stage 2B focused on hand-shape errors and state coverage. A larger model, more random teacher games, a separate GRP model, or paid GPUs would currently be an untested response to the measured weakness.
