# Stage 3B: fixed action-support and gradient diagnostic

Authorized on 20 October 2026: at most **300 cumulative numerical seconds**, **512 saved development rows**, **32 gradient/backend probe rows**, **zero new matches**, **zero optimizer updates** and **zero checkpoint writes**. This executes the Stage 3A review's proposed diagnostic. It is not a playing-strength evaluation.

The original Stage 2B reference and final Stage 3A boundary are compared on identical saved actor inputs. Historical Stage 3A files remain pinned through their completed Git snapshot. Actors receive saved player-visible tensors only. Full logs may identify provenance but are not model inputs. Neither an action-test split nor a new corpus is read.

Before model outputs are computed, save a deterministic selection stratified across offered concealed/open kan, chi, pon, win, discard and pass contexts. A row belongs to one selection stratum; overlapping offered action types remain available in the report. Prefer rare strata, then fill unused capacity deterministically. Save row identities, source hashes, selection/probe rules and both checkpoint hashes in an immutable plan. Report actual coverage, including absent added-kan examples, rather than fabricating them. The sample overrepresents rare actions and is not an estimate of ordinary match frequencies.

Four probability rules are fixed before execution:

| Rule | Definition |
|---|---|
| Reference | `softmax(legal_logits)` |
| Temperature 10 | `softmax(legal_logits / 10)` |
| Temperature 100 | `softmax(legal_logits / 100)` |
| Uniform mixture | `0.95 * reference + 0.05 * uniform_over_legal_candidates` |

These are diagnostic alternatives, not new deployment defaults. The two temperatures are a coarse, predeclared range, not an optimized search. Record probabilities, stable log-probabilities, zero support, entropy, call mass, legal-action gaps and distortion from the reference on ordinary discard positions. A large common logit offset alone does not imply softmax saturation.

For fixed target actions on the probe rows, differentiate the actual rule's log-probability. Report raw-logit gradient response and parameter gradient norms/groups. For the mixture, record attenuation `0.95 * pi(action) / q(action)`: positive sampling support alone does not imply useful model gradients. Do not use `log pi` as a surrogate for `log q`. Backward probes do not call an optimizer, and model weights must be identical before and after.

CPU execution uses one thread, single-row float32 network forwards, and float64 CPU normalization, matching Stage 3A. Compare the same fixed probe rows on MPS if available, normalizing on CPU because MPS does not support float64. Preserve measured discrepancies and partial/unsupported outcomes. Do not widen tolerances after results or treat backend availability as parity.

Every numerical import, test, checkpoint load, forward/backward calculation and failed attempt is charged through a durable 300-second ledger. The subprocess watchdog reserves shutdown time, samples process-tree memory against 32 GiB, and limits new artifacts to 500 MB. Source authoring, standard-library hashing, document review and manual book inspection are outside the numerical allowance. An operation failure stays in the record; no outcome-dependent replacement of rows or alternative grid is allowed.

Book preparation proceeds separately: convert the twelve corrected cases into a partial-scenario schema with supplied fields, unknown fields, ambiguous red-five identity, conditional/compound recommendations, explanations, source identities and family grouping. Preserve all prior revisions and their arithmetic evidence. Keep answers separate from observation exports and prohibit full-policy training eligibility until required context is resolved. No book-derived model training is authorized.

Deliver a measured report, machine-readable row/probe results, closed budget, source/checkpoint integrity evidence, tests and a reviewable next experiment. A subsequent learning proposal must specify a fixed endpoint, finite batches and budget, a matching initialized-policy control, and fresh paired evaluation with identical sampling rules. This diagnostic cannot promote a policy or establish a strength gain.
