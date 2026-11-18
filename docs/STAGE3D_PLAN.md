# Stage 3D — fixed evaluation-only recovery

Authorized by the user's 21 October 2026 instruction to continue after reviewing the Stage 3C checkpoint. Execute the [proposed recovery](STAGE3C_REVIEW.md) as a separate experiment. Stage 3C remains stopped and is not resumed, retried or pooled into this evaluation.

## Question and unchanged endpoints

Compare the fixed Stage 3C four-update call-head endpoint with its identically initialized, untrained control. Both use the original frozen Stage 2B base, the same call gate/conditional head, the same sampler and three original deterministic argmax opponents. No optimizer update, new checkpoint, policy-rule change or promotion is included.

| Input | SHA-256 |
|---|---|
| Candidate: Stage 3C batch-3.pt | `4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967` |
| Baseline: Stage 3C initial.pt | `3baca74ddcb011ea7346396621c709e7499835c2cf4b9e6e201688aa52e75bf2` |
| Original opponent checkpoint | `5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f` |
| Behavior protocol retained in records | `1fb067d7b76a02eb286c9ab560678a78f8623cc6f59a75cbfa64016fe2936776` |

The new configuration hash identifies the recovery campaign separately from the old behavior protocol. Every raw match is evaluation-only, with explicit ineligible-for-training flags. Full logs and physical-wall commitments remain auditor-only; policy inputs and simultaneous-response ordering are unchanged.

## Frozen budget and seed plan

The [configuration](../configs/stage3d.json) contains all 32 fresh wall seeds, 128 paired sampling seeds and match IDs 320–575 before any game. SHA-256 namespace `riichi-ai-stage3d-20261021` separates the new campaign. Preparation checks disjointness against every Stage 3C training/evaluation wall and sampling seed.

Each of 32 blocks shares one wall seed across four focal-seat rotations and both arms. Each pair shares a sampling seed. Compare actual physical-wall commitments for every common ordinal hand; matching numeric seeds alone is insufficient. Preserve every score, placement, choice, probability and raw actor encoding.

| Constraint | Fixed value |
|---|---|
| Total numerical budget | 400 cumulative seconds, including preparation, tests, failed attempts, book extraction/rendering, matches and leased evidence audit |
| Preparation before formal games | At most 45 actual seconds |
| Shutdown reserve | 30 seconds |
| Memory and artifacts | 32 GiB sampled process-tree RSS; 500,000,000 new bytes |
| Formal matches | 32 blocks ×4 seats ×2 arms = 256, no replacements |
| Per-block reservation | 30 seconds; worker deadline 20 seconds; watchdog 25 seconds with bounded cooperative shutdown |
| Precision and device | CPU, one Torch thread, single-row fp32 network and staged CPU fp64 probabilities |
| Training / checkpoint writes | Zero |

The controller refuses admission unless the full declared reservation fits after the shutdown reserve. A failed block stops the chain; it is not replaced or extended after inspecting outcomes. A single file-locked ledger prevents concurrent numerical leases. Storage and sampled RSS are monitored during each operation.

## Workflow correction and preflight

The new controller hashes an **explicit list of runtime and test paths**, including the conservative historical dependency set and only the named new collector/controller tests. It ignores unrelated additions while continuing to reject changes or deletion of a declared file. The new configuration is separately pinned. No historical runtime file is edited.

Before each match, write an exclusive start record binding campaign, behavior protocol, plan, block/arm, seat, seeds and input state. Synchronize it to disk before collection. The new collector checks cancellation between complete response steps. SIGTERM sets a flag rather than interrupting a response halfway through. A cancelled collector returns a partial trajectory without terminal utility; the controller preserves it and stops. JSON/NPZ artifacts are synchronized before an exclusive finish marker is published. A start without finish remains an explicit unresolved interruption, never a completed match.

Preflight covers unrelated-file creation, modified/missing declared files, seed disjointness, fixed caps, no retries, exclusive synchronized markers, original-versus-recovery behavior on short real-engine trajectories, cancellation before/during collection, partial evidence and evaluation exclusion from training. The original genuine post-call fixtures are also run. Passing current-source preflight is required before loading the fixed endpoints and freezing the formal plan.

All 32 completed block receipts must be closed before the final summary is written. Both parameters and optimizer-state identities are compared before/after each block; original checkpoint files are rehashed. The independent auditor checks saved raw data, terminal accounting, sampler RNG, provenance, lifecycle markers, source membership and the fixed comparison.

## Analysis fixed before outcomes

For each block average the four paired seat utility differences, with utility `[1.5,0.5,-0.5,-1.5]`. Each block difference lies in [-3,3]. Report the mean across all 32 blocks and a fixed two-sided 95% Hoeffding interval with half-width `6*sqrt(log(40)/(2*32))`, clipped to [-3,3]. This interval is wide; completion alone cannot establish a small strength gain. Do not reinterpret identical sampled trajectories as proof of policy equality.

This comparison isolates the four learning updates from the initial introduction of a 5% call gate. It does not measure the net effect of adding that gate versus the original no-gate policy. No model is selected or tuned from this evaluation, and no result automatically promotes it.

## Book work alongside the recovery

Use the already supplied local books. The narrow extraction plan is one text extraction and at most eight page renders, with six selected pages from Seventh9's call chapter initially rendered. Prepare at most two source-linked partial call packets and a small locator list. Keep offered tile/source seat, consumed tiles, existing melds and pass availability explicit or unknown. Explanations remain separate from question fields. No full-policy conversion, invented missing state, OCR expansion, model evaluation or book training is included.
