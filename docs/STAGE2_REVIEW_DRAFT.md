# Proposed Stage 2 instructions — for review

This is a proposed next task, not a training run already started. Stage 1 provides rule fixtures, audited player inputs, three heuristic policies, and full-match evaluation. No neural policy or training corpus exists yet.

## Recommended first bounded task

Build a reproducible synthetic-data and imitation-learning pilot on the M5 Pro. Keep the four-player red-five hanchan profile and average-rank utility. Use `shanten` as the first named imitation teacher; retain `defense` and `random` as distinct evaluation/reference policies. Do not call any of them expert or world-class teachers. Preserve a frozen imitation checkpoint for later reinforcement learning.

1. **Fix schemas and provenance.** Save one canonical match event stream, policy requests separately from resolved outcomes, per-seat policy versions, rules hash, seeds, match termination, and final results. Keep auditor secrets in a distinct artifact that actor-training readers cannot load. Derive player views through the same tested adapter. Do not copy a complete history into every saved raw record.
2. **Pilot data before scaling.** Start with at most 1,000 complete synthetic matches. Assign entire matches and any related shared-wall seed blocks to train/validation/test before extracting seat views or augmenting them. Use an initially declared 80/10/10 split and preserve it in a manifest. Exclude Stage 1 development seeds. Reject incomplete matches from terminal-target learning and record rejection reasons. Never quietly convert a time cap into a terminal reward.
3. **Measure the dataset.** Report source mixture, complete hands/matches, each action kind, forced versus optional decisions, per-seat samples, duplicates, rejected records, estimated disk size, and generation/encoding speed. Synthetic logs contain the actual simultaneous requests; imported resolved-only logs would require censoring rules and separate permission checks.
4. **Implement a compact network.** Begin in the approximate 1–5 million parameter range, then choose the concrete size from measured memory/latency. Use suit-aware tile features, public match state, current-hand history, and a legal-candidate scoring head. Preserve red-five and consumed-tile semantics. Stage 0's 46 benchmark outputs are not a gameplay action contract. Treat raw physical-copy distinctions carefully so arbitrary copy numbers do not become a spurious strategic signal.
5. **Prove learning on a tiny batch.** First overfit a small, inspected set of decisions. Check legal masking, finite loss/gradients, model-specific CPU/MPS output/gradient parity, deterministic inference, and checkpoint/optimizer resume. Keep policies, encoders, and training targets versioned. Do not judge quality by action accuracy alone or count forced moves as meaningful policy success.
6. **Run one bounded pilot.** Proposed initial limits: one hour of training, approximately 32 GB process memory, and 5 GB new data/checkpoints, using local CPU/MPS. Measure before committing to those limits; reduce workload when needed. Store bulk data/checkpoints in a configured local directory outside iCloud. Stop and report if a limit, numerical failure, or correctness regression is reached. No paid compute or unattended overnight run is included.
7. **Evaluate the frozen checkpoint.** Use fresh full-match blocks against the fixed heuristic panels, reporting raw rank utility, uncertainty, illegal choices/failures, latency, and action-type behavior. Save training configuration, data manifest, seeds, all source/dependency hashes, optimizer/resume state, checkpoint hash, and the evaluation report. Keep evaluation budgets fixed and distinguish development from later confirmation.

## Suggested milestone clarification

The current implementation plan requires Stage 2 to produce a policy that beats its declared heuristic panel. Keep that playing-strength objective visible, but distinguish it from **pipeline readiness**: imitation can successfully learn a limited teacher without outperforming that teacher. A lack of improvement may reflect the teacher's ceiling rather than an implementation defect.

Recommended refinement for your review: make Stage 2A end when the pipeline, small learning test, reproducible checkpoint, and honest full-match evaluation work; reserve Stage 2B/Stage 3 promotion for a statistically supported playing-strength gain. This refinement has not silently replaced the existing Stage 2 exit condition. A higher action-match percentage alone must never be presented as stronger Mahjong.

The current heuristics seldom or never choose useful calls/kan, and they lack nuanced hand-value and placement strategy. Blindly copying them indefinitely is not a route to world leadership. After the pilot, consider stronger permitted teaching data, carefully justified synthetic policy diversity, or measured RL exploration. Choose the next expansion from the observed failure modes.

## Instructions you can change

| Item | Proposed default | Useful alternative |
|---|---|---|
| Next scope | Data pipeline plus one bounded imitation pilot | Data pipeline only, with a review before any training |
| Resource limit | Local M5 Pro; one-hour training cap; about 32 GB memory; 5 GB new artifacts | A smaller time/disk cap, or a longer explicitly approved run after profiling |
| Objective | Average placement under the current hanchan profile | A specified tournament/ladder utility as a separate experiment |
| Initial teacher | Frozen `shanten` baseline | A documented mixture, or an identified external teacher after access/usage review |
| Readiness gate | Separate reproducible learning from strength promotion | Keep the existing single Stage 2 gate, accepting that it may remain open after a working pilot |
| Data route | Locally generated synthetic games | A specifically authorized external source with recorded provenance |

A possible next instruction is: “Proceed with the Stage 2A synthetic-data and imitation pilot using these defaults. Keep the proposed limits, report the data audit before scaling, and show me the checkpoint evaluation and remaining weaknesses when finished.”

Any changed limits, objectives, or preferences you provide should replace these proposed defaults in the next task specification.
