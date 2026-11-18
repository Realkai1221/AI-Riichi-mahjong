# Stage 2A accepted implementation plan

Authorized on 17 October 2026 after review of revision 2. This plan adopts the reviewed corrections and supersedes proposed defaults where they differ. The original [first draft](STAGE2_REVIEW_DRAFT.md) and revision 2 remain historical proposals. Executable limits and seed namespaces are in [stage2_pilot.json](../configs/stage2_pilot.json).

## Scope and resource limits

Build the synthetic-data pipeline and run a bounded local imitation pilot. Use the existing four-player red-five hanchan profile and average-rank utility. Preserve Stage 1 in Git; record the implementation commit, dirty status, dependency/source hashes, teacher version, encoder schema, dataset/checkpoint hashes, and actual training work in reports.

Generate at most 1,000 full synthetic matches, split into 800 training, 100 validation, and 100 test matches before extracting player views. All seats of a match and any shared-wall relatives remain in the same split. Stage 1 seeds are development-only. Learning curves use nested 100/300/800 training subsets with fixed holdouts. Keep total training within one hour, approximately 32 GiB aggregate memory, and 5 GB new data/checkpoints. Bulk artifacts live outside iCloud. No external corpus, paid compute, public bot play, or RL is included.

## Required implementation

1. Preserve existing v1 heuristics and historical results. Add versioned v2 teachers with documented semantic tie-breaking. Group interchangeable physical candidates while retaining red tiles, drawn-versus-held discards, and consumed-tile semantics. Log full rule priorities and tie sets separately from actual deterministic behavior.
2. Define observer-specific hashes that cover the actual safe history and hand ID. Measure serialization/hash/finalization costs separately. Retain full-view reference checks. Parallel CPU blocks must reproduce deterministic results; timings may differ. Start with one/four-worker measurements rather than assuming 16 workers help.
3. Save canonical per-match auditor events and requested actions separately from numeric actor inputs and targets. No teacher scores, hidden hands, wall order, or outcome targets enter actor features. Record complete-match rank targets only after true termination. Forced semantic decisions stay in trajectories and rank loss, but not policy loss.
4. Implement a compact suit-aware actor and four-way rank head with legal semantic candidate scores. Record any history compression/truncation explicitly. The rank distribution supports rank-only utilities under the recorded behavior/opponent mixture; it is not a policy optimized for arbitrary new objectives.
5. Verify tiny-batch distribution learning, masking, finite updates, CPU/MPS numerical behavior, exact same-stack CPU resume, and bounded MPS resume tolerance. Check complete optimizer, sampler, and RNG restoration. Compare one-hot with semantic tie-aware targets as a bounded validation experiment if the budget permits.
6. Report data quality and training curves before increasing scale. Count observable settlements with denominators; mark uninstrumented causes unknown. The closed-hand teacher's absent kan/call paths are a coverage limitation, not validation of those rules.
7. Freeze a checkpoint using validation only. Evaluate legal deterministic deployment on fresh paired full-match blocks against the versioned teacher panel. No post-evaluation tuning may reuse confirmation blocks.

## Evaluation and milestones

The student-minus-teacher utility is positive when average rank improves. The proposed non-inferiority margin is 0.10. Use a separate 30-block development pilot to estimate variability. Confirmation, if warranted, uses a predeclared one-sided normal-approximation lower bound at nominal 95% confidence, with independent block means. This is an approximate statistical method, not a finite-sample distribution-free guarantee.

For planning, use the larger of Stage 1's block SD 0.546017 and 1.25 times pilot SD. Assume true difference zero and target 80% power. Fix the confirmation count before reading fresh confirmation outcomes, with a minimum of 185 and maximum of 256 blocks. If the pilot mean is at or below -0.10, retain it as a development result and omit confirmation; if the calculated count exceeds the cap, report inconclusive. A failed non-inferiority test does not by itself prove inferiority. Existing conservative Hoeffding intervals remain diagnostics and are not substituted for this method.

Stage 2A distinguishes engineering readiness (audited data, learning/checkpoint/resume, honest evaluation) from teacher-strength retention. Report each criterion separately, including the proposed 80% optional-discard agreement diagnostic. A weak or inconclusive pilot must not be labeled a successful strength result. Stage 2A readiness permits proposing the next controlled RL experiment; the stronger-than-panel milestone remains outstanding until measured improvement, potentially during Stage 3.

Belief auxiliary training and a separate GRP model are deferred until the first policy/rank pilot is interpretable. Neither becomes a mandatory prerequisite for this stage or changes rewards during this pilot.
