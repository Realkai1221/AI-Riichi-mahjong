# Accepted Stage 2B experiment

Authorized by the user on 17 October 2026. This adopts the [revision review](../STAGE2B_REV2_REVIEW.md); the original proposals remain historical. The exact protocol, seed roots, selection rules, and limits are in [stage2b.json](../../configs/stage2b.json). Implementation and measurement will be reported separately.

## Scope and budget

Build exact actor-computable shanten/ukeire/improving-type features with a 128-event history, preserve runnable v1 checkpoints, and compare feature-assisted learning against matched raw controls. Run two seeds each for feature-assisted rank weight 0.1, raw rank weight 0.1, and feature-assisted rank weight zero. At most one conditional two-seed comparison follows if its predeclared diagnostic warrants it and both runs fit the budget.

The ceilings are 3,600 seconds cumulative training-related work, approximately 32 GiB sampled memory, 5 GB new artifacts, eight full training runs, 600 newly collected corpus matches, four 30-block development evaluations, and one conditional retention confirmation of at most 256 blocks. The separate all-neural throughput benchmark has an aggregate 1,800-second ceiling including worker sweeps and a planned 1,200-second sustained run. Bulk files stay outside iCloud.

Feature derivation/alignment, learning checks, loading, optimization, validation and test inference count toward the training-related ledger. Dataset collection, non-learning engineering tests, full-match evaluation, and benchmark work have their own recorded counters/timers. A resource limit can leave a comparison incomplete; it cannot justify extending scope or presenting a partial comparison as a pass.

## Data and information boundary

The original 800 training matches remain the common corpus. Derive feature arrays from safe actor counts without changing the original files. Exact alignment is required on all applicable original training and validation candidates. Teacher targets remain outside the network inputs. A deterministic reconstruction verifies teacher preference ordering separately from learned model accuracy.

Generate 100 fresh old-student diagnostic matches, 100 new validation matches, and 100 new test matches. Validation/test each use 50 teacher-focal and 50 frozen old-student-focal matches against the unchanged opponent panel. Save focal-seat training records and complete auditor provenance; distinguish the actual behavior request from a teacher query at that state. Test performance remains sealed until the final candidate is frozen. Optional aggregation adds at most 300 complete matches and never imports validation/test rows.

## Controls and selection

Copy identical shared parameter tensors explicitly across paired raw/feature initializations, preserve paired sampling streams, and record hashes. Use a fixed stratified validation sample for checkpoint selection and full validation for recipe selection. The ordering is highest equal-stratum optional-discard best-set agreement, lowest shanten-worsening rate, then lowest optional policy KL. Recipe scores average both seeds. The lower-numbered primary seed of the winning recipe supplies the final checkpoint before action-test or development match results.

Use the top two complete eligible recipes for four development campaigns, one campaign per recipe/seed, on common block seeds. The final candidate alone can proceed to fresh confirmation under the retained 0.10-margin, variance/power/cap/futility rules. Validation accuracy, engineering readiness, and teacher retention remain distinct outcomes. No conditional experiment uses confirmation results.

## Preserved history and deferred work

Stage 2A's snapshot is `e19a6cb959a72bdfbf3bb62d7b9c327dd01337a7`; its external artifacts remain immutable. Run `.venv/bin/python scripts/verify_stage2a_snapshot.py` to verify its original inventory against Git while current sources evolve. Do not rewrite historical hashes to match new code.

History-window expansion, calls_v1 implementation/evaluation, RL, search, external data/checkpoints/contact, paid compute, public play, and overnight runs remain outside this stage. Document the safe long-run routine for a later proposal. Any Stage 3 recommendation follows the measured result rather than launching automatically.
