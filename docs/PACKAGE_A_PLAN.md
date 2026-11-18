# Package A — retention confirmation and stability study

Authorized by the user after the [revision 2 assessment](stage2b/NEXT_REVIEW_REV2_ASSESSMENT.md). This is a bounded follow-up to Stage 2B, not the start of Stage 3. The executable settings are [package_a.json](../configs/package_a.json). Results will be written under `reports/package_a` and experimental checkpoints under the local non-iCloud RiichiAI directory.

## A1: unchanged-model retention

Run the frozen Stage 2B primary against the unchanged baseline and panel: 319 independent paired wall-seed blocks, four focal-seat rotations and two arms per block, totaling 2,552 match executions. Candidate SHA-256 is `5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f`. Keep utility `[1.5, 0.5, -0.5, -1.5]`, margin 0.10, one-sided alpha 0.05, the original normal approximation and original rules. No candidate switching, interim strength-based decisions, pilot pooling or count adjustment.

The unused root namespace `2026101718` supplies all 319 precommitted wall seeds. Check actual historical corpus, development and benchmark seeds for overlap, including the 63 blocks beyond the earlier 256-block ceiling. Save the complete list before outcomes. The evaluation package remains byte-identical to completed Stage 2B throughout A1; only new external orchestration supplies the deadline.

Ceilings: 600 cumulative evaluation-related seconds including setup, imports, failed setup, model loading and serialization; 32 GiB sampled aggregate process RSS; 1 GB new artifacts. Reserve 15 seconds for shutdown/finalization. A killed, invalid or incomplete campaign cannot pass. A pass supports retention within the stated margin against this local panel only; it does not establish superiority or human rank.

The initial sandbox launch failed at process enumeration, with an empty console and no saved game outcomes. Its child process group was terminated. Preserve the initial plan/log and charge a conservative five-second setup allowance to the same 600-second ceiling. The monitored retry uses the identical model, seeds and hypotheses. This is an explicitly recorded setup failure, not a new statistical sample chosen after results.

## A2: four attempts, one intervention

Use two instrumented rank-zero controls and two rank-zero runs with epsilon 0.01 legal non-best label smoothing, paired by seeds `202610171` and `202610172`. Reuse the original data, matched initialization, sample streams, AdamW settings, batch size 256 and 18,500-update horizon. Preserve original teacher labels and report all diagnostic agreement/KL against them. A separate training target allocates 99% mass to the original best set and 1% uniformly to semantic legal non-best actions when such actions exist; forced rows remain excluded and padding receives no mass.

The four-attempt maximum includes failed and safety-stopped formal attempts. No automatic replacements or second intervention. The cumulative training-related limit is 2,700 seconds, including numerical tests and failed tests, imports, loading, probes, updates, validation, entropy measurement and finalization. Ceilings are 32 GiB sampled RSS and 2 GB new artifacts. Source editing and model-free document/hash checks are recorded separately. Persistent leases charge interrupted operations rather than silently refunding them. A timed probe determines whether the full comparison can fit before formal attempts start.

Measure actual shared-context norms, legal raw and centered logits, best/non-best gaps, frozen placement-probe loss, gradient tails, and module parameter/update norms. Exclude padded candidates. Fixed diagnostic indices use an independent RNG; hooks must preserve forward/backward behavior and training RNG. Verify those properties and CPU/MPS numerical agreement before formal runs.

Freeze the same experimental guards for every run: diagnostics every 100 updates; median reference from the first 1,000 updates; scale factor 10 with scale floor 1; three consecutive scale exceedances; raw-logit threshold additionally floored at 100. Immediate stop for nonfinite quantities; sustained pre-clipping gradient above 50 for three updates or unsmoothed policy CE above 10 for 50 updates is a backstop. These are prospective safety heuristics, not previously calibrated predictors or proof of a collapse mechanism. Preserve warm-up and alarm traces, and compare alarm timing with the historical collapse region. Do not tune thresholds after observing a control.

An intervention passes the stated durability target only if both seeds complete 18,500 updates without a safety alarm and each terminal checkpoint reaches at least 99.9% fixed-validation best-set discard agreement. Keep selected and terminal checkpoints separate; an earlier good checkpoint cannot hide a bad final one. Report full validation, late trajectory, unsmoothed KL, error severity and probability mass. No convergence early stopping applies to this durability test.

Measure entropy, normalized entropy where defined, teacher-best mass and call/kan mass by population, legal-choice count and decision availability. Keep forced decisions separate. These measurements describe exploration; they do not certify useful call behavior or authorize RL. No action-test reuse for tuning.

## Evidence and stopping boundary

Preserve completed Stage 2B inventory bytes and bulk artifacts. `scripts/verify_stage2b_snapshot.py` checks historical project files at commit `60ceff6b7fd8d4d40f96d5ceec54a2cac720ba3c`, allowing current status/decision documents to evolve without rewriting history. Stage 2A retains its own snapshot verifier.

Package A ends with results, accounting, an integrity inventory and a review report. Calls expansion, a large outcome corpus, compiled efficiency work, reinforcement learning, external data/contact and paid compute remain outside this package.
