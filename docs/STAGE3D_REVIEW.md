# Stage 3D review — evaluation recovery and the next learning question

The fresh evaluation finished all **32 paired blocks / 256 complete matches**. The four-update candidate and the initialized, untrained control with the same frozen base both averaged **2.6328125 place**. Their measured average-rank utility difference is **0**, with the predeclared two-sided 95% Hoeffding interval **[-1.440484, +1.440484]**. This is a completed comparison with no demonstrated strength improvement. The interval is too wide to establish equivalence or rule out a meaningful difference.

The [measured report](../reports/stage3d/REPORT.md) gives the final audit, resource totals and artifact links. The [fixed plan](STAGE3D_PLAN.md) was frozen before these games. No training, checkpoint writing, historical-result pooling or policy promotion took place in Stage 3D.

## What changed and what worked

The controller now monitors an explicit list of runtime and test files. Adding an unrelated analysis file cannot change that membership; editing or deleting a declared file still stops execution. A small evaluation-specific collector preserves the original policy behavior and simultaneous-response ordering while adding cooperative cancellation and synchronized start/finish records for every match. Interrupted trajectories cannot acquire a terminal reward or enter training.

All fresh blocks completed without replacements. Both input checkpoint files and their parameter/optimizer identities stayed fixed. This resolves the Stage 3C workflow failure. It does not retroactively complete Stage 3C: its failed block and incomplete comparison remain preserved as a separate experiment.

## What the behavior tells us

Of 128 paired matches, **127 had identical complete action sequences**. One pair differed first at response step 197: the learned candidate passed where the initialized control requested PON. Its later actions and final scores differed, but all four placements were the same. Every pair therefore had zero focal utility difference.

The policies are not numerically identical. On **3,913 call-opportunity rows with matching pre-response states**, every saved distribution changed. The average total call probability decreased by **0.00018448**, or about **0.01845 percentage points**. The largest per-action probability change was **0.00019942**. On all **15,305 matching rows without a call opportunity**, the saved distributions were exactly unchanged. Comparisons stop at the first divergent response; later mismatched states are not treated as matched observations.

These measurements support a narrow conclusion: the four training updates changed the call mechanism, but usually by too little to change the sampled action under shared random draws. They do not show useful strategic learning. A decrease in calling is not inherently an improvement, and a frozen base trained largely on closed hands may make post-call play a limiting factor.

The candidate resolved 175 calls/kans and the control 176. DAIMINKAN appeared twice in each arm; neither arm requested KAKAN. This adds exercised behavior, not comprehensive rare-rule coverage. Both arms selected RON in all four sampled win-versus-call opportunities in each arm, although the gate still assigns approximately 5% probability away from that win. Four observed wins do not establish that this issue is solved.

## Recommended next experiment — proposed, not started

The next useful question is whether a longer, fixed learning horizon produces useful behavioral change. Another evaluation-only recovery or a learning-rate sweep would not answer that question as directly.

1. Restore the Stage 3C update-4 model **and optimizer**. Continue for exactly **28 batches of 16 fresh complete matches**, ending at update 32. Keep the architecture, learning rate, utility, sampler, frozen base and opponent panel unchanged.
2. Evaluate only that fixed endpoint against the starting update-4 checkpoint, using **32 fresh paired blocks / 256 matches**. Do not choose a best checkpoint, add seeds after seeing results, or pool the present evaluation. Total new games would be **448 training + 256 evaluation = 704**.
3. Record the same shared-state probability changes, action divergence, call requests/resolutions, post-call behavior and fixed utility uncertainty. Retain the win-versus-call diagnostic. Training rewards and call counts are diagnostic evidence, not the selection criterion.
4. If behavior still barely changes, stop extending the same loop. Investigate credit-assignment variance and the frozen base's post-call competence as separate interventions. If behavior changes without a supported utility gain, report that result without promotion. A more diverse and stronger opponent panel remains necessary before any competitive-strength claim.

A **provisional 1,200-second numerical cap** is reasonable to assess before freezing this proposal: the four historical training batches each took 21.4–22.4 seconds, and this evaluation used about 323.3 seconds including its per-block runtime overhead. Extrapolating 28 similar training batches plus the same evaluation suggests roughly 15–16 minutes before preparation and audits. That is an estimate, not a sustained-run benchmark or an already authorized budget. The next executable plan should reserve evaluation and audit capacity explicitly, retain a shutdown reserve, and stop without replacing failed work.

This experiment isolates more learning from update 4. It still would not measure the net benefit of introducing the call gate compared with the original no-gate reference; that needs a separate declared comparison. It also cannot establish world-class play against the current narrow opponents.

## Book preparation and review points

The book review locates **eight call-related cases across six families** and prepares **two detailed partial source packets**. These are local development material. Neither packet is a complete engine state or an accepted policy-training example.

One example supplies a call-versus-decline preference; the other assumes calling and teaches which CHI to select. Keeping those labels separate is essential for the two new heads. The first example's dora label appears only in its explanation and is excluded from the question observation. Unknown rivers, scores, meld declarations and physical red identities remain unknown.

The next book task should implement a strict partial-call validator and information-boundary checks before expanding extraction or introducing any book loss. The most useful feedback now is on the proposed longer learning horizon, the next opponent-panel priority, and whether to prioritize this book validator alongside the learning experiment.
