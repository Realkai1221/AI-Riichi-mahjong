# Stage 3C review checkpoint

**The call-learning mechanism works in this pilot. Training is complete; the planned evaluation is incomplete because of a workflow error. No model is promoted.**

The four fixed batches completed all 64 training matches and four heads-only updates. The learner recorded 10,010 decisions across 640 hands, including 100 resolved calls: 63 chi, 34 pon, two concealed kans and one added kan. There were 740 later decisions with a meld present. This is substantial new coverage relative to the previous no-call training path, although no open kan was selected in training.

Both new heads received finite gradients and changed on every update. The base and rank head remained identical, and pre-update probabilities replayed exactly. These results support retaining the trainable call-gate design as an experimental mechanism. Four updates do not establish useful strategy or stronger play.

## Why evaluation stopped

Eight of the planned 32 paired blocks completed. During block 8, a newly created, unrelated analysis test matched the controller's dynamic source-file glob. Its addition changed the monitored file set, triggering termination. This was a coordination and source-scope mistake in our implementation workflow. Every originally pinned runtime and test file remained byte-identical; there was no detected model mutation or game-rule failure.

The failed attempt and two fully saved matches from that block are preserved. The controller stopped the chain as designed. The new independent test was moved outside the frozen filename pattern, but no games were resumed or replaced. The failed block is excluded from paired summaries. There are 130 fully persisted matches in total: 64 training and 66 evaluation, including those two partial-block records. A hard watchdog termination may leave the next in-flight match without a complete trajectory; it is not counted as a completed match.

The 32 complete pairs from the eight valid blocks produced identical action sequences and placements for trained and untrained heads. This is a descriptive prefix, not the prescribed fixed endpoint, a confidence interval, a proof of equivalence or evidence of improved strength. The model probabilities did change slightly; common random draws simply selected the same actions in those pairs.

## Keep, change and defer

Keep the stable call/noncall normalization, independent conditional call head, explicit learner-only provenance, complete-match utility, frozen base, exact replay, genuine post-call fixtures and physical-wall pairing. Retain both initialized and final checkpoints.

Change source-freeze implementation in a new controller version: declare an explicit immutable runtime/test file set before the experiment. Continue detecting edits/deletions of those files, while allowing unrelated analysis documents and tests to be created. Test both cases. Add a durable match-start marker and cooperative termination evidence so an interrupted match is distinguishable from an unstarted one.

Do not raise the learning rate, add training batches or select a different checkpoint from this partial evaluation. Do not silently force wins in the existing sampler: that would change the intervention. The five training positions where calls competed with a win retained about 95% win probability; every one happened to select the win. Any future always-win rule must be specified and compared separately.

## Proposed next stage: evaluation recovery only

Recommended Stage 3D is a separate fixed evaluation campaign after the workflow fix, with **no optimizer updates**. It would use the unchanged Stage 3C final endpoint and unchanged initialized control, 32 entirely fresh paired wall seeds, four learner-seat rotations and both arms: 256 matches. Do not pool this stopped prefix with the fresh campaign.

Proposed envelope: 400 cumulative numerical seconds, at most 45 seconds of preparation/checks, 32 GiB sampled RSS, 500 MB new artifacts, and a 30-second shutdown reserve. The eight completed Stage 3C blocks took approximately 7.2–10.4 seconds each, so throughput suggests this is feasible, but admission and stopping must still be enforced. All seeds, explicit file membership, check identities and endpoint hashes must be frozen before the first game. No replacement blocks or outcome-based extension.

Use the same declared block-level mean utility difference and two-sided 95% Hoeffding interval. Its wide interval will still limit conclusions. Even a favorable trained-versus-initialized comparison would not establish net improvement over the original policy without the 5% gate; that needs a separate, prespecified comparison later. This recovery proposal is not started.

After a complete comparison, consider a longer learning horizon, stronger opponent diversity or a separately validated value baseline one at a time. The current result is sufficient to show learnability; it is not sufficient to choose those changes from performance evidence.

## Books and review files

The [measured report](../reports/stage3c/REPORT.md) gives the training and stopped-evaluation evidence. The book notation review adds source evidence for the existing red-five candidates while preserving unknown physical identities. The call-scenario intake proposal specifies a small next extraction batch with offered tile, source seat, consumed tiles, existing melds, pass choice and conditional explanation kept explicit. None of the books has entered training.

Useful feedback now is whether to prioritize the evaluation-only recovery, which call situations deserve the next book batch, and whether you want a separate later experiment on always taking a legal win. The current proposal preserves the existing behavior rule and endpoints.
