# Stage 2B collection observations

The initial focal corpus completed **300 matches, 2,858 hands and 48,088 focal decision rows**, with **191,475 requests across all four seats**. All 300 requested matches completed; none was rejected, truncated or omitted. Match seeds and complete-game hashes are each unique across the 300 matches. These are collection and integrity findings, not model accuracy or playing-strength results.

The evidence is the collection manifest, verification report and resource report. The external dataset manifest has SHA-256 `1e82fcac042676abf1e25360250605154ed9549ca74daf0cb429f69d1a2728fe`; the formatted report copy has SHA-256 `af750f80c8c1978a69521e2c1bcb7af39bebd126d989030a65546c8220af9797`. Their parsed contents are equal; their serialization differs.

## Split and population inventory

| Group | Matches | Focal rows | Hands | Focal seats 0/1/2/3 |
|---|---:|---:|---:|---|
| Historical-student diagnostic | 100 | 16,337 | 964 | 25/25/25/25 |
| Teacher validation | 50 | 8,049 | 486 | 13/13/12/12 |
| Historical-student validation | 50 | 7,975 | 470 | 13/13/12/12 |
| Teacher test | 50 | 7,654 | 459 | 13/13/12/12 |
| Historical-student test | 50 | 8,073 | 479 | 13/13/12/12 |

Validation therefore contains 16,024 rows, and test contains 15,727. Equal match allocation does not imply equal row counts; selection must continue to average the two strata's rates equally. There are **zero training matches in this initial fresh corpus**. The diagnostic, validation and test groups are not extra training data. Test counts above describe inventory only: this review ran no model predictions, action-agreement measurements or test-performance analysis.

## Behavior, targets and integrity

The collector used implementation commit `a28a19c559542dfe32713943eb45fc9f6e1bf312`, with its tracked worktree clean and source fingerprints unchanged during collection. Dependencies were Python 3.14.8, RiichiEnv 0.4.10, NumPy 2.5.4, PyTorch 2.14.1 and mahjong 2.0.0. The recorded rule-profile hash is `c3f043568af2dcefc1be8de65fd241311e53347e56401f3ac09f5edcde3dbe98`.

The teacher stratum acted with `shanten_v2`; the historical-student stratum acted with checkpoint SHA-256 `09d42bf7c2b881f57d76f4ed38b33fa04070a90c7294d183338e5d0a30d1537d`. Every focal row was separately labeled by `shanten_v2` from its pre-action view. The clockwise opponent panel remained `shanten_v2`, `defense_v2`, `shanten_v2`. Loaded checkpoint identities and all auditor panel identities were checked; the checkpoint stayed unchanged through collection. Final-rank targets belong to the recorded actual behavior.

Verification checked **900 artifacts and every 48,088 numeric row**, then reproduced one full match from each of the five groups: five matches comprising 889 focal rows. All checks passed with no reported errors. Replays compared actor, target and complete auditor artifact bytes. This checks internal reproducibility; it is not an independent rules certification or a replay of all 300 matches.

There are 3,175 physically forced rows and 3,333 semantically forced rows, leaving **44,755 optional semantic decisions**. The difference of 158 reflects physical alternatives collapsing into one semantic class. Aggregate behavior and teacher **action-kind** counts coincide: 36,497 discards, 10,161 passes, 938 riichi requests, 330 ron requests and 162 tsumo requests. This does not establish agreement on discard identities or membership in the teacher's best set; those are separate model-performance measurements.

## Coverage limits

The actual canonical event streams contain **no executed chi, pon or kan**. Focal legal-candidate inventories nevertheless include 8,430 chi classes, 3,836 pon classes, 261 ankan classes and 183 daiminkan classes, plus ten nine-terminals abort candidates. There are no kakan candidates. Candidate counts count offered semantic classes, not distinct decisions or executed transitions. The corpus can test rejection of offered calls; it does not demonstrate correct behavior after taking those calls or validate unseen kan paths.

Observed settlement counters include **14 double-ron settlements across 13 matches**, 530 exhaustive draws, nine four-riichi aborts and five four-wind aborts. The logs contain 2,328 winning events and 544 draw/abort events; multiple winners explain why winning-event plus draw-event counts exceed the number of hands. No triple-ron abort was observed. Chankan, rinshan, haitei, houtei, nagashi mangan, pao and riichi refusal due to remaining wall length remain explicitly **not instrumented**, with unknown counts rather than zero. These figures describe this particular closed-hand policy mixture.

## Collection resources

Four bounded workers collected the corpus in **44.10 seconds**. Verification took **4.89 seconds**, and the full collection command took **49.06 seconds**. Sampled peak aggregate process RSS was **1,428,078,592 bytes** (about 1.33 GiB). Numeric training arrays occupy 179,494,949 uncompressed bytes; the collector wrote 24,883,501 bytes before its final manifest. Combined Stage 2B data and reports occupied 26,692,254 logical bytes immediately before writing the collection resource report. These are collection measurements, separate from feature-derivation and training time, and are not sustained inference benchmark results.
