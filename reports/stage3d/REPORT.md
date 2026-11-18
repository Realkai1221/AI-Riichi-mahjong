# Stage 3D measured results — 21 October 2026

The evaluation-only recovery completed **32 fresh paired blocks / 256 full matches**. Both fixed policies averaged **2.6328125 place**. The mean candidate-minus-control utility difference is **0**, with the predeclared two-sided 95% Hoeffding interval **[-1.44048418698, +1.44048418698]**. No playing-strength improvement is demonstrated and no policy is promoted.

The [review and next-step proposal](../../docs/STAGE3D_REVIEW.md) explains the implications. The [plan](../../docs/STAGE3D_PLAN.md), [configuration](../../configs/stage3d.json) and frozen preparation record establish the scope before formal play.

## Experiment identity and completed scope

| Item | Measured or fixed value |
|---|---|
| Candidate | Stage 3C batch-3.pt, four call-head updates |
| Control | Stage 3C initial.pt, initialized call heads with the same frozen base |
| Opponents | Three original Stage 2B deterministic argmax policies |
| Utility | `[1.5, 0.5, -0.5, -1.5]` from complete-match placement |
| Design | 32 fresh wall blocks ×4 focal seats ×2 arms; shared sampling seed per pair |
| New matches | 256 complete, zero partial/unresolved admissions |
| New optimizer updates / checkpoints | 0 / 0 |
| Failed blocks / replacements | 0 / 0 |
| Historical evaluation pooled | None |
| Runtime | CPU, one Torch thread, single-row fp32 network with staged CPU fp64 probabilities |
| Source freeze | Explicit file membership; implementation commit `2e1887e8fe1cc6e9bb50baf40d6772f85ff03cc3` |

The configuration SHA-256 is `643ebf51fd6d6daee9e0fd8819bf6d20dac9f9455ea69a2aa9dfe3e73770443f`. The preparation record SHA-256 is `0752be666c0dffa4bf59ce6bc35855dfae408cc00faa4237eba3e4e469ae571c`. The original behavior protocol remains separately recorded in every trajectory; the recovery campaign does not change the saved policies' behavior definition.

Both input checkpoint files stayed byte-identical:

- Candidate: `4b14be5956c53a7665823a23f9d91002d0264b4687482133bb3feb106f91a967`.
- Initialized control: `3baca74ddcb011ea7346396621c709e7499835c2cf4b9e6e201688aa52e75bf2`.
- Original opponent/reference: `5130185ba8a397b84446ba8dc2d02f068a9448db5c0373f2aeb357e4af525a6f`.

The Stage 3C **383-file historical snapshot** verifies unchanged at commit `fbba75bc5e2301b67f1addd68844d3b4589cbfdf`. Its stopped evaluation stays incomplete and separate. See historical verification.

## Fixed endpoint

| Measure | Four-update candidate | Initialized control |
|---|---:|---:|
| Full matches | 128 | 128 |
| Hands, including dealer continuations | 1,245 | 1,245 |
| Mean placement | 2.6328125 | 2.6328125 |
| First / second / third / fourth | 21 / 38 / 36 / 33 | 21 / 38 / 36 / 33 |
| Learner decisions | 19,334 | 19,308 |
| Call-opportunity rows | 3,943 | 3,930 |
| Multiple-call opportunity rows | 564 | 560 |
| Post-call learner rows | 1,412 | 1,413 |
| Open-hand learner rows | 1,386 | 1,387 |

All 32 block utility differences are zero. The interval uses the fixed block-level bound `D ∈ [-3,3]` and half-width `6*sqrt(log(40)/(2*32))`. The four seat rotations are averaged within each block; 256 games are not treated as 256 independent replicates. The interval is wide and does not establish equality or a useful noninferiority margin. Machine-readable endpoint.

In **127 of 128 pairs**, every physical action and final score matched. In one pair—block 2, focal seat 1, match IDs 338/339—the candidate selected PASS while the control selected PON at response step 197. Subsequent play and scores diverged, but every seat's final placement matched. All 128 focal utility differences are therefore zero. This is evidence of a small behavioral change, not proof that the policies are identical.

## Descriptive call behavior

| Kind | Candidate requested / resolved | Control requested / resolved |
|---|---:|---:|
| CHI | 100 / 95 | 99 / 94 |
| PON | 73 / 72 | 75 / 74 |
| DAIMINKAN | 2 / 2 | 2 / 2 |
| ANKAN | 6 / 6 | 6 / 6 |
| KAKAN | 0 / 0 | 0 / 0 |
| Total | 181 / 175 | 182 / 176 |

Requested responses are distinct from actions selected by the engine's response resolution. Opportunity categories can overlap. These populations differ after the one action divergence, so per-arm counts alone are not a causal estimate of improved calling.

The saved-data analysis compares probabilities only where paired histories, player-visible inputs, candidate ordering and random draws match. It includes the first divergent response and excludes all later states in that pair:

- **19,218** matched learner rows in total.
- All **3,913** matched call-opportunity rows changed probability; mean call-mass change **-0.00018448185136** (about -0.01845 percentage points).
- Maximum absolute per-action probability change **0.00019941584069**.
- All **15,305** matched rows without call candidates had exactly unchanged distributions.
- One selected action changed in these matched states.

Both arms selected RON in all **four win-versus-call opportunities per arm**. Candidate average actual win mass was 0.95018253 versus 0.95000000 for the control. Thus the gate still assigns probability away from a legal win in these states; four sampled wins per arm do not establish that this design concern is resolved. KAKAN remains unexercised in this evaluation. Saved-data behavior analysis.

## Validation and resource accounting

The final preflight passed **91 tests**. Earlier passing preflight evidence remains saved; the release run incorporates the final directory-synchronization fix and independent audit tests. After formal games, **15 analysis tests** and **3 audit-closure tests** passed. These targeted suites validate the changed paths; they are not a full engine rule certification.

The independent full audit passed **2,115,700 data-level assertions** over all 256 raw JSON/NPZ pairs and their lifecycle records. It checks actor/provenance boundaries, terminal score reconstruction, sampling evidence, evaluation-only eligibility, ordinal wall commitments, fixed endpoints, source identity and budget history. Policies consume audited player-visible features; complete engine logs remain auditor inputs. No frozen-opponent action becomes learner training data.

After the leased audit completed, a hash-only closure revalidated **1,152 evidence inputs** and added **1,451 integrity/bookkeeping checks**. It retained the prior numerical findings and verified that precisely the audit's own successful operation closed; it did not rerun trajectory arithmetic outside the budget. Full audit prefix, final audit and closure.

| Resource | Recorded result | Cap |
|---|---:|---:|
| Cumulative numerical time | **350.845432457 s** | 400 s |
| Preparation before formal matches | 17.605166833 s | 45 s |
| Fixed 32-block evaluation | 323.319083291 s | Included above |
| Local book text extraction + six renders | 12.692085708 s | Included above |
| Closed operations | 41 successful, 0 failed | One lease at a time |
| Sampled process-tree RSS peak | 613,236,736 bytes, about 0.571 GiB | 32 GiB |
| Artifact bytes at audit closure | 266,340,901, before final report/inventory packaging | 500,000,000 bytes |
| Remaining numerical allowance | 49.154567543 s, unused | Includes 30 s shutdown reserve |

The closed ledger includes book preparation, both preflights, endpoint loading, all evaluation blocks, analysis/tests and the full audit. Model computation, test execution and extraction/rendering were leased. Subsequent source/report editing and hash-only packaging are separate administrative work. Sampled RSS is not a continuous system-memory measurement. The final inventory records exact final file sizes and hashes.

Recorded environment: Apple M5 Pro / 64 GB, macOS ARM64, project `.venv` Python 3.14.8; Torch 2.14.1, NumPy 2.5.4, RiichiEnv 0.4.10, mahjong 2.0.0, psutil 7.2.2 and pytest 9.1.1. Dependency pins remain in [requirements-macos-arm64.lock](../../requirements-macos-arm64.lock). No MPS/CUDA throughput or competitive-strength conclusion is inferred from this CPU run.

## Books and files for review

Book work prepared **eight locators across six families** and **two detailed partial source packets** from the supplied Seventh9 revision. Six pages were visually inspected. One packet describes declining a red-dragon call; the second chooses 6p+8p consumption for a CHI on 7p conditional on calling. The first question excludes dora information supplied only by its explanation. Unknown state, physical-copy ambiguity and attributed strategic recommendations remain explicit.

The earlier book artifacts and source PDF remain unchanged. The new records stay local under `Books/stage3d/`, outside Git. They are not certified legal-action sets, complete policy observations or adopted training labels. No book training, model evaluation on books, OCR expansion, external acquisition, account registration or paid compute occurred. Book review and local source links.

For review, start with [STAGE3D_REVIEW.md](../../docs/STAGE3D_REVIEW.md). It proposes a fixed longer learning horizon and a strict partial-call book validator, both unstarted. The present comparison isolates four updates from an initialized call policy; it does not estimate the net effect of the call gate versus the original no-gate reference or strength against expert opponents.

The final bundle is recorded in manifest.json and checked by inventory-verification.json. Recheck its live identities with `.venv/bin/python scripts/stage3d_inventory.py --verify`. Do not rerun numerical commands into the preserved campaign directories.
