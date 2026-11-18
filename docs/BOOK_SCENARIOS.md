# Book scenario intake and evaluation

Mahjong exercises with published answers can help identify specific weaknesses, especially when the current heuristic teacher cannot supply richer strategy. The first useful application is a separate, reviewed scenario benchmark. A printed answer is a source claim under the exercise's assumptions; it is not automatically a unique optimum for our average-rank objective `[1.5, 0.5, -0.5, -1.5]`.

**Implemented on 18 October 2026:** a small local intake format, structural validator, and original format example. Five supplied books have been assessed, with three Seventh9 partial exercises kept in quarantine and six richer question/explanation packets preserved separately for review; see the source assessment. No book-derived training has run and no scenario has been evaluated by a model. The first version covers discard exercises. Calls, riichi choices, and multi-step questions need a later schema extension with appropriate legal-action details.

## What to provide

For additional sources, retain the title, author, edition, and a small set of exercise pages with their answer pages. A PDF, image, or structured text can work. Page and problem numbers must remain attached to each transcription. Record the intended local use and its permission basis; owning a copy does not itself establish every training or redistribution permission. We will not upload source pages or publish excerpts as part of intake.

Start with a small, varied sample, such as 10–20 exercises, before transcribing a whole book. Preserve the printed hand, red tiles, drawn tile if shown, question, answer alternatives, assumptions, and reasoning. Unknown information stays unknown. OCR and diagram extraction should be checked against the supplied page before any label is accepted.

## Two kinds of position

| Intake kind | What it represents | Permitted interpretation |
|---|---|---|
| `partial_exercise` | Explicitly supplied concealed tiles and optional drawn tile, plus a list of missing context. | A question about the stated partial hand. It may support a constrained efficiency exercise, but cannot be scored as a complete policy decision. |
| `player_view_reference` | A path and SHA-256 for a separately prepared canonical player-view candidate, with its schema and explicit rule profile. | A proposed complete visible state. The intake validator does not open the reference, certify completeness, check legal actions, or reconstruct an engine state. |

A complete policy evaluation needs the actor's hand and public context: scores, seat and round, continuations and deposits, visible discards and melds, riichi declarations, dora indicators, relevant preceding events, and the exact legal candidate set. A complete **player-visible** state does not require exposing opponents' concealed hands or the wall. Any simulator world used for replay or value estimation must be kept in the auditor layer and must be consistent with that visible state.

Partial exercises should retain their own evaluation category. Do not fill missing scores, dora, rivers, or drawn tiles with convenient defaults and then report agreement as full-policy accuracy. If an explicit hypothetical completion is useful, save it as a new variant, state the assumptions, and keep it in the original problem family.

## Record format

The [authored JSONL example](../examples/book_scenarios/example.jsonl) is a template, not copied book content. Each line is one `riichi-ai-book-exercise-v1` record. Required sections are:

| Section | Required meaning |
|---|---|
| `source` | Source kind, stable book identity, title, authors, edition, chapter, page, and problem number. Related translations and reprints also need shared split groups. |
| `permissions` | Recorded basis and separate booleans for local training, local evaluation, and redistribution. These are provenance assertions, not a legal determination by the validator. |
| `split` | Role (`quarantine`, `train`, `validation`, or `heldout`), book group, chapter group, problem family, and optional parent variant ID. Newly transcribed records should start in quarantine. |
| `objective` and `rules` | The exercise's objective, explicit rule profile if available, and assumptions. Expected points, winning probability, tile efficiency, and average match rank are distinct targets. |
| `state` | Only supplied visible information or the reference to a candidate complete view; missing context is explicit. |
| `source_answer` | The source's acceptable-action set, rationale, and precise claim. Preserve alternatives rather than choosing an arbitrary single answer. |
| `label_review` | A separate pending, supported, or disputed review, with reviewer, method, evidence, and reviewed acceptable actions. Never silently replace the author's answer. |

Tiles use `1m`–`9m`, `1p`–`9p`, `1s`–`9s`, red fives `5mr`, `5pr`, `5sr`, and honors `E`, `S`, `W`, `N`, `P`, `F`, `C`. Discards distinguish a drawn tile, held tile, or an answer that does not specify that distinction. Red and ordinary copies remain distinct in the answer but share the four-copy tile-type limit.

The first version intentionally cannot represent the whole variety of book diagrams. An open-hand exercise, call problem, or richer public-state diagram should remain unconverted until the schema can preserve its relevant information. Do not omit depicted information just to fit the format.

## Answer separation and split protection

`exercise_inputs()` returns only the supplied hand and drawn tile for a future partial-exercise evaluator. Source answers, explanations, page identifiers, permissions, review results, and missing-context assumptions are excluded. This output is not compatible with the existing full policy encoder. `player_view_reference` records require a separately audited adapter before any actor input is produced.

The collection validator conservatively keeps each whole book group in one active split. Chapter groups and problem families cannot cross training, validation, and held-out roles either. All variants, suit permutations, changed-score versions, translations, and paraphrases of a problem belong to the same family. Variant parents must exist in the collection, and cyclic references are rejected. An exact-content check catches identical partial hands with reordered tiles and matching context, rules, and objective. It cannot recognize all semantic paraphrases; manual grouping remains necessary.

Hold out books or problem families before inspecting model results. A collection used to choose learning rates, modify features, or select checkpoints is development data, even if it was previously named held-out. Preserve an untouched confirmation collection for later evaluation. Do not mix book questions with synthetic full-match records in an aggregate metric without separate denominators and clear task definitions.

## Validation and later integration

From the project environment:

```text
.venv/bin/python -m riichi_ai.book_scenarios examples/book_scenarios/example.jsonl
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_book_scenarios.py'
```

The validator checks required fields, tile spelling and copy limits, supplied-hand consistency for discard labels, review metadata, permissions for assigned roles, duplicate IDs, and split/variant consistency. It rejects duplicate JSON keys and nonstandard JSON numbers. It uses only the Python standard library. Every report currently returns `policy_training_ready: false`; passing this check does not launch training, follow references, or validate the strategic answer.

The next integration steps depend on the supplied material:

1. Transcribe and visually verify a small sample; retain source references and the original answer separately from any interpretation.
2. Review whether each answer follows from the stated objective. For efficiency-only questions, use independent shanten/ukeire checks where applicable. For strategic claims, record whether evidence is expert reasoning, simulation under declared assumptions, or a stronger reference policy. Disagreements remain visible.
3. Classify partial exercises separately from complete player-view candidates. Audit any conversion into `PlayerView`, its legal action mapping, and the actor information boundary before using our policy.
4. Freeze groups, roles, labels, and denominators. Evaluate acceptable-set agreement by exercise type, including abstentions and unsupported questions. Agreement with a book is distinct from measured playing strength.
5. Only after that review, propose a bounded supervised experiment using explicitly permitted training records. Keep reviewed evaluation groups out of optimization and maintain synthetic-data regression checks.

Book exercises can add useful strategic targets and interpretability. Their selection, missing context, and educational simplifications make them a complement to complete-match evaluation, not a replacement for it.
