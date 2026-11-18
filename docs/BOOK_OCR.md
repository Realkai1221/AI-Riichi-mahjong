# Local book prose OCR proof

`scripts/book_page_ocr.swift` implements a local Apple Vision extraction probe for upright rendered PNG pages. It is deliberately separate from the model and scenario-import paths. The first measured run completed eight pages from four books: horizontal prose was partly useful, but the two pages with vertical Japanese explanations yielded no substantive explanation text. This is a working extraction probe, not a reliable book-to-training pipeline.

The probe pins text recognition revision 3 and accurate mode, queries supported languages, and rejects an unsupported requested language without falling back. It requires macOS 13 or later. Japanese pages should request `ja-JP`; simplified Chinese pages should request `zh-Hans`. Language correction is enabled and automatic language detection is disabled. This configuration is recorded in every output.

Example invocation, to be run only under the session's bounded controller:

```sh
/usr/bin/swift \
  -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  -module-cache-path /tmp/riichi-book-ocr-cache-sdk26 \
  scripts/book_page_ocr.swift \
  --outputdir /tmp/riichi-book-ocr-output \
  --language ja-JP \
  /tmp/riichi-book-review/uzaku-300-page-012.png \
  /tmp/riichi-book-review/uzaku-300-page-013.png
```

Use a fresh output directory for each invocation. The program writes one `<input-basename>.ocr.json` per page, rejects duplicate basenames, and refuses to overwrite an existing output. A failed later page retains earlier completed outputs and exits nonzero; a batch is complete only when the controller records successful process completion and all expected files. File writes are exclusive but not atomic, so a killed process can leave an incomplete JSON file; never count an unparseable file as a result. Re-run into a new directory if recovery is necessary.

Each JSON records the SHA256 of the exact image bytes decoded, source dimensions, explicit upright orientation, OS and Vision runtime metadata, recognition settings, elapsed time, recognized text, candidate confidence, and normalized bounding boxes. The script source hash is included when its source remains readable. The controller should separately capture `/usr/bin/swift --version`, the command, total wall time including compilation, exit status, and final file hashes. The per-page elapsed time includes loading, decoding and recognition but excludes initial compilation and output serialization.

Text observations remain in Vision's returned order. Bounding-box coordinates have a lower-left origin, with widths and heights normalized to the image. No algorithm reconstructs Japanese vertical reading order, multi-column order, question/answer pairing, or relationships between prose and diagrams. Missing candidates remain explicit null entries. Confidence is a model score, not a measured accuracy rate. Counts of extracted blocks or characters establish extraction throughput only.

All outputs carry `review_status: unreviewed`, `training_ready: false`, `tile_parsing_performed: false`, and `complete_question_assessed: false`. Recognition may turn tile glyphs into meaningless text; these strings are not tile identities or answer labels. A human review must compare the source page, question scope, solution explanation and relevant diagram before any scenario can leave quarantine. A successful OCR call does not establish correct labels, complete questions, corpus coverage, usage permissions, or training readiness.

## Measured sample and review

The successful run used Apple Swift 6.3.3 (`swiftlang-6.3.3.1.3`, clang `2100.1.1.101`), the explicitly selected installed macOS 26.5 SDK, macOS 26.6 build 25G72, and Vision bundle `com.apple.VN` version 9.5.4. All eight pages requested `ja-JP`. The initial attempt selected SDK 27.0, whose Swift 6.4 interfaces were incompatible with the installed compiler; it failed during compilation and produced no page outputs. Selecting the already installed 26.5 SDK allowed the unchanged script to complete. No toolchain was installed or global selection changed. The explicit SDK path is machine-specific and should be rechecked on another installation.

The failed attempt charged **8.797282 seconds** and the successful compile-plus-recognition attempt charged **27.947761 seconds**, totaling **36.745043 seconds**. Both remain in the session ledger. The successful watchdog sampled a peak process-tree RSS of **1,505,067,008 bytes**; this is a sampled process measurement, not a guaranteed peak or accelerator-memory total. The eight per-page timers total **2.460268 seconds**, excluding compilation, startup and output serialization, so they must not replace the controller total. The resulting eight JSON files contain **270 observations and 1,701 recognized characters**, totaling **122,327 bytes**. These counts include navigation text, watermarks and tile-like noise.

| Rendered PDF page | Blocks | Characters | Per-page seconds | Visual comparison with source |
| --- | ---: | ---: | ---: | --- |
| Uzaku 300, 12 | 48 | 161 | 0.493555 | Question identifiers, some round/seat/turn text and the closing prose appear; tile rows become noise. Some turn labels have character substitutions. |
| Uzaku 300, 13 | 56 | 407 | 0.452495 | All three horizontal explanation areas contribute readable prose. Embedded tiles, answer tiles and count formulas are corrupted or incomplete; sidebar numbers interrupt the text stream. |
| Uzaku 301, 10 | 52 | 189 | 0.275013 | Question/context labels and closing prose appear, with fragmented ordering. Hand diagrams are not transcribed reliably. |
| Uzaku 301, 11 | 48 | 351 | 0.363913 | Horizontal explanation paragraphs are substantially represented. Context labels, embedded tiles and some mahjong terms contain substitutions. |
| Super Digital, 29 | 11 | 50 | 0.148417 | No substantive vertical explanation recovered. Output consists chiefly of watermarks, diagram labels, tile-like noise and page numbering; the large vertical title is also absent. |
| Super Digital, 30 | 3 | 31 | 0.109002 | No substantive vertical explanation recovered. Only watermark/navigation fragments appear despite extensive visible prose. |
| Statistics, 44 | 30 | 134 | 0.259674 | Question headings and turn/context labels appear; the heading has a lexical error. Tile diagrams do not yield usable tile identities. |
| Statistics, 45 | 22 | 378 | 0.358199 | Horizontal answer paragraphs and several numerical claims appear. Answer tiles are missing; prose/action text is interleaved and includes lexical errors and altered negative-value notation. |

An independent model-free review opened all eight source images and compared their layout and prose with every returned JSON. It verified JSON parsing, source-image hashes and dimensions, source-packet associations and live PDF hashes, logged output hashes, script hash, runtime/settings consistency, consecutive observation indices, block/character counts, finite box coordinates and confidence ranges. Four angled watermark boxes extend slightly beyond the right edge, reaching a maximum normalized right extent of approximately 1.00223; the original geometry is preserved without clamping and listed in the summary. The review did **not** create a complete ground-truth transcription or measure character accuracy, precision, recall or label correctness. Confidence scores do not remedy missing vertical text.

The measured recommendation is to keep local Vision as a draft aid for horizontal prose with mandatory source comparison. The present full-page configuration is unsuitable for these vertical pages. A later separately budgeted comparison could test a vertical-text-capable recognizer or carefully designed layout preprocessing against a manually reviewed reference. Simply treating these outputs as complete questions, or assuming a rotated page will solve vertical Japanese recognition, is unsupported. Preserve original image provenance and existing source packets; OCR does not replace them.

The machine-readable summary contains hashes, source associations and descriptive coverage observations without reproducing raw book text. Raw OCR remains under `Books/pilot/ocr-ja-v1-sdk26/` and every output retains its unreviewed status. The failed attempt record, failure log, successful attempt record, successful log, and compiler-version record preserve the failure/retry history.

The implementation uses the installed macOS SDK interfaces in `Vision.framework/Headers/VNRecognizeTextRequest.h`, `VNRequest.h`, `VNObservation.h` and `VNRequestHandler.h`, plus Foundation's exclusive `Data.write(..., .withoutOverwriting)`. No external service, network request, PDF scraping or training is performed by this script.
