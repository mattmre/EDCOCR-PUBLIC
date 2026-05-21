# International OCR Limitations

The low-resource language tranche expands the model baseline beyond the
original 27-language set, but searchable-PDF text-layer behavior is still
constrained by the current Helvetica-based insertion path in `ocr_gpu_async.py`.

## Right-to-Left (RTL) Scripts
- **Arabic, Hebrew, Farsi, Urdu**: PaddleOCR extracts text correctly but text overlay positioning in the PDF uses LTR coordinates. The invisible text layer (render_mode=3) positions text at bounding box coordinates — for RTL scripts, the reading order in the PDF text layer may be reversed.
- **Impact**: Search functionality in PDF viewers may not work correctly for RTL text. Extracted .txt files should be correct.
- **Affected models**: ar, fa, ur, ug
- **Remediation**: Apply RTL text direction metadata when inserting text layer

## Vertical CJK Text
- **Japanese, Chinese**: A conservative post-processing path now reorders
  clearly vertical CJK columns right-to-left before text emission.
- **Current limitation**: Mixed pages with both horizontal and vertical CJK
  blocks may still need analyst review because the heuristic only activates
  when vertical lines are the clear majority of the page.
- **Impact**: Pure vertical pages should now preserve intended column order more
  reliably, but mixed-layout pages may still read imperfectly.
- **Affected models**: japan, ch, chinese_cht
- **Remediation**: Continue improving mixed-layout detection and table/form
  handling for vertical blocks

## Mixed-Script Pages
- **Issue**: FastText language detection operates on the majority language of extracted text. Pages with multiple scripts (e.g., English headers with Japanese body, Arabic with French) will use only one OCR model.
- **Impact**: Minority script text will be OCR'd with wrong model, producing garbage text or low confidence
- **Current threshold**: 40% confidence for language switch (detect_language_from_text)
- **Remediation**: Per-region language detection (requires PP-StructureV3 layout analysis first, then per-region OCR)

## Diacritical Marks
- **Affected languages**: Hungarian (ő, ű), Polish (ą, ę, ź), Czech (ř, ů), Romanian (ș, ț), Turkish (ğ, ş), Vietnamese (ấ, ề, ở)
- **Issue**: PDF text overlay uses Helvetica font (fontname="helv") which may not contain all diacritical combinations
- **Impact**: Characters may render as replacement glyphs or be dropped from text layer
- **Remediation**: Use a Unicode-complete font (e.g., DejaVu Sans, Noto Sans) for text insertion

## Indic Scripts
- **Affected**: Devanagari (Hindi, Marathi), Tamil, Telugu, Bengali, Kannada
- **Issue**: Complex ligature rendering — multiple Unicode codepoints combine into single visual glyphs. Helvetica font cannot render these.
- **Impact**: Text overlay will be garbled; extracted text may be correct but PDF text layer unusable
- **Remediation**: Use Noto Sans Devanagari/Tamil/etc. for text insertion; requires font embedding in PDF

## Georgian
- **Affected**: Georgian
- **Issue**: OCR model support can extract text, but the current PDF text layer
  still relies on Helvetica and may not preserve glyph fidelity.
- **Impact**: Extracted text output should be usable, while searchable PDF text
  rendering may degrade or drop characters.
- **Remediation**: Wire `font_selector.py` into PDF text insertion and embed
  `NotoSansGeorgian-Regular.ttf` for Georgian pages.

## Low-Resource Tranche Status
- **Current baseline**: 34 pre-baked language models
- **Added in the current tranche**: fa, ur, ug, ta, te, kn, ka
- **Still open for **: broader low-resource expansion follow-on,
  PDF font embedding integration, and runtime validation against representative
  corpora for the new scripts

## Unicode Normalization
- **Issue**: No NFC/NFD normalization is applied to extracted text
- **Impact**: The same visual character (e.g., é) may be stored as single codepoint (U+00E9) or combining sequence (U+0065 + U+0301) depending on OCR engine
- **Impact on search**: PDF text search and downstream text processing may miss matches
- **Remediation**: Apply `unicodedata.normalize('NFC', text)` before writing text output

## Bi-directional Text in Tables
- **Issue**: Tables containing both LTR and RTL text (e.g., Arabic invoice with English product codes) may have cell content ordering issues
- **Remediation**: Requires table-level RTL detection and per-cell direction metadata
