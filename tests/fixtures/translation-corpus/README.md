# Translation Test Corpus

Seed data from FLORES-200 (CC-BY-SA-4.0).
https://github.com/facebookresearch/flores

## Attribution

FLORES-200: Goyal et al., 2022. "The Flores-200 Evaluation Benchmark for
Low-Resource and Multilingual Machine Translation."
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

## Contents

- `flores200/` -- sentence pairs sampled from the devtest split.
  - Each file: `<src>_<tgt>.tsv` with columns: `src_text TAB tgt_text`.
  - Minimum 10 pairs per language direction.

## Loader

Use ``tests.fixtures.translation_corpus.load_flores_pairs(src, tgt)`` to
load the pairs in tests; the loader returns ``[]`` for unknown
directions so callers can degrade gracefully.
