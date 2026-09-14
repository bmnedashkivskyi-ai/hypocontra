# Round 14: PDF-based pipeline for ACL-only surveys — design

Status: approved by user 2026-09-09, ready for implementation plan.

## Context

Round 13 (`docs/next_scaling_directions.md`, item #1 as ranked after round 12)
tried ACL Anthology as a candidate source, but scoped to the 23/606 ACL
survey-like titles that are *also* on arXiv (`results/acl_anthology/acl_found_on_arxiv.csv`),
so the existing LaTeX `.bbl`/`.bib` pipeline needed no changes. Yield was low
(1 new pair, 1 duplicate of an existing round-9 dispute, 5 false-positive
skips) because that subset mostly overlaps what the round 9-12 cs.CL arXiv
title search already found.

The unexploited reserve is the other **285 titles**
(`results/acl_anthology/acl_not_on_arxiv.csv`) — ACL/EMNLP/NAACL/COLING
survey papers with no arXiv presence, available only as PDF, no LaTeX
source. This spec covers building a new front-end that mines and resolves
citation pairs from these PDFs, feeding into the existing round-based
Claude/Gemma pipeline unchanged.

Corpus state going in: 88 curated pairs, combined round 7-13 kappa=0.590
(n=70).

## Goals

- Produce `results/candidate_sentences_resolved_acl_pdf.csv` in the exact
  schema `resolve_survey_citations.py`'s `resolve_row()` already produces,
  so `prepare_claude_pair_batch.py --input candidate_sentences_resolved_acl_pdf.csv --round 14`
  runs with zero code changes to any existing downstream script.
- Process all 285 candidates in one run (user's explicit choice — no pilot
  subset), with enough logging/caching that a mid-run interruption (as
  happened in round 13) costs nothing to resume.
- Reuse existing infrastructure wherever it already fits: the disagreement
  regex taxonomy (`DISAGREEMENT_PATTERNS`/`EXCLUSION_PATTERNS`/`ARTIFACT_PATTERN`),
  the `monitor_progress.py` `[i/N]` convention, the OpenAlex/Crossref
  `mailto` pattern already used in the sibling Kros project's bibliography
  verification scripts (`CONTACT_EMAIL = "b.m.nedashkivskyi@gmail.com"`).

## Non-goals

- No changes to `prepare_claude_pair_batch.py`, `ingest_claude_pairs.py`,
  `prepare_claude_label_batch.py`, `ingest_claude_labels.py`,
  `annotate_with_gemma.py`, or `merge_and_compute_kappa.py`.
- No changes to the taxonomy or disagreement-marker regex list itself
  (only imported and reused with a different citation-detection regex).
- No attempt to achieve `.bbl`-level citation-resolution precision — PDF
  reference-list parsing is explicitly heuristic and is expected to have a
  non-trivial drop rate, logged like every other filtering step in this
  project (never silent).

## Architecture

Three new scripts, run in sequence, followed by the existing round
machinery starting at round 14:

### 1. `src/fetch_acl_pdfs.py`

- Input: `results/acl_anthology/acl_not_on_arxiv.csv` (`key, year, title, url`).
- For each row, download `<url>.pdf` (ACL Anthology's landing-page URL plus
  `.pdf` is the direct PDF link) to `results/acl_pdfs/<key>.pdf`.
- Skip rows whose PDF is already cached (resumable).
- `[i/N]` progress printing (monitor_progress.py-compatible), politeness
  delay between requests, exponential backoff on HTTP errors matching the
  pattern already used in `check_acl_arxiv_overlap.py`.
- Failures logged to `results/acl_pdf_download_failed.csv` with a reason
  column — never silently skipped.

### 2. `src/mine_disagreement_from_acl_pdfs.py`

- Input: PDFs cached by step 1.
- New pip dependency: `pymupdf` (better multi-column reading-order handling
  than the already-installed `pypdf`). Extract text per PDF via
  `fitz.open(path)` / `page.get_text("text")` (or `"blocks"` if plain text
  reading order proves unreliable on inspection — implementer's call during
  execution, both are compatible with the sentence-splitting step below).
- Split into pseudo-sentences with the same
  `re.split(r"(?<=[.!?])\s+", text)` approach `mine_disagreement_from_surveys.py`
  uses for LaTeX (keep it consistent rather than introducing spaCy
  segmentation here — spaCy is a project dependency already, but changing
  segmentation approach for one source and not the other would make round
  comparisons noisier for no clear benefit).
- Reuse `COMBINED_PATTERN` (from `DISAGREEMENT_PATTERNS`) and
  `EXCLUSION_PATTERN` by importing them from `mine_disagreement_from_surveys.py`
  (refactor that module minimally if needed so the patterns are importable
  without triggering its `main()` / arXiv-specific state) rather than
  duplicating the regex list.
- New citation-detection regex (replaces `CITE_PATTERN`, which only matches
  `\cite{key}`): match parenthetical author-year citations, e.g.
  `(Smith, 2020)`, `(Smith et al., 2020)`, `(Smith and Jones, 2020)`,
  including `;`-separated clusters `(Smith, 2020; Jones, 2019)`, plus the
  narrative form `Smith et al. (2020)`. Exact regex is an implementation
  detail — validate against a handful of real extracted ACL sentences
  before trusting it at scale.
- Same `ARTIFACT_PATTERN` filter reused as-is (table/figure/tikz artifacts
  — less relevant for PDF text than LaTeX, but cheap to keep for safety
  since PDF extraction can produce similar table-fragment garbage).
- Output: same shape as `survey_disagreement_sentences*.csv`
  (`file, sentence, cited_keys, survey_arxiv_id, survey_title`), written as
  `results/survey_disagreement_sentences_acl_pdf.csv`. Here `cited_keys` are
  the raw matched citation strings (e.g. `"Smith, 2020"`), not `.bbl` keys —
  resolved to a stable key format in step 3.
- `[i/N]` progress printing per PDF.

### 3. `src/resolve_acl_pdf_citations.py`

- Input: `results/survey_disagreement_sentences_acl_pdf.csv` (step 2) +
  the cached PDFs (step 1) for parsing each paper's own reference list.
- Per paper (cached across all its candidate sentences, not re-parsed per
  sentence): extract the References/Bibliography section text and split it
  into individual entries via a heuristic (paragraph-break or
  hanging-indent boundary detection; entries are expected to roughly match
  `<authors>. <year>. <title>. In <venue>.` for ACL-style bibliographies).
  Parse each entry into `(first_author_surname, year, title)`.
- Match each in-text citation from step 2 against that paper's parsed
  reference list by `(first_author_surname, year)`. On no match or
  ambiguous match (same surname+year, multiple entries — a real risk with
  common surnames), log a drop reason (`citation_unmatched` /
  `citation_ambiguous`) rather than guessing.
- For matched entries, look up an abstract:
  1. **OpenAlex** (`https://api.openalex.org/works?search=<title>&mailto=b.m.nedashkivskyi@gmail.com`,
     same `CONTACT_EMAIL` constant as `verify_bibliography.py`). Reconstruct
     plain text from `abstract_inverted_index` (OpenAlex doesn't return
     abstracts as plain text — this project's existing verification script
     doesn't do this reconstruction yet, so it's new code, but the API
     shape is already known from that script).
  2. **Crossref fallback** if OpenAlex has no abstract
     (`https://api.crossref.org/works?query.bibliographic=<title>`, same
     `User-Agent: ...(mailto:b.m.nedashkivskyi@gmail.com)` header pattern).
     Crossref abstracts, when present, carry JATS-ish XML tags — strip them.
  3. If neither has an abstract, leave the `abstract` field empty (matches
     existing precedent: `prepare_claude_pair_batch.py`'s instructions
     already tell Claude to use `skip_reason=insufficient_context` when a
     field is empty — no new failure mode for downstream steps).
  - Accept a lookup match only above a title-similarity threshold (e.g.
    normalized token overlap or a simple ratio via `difflib.SequenceMatcher`)
    to avoid attaching the wrong paper's abstract to a similarly-titled
    reference. Below threshold: treat as "not found," same as no lookup hit.
  - Cache every OpenAlex/Crossref response to disk (e.g.
    `results/acl_pdf_bib_lookup_cache.json`, keyed by normalized title) so
    a re-run after interruption doesn't re-query already-resolved titles.
  - Rate limiting: same 0.4s-between-calls pattern as `verify_bibliography.py`.
- Apply the same `MIN_KEYS=2` / `MAX_KEYS=6` gate as `resolve_row()` in
  `resolve_survey_citations.py` (fewer than 2 resolved citations in a
  sentence isn't a pair candidate; more than 6 is almost always a
  related-work list, not a real two-way contrast).
- Output: `results/candidate_sentences_resolved_acl_pdf.csv`, matching the
  exact column set `resolve_survey_citations.py` produces (`sentence_id,
  file, sentence, cited_keys, survey_arxiv_id, survey_title,
  cited_keys_split, n_cited_keys, n_resolved, resolved_titles,
  resolved_authors, resolved_years, resolved_abstracts, resolved_context,
  resolution_source`), FIELD_SEP (`\x1f`)-joined multi-value columns exactly
  as today. `survey_arxiv_id` holds the ACL key (e.g.
  `ollagnier-2026-antisocial`) instead of an arXiv ID — every downstream
  script already treats this column as an opaque survey identifier, never
  parses it as an arXiv-formatted ID.
- Dropped/unresolved rows logged to
  `results/citation_resolution_dropped_acl_pdf.csv`, same shape as the
  existing `citation_resolution_dropped*.csv` files, with reasons including
  the two new PDF-specific ones (`citation_unmatched`,
  `citation_ambiguous`) alongside the existing `single_cite_key` /
  `likely_latex_artifact` reasons where they still apply.

### 4. Existing round machinery (round 14)

No new code. Run in order:
`prepare_claude_pair_batch.py --input candidate_sentences_resolved_acl_pdf.csv --round 14`
→ Claude constructs pairs → `ingest_claude_pairs.py --round 14` →
`prepare_claude_label_batch.py --round 14` → Claude blind-labels →
`ingest_claude_labels.py --round 14 --annotator A` →
`annotate_with_gemma.py --round 14` → `merge_and_compute_kappa.py --round 14`.

## Storage / gitignore

- `results/acl_pdfs/` (raw PDF cache, ~285 files) — gitignore, same
  treatment as `results/survey_sources/` and `results/acl_anthology/anthology.bib*`.
- `results/acl_pdf_bib_lookup_cache.json` — small, keep tracked (like other
  `results/*.csv` intermediate artifacts) unless it grows unexpectedly
  large; revisit if so.

## Risks (carried over from the design discussion, restated for the record)

- **Citation-resolution reliability is unproven.** No `.bbl` ground truth
  to check against; expect a meaningfully higher `citation_unmatched` rate
  than the `.bbl` pipeline's near-zero unresolved rate. This could mean the
  effective yield from 285 titles ends up well below round 7-12's ~11-15%
  candidate-to-pair conversion, even before Claude/Gemma review.
- **Title-based OpenAlex/Crossref matching can mismatch** (same-title
  different paper, or a preprint vs. camera-ready title drift) — mitigated
  by the similarity threshold above, but not eliminated.
- **PDF reference-section boundary detection is heuristic** — venues and
  years in this set vary in bibliography formatting; a single regex won't
  cover every entry style. Expect partial coverage per paper, logged
  explicitly rather than silently skipped.

None of these risks block starting the run — they're exactly the kind of
signal the `[i/N]` progress output and the dropped/failed CSVs during a
full 285-title run are meant to surface early, per the user's choice to run
the full batch rather than a pilot.
