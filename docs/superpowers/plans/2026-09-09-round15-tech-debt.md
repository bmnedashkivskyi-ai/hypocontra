# Round 15 phase 1: close round-14 technical debt — recovery checkpoint

**Status: COMPLETE (2026-09-09).** All 8 fixes implemented, individually
committed, and verified against a full real re-run of the pipeline (see
**Verification results** below). Phase 2 (corpus scaling, see the bottom
section) has not started yet.

This file exists so this exact plan survives a session interruption (PC
stop, overload, crash) — the same failure mode that started the round-13→14
work in this project. If you are picking this up cold: read the **Progress**
section right below first — it is updated after every completed stage, with
the commit SHA that closed it. Any fix not listed there as done has not been
started. Cross-check against `git log --oneline -15` in `~/prjs/hypocontra`.

## Progress

Each row's commit message starts with "Round 15 tech-debt fix N:" — find it
with `git log --oneline --grep="tech-debt fix"` if the SHA below ever looks
stale (this table is updated by hand right after each commit, so it should
not lag, but trust `git log` over this table if they ever disagree).

- [x] Fix 1 (incremental bib-cache save) — done
- [x] Fix 2 (per-file exception isolation in mining) — done
- [x] Fix 3 (centralize control-char stripping) — done
- [x] Fix 4 (Crossref path + HTML-entity unescaping) — done (`_strip_jats_tags` fixed and verified with a synthetic JATS+entities string; the Crossref *query* path itself is still never exercised by a real successful response — that risk is unchanged, only the entity-unescaping bug is closed)
- [x] Fix 5 (disambiguate resolution_source vocabulary) — done (`no_match` vs `no_abstract_found`, replacing the overloaded `unresolved`)
- [x] Fix 6 (guard year_prefix) — done, verified with `'2023a'->'2023'`, `'2019'->'2019'`, `'n/a'->'n/a'` (no crash)
- [x] Fix 7 (remove dead single_cite_key branch) — done
- [x] Fix 8 (URL-scheme allowlist) — done, verified `file:///etc/passwd` is rejected before any `urlopen` call
- [x] Verification plan — done, see **Verification results** below

## Verification results (2026-09-09)

Ran the full pipeline against real data after all 8 fixes landed:

1. **Full re-fetch** (`fetch_acl_pdfs.py`): 272/284 downloaded, 12 failed
   (same 12 pre-2013 404s as round 14's original run) — matches exactly.
2. **Full mining** (`mine_disagreement_from_acl_pdfs.py`): 24 candidate
   sentences, **0 failed to process** (the new exception-isolation counter
   never fired — no corrupt PDFs in this batch, consistent with round 14's
   final review finding zero exceptions across all 272 files). Confirms
   moving control-char stripping into `acl_pdf_text.extract_pdf_text`
   (fix #3) introduced no regression.
3. **Full resolution** (`resolve_acl_pdf_citations.py`): 14 eligible / 10
   dropped, same `4 too_many_citations / 4 citation_unmatched / 2
   citation_ambiguous` split as round 14's original run.
4. **Incremental cache save (fix #1), directly measured**: instrumented
   `save_cache` with a call counter and re-ran resolution on a
   3-entries-removed cache — `save_cache` was called **14 times** (once per
   kept row), not once at the end. Confirms the fix, not just the code
   placement.
5. **`_strip_jats_tags` (fix #4)**: verified with a synthetic
   `"<jats:p>Foo &amp; bar &lt;baz&gt;</jats:p>"` → tags replaced with
   spaces (existing behavior), entities unescaped to `Foo & bar <baz>`.
6. **`resolution_source` vocabulary (fix #5)**: final `git diff` against
   the committed `candidate_sentences_resolved_acl_pdf.csv` shows exactly 8
   changed lines, all `unresolved` → `no_match`/`no_abstract_found`,
   **zero** remaining `unresolved` values, and no other column touched.

**One real gotcha hit and corrected during verification, worth recording
for next time:** step 4's cache-entry-deletion test forced 3 fresh live
OpenAlex re-queries. Two of those three came back with a *different*
match/abstract result than the original round-14 run (OpenAlex's search
ranking is not perfectly stable between calls at different times — the
same title-search returned no qualifying abstract on the retry where it
had found one before). This is not a bug introduced by any of the 8 fixes
(none of them touch `_query_openalex` or the OpenAlex request path at
all) — it's the same live-API non-determinism this project already
tolerates elsewhere (`check_acl_arxiv_overlap.py`'s retry/backoff, the
round-14 plan's own "re-run once on network error" guidance). The fix:
restored `results/acl_pdf_bib_lookup_cache.json` from its last committed
version (`git checkout --`) before the final regeneration, so the
committed `candidate_sentences_resolved_acl_pdf.csv` reflects only the
intended fix #5 vocabulary change, not incidental re-query drift. Lesson
for round 16+: never delete-and-re-query cache entries for a live
verification test against data you intend to keep — copy the cache to a
scratch path and test there, or accept that a forced re-query may not be
bit-for-bit reproducible.

## Context

Round 14 (PDF-based mining pipeline for 284 ACL-only survey papers) is fully
implemented, reviewed (task-scoped + final whole-branch review on Opus), and
merged to `master`. Round-14 pair construction is also done: 89 curated pairs
total, κ=0.578 (n=71). The final whole-branch review deferred 3 Important-tier
findings and several Minor findings as explicit follow-up work rather than
blocking round 14's merge (see `docs/superpowers/plans/2026-09-09-acl-pdf-pipeline.md`'s
commit history / the `master` git log around commits `52abb58`..`66ce37c` for
the original review discussion). This plan closes that follow-up work before
starting the next corpus-scaling phase.

**Known data-state gap:** the 272 PDFs fetched during round 14 lived only in
the now-deleted git worktree (`results/acl_pdfs/`, gitignored, never
committed) and are gone. `results/acl_pdf_bib_lookup_cache.json` (committed)
survived. Re-verifying these fixes against real data requires re-running
`src/fetch_acl_pdfs.py` first (idempotent — re-downloads the same 272,
retries the 12 that 404'd before, ~8-12 minutes).

## The 8 fixes (bounded — no new architecture, all touch already-existing round-14 files)

### 1. Incremental bib-lookup cache save (resumability)

**File:** `src/resolve_acl_pdf_citations.py`

Currently `save_cache(BIB_CACHE_PATH, bib_cache)` is called once, after the
entire `for survey_key, group in df.groupby(...)` loop finishes. If the run
crashes or is interrupted mid-way, every OpenAlex/Crossref lookup performed
in that run is lost, defeating the resumability goal stated in the design
spec (`docs/superpowers/specs/2026-09-09-acl-pdf-pipeline-design.md`, the
"Cache every OpenAlex/Crossref response to disk" requirement).

**Fix:** move `save_cache(BIB_CACHE_PATH, bib_cache)` inside the per-sentence
loop, called once per completed row (after each `kept_rows.append(...)` —
simplest correct placement, no need to track "did anything actually change"
this iteration; at round-14 scale, ~30-40 total lookups, saving every row is
cheap).

### 2. Per-file exception isolation in mining

**File:** `src/mine_disagreement_from_acl_pdfs.py`

`main()`'s loop calls `find_disagreement_sentences(pdf_path)` with no
exception handling — one unreadable/corrupt PDF (a bad pymupdf parse, e.g.)
aborts the entire run, discarding every other file's already-completed work.
Empirically this did not fire against the real 272 PDFs (independently
verified during the final review), but it's a real gap for a future,
larger-scale run.

**Fix:** wrap the per-PDF body of the loop in `try/except Exception`, print
an error line naming the key and the exception, increment a counter, and
`continue` to the next PDF. Report the failure count in the final summary
print (`n_failed`, alongside the existing `n_missing_pdf`). Do not add a
separate failure-log CSV for this — a print + counter is proportionate; this
is a defensive fix for an unobserved failure mode, not a new feature.

### 3. Centralize control-character stripping

**Files:** `src/acl_pdf_text.py` (add), `src/mine_disagreement_from_acl_pdfs.py` (remove now-redundant copy)

The round-14 fix-round added `CONTROL_CHAR_PATTERN` stripping only inside
`mine_disagreement_from_acl_pdfs.py`'s `find_disagreement_sentences`.
`resolve_acl_pdf_citations.py`'s `build_reference_index` independently
re-extracts the same PDFs via `acl_pdf_text.extract_pdf_text` for reference-
list parsing, with no such stripping — same bug class, latent (verified zero
control characters across all 272 real PDFs' parsed reference titles/
surnames during the final review, but not structurally prevented).

**Fix:** move the `CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")`
constant and a `.sub("", text)` call into `acl_pdf_text.extract_pdf_text`
itself (strip once, at the source, right after `pymupdf` extraction, before
returning). Remove the now-redundant stripping line from
`mine_disagreement_from_acl_pdfs.py`'s `find_disagreement_sentences` (it
would otherwise run twice, harmlessly but pointlessly). Both consumers then
inherit the protection automatically.

### 4. Crossref path: real test + HTML-entity unescaping

**File:** `src/bib_lookup.py`

The Crossref fallback (`_query_crossref`, `_strip_jats_tags`) has never
executed successfully against a real response in testing (all 26 real
lookups during round 14 hit OpenAlex; Crossref was tried 7 times and found
nothing matching the similarity threshold). Two issues: (a) no test has
actually exercised a real Crossref-abstract response, so latent bugs in that
path are unverified; (b) `_strip_jats_tags` strips `<...>` tags but does not
unescape HTML entities (`&amp;`, `&lt;`, etc.), so a real Crossref abstract
would ship with literal escaped entities in committed data.

**Fix:**
- Add `import html` and change `_strip_jats_tags` to
  `return html.unescape(re.sub(r"<[^>]+>", " ", text))`.
- Add a direct unit-style check (not dependent on live API luck) that feeds
  `_strip_jats_tags` a synthetic string containing both JATS tags AND HTML
  entities, e.g. `"<jats:p>Foo &amp; bar &lt;baz&gt;</jats:p>"`, and asserts
  the output is `"Foo & bar <baz>"` (tags gone, entities unescaped). This
  can be a throwaway verification script (this project's established
  testing convention — no pytest), doesn't need to be committed as a
  permanent test file.

### 5. Disambiguate `resolution_source` vocabulary

**File:** `src/resolve_acl_pdf_citations.py`

Currently `"unresolved"` is used for two different situations in the
per-citation loop: (a) `entry is None` (no reference-list entry matched this
citation at all) and (b) `entry` matched but `bib_lookup.lookup_abstract`
returned `None` (no abstract found via OpenAlex/Crossref). Nothing
downstream currently reads this column, so this is a low-urgency
documentation-quality fix, not a functional bug.

**Fix:** in the `if entry is None:` branch, append `"no_match"` instead of
`"unresolved"`. In the `bib is None` case (inside the `else` branch, when
`entry` is not `None` but `lookup_abstract` returned `None`), append
`"no_abstract_found"` instead of `"unresolved"`.

### 6. Guard `year_prefix()` against a non-matching year string

**File:** `src/resolve_acl_pdf_citations.py`

```python
def year_prefix(year: str) -> str:
    return re.match(r"(\d{4})", year).group(1)
```

Safe by construction today (every year string traces back to
`acl_ref_parsing.YEAR_PATTERN`, which always starts with 4 digits), but
unguarded — a future caller or data-format change could crash this with
`AttributeError` on `None`.

**Fix:**
```python
def year_prefix(year: str) -> str:
    m = re.match(r"(\d{4})", year)
    return m.group(1) if m else year
```

### 7. Remove the structurally-dead `single_cite_key` branch

**File:** `src/resolve_acl_pdf_citations.py`

```python
dropped_rows.append({**row.to_dict(), "sentence_id": sentence_id, "drop_reason": "single_cite_key" if n_keys < MIN_KEYS else "too_many_citations"})
```

`n_keys < MIN_KEYS` (i.e. `< 2`) can never be true here: `main()` only ever
reads `results/survey_disagreement_sentences_acl_pdf.csv`, which
`mine_disagreement_from_acl_pdfs.py` already produces with its own
`MIN_CITATIONS = 2` gate applied before a row is ever written. Confirmed
dead by the final review (zero impact, zero occurrences).

**Fix:** simplify to `"drop_reason": "too_many_citations"` directly (the
`if n_keys < MIN_KEYS` branch and the `MIN_KEYS` check in the surrounding
`if` can stay as-is for defensive clarity — only the now-pointless ternary
in the dict literal changes). Add a one-line comment noting `MIN_CITATIONS`
is already enforced upstream by the miner, so only the "too many" side is
reachable in practice.

### 8. URL-scheme allowlist in the PDF downloader

**File:** `src/fetch_acl_pdfs.py`

`download_one(url, dest)` calls `urllib.request.urlopen(req, ...)` on a URL
built directly from the input CSV's `url` column with no scheme/host check.
Theoretical risk only (`results/acl_anthology/acl_not_on_arxiv.csv` is
locally curated, not user-supplied), but cheap to close.

**Fix:** at the top of `download_one`, before constructing the request:
```python
if not url.startswith("https://aclanthology.org/"):
    return f"rejected: unexpected URL scheme/host ({url})"
```

## Verification plan (after implementing all 8)

1. `cd ~/prjs/hypocontra && .venv/bin/python3 src/fetch_acl_pdfs.py` — full
   re-fetch (idempotent; nothing is cached on disk right now since the
   worktree that held the 272 PDFs was deleted after round 14's merge).
   Expect similar counts to round 14's original run (272 fetched, 12 failed
   — same 12 pre-2013 404s, unless ACL Anthology's serving changed).
2. `.venv/bin/python3 src/mine_disagreement_from_acl_pdfs.py` — confirm it
   still produces ~24 candidate sentences (small variance possible if the
   control-char fix changes which sentences pass the 500-char truncation
   boundary — check, don't assume identical).
3. `.venv/bin/python3 src/resolve_acl_pdf_citations.py` — confirm ~14
   eligible / ~10 dropped with the same reason breakdown, AND confirm the
   `results/acl_pdf_bib_lookup_cache.json` file's mtime updates incrementally
   during the run (not just once at the end) — e.g. `watch -n 2 stat
   results/acl_pdf_bib_lookup_cache.json` in a second terminal, or just diff
   its content against the pre-run committed version to confirm growth.
4. Run the Finding-4 verification script (synthetic JATS+entities string)
   directly — confirm `_strip_jats_tags` output.
5. Confirm `results/candidate_sentences_resolved_acl_pdf.csv`'s
   `resolution_source` column now contains `no_match`/`no_abstract_found`
   instead of the old overloaded `unresolved` value (this file WILL be
   regenerated with different content in this column — expected, not a
   regression, since round-14's actual pair-construction step already
   happened and doesn't depend on this column).
6. Diff-review the full changeset before committing — this touches 3 files
   (`acl_pdf_text.py`, `mine_disagreement_from_acl_pdfs.py`,
   `resolve_acl_pdf_citations.py`) plus `fetch_acl_pdfs.py`, all already
   merged to `master` — no worktree/branch needed for a fix this size,
   direct commits to `master` are consistent with this project's established
   pattern (round 13, spec/plan authoring, and the pre-round-14 setup were
   all committed directly to `master`).

## After this: phase 2 — corpus scaling

Once the above is closed and committed, proceed to
`docs/next_scaling_directions.md`'s ranked list (as of round 14: cs.CL
title-synonyms is #1, DISAGREEMENT_PATTERNS expansion #2, citation-graph
expansion #3, periodic re-run #4) — this needs its own scoping discussion
(which direction(s) to pursue, one at a time or several) before
implementation starts. Not part of this plan.
