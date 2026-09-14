# Round 17: citation-graph expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find new candidate survey papers by mining the bibliographies of the 3720+ surveys already downloaded — extracting cited works whose own titles look like a survey/review, resolving them against arXiv, downloading the genuinely new ones, and pushing them through the existing mining → citation-resolution → batch-issuing pipeline, exactly like every prior round.

**Architecture:** Four sequential offline→online→online→offline stages, each producing a file the next consumes, matching this project's established multi-step round structure (e.g. round 14's PDF pipeline, round 16's rescan pipeline):

1. **Offline** — scan already-downloaded `.bbl`/`.bib` files for candidate survey-like cited titles (no network).
2. **Online, rate-limited, resumable** — resolve each unique candidate title against arXiv (`ti:"<title>"`), keep only genuine matches not already in the corpus.
3. **Online** — download + mine the genuinely new surveys, reusing `mine_disagreement_from_surveys.main()`'s existing `survey_dict` parameter unchanged.
4. **Offline** — resolve citations and issue the Claude pair-construction batch, reusing `resolve_survey_citations.py`/`prepare_claude_pair_batch.py` unmodified (identical to round 16's Task 4).

Key design decision: **"already in the corpus" is checked once, by arXiv ID against the `results/survey_sources/` directory listing, in Task 2** — not by fuzzy title matching in Task 1. Bibliography-extracted titles are noisy (LaTeX artifacts, truncation); arXiv IDs are exact. Task 1 does pure candidate generation (extract + classify + dedupe by title string); Task 2 is the single authoritative dedup point, checked only after a title has been verified to resolve to a real arXiv paper.

**Tech Stack:** Python 3, pandas, stdlib `re`/`urllib`/`argparse`/`pathlib`/`json`, arXiv Atom API. No pytest — verify with a throwaway script, delete after (established convention, see round 15/16 plans).

**Spec:** `docs/next_scaling_directions.md` (candidate #1, "Citation-graph розширення від уже відомих оглядів") — confirmed as the round-17 direction by the user, sequenced before candidate #2 (periodic re-run).

## Global Constraints

- Every place an `arxiv_id`/`survey_arxiv_id` column is read from CSV, pass `dtype={"...": str}` — the established float-truncation bug (e.g. `2306.1053` vs `2306.10530`) recurs project-wide if this is skipped.
- `results/survey_sources/` is gitignored bulk source data — read-only in Tasks 1-2, only Task 3's `download_and_extract` (already-existing, unmodified function) may write new subdirectories there.
- Task 1 must not make any network call — it is pure local file scanning. Task 2 is the only network-touching task in this plan besides Task 3's downloads.
- Task 2 must be resumable: cache each title's arXiv lookup result to disk incrementally (one JSONL line per query, matching `bib_lookup.py`'s and round 15's `save_cache`-per-row precedent), not held in memory until the end. A long-running title-resolution job that crashes partway must not lose completed lookups.
- Reuse existing code by **import** for generic utilities (`fetch_page`, `ARXIV_NS`, `PAGE_DELAY` from `find_surveys_bulk.py`; `BIBITEM_PATTERN`, `HREF_TITLE_PATTERN`, `QUOTED_TITLE_PATTERN`, `clean_latex`, `parse_bib` from `resolve_survey_citations.py`). Reuse the survey-title phrase list and `EXCLUDE_PATTERN` by **copying**, not importing — this project's own established convention (see the comment directly above `EXCLUDE_PATTERN` in `find_surveys_bulk.py`: copied deliberately so each script stays independently runnable without cross-script coupling).
- Do not modify `src/mine_disagreement_from_surveys.py`, `src/resolve_survey_citations.py`, `src/prepare_claude_pair_batch.py`, or `src/find_surveys_bulk.py` — every task in this plan calls their existing functions/CLIs unchanged.
- **Volume checkpoint:** Task 1's candidate count is unknown until it runs (typical survey has ~100-170 references; ~3100 surveys have at least one of `.bbl`/`.bib`). Task 2 costs ≥5s per unique candidate title (rate limit) — a few hundred candidates is a manageable ~20-40 minute job, a few thousand is multiple hours. Task 1's own verification step must report the exact count; if it is unexpectedly large (order of magnitude above a few hundred), stop and reassess Task 2's scope with the controller before committing to a full run — this is a deliberate checkpoint, not a step to automate past.
- Direct commits to `master`, no worktree/branch — same reasoning as round 15/16 (small, reversible, local-only unpushed repo).

---

### Task 1: Extract and classify candidate survey-like titles from downloaded bibliographies

**Files:**
- Create: `src/extract_cited_survey_titles.py`

**Interfaces:**
- Consumes: `BIBITEM_PATTERN`, `HREF_TITLE_PATTERN`, `QUOTED_TITLE_PATTERN`, `clean_latex`, `parse_bib` (all importable, unchanged, from `src/resolve_survey_citations.py`); `SRC_CACHE_DIR` (importable from `src/mine_disagreement_from_surveys.py`, equal to `results/survey_sources/`).
- Produces: `results/round17_candidate_survey_titles.csv`, columns `title, n_citing_surveys, citing_survey_ids, extraction_source` — consumed by Task 2.

This task intentionally does NOT reuse `resolve_survey_citations.build_survey_ref_index()` wholesale: that function's `.bbl` path falls through to a messy "first `\newblock` block truncated to 300 chars" tier when neither `HREF_TITLE_PATTERN` nor `QUOTED_TITLE_PATTERN` matches (acceptable there, since a human/Claude reads the full `context` field too — not acceptable here, since round 17 treats the extracted string as an exact title to search arXiv for). This task calls the two precise `.bbl` patterns directly and skips an entry if neither matches, rather than falling back.

- [ ] **Step 1: Write the script**

```python
"""Round 17 (docs/next_scaling_directions.md candidate #1): extract cited-work
titles from all already-downloaded survey .bbl/.bib files, classify which
ones look like a survey/review themselves, and count how many different
downloaded surveys cite each one. Pure local file scan -- no network calls.
"a candidate is already in the corpus" is NOT checked here (bibliography
titles are noisy -- LaTeX artifacts, truncation); that check happens in
src/resolve_candidate_titles_to_arxiv.py, once a candidate has resolved to
an exact arXiv ID.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from resolve_survey_citations import (
    BIBITEM_PATTERN,
    HREF_TITLE_PATTERN,
    QUOTED_TITLE_PATTERN,
    clean_latex,
    parse_bib,
)
from mine_disagreement_from_surveys import SRC_CACHE_DIR

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Copied (not imported) from find_surveys_bulk.py's QUERY_MODES phrase list --
# this project's established convention for this list is to copy it per
# script rather than import, so each script stays independently runnable.
SURVEY_TITLE_PATTERN = re.compile(
    r"\bsurvey\b|systematic review|literature review|overview of|review of|\ba review\b|primer on|tutorial on",
    re.IGNORECASE,
)
# Copied verbatim from find_surveys_bulk.py's EXCLUDE_PATTERN (same reasoning).
EXCLUDE_PATTERN = re.compile(
    r"survey generation|generat\w+ (academic )?survey|survey (automation|writing|writer|benchmark|evaluator)"
    r"|survey response|survey simulation|simulate\w* .*survey|synthetic survey|social survey"
    r"|questionnaire|opinion survey|persona-grounded|automatic survey|survey item",
    re.IGNORECASE,
)


def guess_precise_bbl_title(entry_text: str) -> str | None:
    """Only the two reliable .bbl title patterns -- no messy newblock fallback."""
    href_match = HREF_TITLE_PATTERN.search(entry_text)
    if href_match:
        return clean_latex(href_match.group(1))
    quoted_match = QUOTED_TITLE_PATTERN.search(entry_text)
    if quoted_match:
        return clean_latex(quoted_match.group(1))
    return None


def extract_bbl_titles(bbl_path: Path) -> list[str]:
    try:
        text = bbl_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    titles = []
    for match in BIBITEM_PATTERN.finditer(text):
        title = guess_precise_bbl_title(match.group(2))
        if title:
            titles.append(title)
    return titles


def extract_bib_titles(bib_path: Path) -> list[str]:
    return [v["title"] for v in parse_bib(bib_path).values() if v.get("title")]


def main() -> None:
    survey_dirs = sorted(p for p in SRC_CACHE_DIR.iterdir() if p.is_dir())

    # normalized title -> {"display_title": str, "citing_ids": set[str], "sources": set[str]}
    candidates: dict[str, dict] = {}
    n_surveys_scanned = 0

    for survey_dir in survey_dirs:
        arxiv_id = survey_dir.name
        titles_this_survey: list[tuple[str, str]] = []  # (title, source)
        for bbl_path in survey_dir.rglob("*.bbl"):
            titles_this_survey.extend((t, "bbl") for t in extract_bbl_titles(bbl_path))
        for bib_path in survey_dir.rglob("*.bib"):
            titles_this_survey.extend((t, "bib") for t in extract_bib_titles(bib_path))
        if titles_this_survey:
            n_surveys_scanned += 1

        for title, source in titles_this_survey:
            if not SURVEY_TITLE_PATTERN.search(title) or EXCLUDE_PATTERN.search(title):
                continue
            key = re.sub(r"\s+", " ", title).strip().lower()
            if not key:
                continue
            entry = candidates.setdefault(key, {"display_title": title, "citing_ids": set(), "sources": set()})
            entry["citing_ids"].add(arxiv_id)
            entry["sources"].add(source)

    rows = [
        {
            "title": v["display_title"],
            "n_citing_surveys": len(v["citing_ids"]),
            "citing_survey_ids": ";".join(sorted(v["citing_ids"])),
            "extraction_source": "+".join(sorted(v["sources"])),
        }
        for v in candidates.values()
    ]
    df = pd.DataFrame(rows, columns=["title", "n_citing_surveys", "citing_survey_ids", "extraction_source"])
    df = df.sort_values("n_citing_surveys", ascending=False).reset_index(drop=True)
    out_path = RESULTS_DIR / "round17_candidate_survey_titles.csv"
    df.to_csv(out_path, index=False)

    print(f"Scanned {len(survey_dirs)} survey directories ({n_surveys_scanned} had a .bbl or .bib file).")
    print(f"{len(df)} unique candidate survey-like titles -> {out_path}")
    if len(df) > 0:
        print(f"Top citing-count: {df['n_citing_surveys'].iloc[0]}, median: {df['n_citing_surveys'].median():.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and report the count**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/extract_cited_survey_titles.py`

(No `sys.path` manipulation is needed: running a script as `python3 src/foo.py` makes Python add `src/` to `sys.path[0]` automatically, which is why `mine_disagreement_from_acl_pdfs.py`'s `from mine_disagreement_from_surveys import ...` — and every other cross-importing script in this project — already works with a plain top-level `import`, invoked exactly this way from the repo root. Do not add `sys.path.insert` calls.)

Expected: completes in well under a minute (pure local regex scan of already-cached files, no network). Report the exact candidate count from the printed output — **this number gates Task 2's scope per the Global Constraints volume checkpoint**.

- [ ] **Step 3: Commit**

```bash
cd ~/prjs/hypocontra
git add src/extract_cited_survey_titles.py results/round17_candidate_survey_titles.csv
git commit -m "Round 17: extract candidate survey-like titles from downloaded bibliographies"
```

---

### Task 2: Resolve candidate titles against arXiv (rate-limited, resumable)

**Files:**
- Create: `src/resolve_candidate_titles_to_arxiv.py`

**Interfaces:**
- Consumes: `results/round17_candidate_survey_titles.csv` (Task 1's output); `fetch_page`, `ARXIV_NS`, `PAGE_DELAY` (importable, unchanged, from `src/find_surveys_bulk.py`); `SRC_CACHE_DIR` (from `mine_disagreement_from_surveys.py`, for the authoritative "already downloaded" check).
- Produces: `results/round17_title_resolution_cache.jsonl` (incremental, resumable cache — one line per queried title) and `results/round17_resolved_new_surveys.csv` (columns `arxiv_id, title` — only rows that matched AND are not already in `results/survey_sources/`), consumed by Task 3.

**Before running the full job:** confirm Task 1's reported candidate count with the controller per the Global Constraints checkpoint. If it's in the low hundreds, run in full. If it's in the thousands, ask before committing multiple hours of rate-limited network calls — this step is a judgment call, not something to push through unilaterally.

- [ ] **Step 1: Write the script**

```python
"""Round 17, step 2: resolve each unique candidate survey-like title
(src/extract_cited_survey_titles.py's output) against arXiv via an exact
`ti:"<title>"` phrase query, keeping only genuine matches whose arXiv ID is
NOT already a downloaded survey under results/survey_sources/. Rate-limited
(PAGE_DELAY between requests, exponential backoff on 429 -- both inherited
unchanged from find_surveys_bulk.fetch_page) and resumable: every query's
result (match or no-match) is appended to a JSONL cache immediately, so a
crash or interruption loses at most the one in-flight request.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from find_surveys_bulk import ARXIV_NS, PAGE_DELAY, fetch_page
from mine_disagreement_from_surveys import SRC_CACHE_DIR

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CACHE_PATH = RESULTS_DIR / "round17_title_resolution_cache.jsonl"


def normalize_title(title: str) -> str:
    text = re.sub(r"[^\w\s]", "", title.lower())
    return re.sub(r"\s+", " ", text).strip()


def load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["title"]] = rec
    return cache


def append_to_cache(record: dict) -> None:
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_one(title: str) -> dict:
    escaped = title.replace('"', "'")
    query = f'ti:"{escaped}"'
    root = fetch_page(0, 3, query)
    entries = root.findall("atom:entry", ARXIV_NS)
    target_norm = normalize_title(title)
    for entry in entries:
        entry_title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
        if normalize_title(entry_title) == target_norm:
            arxiv_id = entry.find("atom:id", ARXIV_NS).text.strip().split("/")[-1].split("v")[0]
            return {"title": title, "match": True, "arxiv_id": arxiv_id, "matched_title": entry_title}
    return {"title": title, "match": False, "arxiv_id": None, "matched_title": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N candidates (by n_citing_surveys descending).")
    args = parser.parse_args()

    candidates = pd.read_csv(RESULTS_DIR / "round17_candidate_survey_titles.csv", dtype=str)
    if args.limit is not None:
        candidates = candidates.head(args.limit)

    cache = load_cache()
    known_ids = {p.name for p in SRC_CACHE_DIR.iterdir() if p.is_dir()}

    n_processed = 0
    n_matched = 0
    n_new = 0
    for _, row in candidates.iterrows():
        title = row["title"]
        if title in cache:
            record = cache[title]
        else:
            record = resolve_one(title)
            record["timestamp"] = time.time()
            append_to_cache(record)
            time.sleep(PAGE_DELAY)
        n_processed += 1
        if record["match"]:
            n_matched += 1
            if record["arxiv_id"] not in known_ids:
                n_new += 1
        if n_processed % 50 == 0:
            print(f"[{n_processed}/{len(candidates)}] matched={n_matched} new={n_new}", flush=True)

    # Rebuild the final output from the full cache (covers resumed runs too).
    new_rows = []
    seen_ids = set()
    for title in candidates["title"]:
        record = cache.get(title) or load_cache().get(title)
        if record and record["match"] and record["arxiv_id"] not in known_ids and record["arxiv_id"] not in seen_ids:
            new_rows.append({"arxiv_id": record["arxiv_id"], "title": record["matched_title"]})
            seen_ids.add(record["arxiv_id"])

    out_path = RESULTS_DIR / "round17_resolved_new_surveys.csv"
    pd.DataFrame(new_rows, columns=["arxiv_id", "title"]).to_csv(out_path, index=False)
    print(f"\nProcessed {n_processed} candidates: {n_matched} matched an arXiv paper, "
          f"{len(new_rows)} are genuinely new (not already in results/survey_sources/) -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on 5 candidates**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/resolve_candidate_titles_to_arxiv.py --limit 5`
Expected: exits 0, prints a `[5/5]`-or-fewer progress line and a summary; `results/round17_resolved_new_surveys.csv` exists with the 2-column header even if 0 rows (5 titles may genuinely not all match or may already be known).

- [ ] **Step 3: Run the full job** (or a controller-agreed `--limit`, per the volume checkpoint)

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/resolve_candidate_titles_to_arxiv.py`
This is resumable — if interrupted, re-running the same command picks up from the cache (`results/round17_title_resolution_cache.jsonl`) and only queries titles not already cached.

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/resolve_candidate_titles_to_arxiv.py results/round17_title_resolution_cache.jsonl results/round17_resolved_new_surveys.csv
git commit -m "Round 17: resolve candidate titles against arXiv, isolate genuinely new surveys"
```

---

### Task 3: Download and mine the newly-found surveys

**Files:**
- Create: `src/mine_round17_citation_graph_surveys.py`

**Interfaces:**
- Consumes: `results/round17_resolved_new_surveys.csv` (Task 2's output); `main(survey_dict, out_suffix)` (importable, unchanged, from `src/mine_disagreement_from_surveys.py` — already supports a custom `{arxiv_id: title}` dict, downloading and mining each one exactly like every prior round's driver scripts).
- Produces: `results/survey_disagreement_sentences_round17.csv` (Task 4 consumes this via `resolve_survey_citations.py --input`).

- [ ] **Step 1: Write the thin driver script**

```python
"""Round 17, step 3: download and mine the surveys src/resolve_candidate_titles_to_arxiv.py
found via the citation graph. Thin wrapper -- mine_disagreement_from_surveys.main()
already accepts a custom survey_dict and does the download+mine in one pass,
same pattern this project has used since round 7's SURVEY_ARXIV_IDS dict.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mine_disagreement_from_surveys import main as mine_main

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "round17_resolved_new_surveys.csv", dtype=str)
    survey_dict = dict(zip(df["arxiv_id"], df["title"]))
    print(f"Downloading and mining {len(survey_dict)} newly-found surveys...")
    mine_main(survey_dict=survey_dict, out_suffix="_round17")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/mine_round17_citation_graph_surveys.py`
This downloads each new survey's LaTeX source (network, `time.sleep(2)` between each per `mine_disagreement_from_surveys.main()`'s existing behavior — expect a few minutes per ~100 surveys) and mines it with the current `DISAGREEMENT_PATTERNS`/`EXCLUSION_PATTERNS`/`ARTIFACT_PATTERN` (already includes round 16's 7 new markers, since those are now the module's standing patterns). Note the printed candidate-sentence count.

- [ ] **Step 3: Commit**

```bash
cd ~/prjs/hypocontra
git add src/mine_round17_citation_graph_surveys.py results/survey_disagreement_sentences_round17.csv
git commit -m "Round 17: download and mine newly-found citation-graph surveys"
```

---

### Task 4: Resolve citations and issue the round-17 Claude batch

**Files:**
- Creates (data, committed): `results/candidate_sentences_resolved_round17.csv`, `results/citation_resolution_dropped_round17.csv`, `results/claude_pair_batch_round17.jsonl`, `results/claude_batch_state_round17.json`
- Modify: `docs/next_scaling_directions.md` (record round 17's outcome)

**Interfaces:**
- Consumes: `results/survey_disagreement_sentences_round17.csv` (Task 3's output), via the existing, unmodified `src/resolve_survey_citations.py --input` and `src/prepare_claude_pair_batch.py --input`.
- Produces: `results/claude_pair_batch_round17.jsonl`, ready for a future live Claude session to construct pairs from — pair construction itself is out of scope for this plan, matching every prior round's split.

- [ ] **Step 1: Resolve citations**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/resolve_survey_citations.py --input survey_disagreement_sentences_round17.csv --output-suffix _round17`

Note the printed eligible/dropped counts and `drop_reason` breakdown.

- [ ] **Step 2: Issue the batch**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/prepare_claude_pair_batch.py --round 17 --input candidate_sentences_resolved_round17.csv --all`

Confirm the printed `Issued <N> sentences into results/claude_pair_batch_round17.jsonl` line.

- [ ] **Step 3: Record the outcome in docs/next_scaling_directions.md**

Add an entry to the "Вже перевірено" table (matching round 16's row format) recording: how many candidate titles were extracted (Task 1), how many resolved to a genuinely new arXiv survey (Task 2), how many surveys were downloaded/mined (Task 3), and the eligible/dropped counts from Task 4 Step 1. Do not claim a pair count — unknown until a future pair-construction session.

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add results/candidate_sentences_resolved_round17.csv results/citation_resolution_dropped_round17.csv results/claude_pair_batch_round17.jsonl results/claude_batch_state_round17.json docs/next_scaling_directions.md
git commit -m "Round 17: resolve citations and issue Claude pair-construction batch"
```

---

## Self-Review

**Spec coverage:** `docs/next_scaling_directions.md` candidate #1 specifies (a) extract cited works whose titles look like a survey — Task 1; (b) check arXiv presence via `ti:"<exact title>"` — Task 2; (c) the "human-vetted comparability" principle (a survey author already judged the cited work relevant) is preserved by construction — Task 1 only extracts titles a real survey actually cited, no independent judgment substituted. Batch issuance (not pair construction) is the correct stopping point, matching round 16's precedent exactly.

**Placeholder scan:** no TBD/TODO; every step has runnable code or an exact command. Task 1 Step 2's import-path instruction is deliberately left to match an existing precedent (`mine_disagreement_from_acl_pdfs.py`'s cross-module import) rather than guessing a path convention — flagged explicitly for the implementer to verify against real code, not a placeholder for missing logic.

**Type consistency:** `SRC_CACHE_DIR`, `main(survey_dict, out_suffix)` used in Task 3 exactly as defined in `mine_disagreement_from_surveys.py` (verified against that file's actual source during planning, matching round 16's plan's own verification standard). `BIBITEM_PATTERN`/`HREF_TITLE_PATTERN`/`QUOTED_TITLE_PATTERN`/`clean_latex`/`parse_bib` used in Task 1 exactly as defined in `resolve_survey_citations.py` (verified line-by-line against that file's actual source, not assumed from memory). `fetch_page`/`ARXIV_NS`/`PAGE_DELAY` used in Task 2 exactly as defined in `find_surveys_bulk.py`. CSV schemas match across every task boundary (`round17_candidate_survey_titles.csv` → Task 2's input; `round17_resolved_new_surveys.csv` → Task 3's input; `survey_disagreement_sentences_round17.csv` → Task 4's input, same 5-column schema every prior round's mining output uses).
