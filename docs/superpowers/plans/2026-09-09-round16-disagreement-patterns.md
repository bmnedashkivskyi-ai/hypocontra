# Round 16 direction 1: DISAGREEMENT_PATTERNS expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the contrast-marker regex list (`DISAGREEMENT_PATTERNS`) that drives HypoContra's survey-mining step, re-scan every already-downloaded survey LaTeX source with the wider pattern set (no new network fetches), isolate the sentences the old patterns missed, and push those through the existing citation-resolution + batch-issuing steps so a future session can hand them to Claude for pair construction.

**Architecture:** This is a data-pipeline scaling round, not a new subsystem. It touches one existing module (`src/mine_disagreement_from_surveys.py`, shared by both the LaTeX-survey and ACL-PDF mining paths via import) and adds two small, generic driver scripts that reuse existing functions rather than duplicating mining/resolution logic. No downstream script (`resolve_survey_citations.py`, `prepare_claude_pair_batch.py`) is modified — round 16 output is shaped to match what they already expect, exactly as rounds 7-15 did.

**Tech Stack:** Python 3, pandas, stdlib `re`/`argparse`/`pathlib`. No pytest in this project — verification is a throwaway script run directly and its output inspected (see `docs/superpowers/plans/2026-09-09-round15-tech-debt.md` for the established precedent).

**Spec:** `docs/next_scaling_directions.md` (candidate #1, "Розширення набору маркерів контрасту (DISAGREEMENT_PATTERNS)") — confirmed as the chosen round-16 direction by the user in the current session.

## Global Constraints

- No pytest. Verify with a throwaway script (run via `.venv/bin/python3`), inspect its printed output directly, delete the script after — do not commit it as a permanent test file (matches round 15's convention).
- Every place a CSV's `arxiv_id`/`survey_arxiv_id` column is read, pass `dtype={"...": str}` — this column has a known float-truncation bug (e.g. `2306.1053` instead of `2306.10530`) if read as a numeric dtype (documented in `src/resolve_survey_citations.py`'s module docstring).
- `results/survey_sources/` is gitignored (27GB, bulk re-downloadable source data) — never delete or modify it; only read from it. Everything else this plan writes under `results/` and `docs/` is tracked and gets committed normally.
- Direct commits to `master`, no worktree/branch: this round touches one existing file plus two new, independent, reversible scripts, makes zero new network calls (it re-scans already-cached sources), and is the same size/risk class as round 15 phase 1, which used direct commits to `master` by explicit precedent (see that plan's step 6).
- `COMBINED_PATTERN`/`EXCLUSION_PATTERN`/`ARTIFACT_PATTERN` in `src/mine_disagreement_from_surveys.py` are imported directly (not copied) by `src/mine_disagreement_from_acl_pdfs.py` — editing the pattern lists in Task 1 changes both mining paths' behavior for any future run of either. This plan only re-runs the LaTeX-survey path (Task 3); it does not re-run the ACL-PDF path (that source is documented as exhausted and is out of scope here).
- Do not preemptively invent new `EXCLUSION_PATTERNS` for hypothetical false senses of the new markers. This project's established methodology (see the round 3/4/5/12/15 comments already in `mine_disagreement_from_surveys.py` and in `docs/next_scaling_directions.md`) discovers false-sense categories empirically from real mining output, then adds a targeted fix — never speculatively. Task 1's verification only checks that the *existing* exclusion filter still generalizes to the *new* trigger words for the false senses already known (§8 contradiction-detection systems, annotator disagreement, model self-correction). Anything new Task 3's real run turns up is a follow-up, not part of this plan.

---

### Task 1: Expand DISAGREEMENT_PATTERNS in mine_disagreement_from_surveys.py

**Files:**
- Modify: `src/mine_disagreement_from_surveys.py:49-62` (the `DISAGREEMENT_PATTERNS` list)

**Interfaces:**
- Consumes: nothing new — extends the existing `DISAGREEMENT_PATTERNS` list, which is `"|".join(...)`-compiled into the existing `COMBINED_PATTERN` (module-level, unchanged name/shape).
- Produces: `COMBINED_PATTERN` now also matches the 7 new phrases. `EXCLUSION_PATTERN` and `ARTIFACT_PATTERN` are untouched by this task — Task 3's real run is what tells us whether they need extending, per the Global Constraints note above.

- [ ] **Step 1: Add the 7 new patterns with a provenance comment**

Edit `src/mine_disagreement_from_surveys.py`, replacing the `DISAGREEMENT_PATTERNS` block (currently lines 48-62):

```python
# Лінгвістичні маркери контрасту/незгоди -- englitude regex, case-insensitive
DISAGREEMENT_PATTERNS = [
    r"in contrast to",
    r"unlike \\?cite",
    r"contrary to",
    r"however,?\s+\\?cite",
    r"while \\?cite\w*\{[^}]+\}[^.]{0,150}\\?cite\w*\{[^}]+\}",
    r"disagree",
    r"conflicting (results|findings|evidence)",
    r"mixed (results|findings|evidence)",
    r"contradict",
    r"inconsistent (results|findings)",
    r"different from \\?cite",
    r"in disagreement with",
    # Round 16 (docs/next_scaling_directions.md candidate #1): DISAGREEMENT_PATTERNS
    # hadn't been revisited since round 3, and acts on the 3700+ surveys already
    # downloaded into results/survey_sources/ -- an independent lever from finding
    # new survey sources (all of which are now exhausted, see that doc). Risk
    # (documented there too): these markers don't all contain "contradict", so a
    # sentence can hit COMBINED_PATTERN via one of these without also hitting the
    # existing EXCLUSION_PATTERNS' §8 contradiction-detection-system guard unless
    # that sentence *also* separately contains a contradiction/detect phrase --
    # verified still true for the known false-sense categories (see the round-16
    # plan's Task 1 verification script), but a real run may surface new ones.
    r"on the other hand",
    r"conversely",
    r"runs counter to",
    r"at odds with",
    r"diverges from",
    r"challenges the claim that",
    r"counter to",
]
COMBINED_PATTERN = re.compile("|".join(DISAGREEMENT_PATTERNS), re.IGNORECASE)
```

- [ ] **Step 2: Write and run the verification script**

Create `/tmp/verify_round16_patterns.py` (throwaway, not committed):

```python
import sys
sys.path.insert(0, "/home/consul/prjs/hypocontra/src")
from mine_disagreement_from_surveys import COMBINED_PATTERN, EXCLUSION_PATTERN

# (sentence, expect_combined_match, expect_excluded)
CASES = [
    # New patterns must fire on a genuine-looking contrast sentence
    (r"On the other hand, \cite{smith2020} found no significant effect, unlike \cite{jones2019}.", True, False),
    (r"\cite{a2020} reported positive results; conversely, \cite{b2021} found the opposite outcome.", True, False),
    (r"This finding runs counter to \cite{lee2022}'s conclusions.", True, False),
    (r"The results reported by \cite{x2020} are at odds with \cite{y2021}.", True, False),
    (r"\cite{p2023}'s account diverges from \cite{q2019} on this point.", True, False),
    (r"\cite{r2020} challenges the claim that \cite{s2018} made about tokenization.", True, False),
    (r"This is counter to \cite{t2021}'s earlier finding.", True, False),
    # New trigger words + an already-known false sense -- EXCLUSION_PATTERN must
    # still catch these even though the OLD trigger word isn't the one that fired
    (r"On the other hand, the proposed classifier detects the contradiction between two sentences reliably.", True, True),
    (r"Conversely, annotators disagreed on whether the two statements were contradictory.", True, True),
    (r"This runs counter to the model's own reasoning, which corrects itself after further iterations.", True, True),
]

failures = 0
for sentence, expect_combined, expect_excluded in CASES:
    got_combined = bool(COMBINED_PATTERN.search(sentence))
    got_excluded = bool(EXCLUSION_PATTERN.search(sentence))
    ok = got_combined == expect_combined and got_excluded == expect_excluded
    status = "OK" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{status}] combined={got_combined} excluded={got_excluded} :: {sentence}")

print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
sys.exit(1 if failures else 0)
```

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 /tmp/verify_round16_patterns.py`
Expected: `10/10 passed`, every line `[OK]`.

- [ ] **Step 3: Delete the throwaway script**

Run: `rm /tmp/verify_round16_patterns.py`

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/mine_disagreement_from_surveys.py
git commit -m "Round 16: expand DISAGREEMENT_PATTERNS with 7 new contrast markers"
```

---

### Task 2: Build a reusable cached-survey rescan driver

**Files:**
- Create: `src/rescan_cached_surveys.py`

**Interfaces:**
- Consumes: `find_disagreement_sentences(tex_dir: Path) -> list[dict]`, `RESULTS_DIR: Path`, `SRC_CACHE_DIR: Path` — all already exported, unchanged, from `src/mine_disagreement_from_surveys.py`.
- Produces: a CSV at `results/<--out>` with columns `file, sentence, cited_keys, survey_arxiv_id, survey_title` — identical schema to every existing `survey_disagreement_sentences*.csv`, so it's a drop-in input to `src/resolve_survey_citations.py --input`.

This script exists because `mine_disagreement_from_surveys.main()` is built for *fresh* downloads: it calls `download_and_extract` (which no-ops for a dir that already exists, so it's not wrong to reuse it) but then unconditionally sleeps 2s per survey regardless of whether a network call happened — fine for ~50-900 fresh downloads per round, but a 3700+-directory rescan of *already-cached* sources would waste over 2 hours doing nothing. This script instead calls `find_disagreement_sentences` directly against each cached directory, with no download step and no sleep.

- [ ] **Step 1: Write the script**

```python
"""Round 16: re-scan every already-downloaded survey LaTeX source under
results/survey_sources/ against the CURRENT DISAGREEMENT_PATTERNS/
EXCLUSION_PATTERNS/ARTIFACT_PATTERN, with no new network calls. Exists
because mine_disagreement_from_surveys.main() is built for fresh downloads
(one download_and_extract + 2s sleep per survey) -- fine for a few hundred
new surveys per round, wasteful for rescanning thousands of already-cached
ones with an updated pattern set. Reusable for any future pattern-only
round, not just round 16.

Title lookup: results/survey_sources/<id>/ is gitignored bulk source data
with no title metadata of its own, so titles are joined in from the union
of every arxiv-search registry CSV this project has ever written (each has
an arxiv_id/title column). A directory with no matching registry row keeps
survey_title="" -- rare (only if a source was ever downloaded through a
path this script doesn't know about) and reported in the summary, not
fatal: resolve_survey_citations.py only reads survey_title for display,
never for matching.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mine_disagreement_from_surveys import RESULTS_DIR, SRC_CACHE_DIR, find_disagreement_sentences

TITLE_REGISTRY_FILES = [
    "survey_bulk_search.csv",
    "survey_bulk_search_part2.csv",
    "survey_bulk_search_overview.csv",
    "survey_bulk_search_broad.csv",
    "survey_bulk_search_title_synonyms.csv",
    "survey_bulk_search_title_synonyms_filtered.csv",
    "survey_bulk_search_title_synonyms_excluded_manual.csv",
    "additional_surveys_found.csv",
    "additional_surveys_found_v1_81.csv",
    "survey_batch2_truly_new.csv",
]


def build_title_map() -> dict[str, str]:
    title_map: dict[str, str] = {}
    for name in TITLE_REGISTRY_FILES:
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"arxiv_id": str})
        for arxiv_id, title in zip(df["arxiv_id"], df["title"]):
            title_map.setdefault(arxiv_id, title)
    return title_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output CSV filename under results/.")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N cached directories (smoke testing).")
    args = parser.parse_args()

    title_map = build_title_map()
    survey_dirs = sorted(p for p in SRC_CACHE_DIR.iterdir() if p.is_dir())
    if args.limit is not None:
        survey_dirs = survey_dirs[: args.limit]

    all_hits = []
    n_no_title = 0
    for i, tex_dir in enumerate(survey_dirs, 1):
        arxiv_id = tex_dir.name
        title = title_map.get(arxiv_id, "")
        if not title:
            n_no_title += 1
        hits = find_disagreement_sentences(tex_dir)
        for h in hits:
            h["survey_arxiv_id"] = arxiv_id
            h["survey_title"] = title
        all_hits.extend(hits)
        if i % 200 == 0:
            print(f"[{i}/{len(survey_dirs)}] {len(all_hits)} candidate sentences so far", flush=True)

    df = pd.DataFrame(all_hits, columns=["file", "sentence", "cited_keys", "survey_arxiv_id", "survey_title"])
    out_path = RESULTS_DIR / args.out
    df.to_csv(out_path, index=False)

    print(f"\nScanned {len(survey_dirs)} cached survey directories ({n_no_title} with no known title).")
    print(f"{len(df)} candidate disagreement sentences -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on 5 cached directories**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/rescan_cached_surveys.py --out tmp_smoke_round16.csv --limit 5`
Expected: exits 0, prints `Scanned 5 cached survey directories (...)`, and `results/tmp_smoke_round16.csv` exists with the 5-column header (`file,sentence,cited_keys,survey_arxiv_id,survey_title`) even if it has 0 data rows (5 surveys may genuinely have no contrast-marker hits — that's fine, this step only checks the script runs and writes the right shape). Note: `--out` is joined under `results/` via `RESULTS_DIR / args.out` — pass a bare filename, not a leading-slash path, or pathlib will treat it as absolute and write outside `results/` entirely.

Run: `rm ~/prjs/hypocontra/results/tmp_smoke_round16.csv` (smoke-test output, not the real round-16 result — don't commit it).

- [ ] **Step 3: Commit**

```bash
cd ~/prjs/hypocontra
git add src/rescan_cached_surveys.py
git commit -m "Round 16: add rescan_cached_surveys.py -- rescan cached sources without re-downloading"
```

---

### Task 3: Run the full rescan and isolate genuinely new candidates

**Files:**
- Create: `src/filter_new_candidates.py`
- Creates (data, committed): `results/survey_disagreement_sentences_round16.csv`, `results/survey_disagreement_sentences_round16_new.csv`

**Interfaces:**
- Consumes: `results/survey_disagreement_sentences_round16.csv` (Task 3 Step 1's output) plus every existing `results/survey_disagreement_sentences_*.csv` from prior rounds (read-only).
- Produces: `results/survey_disagreement_sentences_round16_new.csv` — same 5-column schema, containing only `(survey_arxiv_id, sentence)` pairs no earlier round's file already contains. This is the file Task 4 feeds into `resolve_survey_citations.py`.

- [ ] **Step 1: Run the full rescan**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/rescan_cached_surveys.py --out survey_disagreement_sentences_round16.csv`

This scans all ~3700 cached directories in `results/survey_sources/` with no network calls; expect it to complete in well under the ~2 hours a fresh-download round takes (pure local regex scanning). Note the printed total candidate-sentence count.

- [ ] **Step 2: Write the new-candidate filter script**

```python
"""Round 16: isolate the candidate sentences a pattern update caught that
no earlier round's narrower pattern set already saw, so already-resolved/
already-labeled sentences aren't re-issued to Claude for pair construction
a second time. Compares on the exact (survey_arxiv_id, sentence) pair --
the same key prepare_claude_pair_batch.py's downstream dedup already uses.
Reusable for any future pattern-only round, not just round 16.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="New round's rescan CSV, under results/.")
    parser.add_argument("--history-glob", type=str, default="survey_disagreement_sentences_*.csv")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    new_df = pd.read_csv(RESULTS_DIR / args.input, dtype={"survey_arxiv_id": str})

    seen: set[tuple[str, str]] = set()
    for path in sorted(RESULTS_DIR.glob(args.history_glob)):
        if path.name == args.input:
            continue
        hist_df = pd.read_csv(path, dtype={"survey_arxiv_id": str})
        seen.update(zip(hist_df["survey_arxiv_id"], hist_df["sentence"]))

    is_new = [
        (row.survey_arxiv_id, row.sentence) not in seen
        for row in new_df.itertuples()
    ]
    new_only = new_df[is_new].reset_index(drop=True)
    new_only.to_csv(RESULTS_DIR / args.output, index=False)

    print(f"{len(new_df)} total hits in {args.input}")
    print(f"{len(new_df) - len(new_only)} already seen in an earlier round's output")
    print(f"{len(new_only)} genuinely new -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/filter_new_candidates.py --input survey_disagreement_sentences_round16.csv --output survey_disagreement_sentences_round16_new.csv`

Note the three printed counts (total / already-seen / genuinely new) for the plan's final report to the user.

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/filter_new_candidates.py results/survey_disagreement_sentences_round16.csv results/survey_disagreement_sentences_round16_new.csv
git commit -m "Round 16: full rescan of cached surveys, isolate genuinely new candidate sentences"
```

---

### Task 4: Resolve citations and issue the round-16 Claude batch

**Files:**
- Creates (data, committed): `results/candidate_sentences_resolved_round16.csv`, `results/citation_resolution_dropped_round16.csv`, `results/claude_pair_batch_round16.jsonl`, `results/claude_batch_state_round16.json`
- Modify: `docs/next_scaling_directions.md` (record round 16's mining outcome)

**Interfaces:**
- Consumes: `results/survey_disagreement_sentences_round16_new.csv` (Task 3's output), via the existing, unmodified `src/resolve_survey_citations.py --input` and `src/prepare_claude_pair_batch.py --input`.
- Produces: `results/claude_pair_batch_round16.jsonl` — same shape as `claude_pair_batch_round14.jsonl` (one JSON object per sentence, ready for a live Claude Code session to read and construct pairs from, per `docs/claude_batch_instructions.md`). Pair construction itself is a separate, later, human-in-the-loop session step (same split every prior round used) and is explicitly out of scope for this plan.

- [ ] **Step 1: Resolve citations**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/resolve_survey_citations.py --input survey_disagreement_sentences_round16_new.csv --output-suffix _round16`

Note the printed eligible/dropped counts and the `drop_reason` breakdown.

- [ ] **Step 2: Issue the batch**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/prepare_claude_pair_batch.py --round 16 --input candidate_sentences_resolved_round16.csv --all`

(`--all` because this is a fresh round with no prior `claude_batch_state_round16.json` to resume from — matches how every prior round's first batch was issued.) Confirm the printed line `Issued <N> sentences into results/claude_pair_batch_round16.jsonl`.

- [ ] **Step 3: Record the outcome in docs/next_scaling_directions.md**

Add a new row to the "Вже перевірено" table (or a new paragraph directly above it, matching how round 15's entry was written) recording: pattern list added, total rescan hits, how many were already seen under the old patterns, how many were genuinely new, and how many survived citation resolution into the round-16 batch. Do not claim a pair count yet — that's unknown until a future session runs Claude pair construction on `claude_pair_batch_round16.jsonl`.

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add results/candidate_sentences_resolved_round16.csv results/citation_resolution_dropped_round16.csv results/claude_pair_batch_round16.jsonl results/claude_batch_state_round16.json docs/next_scaling_directions.md
git commit -m "Round 16: resolve citations and issue Claude pair-construction batch"
```

---

## Self-Review

**Spec coverage:** `docs/next_scaling_directions.md` candidate #1 specifies (a) adding the 7 named patterns — Task 1; (b) that it "acts on the already-mined 3019+ surveys" (independent of new source search) — Task 2/3 rescan cached sources with zero new downloads; (c) the stated risk requires "EXCLUSION_PATTERNS will have to be expanded in parallel" — addressed via the Global Constraints note: verified the *existing* exclusion filter still generalizes (Task 1 Step 2), left further expansion to a real-data follow-up rather than guessing, consistent with this project's own established methodology. Batch issuance (not pair construction) is the correct stopping point, matching every prior round's split between a scripted mining/issuing step and a separate manual Claude pair-construction step.

**Placeholder scan:** no TBD/TODO; every step has runnable code or an exact command.

**Type consistency:** `find_disagreement_sentences`, `RESULTS_DIR`, `SRC_CACHE_DIR` are used in Task 2 exactly as already defined in `mine_disagreement_from_surveys.py` (verified against that file's current source, not assumed). `filter_new_candidates.py`'s dedup key `(survey_arxiv_id, sentence)` matches `prepare_claude_pair_batch.py`'s own existing dedup key at `resolve_survey_citations.py:262` (`drop_duplicates(subset=["survey_arxiv_id", "sentence"])`). CSV schema (`file, sentence, cited_keys, survey_arxiv_id, survey_title`) is identical across Task 2's output and every existing `survey_disagreement_sentences*.csv`, matching what `resolve_survey_citations.py` already expects unmodified.
