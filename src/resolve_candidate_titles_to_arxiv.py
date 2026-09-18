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

from find_surveys_bulk import ARXIV_NS, PAGE_DELAY, PersistentBlockError, fetch_page
from mine_disagreement_from_surveys import SRC_CACHE_DIR

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CACHE_PATH = RESULTS_DIR / "round17_title_resolution_cache.jsonl"
SKIPPED_LOG_PATH = RESULTS_DIR / "round18_skipped_titles.txt"

# Cap on fetch_page's retry loop for a single title: ~1050s (~17.5 min) of
# backoff before giving up on THIS title and moving on, instead of the
# unbounded retry that stalled the whole run for 4.5+ hours on one query
# (see PersistentBlockError's docstring in find_surveys_bulk.py).
MAX_ATTEMPTS_PER_TITLE = 6


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


def resolve_one(title: str) -> dict | None:
    """Returns None if arXiv persistently refuses this exact query (see
    PersistentBlockError) -- the caller should skip it, not cache it, so a
    later run can retry it fresh."""
    escaped = title.replace('"', "'")
    query = f'ti:"{escaped}"'
    try:
        root = fetch_page(0, 3, query, max_attempts=MAX_ATTEMPTS_PER_TITLE)
    except PersistentBlockError as e:
        print(f"  SKIPPING title after persistent block: {e}", flush=True)
        with SKIPPED_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(title + "\n")
        return None
    entries = root.findall("atom:entry", ARXIV_NS)
    target_norm = normalize_title(title)
    for entry in entries:
        entry_title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
        if normalize_title(entry_title) == target_norm:
            # Round 18 (2026-09-18): .split("/")[-1] silently dropped the
            # category prefix (e.g. "cs/") that pre-2007 old-style arXiv IDs
            # require -- "cs/0504061" became bare "0504061", which then 404s
            # against arxiv.org/e-print/. Splitting on "abs/" instead keeps
            # the prefix for old-style IDs while new-style ("2011.00362")
            # IDs are unaffected (no "abs/" substring inside them to split on).
            raw_id = entry.find("atom:id", ARXIV_NS).text.strip()
            arxiv_id = raw_id.split("abs/")[-1].split("v")[0]
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
            if record is None:
                # Persistently blocked -- not cached, so a later run retries
                # it fresh instead of either stalling here again or wrongly
                # treating it as a permanent no-match.
                time.sleep(PAGE_DELAY)
                continue
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
