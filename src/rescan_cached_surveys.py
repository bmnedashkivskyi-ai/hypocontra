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
