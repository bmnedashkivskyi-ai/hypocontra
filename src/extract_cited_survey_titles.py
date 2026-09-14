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
