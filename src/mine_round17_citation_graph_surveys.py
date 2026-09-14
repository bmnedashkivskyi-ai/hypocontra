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
