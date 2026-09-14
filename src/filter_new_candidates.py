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
        if path.name == args.output:
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
