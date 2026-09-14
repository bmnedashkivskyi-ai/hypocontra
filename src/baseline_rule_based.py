"""Baseline NLI evaluation: rule-based lexical baseline (negation/antonym
patterns) -- the deliberate lower-complexity-bound baseline per
docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md
component 2. Never predicts "Apparent" -- an intentional, documented
limitation of a lexical-only heuristic, not a bug.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

NEGATION_PATTERN = re.compile(r"\b(not|no|never|cannot|neither|nor)\b|n't\b", re.IGNORECASE)

ANTONYM_PAIRS = [
    ("increase", "decrease"), ("improve", "worsen"), ("more", "less"),
    ("better", "worse"), ("higher", "lower"), ("positive", "negative"),
    ("support", "refute"), ("confirm", "contradict"),
]


def predict(hypothesis_a: str, hypothesis_b: str) -> str:
    a_lower, b_lower = hypothesis_a.lower(), hypothesis_b.lower()
    a_neg, b_neg = bool(NEGATION_PATTERN.search(a_lower)), bool(NEGATION_PATTERN.search(b_lower))
    if a_neg != b_neg:
        return "Contradiction"
    for word1, word2 in ANTONYM_PAIRS:
        if (word1 in a_lower and word2 in b_lower) or (word2 in a_lower and word1 in b_lower):
            return "Contradiction"
    return "NotContradiction"


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "baseline_ground_truth_agreement.csv", dtype=str)
    predictions = [predict(row["hypothesis_a"], row["hypothesis_b"]) for _, row in df.iterrows()]
    out = pd.DataFrame({"pair_id": df["pair_id"], "predicted_3way": predictions})
    out_path = RESULTS_DIR / "baseline_rule_based_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"{len(out)} predictions -> {out_path}")
    print(out["predicted_3way"].value_counts().to_string())


if __name__ == "__main__":
    main()
