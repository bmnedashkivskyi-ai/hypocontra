"""Baseline NLI evaluation, step 1
(docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md):
build the two ground-truth eval-set files every later baseline script
consumes. Pure local CSV joins -- no model calls, no network.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

VALID_LABELS = {
    "type1_direct_negation",
    "type2_quantitative_conflict",
    "type3_causal_conflict",
    "type4_apparent_contextual",
    "not_contradiction",
}

THREE_WAY_MAP = {
    "type1_direct_negation": "Contradiction",
    "type2_quantitative_conflict": "Contradiction",
    "type3_causal_conflict": "Contradiction",
    "type4_apparent_contextual": "Apparent",
    "not_contradiction": "NotContradiction",
}


def load_curated_pairs() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS_DIR.glob("pilot_round*_curated_sample.csv")):
        df = pd.read_csv(path, dtype=str)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["pair_id", "topic", "hypothesis_a", "source_a", "hypothesis_b", "source_b"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    master = pd.read_csv(RESULTS_DIR / "hypocontra_pilot_master_dataset.csv", dtype=str)
    combo = master[master["method"] == "survey_mining_dual_llm_pipeline"]
    clean = combo[combo["label_A"].isin(VALID_LABELS) & combo["label_B"].isin(VALID_LABELS)]

    pairs_text = load_curated_pairs()

    # --- n=85 set: labels only, both annotators valid, agreement NOT required ---
    all_labels = clean[["pair_id", "label_A", "label_B"]].copy()
    all_labels["label_A_3way"] = all_labels["label_A"].map(THREE_WAY_MAP)
    all_labels["label_B_3way"] = all_labels["label_B"].map(THREE_WAY_MAP)
    all_labels = all_labels.rename(columns={"label_A": "label_A_5way", "label_B": "label_B_5way"})
    all_labels = all_labels[["pair_id", "label_A_5way", "label_A_3way", "label_B_5way", "label_B_3way"]]
    all_out = RESULTS_DIR / "baseline_all_round7plus_labels.csv"
    all_labels.to_csv(all_out, index=False)

    # --- n=59 set: agreement rows only, joined with hypothesis text ---
    agree = clean[clean["label_A"] == clean["label_B"]].copy()
    agree = agree.merge(pairs_text, on="pair_id", how="left")
    missing_text = agree[agree["hypothesis_a"].isna()]
    if len(missing_text) > 0:
        print(f"WARNING: {len(missing_text)} agreement rows have no matching curated-sample "
              f"text: {missing_text['pair_id'].tolist()}", flush=True)
    agree["label_5way"] = agree["label_A"]
    agree["label_3way"] = agree["label_5way"].map(THREE_WAY_MAP)
    agree = agree[["pair_id", "topic", "hypothesis_a", "source_a", "hypothesis_b", "source_b", "label_5way", "label_3way"]]
    ground_truth_out = RESULTS_DIR / "baseline_ground_truth_agreement.csv"
    agree.to_csv(ground_truth_out, index=False)

    print(f"n={len(all_labels)} (all valid round7+ labels) -> {all_out}")
    print(f"  label_A_5way:\n{all_labels['label_A_5way'].value_counts().to_string()}")
    print(f"  label_B_5way:\n{all_labels['label_B_5way'].value_counts().to_string()}")
    print()
    print(f"n={len(agree)} (agreement round7+, with text) -> {ground_truth_out}")
    print(f"  label_5way:\n{agree['label_5way'].value_counts().to_string()}")
    print(f"  label_3way:\n{agree['label_3way'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
