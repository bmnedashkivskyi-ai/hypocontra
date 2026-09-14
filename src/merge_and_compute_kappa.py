"""Крок 5 масштабування round 7. Зливає annotator_A/B_round{N}_labels.csv у
hypocontra_pilot_master_dataset.csv (ідемпотентно -- рядки з тим самим
round замінюються, не дублюються) і рахує Cohen's kappa + confusion matrix.

method="survey_mining_dual_llm_pipeline" -- НОВЕ значення, відмінне від
попередніх (survey_mining_hand_picked/keyword_search/semantic_proxy/
narrowed_regex), щоб κ цього покоління методології ніколи не змішувався з
попередніми без явного прапорця (round 3-6 -- Claude-Claude AI-AI; round 7 --
Claude-Gemma, різні архітектури).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MASTER_PATH = RESULTS_DIR / "hypocontra_pilot_master_dataset.csv"

VALID_LABELS = {
    "type1_direct_negation",
    "type2_quantitative_conflict",
    "type3_causal_conflict",
    "type4_apparent_contextual",
    "not_contradiction",
}


def compute_kappa(labels_a: pd.Series, labels_b: pd.Series) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(labels_a, labels_b)
    except ImportError:
        categories = sorted(set(labels_a) | set(labels_b))
        n = len(labels_a)
        po = (labels_a.values == labels_b.values).sum() / n
        pe = sum(
            (labels_a == c).sum() / n * (labels_b == c).sum() / n
            for c in categories
        )
        if pe == 1.0:
            return float("nan")
        return (po - pe) / (1 - pe)


def load_and_validate_labels(path: Path, expected_pair_ids: set[str], annotator: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    bad_labels = df[~df["label"].isin(VALID_LABELS)]
    if len(bad_labels) > 0:
        print(f"  WARNING: {len(bad_labels)} rows in annotator {annotator} labels have non-taxonomy labels "
              f"(e.g. PARSE_ERROR/MISSING): {bad_labels['label'].value_counts().to_dict()}", flush=True)
    mismatched = set(df["pair_id"]) ^ expected_pair_ids
    if mismatched:
        print(f"  WARNING: pair_id mismatch between annotator {annotator} labels and curated sample: {mismatched}", flush=True)
    return df


def append_to_master(new_rows: pd.DataFrame, round_num: int) -> None:
    master = pd.read_csv(MASTER_PATH)
    master = master[master["round"] != round_num]
    master = pd.concat([master, new_rows], ignore_index=True)
    master.to_csv(MASTER_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=7)
    parser.add_argument("--method-name", type=str, default="survey_mining_dual_llm_pipeline")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    curated = pd.read_csv(RESULTS_DIR / f"pilot_round{args.round}_curated_sample.csv", dtype=str)
    expected_pair_ids = set(curated["pair_id"])

    labels_a = load_and_validate_labels(RESULTS_DIR / f"annotator_A_round{args.round}_labels.csv", expected_pair_ids, "A")
    labels_b = load_and_validate_labels(RESULTS_DIR / f"annotator_B_round{args.round}_labels.csv", expected_pair_ids, "B")

    merged = labels_a.merge(labels_b, on="pair_id", suffixes=("_A", "_B"))
    merged["both_flag_contradiction"] = (merged["label_A"] != "not_contradiction") & (merged["label_B"] != "not_contradiction")
    merged["labels_agree"] = merged["label_A"] == merged["label_B"]

    new_rows = pd.DataFrame({
        "round": args.round,
        "method": args.method_name,
        "pair_id": merged["pair_id"],
        "label_A": merged["label_A"],
        "label_B": merged["label_B"],
        "both_flag_contradiction": merged["both_flag_contradiction"],
        "labels_agree": merged["labels_agree"],
    })

    clean = merged[merged["label_A"].isin(VALID_LABELS) & merged["label_B"].isin(VALID_LABELS)]
    kappa = compute_kappa(clean["label_A"], clean["label_B"]) if len(clean) > 1 else float("nan")
    raw_agreement = clean["labels_agree"].mean() if len(clean) > 0 else float("nan") if "labels_agree" not in clean.columns else (clean["label_A"] == clean["label_B"]).mean()

    confusion = pd.crosstab(merged["label_A"], merged["label_B"])
    confusion_path = RESULTS_DIR / f"round{args.round}_confusion_matrix.csv"

    print(f"N pairs (both annotators, valid labels): {len(clean)} / {len(merged)} total merged", flush=True)
    print(f"Cohen's kappa: {kappa:.3f}" if kappa == kappa else "Cohen's kappa: NaN (degenerate)", flush=True)
    print(f"Raw agreement: {raw_agreement:.1%}" if raw_agreement == raw_agreement else "Raw agreement: NaN", flush=True)
    print("\nConfusion matrix (label_A rows x label_B cols):", flush=True)
    print(confusion.to_string(), flush=True)

    if not args.dry_run:
        confusion.to_csv(confusion_path)
        append_to_master(new_rows, args.round)
        print(f"\nAppended {len(new_rows)} rows to {MASTER_PATH} (round={args.round}, method={args.method_name})", flush=True)
        print(f"Confusion matrix written to {confusion_path}", flush=True)
    else:
        print("\n--dry-run: master dataset not modified.", flush=True)


if __name__ == "__main__":
    main()
