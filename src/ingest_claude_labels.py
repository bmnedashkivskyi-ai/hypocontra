"""Крок 3b масштабування round 7. Валідує results/claude_label_batch_round{N}_completed.csv
(сліпа розмітка Анотатора A, крок 3a) і пише results/annotator_A_round{N}_labels.csv --
схема ідентична annotator_A_round6_labels.csv (pair_id, label, rationale).
"""
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=7)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--annotator", type=str, default="A", choices=["A", "B"])
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else RESULTS_DIR / f"claude_label_batch_round{args.round}_completed.csv"
    df = pd.read_csv(input_path, dtype=str)

    expected_ids = set(pd.read_csv(RESULTS_DIR / f"pilot_round{args.round}_curated_sample.csv", dtype=str)["pair_id"])

    valid_rows = []
    rejected_rows = []
    seen = set()
    for _, row in df.iterrows():
        pair_id = row.get("pair_id")
        label = str(row.get("label", "")).strip()
        reason = None
        if pair_id in seen:
            reason = "duplicate_pair_id"
        elif pair_id not in expected_ids:
            reason = "pair_id_not_in_curated_sample"
        elif label not in VALID_LABELS:
            reason = f"invalid_label:{label!r}"
        seen.add(pair_id)
        if reason:
            rejected_rows.append({**row.to_dict(), "reject_reason": reason})
        else:
            valid_rows.append(row)

    missing = expected_ids - seen
    for pair_id in missing:
        rejected_rows.append({"pair_id": pair_id, "label": "", "rationale": "", "reject_reason": "missing_from_batch"})

    out_path = RESULTS_DIR / f"annotator_{args.annotator}_round{args.round}_labels.csv"
    rejected_path = RESULTS_DIR / f"annotator_{args.annotator}_round{args.round}_labels_rejected.csv"

    pd.DataFrame(valid_rows, columns=["pair_id", "label", "rationale"]).to_csv(out_path, index=False)
    pd.DataFrame(rejected_rows).to_csv(rejected_path, index=False)

    print(f"Valid labels: {len(valid_rows)} -> {out_path}", flush=True)
    print(f"Rejected/missing: {len(rejected_rows)} -> {rejected_path}", flush=True)


if __name__ == "__main__":
    main()
