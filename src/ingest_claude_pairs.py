"""Крок 2b масштабування round 7. Валідує вручну заповнений
results/claude_pair_batch_round{N}_completed.csv (за інструкцією
docs/claude_batch_instructions.md) і зливає у канонічний
results/pilot_round{N}_curated_sample.csv -- схема ідентична
pilot_round6_curated_sample.csv, щоб round7 був сумісний з існуючим
downstream-кодом (merge_and_compute_kappa.py, майбутні docs/pilot_results_round7.md).

Окремий скрипт від prepare_claude_pair_batch.py навмисно: помилка чи
неповнота ручного проходу ніколи не потрапляє напряму в канонічний файл
без валідації.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

REQUIRED_FOR_PAIR = ["topic", "hypothesis_a", "source_a", "hypothesis_b", "source_b"]


def validate_row(row: pd.Series) -> tuple[bool, str]:
    status = str(row.get("status", "")).strip()
    if status not in {"pair_created", "skipped"}:
        return False, f"unrecognized_status:{status!r}"
    if status == "skipped":
        return True, ""
    for field in REQUIRED_FOR_PAIR:
        val = row.get(field)
        if pd.isna(val) or not str(val).strip():
            return False, f"missing_field:{field}"
    if str(row["hypothesis_a"]).strip() == str(row["hypothesis_b"]).strip():
        return False, "hypothesis_a_equals_hypothesis_b"
    return True, ""


def assign_pair_ids(n: int, round_num: int, existing_ids: set[str]) -> list[str]:
    prefix = f"r{round_num}_"
    ids = []
    counter = 0
    while len(ids) < n:
        candidate = f"{prefix}{counter:03d}"
        if candidate not in existing_ids:
            ids.append(candidate)
            existing_ids.add(candidate)
        counter += 1
    return ids


def load_existing_pair_ids() -> set[str]:
    existing: set[str] = set()
    for path in RESULTS_DIR.glob("pilot_round*_curated_sample.csv"):
        df = pd.read_csv(path)
        if "pair_id" in df.columns:
            existing.update(df["pair_id"].astype(str))
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=7)
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else RESULTS_DIR / f"claude_pair_batch_round{args.round}_completed.csv"
    df = pd.read_csv(input_path, dtype=str)

    seen_sentence_ids: set[str] = set()
    dup_mask = []
    for sid in df["sentence_id"]:
        dup_mask.append(sid in seen_sentence_ids)
        seen_sentence_ids.add(sid)

    valid_rows = []
    rejected_rows = []
    for i, row in df.iterrows():
        if dup_mask[i]:
            rejected_rows.append({**row.to_dict(), "reject_reason": "duplicate_sentence_id"})
            continue
        ok, reason = validate_row(row)
        if not ok:
            rejected_rows.append({**row.to_dict(), "reject_reason": reason})
            continue
        if str(row["status"]).strip() == "skipped":
            rejected_rows.append({**row.to_dict(), "reject_reason": "skipped_by_annotator"})
            continue
        valid_rows.append(row)

    existing_ids = load_existing_pair_ids()
    pair_ids = assign_pair_ids(len(valid_rows), args.round, existing_ids)

    curated_records = []
    for pair_id, row in zip(pair_ids, valid_rows):
        curated_records.append({
            "pair_id": pair_id,
            "topic": row["topic"],
            "hypothesis_a": row["hypothesis_a"],
            "source_a": row["source_a"],
            "hypothesis_b": row["hypothesis_b"],
            "source_b": row["source_b"],
        })

    curated_df = pd.DataFrame(curated_records, columns=["pair_id", "topic", "hypothesis_a", "source_a", "hypothesis_b", "source_b"])
    rejected_df = pd.DataFrame(rejected_rows)

    curated_out = RESULTS_DIR / f"pilot_round{args.round}_curated_sample.csv"
    rejected_out = RESULTS_DIR / f"pilot_round{args.round}_construction_rejected.csv"
    curated_df.to_csv(curated_out, index=False)
    rejected_df.to_csv(rejected_out, index=False)

    print(f"Input rows: {len(df)}", flush=True)
    print(f"Pairs created: {len(curated_df)} -> {curated_out}", flush=True)
    print(f"Rejected/skipped: {len(rejected_df)} -> {rejected_out}", flush=True)
    if len(rejected_df) > 0:
        print(rejected_df["reject_reason"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
