"""Baseline NLI evaluation: LLM few-shot baseline. Reuses ALREADY-COLLECTED
Gemma (Annotator B) labels against Claude (Annotator A) as reference, on
ALL round-7+ valid-labeled pairs (n=85) -- NOT the n=59 agreement-only
ground truth, which would be circular (Gemma's own labels partly DEFINE
that set). NO new model calls. See
docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md
Decision 4 for the full rationale.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from baseline_metrics import compute_macro_prf

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ALL_5WAY_CLASSES = [
    "type1_direct_negation", "type2_quantitative_conflict", "type3_causal_conflict",
    "type4_apparent_contextual", "not_contradiction",
]
ALL_3WAY_CLASSES = ["Contradiction", "Apparent", "NotContradiction"]


def metrics_to_rows(metrics: dict, scheme: str, n: int) -> list[dict]:
    rows = []
    for cls, v in metrics["per_class"].items():
        if v == "n/a":
            rows.append({"scheme": scheme, "class": cls, "precision": "n/a", "recall": "n/a", "f1": "n/a", "support": 0})
        else:
            rows.append({"scheme": scheme, "class": cls, "precision": v["precision"], "recall": v["recall"], "f1": v["f1"], "support": v["support"]})
    rows.append({
        "scheme": scheme, "class": "MACRO_AVG",
        "precision": metrics["macro_precision"], "recall": metrics["macro_recall"], "f1": metrics["macro_f1"],
        "support": n,
    })
    return rows


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "baseline_all_round7plus_labels.csv", dtype=str)

    metrics_3way = compute_macro_prf(df["label_A_3way"].tolist(), df["label_B_3way"].tolist(), ALL_3WAY_CLASSES)
    metrics_5way = compute_macro_prf(df["label_A_5way"].tolist(), df["label_B_5way"].tolist(), ALL_5WAY_CLASSES)

    rows = metrics_to_rows(metrics_3way, "3way", len(df)) + metrics_to_rows(metrics_5way, "5way", len(df))
    out = pd.DataFrame(rows, columns=["scheme", "class", "precision", "recall", "f1", "support"])
    out_path = RESULTS_DIR / "baseline_llm_fewshot_metrics.csv"
    out.to_csv(out_path, index=False)

    note_path = RESULTS_DIR / "baseline_llm_fewshot_metrics_NOTE.md"
    note_path.write_text(
        "# LLM few-shot baseline -- methodology note\n\n"
        "This baseline reuses Gemma's (Annotator B) already-collected labels from the "
        "corpus's own annotation rounds -- NO new model calls were made. Metrics are "
        "computed treating Claude (Annotator A) as the reference label, across all "
        f"n={len(df)} round-7+ pairs with valid labels from both annotators (not the "
        "smaller agreement-only ground-truth set used for the rule-based and RoBERTa "
        "baselines, since that set is *defined* by Gemma agreeing with Claude and would "
        "make this evaluation circular). These numbers therefore measure *agreement "
        "with Claude's judgment*, not independent accuracy against gold labels, and are "
        "numerically a recomputation of the same combined Cohen's kappa already on "
        "record for round 7-17 (kappa=0.556), expressed as per-class macro "
        "precision/recall/F1 instead of a single kappa statistic.\n",
        encoding="utf-8",
    )

    print(f"{len(df)} pairs -> {out_path}")
    print(f"3-way macro F1: {metrics_3way['macro_f1']:.3f}")
    print(f"5-way macro F1: {metrics_5way['macro_f1']:.3f}")
    print(f"Note written to {note_path}")


if __name__ == "__main__":
    main()
