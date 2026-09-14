"""Baseline NLI evaluation: shared macro-P/R/F1 computation with explicit
n/a handling for classes absent from the evaluation set (never a
fabricated 0.0), per
docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md's
Error Handling section.
"""
from __future__ import annotations

from sklearn.metrics import precision_recall_fscore_support


def compute_macro_prf(y_true: list[str], y_pred: list[str], all_classes: list[str]) -> dict:
    present_classes = sorted({c for c in y_true})
    if not present_classes:
        raise ValueError("compute_macro_prf: y_true contains none of the declared all_classes -- nothing to evaluate")
    absent_classes = [c for c in all_classes if c not in present_classes]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_classes, average=None, zero_division=0
    )

    per_class: dict = {}
    for cls, p, r, f, s in zip(present_classes, precision, recall, f1, support):
        per_class[cls] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
    for cls in absent_classes:
        per_class[cls] = "n/a"

    macro_precision = sum(v["precision"] for v in per_class.values() if v != "n/a") / len(present_classes)
    macro_recall = sum(v["recall"] for v in per_class.values() if v != "n/a") / len(present_classes)
    macro_f1 = sum(v["f1"] for v in per_class.values() if v != "n/a") / len(present_classes)

    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "n_classes_evaluated": len(present_classes),
        "n_classes_absent": len(absent_classes),
    }
