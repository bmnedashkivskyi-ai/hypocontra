"""Baseline generalization check (article Section 5.3): fine-tune
roberta-large-mnli on the Alamri & Stevenson (2016) biomedical contradiction
pairs (results/alamri_stevenson_training_pairs.csv, built by
build_alamri_stevenson_training_pairs.py), then zero-shot evaluate on
HypoContra's own n=59 3-way ground-truth eval set
(results/baseline_ground_truth_agreement.csv) -- no fine-tuning on
HypoContra at all. This is the opposite direction from
baseline_finetune_roberta.py (which fine-tunes and evaluates in-domain via
cross-validation): here the eval set is held out entirely from training,
model transfer across domains is what's being measured.

The Alamri & Stevenson pairs never carry an "Apparent" label (their source
scheme is binary YS/NO), so zero-shot recall on that class is expected to
be near zero -- an inherent property of the training data, reported
honestly rather than hidden.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline_finetune_roberta import LABELS, LABEL_TO_ID, MODEL_NAME, build_dataset, predict
from baseline_metrics import compute_macro_prf

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
N_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
DEV_FRACTION = 0.1


def train_with_dev_selection(train_encodings, train_labels, dev_encodings, dev_labels, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), ignore_mismatched_sizes=True
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n = train_labels.shape[0]
    dev_true = [LABELS[i] for i in dev_labels.tolist()]

    best_dev_f1 = -1.0
    best_state = None
    training_log_rows = []

    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch = {k: v[idx].to(device) for k, v in train_encodings.items()}
            batch_labels = train_labels[idx].to(device)
            optimizer.zero_grad()
            outputs = model(**batch, labels=batch_labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item() * len(idx)
        train_loss = total_loss / n

        dev_pred_ids = predict(model, dev_encodings, device)
        dev_pred = [LABELS[i] for i in dev_pred_ids]
        dev_metrics = compute_macro_prf(dev_true, dev_pred, LABELS)
        dev_f1 = dev_metrics["macro_f1"]

        training_log_rows.append({"epoch": epoch, "train_loss": train_loss, "dev_macro_f1": dev_f1})
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} dev_macro_f1={dev_f1:.4f}", flush=True)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, training_log_rows, best_dev_f1


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    torch.manual_seed(42)

    pairs = pd.read_csv(RESULTS_DIR / "alamri_stevenson_training_pairs.csv", dtype=str)
    train_df, dev_df = train_test_split(
        pairs, test_size=DEV_FRACTION, random_state=42, stratify=pairs["label_3way"]
    )
    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    print(f"train={len(train_df)} dev={len(dev_df)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings, train_labels = build_dataset(train_df, tokenizer)
    dev_encodings, dev_labels = build_dataset(dev_df, tokenizer)

    model, training_log_rows, best_dev_f1 = train_with_dev_selection(
        train_encodings, train_labels, dev_encodings, dev_labels, device
    )
    print(f"Best dev macro F1 (Alamri & Stevenson held-out split): {best_dev_f1:.4f}", flush=True)

    log_out = pd.DataFrame(training_log_rows, columns=["epoch", "train_loss", "dev_macro_f1"])
    log_path = RESULTS_DIR / "generalization_alamri_stevenson_training_log.csv"
    log_out.to_csv(log_path, index=False)
    print(f"Training log -> {log_path}")

    eval_df = pd.read_csv(RESULTS_DIR / "baseline_ground_truth_agreement.csv", dtype=str)
    eval_encodings, _ = build_dataset(eval_df, tokenizer)
    pred_ids = predict(model, eval_encodings, device)
    predicted_3way = [LABELS[i] for i in pred_ids]

    out = pd.DataFrame({"pair_id": eval_df["pair_id"], "predicted_3way": predicted_3way})
    out_path = RESULTS_DIR / "generalization_alamri_stevenson_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nn={len(out)} zero-shot predictions on HypoContra -> {out_path}")
    print(out["predicted_3way"].value_counts().to_string())

    metrics = compute_macro_prf(eval_df["label_3way"].tolist(), predicted_3way, LABELS)
    print(f"\nZero-shot transfer macro F1 on HypoContra: {metrics['macro_f1']:.4f}")
    for cls, v in metrics["per_class"].items():
        print(f"  {cls}: {v}")

    write_report(metrics, best_dev_f1, len(train_df), len(dev_df), len(eval_df))


def write_report(metrics: dict, best_dev_f1: float, n_train: int, n_dev: int, n_eval: int) -> None:
    lines = [
        "# HypoContra Baseline Generalization Report (Section 5.3)",
        "",
        "Fine-tuned roberta-large-mnli on the Alamri & Stevenson (2016) biomedical "
        "contradiction corpus (results/alamri_stevenson_training_pairs.csv), then "
        "evaluated zero-shot on HypoContra's own n=59 3-way ground-truth set "
        "(results/baseline_ground_truth_agreement.csv) -- no fine-tuning on HypoContra.",
        "",
        f"Training pairs: n_train={n_train}, n_dev={n_dev} (held out for best-epoch "
        f"selection only, not reported as a result). Best dev macro F1: {best_dev_f1:.4f}.",
        "",
        f"## Zero-shot transfer result on HypoContra (n={n_eval})",
        "",
        f"**Macro F1: {metrics['macro_f1']:.4f}** (macro precision={metrics['macro_precision']:.4f}, "
        f"macro recall={metrics['macro_recall']:.4f})",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for cls, v in metrics["per_class"].items():
        if v == "n/a":
            lines.append(f"| {cls} | n/a | n/a | n/a | n/a |")
        else:
            lines.append(f"| {cls} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")

    lines += [
        "",
        "## Caveat: `Apparent` is structurally absent from training data",
        "",
        "Alamri & Stevenson's source scheme is binary (`ASSERTION=\"YS\"`/`\"NO\"`, "
        "support vs. contradict a shared clinical question) -- there is no equivalent "
        "of HypoContra's type4/`Apparent` (contextual, non-logical) class. Every "
        "training pair is labeled either `Contradiction` or `NotContradiction`; the "
        "model never sees an `Apparent` training example. Weak zero-shot recall on "
        "`Apparent` reflects this gap in the training data's label space, not a flaw "
        "in the model or this script.",
    ]

    out_path = DOCS_DIR / "generalization_report.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
