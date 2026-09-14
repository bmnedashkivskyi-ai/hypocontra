"""Baseline NLI evaluation: RoBERTa fine-tuned baseline, 5-fold stratified
cross-validation on the 3-way task ONLY (n=59 is too small for a single
train/test split -- see
docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md
Decision 3). 5-class fine-tuning is not attempted: type2/type3 have 0
examples in this eval set, so stratified folds cannot be constructed for
them at all.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_NAME = "roberta-large-mnli"
LABELS = ["Contradiction", "Apparent", "NotContradiction"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
N_FOLDS = 5
N_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE = 2e-5


def build_dataset(df: pd.DataFrame, tokenizer, max_length: int = 256):
    encodings = tokenizer(
        df["hypothesis_a"].tolist(), df["hypothesis_b"].tolist(),
        truncation=True, padding=True, max_length=max_length, return_tensors="pt",
    )
    labels = torch.tensor([LABEL_TO_ID[label] for label in df["label_3way"]])
    return encodings, labels


def train_one_fold(train_encodings, train_labels, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), ignore_mismatched_sizes=True
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    n = train_labels.shape[0]
    model.train()
    epoch_losses = []
    for epoch in range(N_EPOCHS):
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
        epoch_losses.append(total_loss / n)
    return model, epoch_losses


def predict(model, encodings, device) -> list[int]:
    model.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in encodings.items()}
        logits = model(**batch).logits
        return logits.argmax(dim=-1).cpu().tolist()


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "baseline_ground_truth_agreement.csv", dtype=str).reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # NOTE: added retroactively (fix-wave for whole-branch review Finding 3) so FUTURE
    # runs of this script are reproducible. The StratifiedKFold split was already seeded,
    # but batch-shuffling (torch.randperm below) was not -- this does not affect any
    # already-computed/reported result, since the run that produced those numbers was not
    # re-executed.
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    y = df["label_3way"].tolist()

    all_predictions: list[str | None] = [None] * len(df)
    all_folds: list[int | None] = [None] * len(df)
    training_log_rows = []

    for fold_idx, (train_idx, eval_idx) in enumerate(skf.split(df, y)):
        print(f"[fold {fold_idx + 1}/{N_FOLDS}] train={len(train_idx)} eval={len(eval_idx)}", flush=True)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        eval_df = df.iloc[eval_idx].reset_index(drop=True)

        train_encodings, train_labels = build_dataset(train_df, tokenizer)
        eval_encodings, _ = build_dataset(eval_df, tokenizer)

        model, epoch_losses = train_one_fold(train_encodings, train_labels, device)
        for epoch, loss in enumerate(epoch_losses):
            training_log_rows.append({"fold": fold_idx, "epoch": epoch, "train_loss": loss})
            print(f"  epoch {epoch}: train_loss={loss:.4f}", flush=True)

        pred_ids = predict(model, eval_encodings, device)
        for local_i, global_i in enumerate(eval_idx):
            all_predictions[global_i] = LABELS[pred_ids[local_i]]
            all_folds[global_i] = fold_idx

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame({"pair_id": df["pair_id"], "predicted_3way": all_predictions, "fold": all_folds})
    out_path = RESULTS_DIR / "baseline_roberta_predictions.csv"
    out.to_csv(out_path, index=False)

    log_out = pd.DataFrame(training_log_rows, columns=["fold", "epoch", "train_loss"])
    log_path = RESULTS_DIR / "baseline_roberta_training_log.csv"
    log_out.to_csv(log_path, index=False)

    print(f"\n{len(out)} pooled out-of-fold predictions -> {out_path}")
    print(out["predicted_3way"].value_counts().to_string())
    print(f"Training log -> {log_path}")


if __name__ == "__main__":
    main()
