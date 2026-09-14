"""Baseline generalization check (article Section 5.3), second corpus:
fine-tune roberta-large-mnli on NLI4CT (results/nli4ct_training_pairs.csv,
built by build_nli4ct_training_pairs.py), then zero-shot evaluate on
HypoContra's own n=59 3-way ground-truth eval set
(results/baseline_ground_truth_agreement.csv) -- no fine-tuning on
HypoContra at all. Mirrors baseline_generalization_alamri_stevenson.py;
the two scripts give two independent crossdomain data points for Section 5.3
(see docs/s53_alternative_generalization_corpora.md for why a second corpus
was added: NLI4CT is more recent (2023-2024 vs. 2016) and larger, but
shares Alamri & Stevenson's binary label scheme -- "Apparent" is absent
from training data here too, for the same structural reason).

train/dev split is NLI4CT's own (SemEval organizers' split), not a fresh
random split like the Alamri & Stevenson script -- dev is still used for
best-epoch selection only, never reported as a result.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from baseline_finetune_roberta import LABELS, MODEL_NAME, build_dataset, predict
from baseline_generalization_alamri_stevenson import train_with_dev_selection
from baseline_metrics import compute_macro_prf

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    torch.manual_seed(42)

    pairs = pd.read_csv(RESULTS_DIR / "nli4ct_training_pairs.csv", dtype=str)
    train_df = pairs[pairs["split"] == "train"].reset_index(drop=True)
    dev_df = pairs[pairs["split"] == "dev"].reset_index(drop=True)
    print(f"train={len(train_df)} dev={len(dev_df)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # max_length=512 (roberta-large-mnli's position-embedding limit), not the
    # 256 default used by the other baseline scripts: NLI4CT's hypothesis_b is
    # evidence text resolved from Clinical Trial Report sections, not a single
    # claim sentence -- median 178 tokens, up to 1954, with ~38% of pairs
    # exceeding 256 combined tokens (vs. 0% for Alamri & Stevenson). Verified
    # this alone does not change the qualitative training outcome (see
    # docs/generalization_report_nli4ct.md) -- kept anyway because needlessly
    # truncating ~38% of pairs is a real defect independent of that outcome.
    train_encodings, train_labels = build_dataset(train_df, tokenizer, max_length=512)
    dev_encodings, dev_labels = build_dataset(dev_df, tokenizer, max_length=512)

    model, training_log_rows, best_dev_f1 = train_with_dev_selection(
        train_encodings, train_labels, dev_encodings, dev_labels, device
    )
    print(f"Best dev macro F1 (NLI4CT held-out dev split): {best_dev_f1:.4f}", flush=True)

    log_out = pd.DataFrame(training_log_rows, columns=["epoch", "train_loss", "dev_macro_f1"])
    log_path = RESULTS_DIR / "generalization_nli4ct_training_log.csv"
    log_out.to_csv(log_path, index=False)
    print(f"Training log -> {log_path}")

    eval_df = pd.read_csv(RESULTS_DIR / "baseline_ground_truth_agreement.csv", dtype=str)
    eval_encodings, _ = build_dataset(eval_df, tokenizer)
    pred_ids = predict(model, eval_encodings, device)
    predicted_3way = [LABELS[i] for i in pred_ids]

    out = pd.DataFrame({"pair_id": eval_df["pair_id"], "predicted_3way": predicted_3way})
    out_path = RESULTS_DIR / "generalization_nli4ct_predictions.csv"
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
        "# HypoContra Baseline Generalization Report (Section 5.3) -- NLI4CT",
        "",
        "Fine-tuned roberta-large-mnli on NLI4CT (SemEval-2023 Task 7 / SemEval-2024 "
        "Task 2, results/nli4ct_training_pairs.csv), then evaluated zero-shot on "
        "HypoContra's own n=59 3-way ground-truth set "
        "(results/baseline_ground_truth_agreement.csv) -- no fine-tuning on HypoContra. "
        "Second, independent crossdomain data point for Section 5.3, alongside "
        "docs/generalization_report.md (Alamri & Stevenson, 2016); see "
        "docs/s53_alternative_generalization_corpora.md for why this corpus was added.",
        "",
        f"Training pairs: n_train={n_train}, n_dev={n_dev} (NLI4CT's own SemEval "
        f"train/dev split; dev held out for best-epoch selection only, not reported "
        f"as a result). Best dev macro F1: {best_dev_f1:.4f}.",
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
        "NLI4CT's source scheme is binary (`Label=\"Entailment\"`/`\"Contradiction\"`, "
        "does a statement match or contradict the evidence from one or two Clinical "
        "Trial Reports) -- there is no equivalent of HypoContra's type4/`Apparent` "
        "(contextual, non-logical) class, the same structural gap as in the Alamri & "
        "Stevenson corpus. Weak zero-shot recall on `Apparent` reflects this gap in "
        "the training data's label space, not a flaw in the model or this script.",
        "",
        "## Note: hypothesis_b is evidence text, not a second claim",
        "",
        "Unlike Alamri & Stevenson (both hypothesis_a and hypothesis_b are claims "
        "from research abstracts), here hypothesis_a is the Statement and "
        "hypothesis_b is evidence sentences resolved from the referenced Clinical "
        "Trial Report section(s) (concatenated primary + secondary trial evidence "
        "for \"Comparison\"-type instances). This is a claim-vs-evidence NLI task, "
        "structurally closer to fact verification (e.g. SciFact) than to "
        "HypoContra's claim-vs-claim pairs -- a difference worth weighing when "
        "interpreting how this result compares to the Alamri & Stevenson crossdomain "
        "point.",
    ]

    out_path = DOCS_DIR / "generalization_report_nli4ct.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
