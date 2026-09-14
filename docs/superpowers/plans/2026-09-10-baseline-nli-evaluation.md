# Baseline NLI-model evaluation for HypoContra (article §5, item "в") — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run all three §5.1 baselines (rule-based, fine-tuned RoBERTa, LLM few-shot) against the HypoContra corpus, compute §5.2's macro P/R/F1 metrics honestly (including explicit `n/a` for classes with zero examples, never a fabricated number), and produce one consolidated report.

**Architecture:** Five new, flat scripts in `~/prjs/hypocontra/src/` (no subdirectory, matching project convention), sharing one eval-set-building step and one shared metrics utility, ending in one report generator. Data flow: `build_baseline_eval_set.py` → {`baseline_rule_based.py`, `baseline_finetune_roberta.py`, `baseline_llm_fewshot_report.py`} → `evaluate_baselines.py`. The three baseline scripts are independent of each other and could in principle run in parallel, but this plan sequences them as separate tasks for individually reviewable, bite-sized commits.

**Tech Stack:** Python 3, pandas, scikit-learn (`StratifiedKFold`, `precision_recall_fscore_support` — already installed), PyTorch + HuggingFace `transformers` (not yet installed — add via `.venv/bin/pip install torch transformers`, no `requirements.txt` exists in this project, dependencies are installed directly into `.venv` by established precedent). GPU available (confirmed via `nvidia-smi` during design). No pytest — verify with printed counts and manual spot-checks, matching every prior round's convention in this project.

**Spec:** `docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md` — read this in full before starting; it is the authority this plan argues from, including the 5 brainstormed methodological decisions this plan implements without re-deriving them.

## Global Constraints

- Every CSV read in this pipeline uses `dtype=str` for all columns (pair_id, labels) — no numeric coercion anywhere, matching this project's established convention against silent ID/label corruption.
- **No new Gemma model calls anywhere in this plan.** The LLM few-shot baseline (Task 4) reuses `label_B` values already collected in `results/hypocontra_pilot_master_dataset.csv` — it must not import or call anything from `annotate_with_gemma.py`, and must not hit `http://172.28.224.1:1234`.
- Do not modify any existing round-based script (`resolve_survey_citations.py`, `prepare_claude_pair_batch.py`, `annotate_with_gemma.py`, `merge_and_compute_kappa.py`, `ingest_claude_pairs.py`, `ingest_claude_labels.py`, or any `find_surveys_bulk.py`/mining script) — every task in this plan only reads their output files.
- Ground truth is fixed as: `results/hypocontra_pilot_master_dataset.csv` rows where `method == "survey_mining_dual_llm_pipeline"` (round 7+) — the n=59 subset where `label_A == label_B` for the rule-based/RoBERTa baselines, the wider n=85 subset (`label_A`/`label_B` both valid, agreement not required) for the LLM few-shot baseline. Do not use any other subset or any pilot round 3-6 data (those used two Claude passes, not Claude-vs-Gemma — a weaker independence claim, excluded from ground truth per the design doc's Decision 1).
- The 3-way label mapping is fixed: `{type1_direct_negation, type2_quantitative_conflict, type3_causal_conflict} → "Contradiction"`, `{type4_apparent_contextual} → "Apparent"`, `{not_contradiction} → "NotContradiction"`. Use this exact mapping everywhere; do not invent a different collapse.
- Never report a metric for a class with zero true examples as `0.0` — report it as the string `"n/a"` and exclude it from the macro average's denominator. This is a hard requirement from the design doc's own Decision 2 and Error Handling section, not a style preference.
- §5.3 (cross-domain transfer) is explicitly out of scope for every task in this plan.
- Direct commits to `master`, no worktree/branch — same reasoning as every prior round in this project (local-only, unpushed repo; every commit individually revertable).
- **Verified exact expected numbers** (computed directly against the real corpus during design, 2026-09-10 — if a task's actual output differs from these, the corpus has changed since this plan was written; report the discrepancy rather than silently trusting either the plan or a run that contradicts it):
  - n=85 (all valid round-7+ labels): `label_A_5way` = type1:36, type4:25, not_contradiction:18, type3:5, type2:1. `label_B_5way` = not_contradiction:39, type1:36, type4:10 (type2/type3 absent — Gemma predicted neither in this set).
  - n=59 (round-7+ agreement, `label_A == label_B`): `label_5way` = type1:32, not_contradiction:18, type4:9 (type2 and type3 **both absent** — confirmed exactly, not approximately, during design).

---

### Task 1: Build the baseline evaluation datasets

**Files:**
- Create: `src/build_baseline_eval_set.py`

**Interfaces:**
- Consumes: `results/hypocontra_pilot_master_dataset.csv` (columns: `round, method, pair_id, label_A, label_B, both_flag_contradiction, labels_agree`); `results/pilot_round*_curated_sample.csv` (columns: `pair_id, topic, hypothesis_a, source_a, hypothesis_b, source_b`, confirmed identical schema across all existing round files).
- Produces: `results/baseline_all_round7plus_labels.csv` (columns: `pair_id, label_A_5way, label_A_3way, label_B_5way, label_B_3way`) and `results/baseline_ground_truth_agreement.csv` (columns: `pair_id, topic, hypothesis_a, source_a, hypothesis_b, source_b, label_5way, label_3way`) — both consumed by later tasks.

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Run it and verify against the Global Constraints' exact expected numbers**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/build_baseline_eval_set.py`

Expected (verified exactly during design — see Global Constraints):
- `n=85 (all valid round7+ labels)`, `label_A_5way`: type1_direct_negation 36, type4_apparent_contextual 25, not_contradiction 18, type3_causal_conflict 5, type2_quantitative_conflict 1. `label_B_5way`: not_contradiction 39, type1_direct_negation 36, type4_apparent_contextual 10.
- `n=59 (agreement round7+, with text)`, `label_5way`: type1_direct_negation 32, not_contradiction 18, type4_apparent_contextual 9.
- No `WARNING: ... have no matching curated-sample text` line should print. If one does, investigate before proceeding — it means a `pair_id` in the master dataset has no corresponding row in any `pilot_round*_curated_sample.csv`, which would silently corrupt every later task's input.

- [ ] **Step 3: Commit**

```bash
cd ~/prjs/hypocontra
git add src/build_baseline_eval_set.py results/baseline_all_round7plus_labels.csv results/baseline_ground_truth_agreement.csv
git commit -m "Baseline eval: build ground-truth datasets (n=59 agreement, n=85 all-valid)"
```

---

### Task 2: Shared macro P/R/F1 metric utility with explicit n/a handling

**Files:**
- Create: `src/baseline_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure utility module).
- Produces: `compute_macro_prf(y_true: list[str], y_pred: list[str], all_classes: list[str]) -> dict`, returning `{"per_class": {cls: {"precision": float, "recall": float, "f1": float, "support": int} | "n/a"}, "macro_precision": float, "macro_recall": float, "macro_f1": float, "n_classes_evaluated": int, "n_classes_absent": int}` — imported by Tasks 4 and 6.

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Write and run the verification script**

Create `/tmp/verify_baseline_metrics.py` (throwaway, not committed):

```python
import sys
sys.path.insert(0, "/home/consul/prjs/hypocontra/src")
from baseline_metrics import compute_macro_prf

# Synthetic example: 3 classes declared, one (C) has zero true examples.
y_true = ["A", "A", "B", "B", "A"]
y_pred = ["A", "B", "B", "A", "A"]
result = compute_macro_prf(y_true, y_pred, all_classes=["A", "B", "C"])

assert result["per_class"]["C"] == "n/a", f"expected C to be n/a, got {result['per_class']['C']}"
assert result["n_classes_absent"] == 1, f"expected 1 absent class, got {result['n_classes_absent']}"
assert result["n_classes_evaluated"] == 2, f"expected 2 evaluated classes, got {result['n_classes_evaluated']}"
assert isinstance(result["per_class"]["A"], dict), "expected A to have real metrics"
assert result["per_class"]["A"]["support"] == 3, f"expected A support=3, got {result['per_class']['A']['support']}"
assert result["per_class"]["B"]["support"] == 2, f"expected B support=2, got {result['per_class']['B']['support']}"
# Macro average must be computed only over A and B (2 classes), not divided by 3.
expected_macro_f1 = (result["per_class"]["A"]["f1"] + result["per_class"]["B"]["f1"]) / 2
assert abs(result["macro_f1"] - expected_macro_f1) < 1e-9, "macro_f1 must average only present classes"

print("All assertions passed.")
print(result)
```

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 /tmp/verify_baseline_metrics.py`
Expected: `All assertions passed.` followed by the printed result dict.

- [ ] **Step 3: Delete the throwaway script**

Run: `rm /tmp/verify_baseline_metrics.py`

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/baseline_metrics.py
git commit -m "Baseline eval: shared macro P/R/F1 utility with explicit n/a handling"
```

---

### Task 3: Rule-based lexical baseline

**Files:**
- Create: `src/baseline_rule_based.py`

**Interfaces:**
- Consumes: `results/baseline_ground_truth_agreement.csv` (Task 1's output; columns include `hypothesis_a`, `hypothesis_b`).
- Produces: `results/baseline_rule_based_predictions.csv` (columns: `pair_id, predicted_3way`), consumed by Task 6.

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Run it and spot-check by hand**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/baseline_rule_based.py`
Expected: prints `59 predictions -> results/baseline_rule_based_predictions.csv` and a 2-value breakdown (`Contradiction`/`NotContradiction` only — `Apparent` must never appear in this baseline's output; if it does, the code has a bug).

Manually pick 2-3 rows from `results/baseline_ground_truth_agreement.csv`, read their `hypothesis_a`/`hypothesis_b` text, and confirm by hand that `predict()`'s output matches what the negation/antonym rules should produce for that exact text — this is the task's required manual verification (no pytest in this project).

- [ ] **Step 3: Commit**

```bash
cd ~/prjs/hypocontra
git add src/baseline_rule_based.py results/baseline_rule_based_predictions.csv
git commit -m "Baseline eval: rule-based lexical (negation/antonym) baseline"
```

---

### Task 4: LLM few-shot baseline (reuses existing Gemma labels, no new model calls)

**Files:**
- Create: `src/baseline_llm_fewshot_report.py`

**Interfaces:**
- Consumes: `results/baseline_all_round7plus_labels.csv` (Task 1's output); `compute_macro_prf` (Task 2's output, imported from `baseline_metrics`).
- Produces: `results/baseline_llm_fewshot_metrics.csv` and `results/baseline_llm_fewshot_metrics_NOTE.md`.

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Run it and cross-check against the already-known combined kappa**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/baseline_llm_fewshot_report.py`
Expected: `85 pairs -> results/baseline_llm_fewshot_metrics.csv`, plus 3-way and 5-way macro F1 values. Sanity-check: this run's raw agreement rate (rows where `label_A_3way == label_B_3way`, computable with a one-line pandas check on `results/baseline_all_round7plus_labels.csv`) should be in the same ballpark as the already-recorded round 7-17 raw agreement figures — if it's wildly different, something is wrong with the label mapping, not with history.

- [ ] **Step 3: Commit**

```bash
cd ~/prjs/hypocontra
git add src/baseline_llm_fewshot_report.py results/baseline_llm_fewshot_metrics.csv results/baseline_llm_fewshot_metrics_NOTE.md
git commit -m "Baseline eval: LLM few-shot baseline (reuses existing Gemma labels, no new calls)"
```

---

### Task 5: Fine-tuned RoBERTa baseline (5-fold stratified cross-validation)

**Files:**
- Create: `src/baseline_finetune_roberta.py`

**Interfaces:**
- Consumes: `results/baseline_ground_truth_agreement.csv` (Task 1's output).
- Produces: `results/baseline_roberta_predictions.csv` (columns: `pair_id, predicted_3way, fold`) and `results/baseline_roberta_training_log.csv` (columns: `fold, epoch, train_loss`), consumed by Task 6.

This is the heaviest task in this plan: it downloads a ~1.4GB pretrained model checkpoint from HuggingFace Hub on first run (one-time, cached under `~/.cache/huggingface` afterward) and fine-tunes it 5 times (once per fold). Expect this to take real wall-clock time even on GPU — do not assume it is hung if it runs for several minutes; check `nvidia-smi`/process state if genuinely unsure, and do not launch a second copy if a run seems slow (a prior round in this project wasted real time on accidentally-duplicated slow-script launches — avoid repeating that).

- [ ] **Step 1: Install dependencies**

Run: `cd ~/prjs/hypocontra && .venv/bin/pip install torch transformers`

- [ ] **Step 2: Write the script**

```python
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


def build_dataset(df: pd.DataFrame, tokenizer):
    encodings = tokenizer(
        df["hypothesis_a"].tolist(), df["hypothesis_b"].tolist(),
        truncation=True, padding=True, max_length=256, return_tensors="pt",
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
```

- [ ] **Step 3: Run it**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/baseline_finetune_roberta.py`

If this needs to run long enough that it risks your turn ending mid-run, launch it as a detached background process (`nohup ... & disown`, matching how this project's prior rounds handled multi-minute jobs) and wait for it rather than re-launching. Verify on completion:
- `results/baseline_roberta_predictions.csv` has exactly 59 rows, `fold` values covering `0` through `4`, and every `pair_id` from `results/baseline_ground_truth_agreement.csv` appears exactly once (pooled out-of-fold predictions — no pair_id should be missing or duplicated).
- `results/baseline_roberta_training_log.csv` has `5 folds × 4 epochs = 20` rows. Spot-check that loss is not exploding (e.g., NaN or wildly increasing across epochs within a fold) — some noise is expected given ~47 training examples per fold, but a collapsed/NaN loss indicates a real problem to fix, not something to paper over in the report.

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/baseline_finetune_roberta.py results/baseline_roberta_predictions.csv results/baseline_roberta_training_log.csv
git commit -m "Baseline eval: fine-tuned RoBERTa baseline (5-fold stratified CV, 3-way)"
```

---

### Task 6: Consolidated evaluation report

**Files:**
- Create: `src/evaluate_baselines.py`
- Creates (data, committed): `docs/baseline_evaluation_report.md`

**Interfaces:**
- Consumes: `results/baseline_ground_truth_agreement.csv`, `results/baseline_rule_based_predictions.csv`, `results/baseline_roberta_predictions.csv`, `results/baseline_roberta_training_log.csv` (Task 1/3/5 outputs); `results/baseline_all_round7plus_labels.csv` (Task 1's output, re-read here for the LLM few-shot table rather than re-parsing Task 4's already-formatted CSV); `compute_macro_prf` (Task 2's output).
- Produces: `docs/baseline_evaluation_report.md` — the final human-readable deliverable of this whole plan.

- [ ] **Step 1: Write the script**

```python
"""Baseline NLI evaluation: final consolidated report generator. Computes
rule-based and RoBERTa metrics with the SAME shared function the LLM
few-shot baseline used (src/baseline_metrics.py), so all three baselines'
numbers are genuinely comparable, and writes the final report per
docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md
component 5.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from baseline_metrics import compute_macro_prf

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

ALL_3WAY_CLASSES = ["Contradiction", "Apparent", "NotContradiction"]


def format_metrics_table(metrics: dict) -> str:
    lines = ["| Клас | Precision | Recall | F1 | n |", "|---|---|---|---|---|"]
    for cls, v in metrics["per_class"].items():
        if v == "n/a":
            lines.append(f"| {cls} | n/a | n/a | n/a | 0 |")
        else:
            lines.append(f"| {cls} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    lines.append(
        f"| **MACRO AVG** | **{metrics['macro_precision']:.3f}** | "
        f"**{metrics['macro_recall']:.3f}** | **{metrics['macro_f1']:.3f}** | — |"
    )
    return "\n".join(lines)


def main() -> None:
    ground_truth = pd.read_csv(RESULTS_DIR / "baseline_ground_truth_agreement.csv", dtype=str)
    y_true = ground_truth.set_index("pair_id")["label_3way"]

    rule_based = pd.read_csv(RESULTS_DIR / "baseline_rule_based_predictions.csv", dtype=str).set_index("pair_id")
    roberta = pd.read_csv(RESULTS_DIR / "baseline_roberta_predictions.csv", dtype=str).set_index("pair_id")

    rule_based_aligned = rule_based.loc[y_true.index]
    roberta_aligned = roberta.loc[y_true.index]

    rule_based_metrics = compute_macro_prf(y_true.tolist(), rule_based_aligned["predicted_3way"].tolist(), ALL_3WAY_CLASSES)
    roberta_metrics = compute_macro_prf(y_true.tolist(), roberta_aligned["predicted_3way"].tolist(), ALL_3WAY_CLASSES)

    llm_labels = pd.read_csv(RESULTS_DIR / "baseline_all_round7plus_labels.csv", dtype=str)
    llm_metrics = compute_macro_prf(llm_labels["label_A_3way"].tolist(), llm_labels["label_B_3way"].tolist(), ALL_3WAY_CLASSES)

    training_log = pd.read_csv(RESULTS_DIR / "baseline_roberta_training_log.csv", dtype=str)
    training_log["train_loss"] = training_log["train_loss"].astype(float)
    final_epoch_losses = training_log.groupby("fold")["train_loss"].last()

    report_lines = [
        "# HypoContra baseline NLI-model evaluation report",
        "",
        f"Спираючись на `docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md` "
        f"(стаття §5, пункт \"в\"). Ground truth: n={len(y_true)} round-7+ пар, де Claude "
        f"(Annotator A) і Gemma (Annotator B) погодились (`label_A == label_B`).",
        "",
        "## §5.3 не виконано",
        "",
        "Крос-доменна перевірка (fine-tune на біомедичному корпусі Alamri & Stevenson 2016, "
        "zero-shot оцінка на HypoContra) свідомо відкладена — див. `Non-goals` design doc.",
        "",
        "## 1. Rule-based baseline (нижня межа складності)",
        "",
        format_metrics_table(rule_based_metrics),
        "",
        "## 2. Fine-tuned RoBERTa (roberta-large-mnli, 5-fold stratified CV, лише 3-way)",
        "",
        format_metrics_table(roberta_metrics),
        "",
        "Фінальний train loss по фолдах:",
        "",
        "```",
        final_epoch_losses.to_string(),
        "```",
        "",
        "## 3. LLM few-shot (Gemma) — методологічне застереження: НЕ незалежна оцінка",
        "",
        "Ця baseline повторно використовує вже зібрані мітки Gemma (Annotator B) проти "
        "Claude як референсу, на всіх n=85 round-7+ парах (не на n=59 ground-truth наборі "
        "— щоб уникнути циркулярності, див. `results/baseline_llm_fewshot_metrics_NOTE.md`). "
        "Ці цифри вимірюють узгодженість із судженням Claude, не незалежну точність.",
        "",
        format_metrics_table(llm_metrics),
    ]

    report_path = DOCS_DIR / "baseline_evaluation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    print(f"Rule-based macro F1: {rule_based_metrics['macro_f1']:.3f}")
    print(f"RoBERTa macro F1: {roberta_metrics['macro_f1']:.3f}")
    print(f"LLM few-shot macro F1 (vs Claude): {llm_metrics['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ~/prjs/hypocontra && .venv/bin/python3 src/evaluate_baselines.py`
Expected: `Report written to docs/baseline_evaluation_report.md`, plus the three macro-F1 lines. Read the generated report file afterward and confirm it renders sensibly (three sections, each with a populated table, no stray `n/a` where a real number was expected, no Python `dict`/object repr leaking into the markdown).

- [ ] **Step 3: Commit**

```bash
cd ~/prjs/hypocontra
git add src/evaluate_baselines.py docs/baseline_evaluation_report.md
git commit -m "Baseline eval: consolidated report (rule-based + RoBERTa + LLM few-shot)"
```

---

## Self-Review

**Spec coverage:** Design doc component 1 (`build_baseline_eval_set.py`) → Task 1. Component 2 (rule-based) → Task 3, with the shared metric utility factored out as its own Task 2 since both Task 4 and Task 6 need it (the design doc's component 5 explicitly requires "the same metric function" across baselines — a dedicated task is the only way to guarantee that without duplicating the function). Component 3 (RoBERTa) → Task 5. Component 4 (LLM few-shot) → Task 4. Component 5 (report) → Task 6. All 5 Decisions from the design doc are implemented exactly as specified: Decision 1 (ground truth = n=59 agreement) in Task 1; Decision 2 (3-way primary, 5-way `n/a` handling) in Tasks 1, 2, 4; Decision 3 (RoBERTa 5-fold CV, 3-way only) in Task 5; Decision 4 (LLM few-shot reuses existing labels, n=85, explicit circularity note) in Task 4; Decision 5 (§5.3 deferred) — explicitly called out in Task 6's report output and nowhere else in this plan attempts it.

**Placeholder scan:** no TBD/TODO; every step has runnable code or an exact command. Task 5's runtime warning ("do not assume it is hung") is guidance, not a placeholder for missing logic — the actual training code is fully specified.

**Type consistency:** `compute_macro_prf`'s signature (Task 2) is used identically in Task 4 and Task 6 — same three positional arguments, same return-dict shape (`per_class`, `macro_precision`, `macro_recall`, `macro_f1`, `n_classes_evaluated`, `n_classes_absent`), verified by re-reading Task 2's code while writing Tasks 4 and 6 rather than reconstructing the signature from memory. `THREE_WAY_MAP`'s three target values (`"Contradiction"`, `"Apparent"`, `"NotContradiction"`) match `ALL_3WAY_CLASSES` in Tasks 4 and 6 exactly (same three strings, same casing). File names produced in each task (`baseline_ground_truth_agreement.csv`, `baseline_all_round7plus_labels.csv`, `baseline_rule_based_predictions.csv`, `baseline_roberta_predictions.csv`, `baseline_roberta_training_log.csv`, `baseline_llm_fewshot_metrics.csv`) are referenced identically by every downstream task that consumes them — cross-checked file name spelling across all 6 tasks during this self-review.
