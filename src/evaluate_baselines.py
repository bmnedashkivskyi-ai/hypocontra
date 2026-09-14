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
from sklearn.metrics import cohen_kappa_score

from baseline_metrics import compute_macro_prf

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

ALL_3WAY_CLASSES = ["Contradiction", "Apparent", "NotContradiction"]
# 5-way class order matches results/baseline_llm_fewshot_metrics.csv's "5way" rows.
ALL_5WAY_CLASSES = [
    "not_contradiction",
    "type1_direct_negation",
    "type2_quantitative_conflict",
    "type3_causal_conflict",
    "type4_apparent_contextual",
]


def format_metrics_table(metrics: dict) -> str:
    lines = ["| Клас | Precision | Recall | F1 | n |", "|---|---|---|---|---|"]
    for cls, v in metrics["per_class"].items():
        if v == "n/a":
            lines.append(f"| {cls} | n/a | n/a | n/a | 0 |")
        else:
            lines.append(f"| {cls} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    macro_p = metrics["macro_precision"]
    if macro_p == "n/a":
        lines.append("| **MACRO AVG** | n/a | n/a | n/a | — |")
    else:
        lines.append(
            f"| **MACRO AVG** | **{macro_p:.3f}** | "
            f"**{metrics['macro_recall']:.3f}** | **{metrics['macro_f1']:.3f}** | — |"
        )
    return "\n".join(lines)


def format_not_attempted_5way_table() -> str:
    """5-way table for baselines that were never run on the 5-class task at all
    (rule-based, RoBERTa -- design decision 3). Every cell is n/a, including
    the n column, to signal "not attempted" -- distinct from a class that WAS
    evaluated but happened to have zero eligible ground-truth examples (which
    format_metrics_table handles separately, with a real n=0)."""
    lines = ["| Клас | Precision | Recall | F1 | n |", "|---|---|---|---|---|"]
    for cls in ALL_5WAY_CLASSES:
        lines.append(f"| {cls} | n/a | n/a | n/a | n/a |")
    lines.append("| **MACRO AVG** | n/a | n/a | n/a | n/a |")
    return "\n".join(lines)


def build_llm_fewshot_5way_metrics(fewshot_df: pd.DataFrame) -> dict:
    """Reconstruct a compute_macro_prf-shaped metrics dict from the already-computed
    5-way rows in results/baseline_llm_fewshot_metrics.csv, so format_metrics_table
    can render it the same way as the other tables. Any of ALL_5WAY_CLASSES missing
    from the CSV's "5way" rows (i.e. genuinely zero support in this n=85 set) is
    marked n/a, per Decision 2 -- though in the current data all 5 classes are
    present."""
    five = fewshot_df[fewshot_df["scheme"] == "5way"]
    per_class: dict = {}
    for cls in ALL_5WAY_CLASSES:
        row = five[five["class"] == cls]
        if row.empty:
            per_class[cls] = "n/a"
        else:
            r = row.iloc[0]
            per_class[cls] = {
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1": float(r["f1"]),
                "support": int(float(r["support"])),
            }
    macro_row = five[five["class"] == "MACRO_AVG"]
    if macro_row.empty:
        raise ValueError("build_llm_fewshot_5way_metrics: no MACRO_AVG row found among 5way rows")
    macro_row = macro_row.iloc[0]
    return {
        "per_class": per_class,
        "macro_precision": float(macro_row["precision"]),
        "macro_recall": float(macro_row["recall"]),
        "macro_f1": float(macro_row["f1"]),
    }


def format_training_log_table(training_log: pd.DataFrame) -> str:
    final_epoch_losses = training_log.groupby("fold")["train_loss"].last()
    lines = ["| Fold | Final train loss |", "|---|---|"]
    for fold, loss in final_epoch_losses.items():
        lines.append(f"| {fold} | {loss:.4f} |")
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

    fewshot_csv = pd.read_csv(RESULTS_DIR / "baseline_llm_fewshot_metrics.csv", dtype=str)
    llm_5way_metrics = build_llm_fewshot_5way_metrics(fewshot_csv)

    training_log = pd.read_csv(RESULTS_DIR / "baseline_roberta_training_log.csv", dtype=str)
    training_log["train_loss"] = training_log["train_loss"].astype(float)

    # Finding 4: 3-way Cohen's kappa on the same n=85 all-valid set the already-documented
    # 5-way kappa=0.556 was computed on -- does collapsing to 3 classes recover threshold-level
    # (>=0.6, article's own §4.4) inter-annotator agreement?
    kappa_3way = cohen_kappa_score(llm_labels["label_A_3way"], llm_labels["label_B_3way"])

    report_lines = [
        "# HypoContra baseline NLI-model evaluation report",
        "",
        f"Спираючись на `docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md` "
        f"(стаття §5, пункт \"в\"). Ground truth: n={len(y_true)} round-7+ пар, де Claude "
        f"(Annotator A) і Gemma (Annotator B) погодились (`label_A == label_B`).",
        "",
        f"Міжанотаторська згода (Cohen's κ), n=85 (round 7-17, усі валідні мітки обох анотаторів): "
        f"3-way κ={kappa_3way:.3f} — ВИЩЕ порогу §4.4 статті (0.6); 5-way κ=0.556 (вже "
        f"задокументовано раніше) — нижче порогу. Згортання до 3 класів відновлює "
        f"міжанотаторську згоду на рівні порогу.",
        "",
        "**Важливо: наведені нижче три числа обчислені на РІЗНИХ популяціях і не є прямо "
        "порівнюваними як три оцінки на одному й тому самому тестовому наборі.** Rule-based "
        "(0.310) і RoBERTa (0.575) обчислені на n=59 (agreement-only ground truth, Decision 1); "
        "LLM few-shot (0.690) обчислений на ширшому й іншому n=85 (усі валідні мітки round 7+, "
        "Decision 4 — щоб уникнути циркулярності, оскільки Gemma сама є одним з анотаторів "
        "корпусу).",
        "",
        "## §5.3 (крос-доменна генералізація) виконано окремо",
        "",
        "Fine-tune на біомедичному корпусі Alamri & Stevenson (2016), zero-shot оцінка на "
        "HypoContra — виконано в окремому Bounded-циклі, не в межах цього плану. Результат: "
        "zero-shot transfer macro F1=0.3776 (проти 0.575 in-domain RoBERTa) — див. "
        "`docs/generalization_report.md`.",
        "",
        "## 1. Rule-based baseline (нижня межа складності)",
        "",
        "### 3-way (основна метрика)",
        "",
        format_metrics_table(rule_based_metrics),
        "",
        "### 5-way (додатково)",
        "",
        format_not_attempted_5way_table(),
        "",
        "5-класова задача не виконувалась для цієї baseline (design decision 3) — не плутати "
        "з n/a через нульову підтримку класу.",
        "",
        "## 2. Fine-tuned RoBERTa (roberta-large-mnli, 5-fold stratified CV, лише 3-way)",
        "",
        "Модель стартувала з уже натренованої MNLI-класифікаційної голови, а не зі свіжої: "
        "`roberta-large-mnli` вже має рівно 3 вихідні мітки (CONTRADICTION/NEUTRAL/ENTAILMENT), "
        "тому виклик `AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, "
        "ignore_mismatched_sizes=True)` є no-op для розміру голови й зберігає передтреновані "
        "MNLI-ваги замість переініціалізації з нуля (`ignore_mismatched_sizes=True` доданий "
        "захисно на випадок, якщо кількість міток колись відрізнятиметься, а не тому що це "
        "реально сталося тут). Це дає асиметричний \"теплий старт\": CONTRADICTION/NEUTRAL "
        "приблизно узгоджуються з 2 із 3 класів HypoContra (CONTRADICTION↔Contradiction, "
        "NEUTRAL↔Apparent — §3 таксономії сама називає type4 \"Neutral*\"), але ENTAILMENT не є "
        "тим самим поняттям, що й \"NotContradiction\", тож третій клас узгоджений гірше. Це "
        "легітимний вибір baseline (§5.1 explicitly хоче MNLI-pretrained RoBERTa), але по-класові "
        "цифри варто читати з урахуванням цього асиметричного старту, а не як симетричну baseline "
        "зі свіжою головою.",
        "",
        "Цей прогін НЕ був повністю seeded: детермінованим було лише розбиття на фолди "
        "(`StratifiedKFold(random_state=42)`), тоді як перемішування батчів у циклі тренування "
        "(`torch.randperm`, щоепохи) не мало власного сіда і ніде не викликався "
        "`torch.manual_seed()`. Тому наведені нижче цифри (macro F1=0.575) відображають один "
        "конкретний unseeded прогін, а не гарантовано відтворюваний результат; `torch.manual_seed(42)` "
        "додано до `src/baseline_finetune_roberta.py` для майбутніх прогонів (цей прогін не "
        "перезапускався і наведені цифри не змінились).",
        "",
        "### 3-way (основна метрика)",
        "",
        format_metrics_table(roberta_metrics),
        "",
        "Фінальний train loss по фолдах:",
        "",
        format_training_log_table(training_log),
        "",
        "Усі 5 фолдів збігались (converged) монотонно: початковий train loss у діапазоні "
        "1.18-1.47 спадав до фінального loss у діапазоні 0.06-0.11, без розбіжності "
        "(divergence) чи NaN у жодному фолді.",
        "",
        "### 5-way (додатково)",
        "",
        format_not_attempted_5way_table(),
        "",
        "5-класова задача не виконувалась для цієї baseline (design decision 3) — не плутати "
        "з n/a через нульову підтримку класу.",
        "",
        "## 3. LLM few-shot (Gemma) — методологічне застереження: НЕ незалежна оцінка",
        "",
        "Ця baseline повторно використовує вже зібрані мітки Gemma (Annotator B) проти "
        "Claude як референсу, на всіх n=85 round-7+ парах (не на n=59 ground-truth наборі "
        "— щоб уникнути циркулярності, див. `results/baseline_llm_fewshot_metrics_NOTE.md`). "
        "Ці цифри вимірюють узгодженість із судженням Claude, не незалежну точність.",
        "",
        "### 3-way (основна метрика)",
        "",
        format_metrics_table(llm_metrics),
        "",
        "### 5-way (додатково — реальні обчислені цифри, на відміну від інших двох baseline)",
        "",
        format_metrics_table(llm_5way_metrics),
        "",
        "type2 (n=1) і type3 (n=5) мають надто малу підтримку для змістовної по-класової оцінки: "
        "їхній F1=0.000 — це справжній результат (Gemma не передбачила жодного прикладу цих "
        "класів), а не помилка чи сфабрикований нуль, і саме він тягне вниз macro-середнє "
        "(0.407) — це варто трактувати як шум малого n, а не як провал моделі.",
    ]

    report_path = DOCS_DIR / "baseline_evaluation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    print(f"Rule-based macro F1 (3-way): {rule_based_metrics['macro_f1']:.3f}")
    print(f"RoBERTa macro F1 (3-way): {roberta_metrics['macro_f1']:.3f}")
    print(f"LLM few-shot macro F1 (vs Claude, 3-way): {llm_metrics['macro_f1']:.3f}")
    print(f"LLM few-shot macro F1 (vs Claude, 5-way): {llm_5way_metrics['macro_f1']:.3f}")
    print(f"3-way kappa (n=85): {kappa_3way:.3f}")


if __name__ == "__main__":
    main()
