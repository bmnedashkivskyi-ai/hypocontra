"""Task 1 (граф суперечностей, вересень 2026): консолідувати всі 103 куровані
пари (round 3-17) в один файл, що поєднує текст гіпотез з фінальною
5-way/3-way міткою типу суперечності.

Джерела (усі -- results/, вже наявні в репозиторії, нічого не перераховується
заново з нуля):

1. `hypocontra_pilot_master_dataset.csv` -- вже злиті через
   `merge_and_compute_kappa.py` мітки Анотатора A (Claude) і Анотатора B
   (другий Claude-прохід round 3-6 / Gemma round 7+) по кожній парі, разом з
   прапорцем `labels_agree`. Рядки з round>=3 -- це РІВНО 103 куровані пари
   (round 1-2 -- некурована популяція з автоматичної кластеризації, окрема
   за методом добору, свідомо виключена тут). Мітки в цьому файлі ВЖЕ
   нормалізовані до 5-way словника round7+ (`type4_apparent_contextual`),
   навіть для round 3-6, де сирі per-round файли анотаторів історично
   писали коротшу форму `type4_apparent` -- див. п.3.

2. `pilot_round{N}_curated_sample.csv` (N=3..17, які фактично існують і
   непорожні -- round 15 існує як файл, але має 0 пар, тому природно не
   зʼявиться в результаті) -- текст пари: topic, hypothesis_a, source_a,
   hypothesis_b, source_b.

3. `annotator_A_round{N}_labels.csv` / `annotator_B_round{N}_labels.csv` --
   RAW пер-раундові мітки й, ЩО ВАЖЛИВО, колонка `rationale` -- вона Є в
   даних для КОЖНОГО раунду 3-17 (перевірено empiричним читанням файлів
   перед написанням цього скрипта), всупереч можливості, allowed for в
   постановці задачі, що rationale-тексту може не бути взагалі. Тому
   rationale_A/rationale_B у результаті -- реальний текст з датасету, не
   заглушка. Виняток: round 3-6 сирі файли використовують коротшу форму
   мітки `type4_apparent` (без `_contextual`) -- вона НЕ використовується
   напряму для label_A_5way/label_B_5way (ті беруться з уже нормалізованого
   master dataset, джерело 1); rationale-текст із сирих файлів береться
   незалежно від того, яку форму мітки той файл використовує.

   Один задокументований дефект джерела: `annotator_B_round11_labels.csv`
   містить `r11_001,PARSE_ERROR,empty_response:length_exhausted` -- Gemma не
   згенерувала валідну відповідь для цієї пари (вичерпання довжини
   генерації). Це НЕ мітка типу суперечності -- це діагностика збою. Рядок
   зберігається як є (label_B_5way="PARSE_ERROR", rationale_B=діагностичний
   рядок) для прозорості; annotators_agreed=False для цієї пари (Анотатор B
   фактично не проголосував, а не "не погодився змістовно").

Правило фінальної мітки: **пріоритет Анотатору A (Claude)** -- консистентне
правило проєкту (Claude -- координуюча сесія, первинний анотатор пар).
Де label_A != label_B (включно з випадком PARSE_ERROR вище) --
annotators_agreed=False і ОБИДВІ мітки зберігаються окремими колонками
(label_A_5way, label_B_5way) -- розбіжність (κ=0.556, 5-way, n=85) НЕ
приховується.

Валідація: `baseline_ground_truth_agreement.csv` (n=59, підмножина round7+,
де A і B ПОГОДИЛИСЬ) не копіюється окремим кодовим шляхом -- натомість
загальна логіка вище (пріоритет A + agreed-прапорець) за побудовою повинна
дати ТОЧНО ті самі 59 рядків з тими самими label_5way/label_3way/текстом,
оскільки для агреемент-рядків label_A==label_B. Скрипт наприкінці звіряє це
й падає з AssertionError, якщо є розбіжність (сигнал, що щось у логіці
зламалось, а не тихо підганяє число).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MASTER_PATH = RESULTS_DIR / "hypocontra_pilot_master_dataset.csv"
GROUND_TRUTH_PATH = RESULTS_DIR / "baseline_ground_truth_agreement.csv"
OUTPUT_PATH = RESULTS_DIR / "hypocontra_full_103_with_text_and_labels.csv"

# Ідентична мапі в build_baseline_eval_set.py -- навмисно продубльована тут
# (не імпортована), щоб build_full_corpus_with_text.py лишався окремим,
# самодостатнім скриптом без залежності від порядку запуску інших скриптів.
THREE_WAY_MAP = {
    "type1_direct_negation": "Contradiction",
    "type2_quantitative_conflict": "Contradiction",
    "type3_causal_conflict": "Contradiction",
    "type4_apparent_contextual": "Apparent",
    "not_contradiction": "NotContradiction",
}


def load_master() -> pd.DataFrame:
    master = pd.read_csv(MASTER_PATH, dtype=str)
    master["round"] = master["round"].astype(int)
    curated = master[master["round"] >= 3].copy()
    return curated


def load_curated_text() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS_DIR.glob("pilot_round*_curated_sample.csv")):
        df = pd.read_csv(path, dtype=str)
        if len(df) > 0:
            frames.append(df)
    text = pd.concat(frames, ignore_index=True)
    dupes = text[text.duplicated("pair_id", keep=False)]
    if len(dupes) > 0:
        raise ValueError(f"Duplicate pair_id across pilot_round*_curated_sample.csv: {dupes['pair_id'].tolist()}")
    return text


def load_rationales() -> pd.DataFrame:
    """pair_id -> rationale_A, rationale_B from the raw per-round annotator files."""
    rat_a_frames, rat_b_frames = [], []
    for path in sorted(RESULTS_DIR.glob("annotator_A_round*_labels.csv")):
        if "_rejected" in path.name:
            continue
        df = pd.read_csv(path, dtype=str)[["pair_id", "rationale"]].rename(columns={"rationale": "rationale_A"})
        rat_a_frames.append(df)
    for path in sorted(RESULTS_DIR.glob("annotator_B_round*_labels.csv")):
        df = pd.read_csv(path, dtype=str)[["pair_id", "rationale"]].rename(columns={"rationale": "rationale_B"})
        rat_b_frames.append(df)
    rat_a = pd.concat(rat_a_frames, ignore_index=True).drop_duplicates("pair_id")
    rat_b = pd.concat(rat_b_frames, ignore_index=True).drop_duplicates("pair_id")
    return rat_a.merge(rat_b, on="pair_id", how="outer")


def main() -> None:
    master = load_master()
    text = load_curated_text()
    rationales = load_rationales()

    df = master.merge(text, on="pair_id", how="left", validate="one_to_one")
    missing_text = df[df["hypothesis_a"].isna()]
    if len(missing_text) > 0:
        print(f"WARNING: {len(missing_text)} pairs have labels but no matching curated-sample text: "
              f"{missing_text['pair_id'].tolist()}", file=sys.stderr)

    df = df.merge(rationales, on="pair_id", how="left")

    df["label_A_5way"] = df["label_A"]
    df["label_B_5way"] = df["label_B"]
    df["annotators_agreed"] = df["labels_agree"] == "True"
    df["label_5way_final"] = df["label_A_5way"]  # priority: Annotator A (Claude)
    df["label_3way_final"] = df["label_5way_final"].map(THREE_WAY_MAP)
    unmapped = df[df["label_3way_final"].isna()]
    if len(unmapped) > 0:
        print(f"WARNING: {len(unmapped)} pairs have label_5way_final outside the 5-way taxonomy "
              f"(no 3-way mapping): {unmapped[['pair_id', 'label_5way_final']].to_dict('records')}", file=sys.stderr)

    out = df[[
        "pair_id", "round", "topic", "hypothesis_a", "source_a", "hypothesis_b", "source_b",
        "label_5way_final", "label_3way_final", "annotators_agreed",
        "label_A_5way", "label_B_5way", "rationale_A", "rationale_B",
    ]].sort_values(["round", "pair_id"]).reset_index(drop=True)

    out.to_csv(OUTPUT_PATH, index=False)

    print(f"n={len(out)} pairs -> {OUTPUT_PATH}")
    if len(out) != 103:
        print(f"NOTE: expected n=103, got n={len(out)} -- see WARNINGs above for the discrepancy source.",
              file=sys.stderr)
    print(f"  annotators_agreed=True:  {(out['annotators_agreed']).sum()}")
    print(f"  annotators_agreed=False: {(~out['annotators_agreed']).sum()}")
    print(f"  label_5way_final distribution:\n{out['label_5way_final'].value_counts().to_string()}")

    # --- Validation against the n=59 ground-truth agreement file (task 1
    #     instruction: those 59 rows are not re-derived by a separate code
    #     path -- the generic priority-A logic above must reproduce them
    #     exactly, since for agreed pairs label_A == label_B by definition).
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH, dtype=str)
    check_cols = ["pair_id", "topic", "hypothesis_a", "source_a", "hypothesis_b", "source_b"]
    ours = out[out["pair_id"].isin(ground_truth["pair_id"])][check_cols + ["label_5way_final", "label_3way_final"]]
    ours = ours.rename(columns={"label_5way_final": "label_5way", "label_3way_final": "label_3way"})
    ours = ours.sort_values("pair_id").reset_index(drop=True)
    theirs = ground_truth.sort_values("pair_id").reset_index(drop=True)[ours.columns.tolist()]
    if len(ours) != len(ground_truth):
        raise AssertionError(
            f"Ground-truth validation FAILED: expected {len(ground_truth)} rows to match "
            f"baseline_ground_truth_agreement.csv pair_ids, found {len(ours)} in our output."
        )
    mismatches = ours.compare(theirs)
    if len(mismatches) > 0:
        raise AssertionError(
            f"Ground-truth validation FAILED: {len(mismatches)} rows differ from "
            f"baseline_ground_truth_agreement.csv:\n{mismatches}"
        )
    print(f"\nValidation OK: all {len(ground_truth)} rows from baseline_ground_truth_agreement.csv "
          f"(n=59 Claude/Gemma-agreed round7+ subset) reproduced exactly by the generic priority-A logic.")


if __name__ == "__main__":
    main()
