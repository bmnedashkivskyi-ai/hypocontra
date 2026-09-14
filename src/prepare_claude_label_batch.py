"""Крок 3a масштабування round 7. Формує batch для СЛІПОЇ розмітки
Анотатором A (Claude) -- на відміну від кроку 2 (побудова пар), тут Claude
бачить ЛИШЕ готову пару (topic/hypothesis_a/source_a/hypothesis_b/source_b),
БЕЗ вихідного речення й контексту побудови. Це узгоджене з користувачем
рішення: розділити "написати гіпотези" і "розмітити гіпотези" на два окремі
проходи, щоб Анотатор A хоч частково наближався до незалежного судження,
а не оцінював те, що сам щойно сформулював.

Вхід: results/pilot_round{N}_curated_sample.csv (крок 2b, ingest_claude_pairs.py).
Вихід: results/claude_label_batch_round{N}.jsonl + docs/claude_label_instructions.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
INSTRUCTIONS_PATH = DOCS_DIR / "claude_label_instructions.md"

TAXONOMY = """## Таксономія (§3 статті HypoContra) -- обери РІВНО одну мітку

- type1_direct_negation -- Гіпотеза A стверджує X, гіпотеза B стверджує НЕ-X за тих самих умов і сутностей.
- type2_quantitative_conflict -- розбіжні числові оцінки одного ефекту чи метрики поза межами похибки/довірчого інтервалу.
- type3_causal_conflict -- протилежний напрямок причинного зв'язку між тими самими змінними.
- type4_apparent_contextual -- твердження різняться лише через відмінності датасету/архітектури/гіперпараметрів/умов оцінювання й насправді сумісні.
- not_contradiction -- пара не суперечить одна одній по суті.
"""

INSTRUCTIONS_TEXT = """# Інструкція сліпої розмітки (Claude, крок 3, round N)

ВАЖЛИВО: розмічай ЛИШЕ на основі полів `topic, hypothesis_a, source_a,
hypothesis_b, source_b` з `results/claude_label_batch_round{N}.jsonl`. НЕ
звертайся до вихідного речення-кандидата чи файлівround{N}-побудови пар,
навіть якщо пам'ятаєш їхній зміст із попереднього кроку сесії -- мета цього
проходу саме в тому, щоб оцінка спиралась лише на текст готової пари, а не
на контекст, у якому вона була написана.

""" + TAXONOMY + """
Для кожного запису признач `label` (одне з 5 значень вище, ТОЧНО як написано)
і коротке `rationale` (1-2 речення, чому саме ця мітка, а не сусідня --
особливо явно розрізняй type1-3 від type4).

## Формат виводу

Заповни `results/claude_label_batch_round{N}_completed.csv` з колонками:
`pair_id, label, rationale`. Один вхідний запис -> один вихідний рядок,
без пропусків.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=7)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--refresh-instructions", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else RESULTS_DIR / f"pilot_round{args.round}_curated_sample.csv"
    df = pd.read_csv(input_path, dtype=str)

    batch_path = RESULTS_DIR / f"claude_label_batch_round{args.round}.jsonl"
    with batch_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "pair_id": row["pair_id"],
                "topic": row["topic"],
                "hypothesis_a": row["hypothesis_a"],
                "source_a": row["source_a"],
                "hypothesis_b": row["hypothesis_b"],
                "source_b": row["source_b"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.refresh_instructions or not INSTRUCTIONS_PATH.exists():
        DOCS_DIR.mkdir(exist_ok=True)
        INSTRUCTIONS_PATH.write_text(INSTRUCTIONS_TEXT.replace("{N}", str(args.round)), encoding="utf-8")

    print(f"Issued {len(df)} pairs for blind labeling into {batch_path}", flush=True)
    print(f"Instructions: {INSTRUCTIONS_PATH}", flush=True)


if __name__ == "__main__":
    main()
