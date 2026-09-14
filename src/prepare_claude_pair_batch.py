"""Крок 2a масштабування round 7. Формує batch-файл для РУЧНОЇ обробки
координуючою Claude-сесією -- жодного LLM-виклику всередині цього скрипта.
Розділення навмисне (§ обговорення плану): роль Claude лишається
"прочитати JSONL -> написати CSV за інструкцією", а не монолітний скриптовий
виклик, аби чесно відрізняти цю частину пайплайну від Gemma-частини
(annotate_with_gemma.py), яка справді повністю скриптована й відтворювана
третьою особою без живої Claude Code сесії.

Вхід: results/candidate_sentences_resolved.csv (крок 1, resolve_survey_citations.py).
Вихід: results/claude_pair_batch_round{N}.jsonl + docs/claude_batch_instructions.md
(статична інструкція, перезаписується лише якщо відсутня або --refresh-instructions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
INSTRUCTIONS_PATH = DOCS_DIR / "claude_batch_instructions.md"

INSTRUCTIONS_TEXT = """# Інструкція побудови пар гіпотез (Claude, крок 2 round N)

Вхід: `results/claude_pair_batch_round{N}.jsonl` -- по одному об'єкту на речення:
`{sentence_id, survey_arxiv_id, survey_title, sentence, cited_refs: [{key, title, authors, year, abstract, context}, ...]}`.

Для КОЖНОГО речення:

1. Прочитай `sentence` і список `cited_refs`. Якщо процитовано більше 2 робіт,
   визнач, ЯКІ САМЕ дві дійсно протиставлені реченням (а не просто перелічені
   поруч) -- решту ігноруй.
2. Якщо жодні дві роботи НЕ протиставлені по суті (речення хибно спрацювало на
   маркер контрасту без реального протиставлення гіпотез) -- познач
   `status=skipped`, вкажи `skip_reason` (напр. "not_genuine_contrast",
   "same_finding_different_wording", "insufficient_context").
3. Якщо протиставлення реальне -- сформулюй `hypothesis_a` і `hypothesis_b` як
   САМОСТІЙНІ, зрозумілі без контексту речення твердження, спираючись на
   реальний зміст `title`/`abstract`/`context` кожної роботи (НЕ вигадуй деталей,
   яких немає в наданих полях; якщо `abstract`/`context` не дають достатньо
   інформації для впевненого формулювання -- це підстава для `skipped` з
   `skip_reason=insufficient_context`, а не для вигадки).
4. Признач `topic` -- коротка назва спільної теми/задачі (2-5 слів).
5. `source_a`/`source_b` -- цитатний ключ (`key`) відповідної роботи.

## Формат виводу

Заповни `results/claude_pair_batch_round{N}_completed.csv` з колонками:
`sentence_id, status, skip_reason, topic, hypothesis_a, source_a, hypothesis_b, source_b`

`status` є `pair_created` або `skipped`. Для `pair_created` заповнюються всі
колонки крім `skip_reason` (лишити порожнім); для `skipped` -- лише
`sentence_id, status, skip_reason`.

Один рядок вхідного JSONL -> один рядок вихідного CSV (навіть для skipped --
не пропускай речення мовчки, це порушить облік прозорості пайплайну).
"""


def load_state(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    return set(json.loads(state_path.read_text(encoding="utf-8")))


def save_state(state_path: Path, issued_ids: set[str]) -> None:
    state_path.write_text(json.dumps(sorted(issued_ids), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=None, help="За замовчуванням -- усі неопрацьовані.")
    parser.add_argument("--all", action="store_true", help="Ігнорувати стан, перевидати все.")
    parser.add_argument("--refresh-instructions", action="store_true")
    parser.add_argument("--input", type=str, default="candidate_sentences_resolved.csv")
    args = parser.parse_args()

    df = pd.read_csv(RESULTS_DIR / args.input, dtype={"survey_arxiv_id": str})

    state_path = RESULTS_DIR / f"claude_batch_state_round{args.round}.json"
    issued_ids = set() if args.all else load_state(state_path)

    pending = df[~df["sentence_id"].isin(issued_ids)]
    if args.batch_size is not None:
        pending = pending.head(args.batch_size)

    FIELD_SEP = "\x1f"  # має збігатись з resolve_survey_citations.py -- див. коментар там
    records = []
    for _, row in pending.iterrows():
        cited_keys = row["cited_keys_split"].split(FIELD_SEP)
        titles = row["resolved_titles"].split(FIELD_SEP) if pd.notna(row["resolved_titles"]) else []
        authors = row["resolved_authors"].split(FIELD_SEP) if pd.notna(row["resolved_authors"]) else []
        years = row["resolved_years"].split(FIELD_SEP) if pd.notna(row["resolved_years"]) else []
        abstracts = row["resolved_abstracts"].split(FIELD_SEP) if pd.notna(row["resolved_abstracts"]) else []
        contexts = row["resolved_context"].split(FIELD_SEP) if pd.notna(row["resolved_context"]) else []

        n = len(cited_keys)

        def safe_get(lst: list, i: int) -> str:
            return lst[i] if i < len(lst) and pd.notna(lst[i]) else ""

        cited_refs = [
            {
                "key": cited_keys[i],
                "title": safe_get(titles, i),
                "authors": safe_get(authors, i),
                "year": safe_get(years, i),
                "abstract": safe_get(abstracts, i),
                "context": safe_get(contexts, i),
            }
            for i in range(n)
        ]
        records.append({
            "sentence_id": row["sentence_id"],
            "survey_arxiv_id": row["survey_arxiv_id"],
            "survey_title": row["survey_title"],
            "sentence": row["sentence"],
            "cited_refs": cited_refs,
        })

    batch_path = RESULTS_DIR / f"claude_pair_batch_round{args.round}.jsonl"
    with batch_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.refresh_instructions or not INSTRUCTIONS_PATH.exists():
        DOCS_DIR.mkdir(exist_ok=True)
        INSTRUCTIONS_PATH.write_text(
            INSTRUCTIONS_TEXT.replace("{N}", str(args.round)), encoding="utf-8"
        )

    issued_ids.update(r["sentence_id"] for r in records)
    save_state(state_path, issued_ids)

    print(f"Issued {len(records)} sentences into {batch_path}", flush=True)
    print(f"Total issued so far this round (state file): {len(issued_ids)}", flush=True)
    print(f"Instructions: {INSTRUCTIONS_PATH}", flush=True)


if __name__ == "__main__":
    main()
