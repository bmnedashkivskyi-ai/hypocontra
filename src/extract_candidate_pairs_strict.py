"""Крок 4.2 методології HypoContra, ЗВУЖЕНИЙ ДОБІР (v2, після пілоту
docs/pilot_results.md): попередня версія (extract_candidate_pairs.py,
TF-IDF по noun-phrases, поріг косинусної подібності) виявилась занадто
permissive -- 93-98% кандидатних пар у пілоті виявились взагалі не
суперечністю, лише спільною тематичною лексикою.

Цей скрипт наближається ближче до оригінального критерію Task+Metric
(§4.2 методології):
  1. TASK-фраза: іменникова фраза, що закінчується типовим суфіксом
     задачі NLP/ML (detection, classification, prediction, recognition,
     extraction, generation, translation, segmentation, tagging,
     parsing, summarization, retrieval, reasoning, inference,
     estimation, forecasting, analysis) -- наближення до сутності Task
     з SciIE-подібної схеми, БЕЗ навченого NER (задокументоване
     спрощення, як і в v1).
  2. Кандидатні пари формуються ТІЛЬКИ всередині груп абстрактів з
     ІДЕНТИЧНОЮ (після нормалізації: lowercase, видалення множини)
     TASK-фразою -- НЕ поріг подібності, а точний збіг нормалізованої
     задачі. Це значно суворіше за v1 і мало б різко підвищити частку
     генуїнно порівнянних тверджень серед кандидатів.
  3. Додатковий M-фільтр (наближення до "Metric" половини критерію):
     обидва абстракти в парі повинні містити числове/метричне
     твердження (regex на % або поширені назви метрик -- F1, accuracy,
     BLEU, MSE, precision, recall, AUC, kappa, perplexity, score) --
     це підвищує ймовірність, що пара робить порівнянне, потенційно
     конфліктне кількісне твердження (типи 1-2 таксономії §3), а не
     лише описує метод без конкретного результату.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import pandas as pd
import spacy

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MAX_PAIRS_PER_TASK_GROUP = 30  # уникає вибуху пар у надто загальних груп (напр. "language modeling")
MIN_GROUP_SIZE = 2
MAX_GROUP_SIZE = 12  # група > 12 майже напевно означає TASK-фразу занадто загальну (напр. "text classification")

TASK_SUFFIXES = (
    "detection", "classification", "prediction", "recognition", "extraction",
    "generation", "translation", "segmentation", "tagging", "parsing",
    "summarization", "summarisation", "retrieval", "reasoning", "inference",
    "estimation", "forecasting", "analysis", "alignment", "identification",
    "understanding", "answering", "disambiguation",
)

# Виявлено на першому прогоні (без цього фільтра): загальні
# описові прикметники перед суфіксом ("qualitative analysis", "such
# analysis", "deep understanding") НЕ позначають справжню спільну
# задачу -- це шаблонна академічна лексика, що знову відтворила б
# проблему round 1 (спільний словник без спільного конкретного
# твердження). Фраза виключається, якщо ЄДИНЕ змістовне слово перед
# суфіксом -- один із цих загальних модифікаторів.
GENERIC_MODIFIERS = {
    "qualitative", "comprehensive", "detailed", "theoretical", "statistical",
    "comparative", "such", "thorough", "further", "additional", "deep",
    "better", "final", "general", "initial", "extensive", "simple", "basic",
    "brief", "careful", "rigorous", "systematic", "empirical", "preliminary",
    "automatic", "manual", "model",
}

METRIC_PATTERN = re.compile(
    r"\b(\d+(\.\d+)?\s?%|f1|f-score|f1-score|accuracy|bleu|rouge|meteor|mse|rmse|"
    r"precision|recall|auc|kappa|perplexity|score|correlation|error rate)\b",
    re.IGNORECASE,
)

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def extract_task_phrases(text: str) -> set[str]:
    doc = nlp(text)
    phrases = set()
    for chunk in doc.noun_chunks:
        words = chunk.text.lower().strip().split()
        if not words:
            continue
        last = words[-1].rstrip("s") if words[-1].endswith("s") and not words[-1].endswith("ss") else words[-1]
        if any(last == suf or last == suf.rstrip("s") for suf in TASK_SUFFIXES):
            # нормалізація: прибираємо детермінатори/займенники на початку
            norm = " ".join(w for w in words if w not in {"a", "an", "the", "this", "that", "our", "their"})
            content_words = norm.split()[:-1]  # усе, крім самого суфікса
            if len(norm.split()) >= 2 and not (
                len(content_words) == 1 and content_words[0] in GENERIC_MODIFIERS
            ):
                phrases.add(norm)
    return phrases


def has_metric_mention(text: str) -> bool:
    return bool(METRIC_PATTERN.search(text))


def main():
    df = pd.read_csv(RESULTS_DIR / "arxiv_cscl_abstracts.csv")
    print(f"Loaded {len(df)} abstracts")

    print("Extracting TASK phrases (spaCy, suffix-pattern match)...", flush=True)
    df["task_phrases"] = df["abstract"].apply(extract_task_phrases)
    df["has_metric"] = df["abstract"].apply(has_metric_mention)

    # task_phrase -> list of doc indices
    task_to_docs: dict[str, list[int]] = {}
    for idx, phrases in df["task_phrases"].items():
        for p in phrases:
            task_to_docs.setdefault(p, []).append(idx)

    group_sizes = {p: len(idxs) for p, idxs in task_to_docs.items()}
    n_groups_total = len(task_to_docs)
    n_groups_usable = sum(1 for s in group_sizes.values() if MIN_GROUP_SIZE <= s <= MAX_GROUP_SIZE)
    print(f"Total distinct TASK phrases found: {n_groups_total}")
    print(f"Usable groups (size {MIN_GROUP_SIZE}-{MAX_GROUP_SIZE}): {n_groups_usable}")

    candidate_pairs = []
    seen_pairs = set()
    for task_phrase, idxs in task_to_docs.items():
        if not (MIN_GROUP_SIZE <= len(idxs) <= MAX_GROUP_SIZE):
            continue
        pairs_in_group = list(itertools.combinations(idxs, 2))[:MAX_PAIRS_PER_TASK_GROUP]
        for i, j in pairs_in_group:
            key = tuple(sorted([df.iloc[i]["arxiv_id"], df.iloc[j]["arxiv_id"]]))
            if key in seen_pairs:
                continue
            row_i, row_j = df.iloc[i], df.iloc[j]
            both_have_metric = bool(row_i["has_metric"]) and bool(row_j["has_metric"])
            seen_pairs.add(key)
            candidate_pairs.append({
                "id_a": row_i["arxiv_id"], "id_b": row_j["arxiv_id"],
                "title_a": row_i["title"], "title_b": row_j["title"],
                "abstract_a": row_i["abstract"], "abstract_b": row_j["abstract"],
                "year_a": row_i["year"], "year_b": row_j["year"],
                "shared_task_phrase": task_phrase,
                "both_have_metric": both_have_metric,
            })

    pairs_df = pd.DataFrame(candidate_pairs).drop_duplicates(subset=["id_a", "id_b"])
    pairs_df.to_csv(RESULTS_DIR / "candidate_pairs_strict_raw.csv", index=False)

    metric_filtered = pairs_df[pairs_df["both_have_metric"]]
    metric_filtered.to_csv(RESULTS_DIR / "candidate_pairs_strict_metric_filtered.csv", index=False)

    print(f"\nCandidate pairs (same TASK phrase): {len(pairs_df)}")
    print(f"Of those, both abstracts have a metric mention: {len(metric_filtered)}")
    print("\nTop 15 most common shared task phrases:")
    print(pairs_df["shared_task_phrase"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
