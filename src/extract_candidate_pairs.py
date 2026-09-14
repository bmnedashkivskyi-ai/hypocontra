"""Крок 4.2 методології HypoContra: добір кандидатних пар гіпотез.

СПРОЩЕННЯ ВІДНОСНО §4.2 ОРИГІНАЛЬНОЇ МЕТОДОЛОГІЇ (чесно розкрито, не
приховано): оригінальний план специфікує видобування сутностей за схемою
SciIE/DyGIE++ (типи Task/Method/Metric/Material, за Jain et al., 2020,
SciREX) з подальшим групуванням за спільними Task+Metric. DyGIE++/SciIE
вимагають окремого навченого NER-конвеєра, недоступного в цьому пілоті.
Натомість використано:
  1. Видобування іменникових фраз (noun chunks, spaCy en_core_web_sm) з
     кожного абстракту як наближення до сутностей Task/Method/Metric --
     менш точне, ніж спеціалізований науковий NER, але не потребує
     додаткового навчання моделі.
  2. TF-IDF по цих фразах для визначення "спільної теми" пари абстрактів
     (наближення до кластера Task+Metric з оригінальної методології).
  3. Кластеризація: пари абстрактів із високою косинусною подібністю
     TF-IDF-векторів (>= SIMILARITY_THRESHOLD) І непорожнім перетином
     топ-фраз -- це схоже, не ідентично, оригінальному критерію "спільні
     сутності Task+Metric".

Це задокументоване звуження пілотного обсягу (як і "лише arXiv, без ACL
Anthology" у collect_abstracts.py), а не приховане припущення.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SIMILARITY_THRESHOLD = 0.25
MAX_PAIRS_PER_ABSTRACT = 3  # обмежує квадратичне зростання (§4.2 застереження)
TOP_PHRASES_PER_DOC = 8

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def extract_noun_phrases(text: str) -> list[str]:
    doc = nlp(text)
    phrases = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        # фільтр надто коротких/загальних фраз (займенники, детермінатори-лише)
        if len(phrase.split()) >= 2 and len(phrase) > 5:
            phrases.append(phrase)
    return phrases


def main():
    df = pd.read_csv(RESULTS_DIR / "arxiv_cscl_abstracts.csv")
    print(f"Loaded {len(df)} abstracts")

    print("Extracting noun phrases (spaCy)...", flush=True)
    df["noun_phrases"] = df["abstract"].apply(lambda t: extract_noun_phrases(t))
    df["phrase_text"] = df["noun_phrases"].apply(lambda ps: " ".join(ps))

    print("Computing TF-IDF over noun-phrase text...", flush=True)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    tfidf = vectorizer.fit_transform(df["phrase_text"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    def top_terms(row_idx: int, k: int = TOP_PHRASES_PER_DOC) -> set[str]:
        row = tfidf[row_idx].toarray().ravel()
        top_idx = row.argsort()[::-1][:k]
        return set(feature_names[top_idx][row[top_idx] > 0])

    print("Computing pairwise similarity (blocked by year window to limit search space)...", flush=True)
    n = len(df)
    top_terms_cache = [top_terms(i) for i in range(n)]

    candidate_pairs = []
    # Обмежуємо пошук парами в межах +/- 2 років (правдоподібніше для
    # "тієї самої теми/задачі" і зменшує квадратичну вартість) -- ще одне
    # задокументоване спрощення §4.2.
    df_sorted = df.sort_values("year").reset_index(drop=True)
    years = df_sorted["year"].values
    sim_matrix = cosine_similarity(tfidf)

    pair_count_per_doc = np.zeros(n, dtype=int)
    for i, j in itertools.combinations(range(n), 2):
        if abs(int(df.iloc[i]["year"]) - int(df.iloc[j]["year"])) > 2:
            continue
        if pair_count_per_doc[i] >= MAX_PAIRS_PER_ABSTRACT or pair_count_per_doc[j] >= MAX_PAIRS_PER_ABSTRACT:
            continue
        sim = sim_matrix[i, j]
        if sim < SIMILARITY_THRESHOLD:
            continue
        shared_terms = top_terms_cache[i] & top_terms_cache[j]
        if not shared_terms:
            continue
        candidate_pairs.append({
            "id_a": df.iloc[i]["arxiv_id"], "id_b": df.iloc[j]["arxiv_id"],
            "title_a": df.iloc[i]["title"], "title_b": df.iloc[j]["title"],
            "abstract_a": df.iloc[i]["abstract"], "abstract_b": df.iloc[j]["abstract"],
            "year_a": df.iloc[i]["year"], "year_b": df.iloc[j]["year"],
            "cosine_similarity": float(sim),
            "shared_cluster_terms": "; ".join(sorted(shared_terms)),
        })
        pair_count_per_doc[i] += 1
        pair_count_per_doc[j] += 1

    pairs_df = pd.DataFrame(candidate_pairs)
    pairs_df = pairs_df.sort_values("cosine_similarity", ascending=False)
    pairs_df.to_csv(RESULTS_DIR / "candidate_pairs_raw.csv", index=False)
    print(f"\nCandidate pairs found: {len(pairs_df)}")
    if len(pairs_df) > 0:
        print(pairs_df["cosine_similarity"].describe().to_string())


if __name__ == "__main__":
    main()
