"""Крок 3/3 round 14 -- PDF-аналог resolve_survey_citations.py. На відміну
від .bbl/.bib (офлайн, без мережі), тут потрібен зовнішній виклик
(bib_lookup.lookup_abstract через OpenAlex/Crossref) для отримання
abstract -- тому спершу застосовується MIN_KEYS/MAX_KEYS-фільтр (офлайн,
на основі зіставлення прізвище+рік) і лише ПОТІМ, лише для речень, що
пройшли фільтр, виконується мережевий lookup -- щоб не витрачати запити
на речення, які однаково будуть відкинуті.

Вихідна схема -- ІДЕНТИЧНА до resolve_survey_citations.py (той самий
FIELD_SEP, той самий MIN_KEYS=2/MAX_KEYS=6), щоб
prepare_claude_pair_batch.py читав results/candidate_sentences_resolved_acl_pdf.csv
без жодної зміни коду.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from acl_pdf_text import extract_pdf_text, find_references_section
from acl_ref_parsing import parse_citation_marker, parse_reference_entries
from bib_lookup import load_cache, save_cache, lookup_abstract

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PDF_CACHE_DIR = RESULTS_DIR / "acl_pdfs"
BIB_CACHE_PATH = RESULTS_DIR / "acl_pdf_bib_lookup_cache.json"
FIELD_SEP = "\x1f"
MIN_KEYS = 2
MAX_KEYS = 6


def year_prefix(year: str) -> str:
    """'2023a' -> '2023' -- допускає розбіжність суфікса a/b/c між тим, як
    рік записаний у цитаті-в-тексті, і як у власному списку літератури.
    Безпечно за конструкцією (кожен year трасується до YEAR_PATTERN,
    що завжди починається 4 цифрами), але guard замість потенційного
    AttributeError на випадок майбутньої зміни джерела даних
    (round 15, tech-debt fix #6)."""
    m = re.match(r"(\d{4})", year)
    return m.group(1) if m else year


def match_citation_to_entry(marker: dict, entries: list[dict]) -> tuple[dict | None, str | None]:
    year_pfx = year_prefix(marker["year"])
    candidates = [
        e for e in entries
        if year_prefix(e["year"]) == year_pfx and (marker["surnames"] & e["surnames"])
    ]
    if not candidates:
        return None, "citation_unmatched"
    if len(candidates) > 1:
        return None, "citation_ambiguous"
    return candidates[0], None


def build_reference_index(cache_by_key: dict) -> dict:
    def get(key: str) -> list[dict]:
        if key not in cache_by_key:
            pdf_path = PDF_CACHE_DIR / f"{key}.pdf"
            text = extract_pdf_text(pdf_path)
            refs_text = find_references_section(text)
            cache_by_key[key] = parse_reference_entries(refs_text) if refs_text else []
        return cache_by_key[key]
    return get


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "survey_disagreement_sentences_acl_pdf.csv", dtype={"survey_arxiv_id": str})

    ref_entries_cache: dict[str, list[dict]] = {}
    get_reference_entries = build_reference_index(ref_entries_cache)
    bib_cache = load_cache(BIB_CACHE_PATH)  # {normalized_title: {"title","abstract","source"} | None}

    kept_rows = []
    dropped_rows = []
    sentence_counter = 0

    for survey_key, group in df.groupby("survey_arxiv_id", sort=False):
        entries = get_reference_entries(survey_key)
        for _, row in group.iterrows():
            sentence_id = f"s{sentence_counter:04d}"
            sentence_counter += 1
            raw_markers = row["cited_keys"].split(";")
            n_keys = len(raw_markers)
            if n_keys < MIN_KEYS or n_keys > MAX_KEYS:
                # n_keys < MIN_KEYS не досягається на практиці: main() читає
                # лише survey_disagreement_sentences_acl_pdf.csv, яку
                # mine_disagreement_from_acl_pdfs.py вже фільтрує за власним
                # MIN_CITATIONS=2 до запису рядка -- лишається "too_many_citations"
                # (round 15, tech-debt fix #7, прибрано мертву гілку "single_cite_key").
                dropped_rows.append({**row.to_dict(), "sentence_id": sentence_id, "drop_reason": "too_many_citations"})
                continue

            matched = []
            drop_reason = None
            for raw in raw_markers:
                marker = parse_citation_marker(raw)
                if marker is None:
                    matched.append(None)
                    continue
                entry, reason = match_citation_to_entry(marker, entries)
                matched.append(entry)
                if entry is None and drop_reason is None:
                    drop_reason = reason

            n_resolved = sum(1 for e in matched if e is not None)
            if n_resolved < MIN_KEYS:
                dropped_rows.append({**row.to_dict(), "sentence_id": sentence_id, "drop_reason": drop_reason or "all_keys_unresolved"})
                continue

            # Мережевий lookup виконується ЛИШЕ тут -- для речень, що вже
            # пройшли офлайн-фільтр MIN_KEYS/MAX_KEYS вище, щоб не марнувати
            # запити на речення, які однаково будуть відкинуті.
            titles, authors, years, abstracts, contexts, sources = [], [], [], [], [], []
            for entry in matched:
                if entry is None:
                    # "no_match" -- цитату НЕ зіставлено з жодним записом
                    # списку літератури; відрізняється від "no_abstract_found"
                    # нижче (зіставлено, але lookup_abstract нічого не знайшов)
                    # -- раніше обидва позначались однаково як "unresolved"
                    # (round 15, tech-debt fix #5).
                    titles.append(""); authors.append(""); years.append(""); abstracts.append(""); contexts.append(""); sources.append("no_match")
                    continue
                bib = lookup_abstract(entry["title"], bib_cache)
                titles.append(entry["title"])
                authors.append("")  # entry["surnames"] is a tolerant matching set, not clean author data -- same "" fallback resolve_survey_citations.py's parse_bbl() uses when it can't cleanly extract authors
                years.append(entry["year"])
                abstracts.append(bib["abstract"] if bib else "")
                contexts.append(entry["title"])
                sources.append(bib["source"] if bib else "no_abstract_found")

            kept_rows.append({
                "sentence_id": sentence_id,
                "file": row["file"],
                "sentence": row["sentence"],
                "cited_keys": row["cited_keys"],
                "survey_arxiv_id": survey_key,
                "survey_title": row["survey_title"],
                "cited_keys_split": FIELD_SEP.join(raw_markers),
                "n_cited_keys": n_keys,
                "n_resolved": n_resolved,
                "resolved_titles": FIELD_SEP.join(titles),
                "resolved_authors": FIELD_SEP.join(authors),
                "resolved_years": FIELD_SEP.join(years),
                "resolved_abstracts": FIELD_SEP.join(abstracts),
                "resolved_context": FIELD_SEP.join(contexts),
                "resolution_source": FIELD_SEP.join(sources),
            })

            # Збереження ПІСЛЯ КОЖНОГО рядка (не один раз наприкінці всього
            # циклу) -- переривання прогону більше не втрачає вже зроблені
            # OpenAlex/Crossref lookup'и (round 15, tech-debt fix #1).
            save_cache(BIB_CACHE_PATH, bib_cache)

    # explicit columns= -- без цього pd.DataFrame([]) (0 рядків, як у пробі
    # на 3 PDF) не має жодної колонки, і to_csv() пише порожній файл без
    # заголовка, що ламає pd.read_csv() на приймальній стороні.
    kept_df = pd.DataFrame(kept_rows, columns=[
        "sentence_id", "file", "sentence", "cited_keys", "survey_arxiv_id",
        "survey_title", "cited_keys_split", "n_cited_keys", "n_resolved",
        "resolved_titles", "resolved_authors", "resolved_years",
        "resolved_abstracts", "resolved_context", "resolution_source",
    ])
    dropped_df = pd.DataFrame(dropped_rows, columns=[
        "file", "sentence", "cited_keys", "survey_arxiv_id", "survey_title",
        "sentence_id", "drop_reason",
    ])
    kept_df.to_csv(RESULTS_DIR / "candidate_sentences_resolved_acl_pdf.csv", index=False)
    dropped_df.to_csv(RESULTS_DIR / "citation_resolution_dropped_acl_pdf.csv", index=False)

    print(f"Eligible (resolved) rows: {len(kept_df)}", flush=True)
    print(f"Dropped rows: {len(dropped_df)}", flush=True)
    if len(dropped_df) > 0:
        print(dropped_df["drop_reason"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
