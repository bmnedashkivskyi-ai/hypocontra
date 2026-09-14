"""Крок 2/3 round 14 -- PDF-аналог mine_disagreement_from_surveys.py.
Той самий COMBINED_PATTERN/EXCLUSION_PATTERN/ARTIFACT_PATTERN (імпортовано,
не скопійовано -- майбутні правки таксономії лишаються в одному місці),
але новий детектор цитат: замість \\cite{key} з LaTeX -- дужкові/наративні
author-year цитати (acl_ref_parsing.find_citations_in_sentence).

Секція референсів (acl_pdf_text.find_references_section) явно виключається
з майнінгу -- сама вона ніколи не містить справжнього протиставлення двох
робіт реченням автора, лише перелічує їх.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from acl_pdf_text import extract_pdf_text, find_references_section
from acl_ref_parsing import find_citations_in_sentence
from mine_disagreement_from_surveys import COMBINED_PATTERN, EXCLUSION_PATTERN, ARTIFACT_PATTERN

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PDF_CACHE_DIR = RESULTS_DIR / "acl_pdfs"
MIN_CITATIONS = 2


def load_survey_metadata() -> dict[str, dict]:
    with open(RESULTS_DIR / "acl_anthology" / "acl_not_on_arxiv.csv", encoding="utf-8") as f:
        return {row["key"]: row for row in csv.DictReader(f)}


def find_disagreement_sentences(pdf_path: Path) -> list[dict]:
    text = extract_pdf_text(pdf_path)
    refs = find_references_section(text)
    body = text[: len(text) - len(refs)] if refs else text
    body = re.sub(r"-\n", "", body)  # зняти перенос слова на межі рядка
    body = re.sub(r"\s+", " ", body)

    sentences = re.split(r"(?<=[.!?])\s+", body)
    hits = []
    for sent in sentences:
        if not (COMBINED_PATTERN.search(sent) and not EXCLUSION_PATTERN.search(sent)
                and not ARTIFACT_PATTERN.search(sent)):
            continue
        cites = find_citations_in_sentence(sent)
        if len(cites) < MIN_CITATIONS:
            continue
        clean_sent = re.sub(r"\s+", " ", sent).strip()[:500]
        hits.append({"sentence": clean_sent, "cited_keys": ";".join(cites)})
    return hits


def main(keys: list[str] | None = None) -> None:
    metadata = load_survey_metadata()
    target_keys = keys if keys is not None else sorted(metadata.keys())
    all_hits = []
    n_missing_pdf = 0
    n_failed = 0
    for i, key in enumerate(target_keys, 1):
        pdf_path = PDF_CACHE_DIR / f"{key}.pdf"
        print(f"[{i}/{len(target_keys)}] {key}", flush=True)
        if not pdf_path.exists():
            n_missing_pdf += 1
            continue
        # Один пошкоджений/нечитабельний PDF не має обривати весь прогін --
        # інші 271+ файлів вже опрацьовані, їх результат не варто втрачати
        # через один збій (round 15, tech-debt fix #2).
        try:
            hits = find_disagreement_sentences(pdf_path)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to process {key}: {e}", flush=True)
            n_failed += 1
            continue
        print(f"  {len(hits)} candidate disagreement sentences found", flush=True)
        row = metadata[key]
        for h in hits:
            h["file"] = f"{key}.pdf"
            h["survey_arxiv_id"] = key
            h["survey_title"] = row["title"]
        all_hits.extend(hits)

    df = pd.DataFrame(all_hits, columns=["file", "sentence", "cited_keys", "survey_arxiv_id", "survey_title"])
    out_path = RESULTS_DIR / "survey_disagreement_sentences_acl_pdf.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTotal candidate sentences across {len(target_keys)} PDFs "
          f"({n_missing_pdf} missing from cache, {n_failed} failed to process): {len(df)}", flush=True)


if __name__ == "__main__":
    main()
