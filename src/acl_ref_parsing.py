"""Regex-евристики для round 14 (PDF-пайплайн, без .bbl/.bib -- на відміну
від resolve_survey_citations.py). Дві незалежні задачі в одному модулі,
бо обидві працюють з тим самим "author-year citation" форматом:

1. find_citations_in_sentence / parse_citation_marker -- виявлення й
   парсинг цитат ПРЯМО В РЕЧЕННІ кандидата (заміна \\cite{key} з LaTeX-версії
   на "(Author, Year)" / "Author (Year)" -- ACL-стиль цитування).
2. parse_reference_entries -- парсинг ВЛАСНОГО списку літератури статті
   (замінює .bbl/.bib-парсинг -- немає структурованого джерела, лише
   текст після заголовка "References").

Обидві емпірично каліброван на 3 реальних PDF з
results/acl_anthology/acl_not_on_arxiv.csv (docs/superpowers/specs/
2026-09-09-acl-pdf-pipeline-design.md) під час написання плану, зокрема
два реальні баги, знайдені й виправлені до першого запуску:

- "et al." НЕ вимагає імені після себе (на відміну від "and Name"/"& Name")
  -- перша версія регексу вимагала, і жодна "et al."-цитата не збігалась.
- surname-токени мінімум 2 символи (не 3+) -- 3+ пропускає короткі
  прізвища на кшталт "Ma", "Wu", часті в цій предметній області.
"""
from __future__ import annotations

import re

NAME_PATTERN = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-']+"
YEAR_PATTERN = r"(?:19|20)\d{2}[a-z]?(?:,[a-z])*"
_ONE_CITATION = rf"{NAME_PATTERN}(?:\s+et\s+al\.|\s+(?:and|&)\s+{NAME_PATTERN})?,?\s+{YEAR_PATTERN}"

CITATION_GROUP_PATTERN = re.compile(rf'\(({_ONE_CITATION}(?:\s*;\s*{_ONE_CITATION})*)\)')
NARRATIVE_CITATION_PATTERN = re.compile(
    rf'({NAME_PATTERN}(?:\s+et\s+al\.|\s+(?:and|&)\s+{NAME_PATTERN})?)\s+\(({YEAR_PATTERN})\)'
)
CITATION_YEAR_SUFFIX_PATTERN = re.compile(rf"({YEAR_PATTERN})$")
NAME_TOKEN_PATTERN = re.compile(NAME_PATTERN)


def find_citations_in_sentence(sentence: str) -> list[str]:
    """Повертає РОЗБИТІ по ';' окремі цитати як сирі рядки, напр.
    "Smith et al., 2020" -- готові для parse_citation_marker()."""
    out: list[str] = []
    for group in CITATION_GROUP_PATTERN.findall(sentence):
        out.extend(part.strip() for part in re.split(r"\s*;\s*", group))
    for name, year in NARRATIVE_CITATION_PATTERN.findall(sentence):
        out.append(f"{name}, {year}")
    seen: set[str] = set()
    deduped = []
    for c in out:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def parse_citation_marker(raw: str) -> dict | None:
    raw = raw.strip()
    ym = CITATION_YEAR_SUFFIX_PATTERN.search(raw)
    if not ym:
        return None
    year = ym.group(1)
    name_part = raw[:ym.start()].rstrip(", ")
    name_part = re.sub(r"\bet\s+al\.?$", "", name_part).strip(", ")
    surnames = set(NAME_TOKEN_PATTERN.findall(name_part))
    if not surnames:
        return None
    return {"surnames": surnames, "year": year}


# Технічний (не бібліографічний) шум, що часто трапляється у вікні
# безпосередньо перед year-маркером через "протікання" з попереднього
# запису (venue/publisher-слова) -- не справжні прізвища авторів. Список
# НЕ претендує на повноту: помилкові прізвища в surnames-множині нешкідливі,
# бо зіставлення (resolve_acl_pdf_citations.py) додатково вимагає збігу
# року, що різко знижує ризик хибного зіставлення.
REFERENCE_STOPWORDS = {
    "The", "In", "Proceedings", "Journal", "International", "Conference",
    "Association", "IEEE", "ACM", "Workshop", "Findings", "Computational",
    "Linguistics", "Language", "Natural", "Processing", "Transactions",
    "Annual", "Meeting", "North", "American", "Chapter", "European", "Web",
    "Science", "Advances", "Neural", "Information", "Systems", "Empirical",
    "Methods",
}
_YEAR_MARKER = re.compile(rf'\b({YEAR_PATTERN})\.\s')
_TITLE_AFTER_YEAR = re.compile(r'([^.]{10,250})\.\s')


def parse_reference_entries(refs_text: str) -> list[dict]:
    """Евристичний парсинг: анкер -- "Year." (типовий кінець поля дати в
    ACL-стилі "Author. Year. Title. Venue."), а не спроба точно визначити
    межі запису. surnames -- МНОЖИНА всіх Title-Case токенів у вікні перед
    роком (не лише перший автор) -- навмисно толерантний до неточних меж
    вікна, див. докстрінг модуля."""
    text = re.sub(r"\n\d{1,4}\n", " ", refs_text)  # прибрати номери сторінок
    text = re.sub(r"-\n", "", text)  # зняти перенос слова на межі рядка
    text = re.sub(r"\s+", " ", text).strip()

    entries = []
    prev_end = 0
    for m in _YEAR_MARKER.finditer(text):
        year = m.group(1)
        window = text[prev_end:m.start()]
        surnames = set(NAME_TOKEN_PATTERN.findall(window)) - REFERENCE_STOPWORDS
        rest = text[m.end():]
        title_match = _TITLE_AFTER_YEAR.match(rest)
        title = title_match.group(1).strip() if title_match else None
        if surnames and title:
            entries.append({"surnames": surnames, "year": year, "title": title})
        prev_end = m.end()
    return entries
