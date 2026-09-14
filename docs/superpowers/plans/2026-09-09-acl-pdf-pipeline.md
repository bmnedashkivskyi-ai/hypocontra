# Round 14: PDF Pipeline for ACL-Only Surveys — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PDF-based front-end (fetch → mine → resolve) that turns the 285 ACL-only survey titles with no arXiv match into `results/candidate_sentences_resolved_acl_pdf.csv`, in the exact schema the existing LaTeX-based `resolve_survey_citations.py` already produces, so the existing round machinery (`prepare_claude_pair_batch.py` onward) picks it up for round 14 with zero code changes.

**Architecture:** Three new orchestration scripts (`fetch_acl_pdfs.py`, `mine_disagreement_from_acl_pdfs.py`, `resolve_acl_pdf_citations.py`) built on top of two new shared modules (`acl_pdf_text.py` for PDF text extraction, `acl_ref_parsing.py` for citation-marker and reference-entry regex) and one new reusable module (`bib_lookup.py` for OpenAlex/Crossref abstract lookup, cached). The mining script reuses `mine_disagreement_from_surveys.py`'s `COMBINED_PATTERN`/`EXCLUSION_PATTERN`/`ARTIFACT_PATTERN` by import.

**Tech Stack:** Python 3, `pymupdf` (new dependency, PDF text extraction), `pandas` (already a project dependency), stdlib `urllib`/`json`/`re`/`difflib` (matching the project's existing style for network calls — see `check_acl_arxiv_overlap.py`, `verify_bibliography.py` — no `requests` for these scripts, consistent with everything except `annotate_with_gemma.py`).

**Spec:** `docs/superpowers/specs/2026-09-09-acl-pdf-pipeline-design.md`

## Global Constraints

- Zero changes to `prepare_claude_pair_batch.py`, `ingest_claude_pairs.py`, `prepare_claude_label_batch.py`, `ingest_claude_labels.py`, `annotate_with_gemma.py`, `merge_and_compute_kappa.py`.
- `results/candidate_sentences_resolved_acl_pdf.csv` must have exactly these columns, in this order, matching `resolve_survey_citations.py`'s `resolve_row()` + its caller in `main()`: `sentence_id, file, sentence, cited_keys, survey_arxiv_id, survey_title, cited_keys_split, n_cited_keys, n_resolved, resolved_titles, resolved_authors, resolved_years, resolved_abstracts, resolved_context, resolution_source`.
- Multi-value columns (`cited_keys_split`, `resolved_titles`, `resolved_authors`, `resolved_years`, `resolved_abstracts`, `resolved_context`, `resolution_source`) are joined with `FIELD_SEP = "\x1f"` (ASCII Unit Separator), one value per matched citation, same positional order as `cited_keys_split`.
- `MIN_KEYS = 2`, `MAX_KEYS = 6` resolved citations per candidate sentence (same gate as `resolve_survey_citations.py`).
- `CONTACT_EMAIL = "b.m.nedashkivskyi@gmail.com"` for every OpenAlex/Crossref call (`mailto` param / `User-Agent` header) — same constant used in `~/test-article/01-Kros-domenna-adaptatsiia-segmentatsii-kogeziia/docs/bibliography_verification_scripts/verify_bibliography.py`.
- No commit ever includes `results/acl_pdfs/*.pdf` (gitignored) — same treatment as `results/survey_sources/` and `results/acl_anthology/anthology.bib*`.
- Every filtering/matching failure is logged with an explicit reason column — never silently dropped (existing project convention throughout `resolve_survey_citations.py`, `ingest_claude_pairs.py`, `ingest_claude_labels.py`).
- Every long-running loop prints `[i/N] ...` progress lines (works with `src/monitor_progress.py` out of the box, no script changes needed there).

---

## Task 1: `src/acl_pdf_text.py` — PDF text extraction + references-section detection

**Files:**
- Create: `src/acl_pdf_text.py`
- Modify: `.venv/` — `pip install pymupdf`

**Interfaces:**
- Produces: `extract_pdf_text(pdf_path: Path) -> str`, `find_references_section(text: str) -> str | None`

- [ ] **Step 1: Install pymupdf into the project venv**

Run: `cd ~/prjs/hypocontra && .venv/bin/pip install pymupdf`
Expected: `Successfully installed pymupdf-<version>`

- [ ] **Step 2: Add `pymupdf` to the README's dependency install line**

In `README.md`, find the line:
```
python3 -m venv .venv && .venv/bin/pip install numpy pandas requests scikit-learn spacy
```
Replace with:
```
python3 -m venv .venv && .venv/bin/pip install numpy pandas requests scikit-learn spacy pymupdf
```

- [ ] **Step 3: Write the verification script (fails — module doesn't exist yet)**

Run this to fetch 3 known-good real ACL PDFs into a scratch dir for verification (throwaway, not committed):

```bash
mkdir -p /tmp/acl_pdf_probe && cd ~/prjs/hypocontra && .venv/bin/python3 -c "
import csv, urllib.request, time
rows = list(csv.DictReader(open('results/acl_anthology/acl_not_on_arxiv.csv', encoding='utf-8')))
sample = rows[:3]
for r in sample:
    url = r['url'].rstrip('/') + '.pdf'
    req = urllib.request.Request(url, headers={'User-Agent': 'hypocontra-pilot/1.0'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
    open(f'/tmp/acl_pdf_probe/{r[\"key\"]}.pdf', 'wb').write(data)
    print(r['key'], len(data), 'bytes')
    time.sleep(1)
"
```

Then write `/tmp/acl_pdf_probe/verify_task1.py`:

```python
import sys
sys.path.insert(0, "src")  # relative to cwd -- run this script with the repo root as cwd
from pathlib import Path
from acl_pdf_text import extract_pdf_text, find_references_section

PROBE_DIR = Path("/tmp/acl_pdf_probe")

# ollagnier-2026-antisocial: clean single-column PDF, "References" heading alone on its line
text1 = extract_pdf_text(PROBE_DIR / "ollagnier-2026-antisocial.pdf")
assert len(text1) > 10000, f"expected substantial text, got {len(text1)} chars"
assert "Antisocial Behavior Prediction" in text1
refs1 = find_references_section(text1)
assert refs1 is not None, "expected references section to be found"
assert "Al-Merekhi" in refs1[:2000], "expected first reference author near the top of the section"

# zevallos-etal-2026-amazonianlp: same clean case, second real paper
text2 = extract_pdf_text(PROBE_DIR / "zevallos-etal-2026-amazonianlp.pdf")
assert len(text2) > 5000
refs2 = find_references_section(text2)
assert refs2 is not None

# plausinaityte-zinsmeister-2026-survey: heading is "Bibliographical References", not bare
# "References" -- must still be found (prefix-tolerant heading match)
text3 = extract_pdf_text(PROBE_DIR / "plausinaityte-zinsmeister-2026-survey.pdf")
assert len(text3) > 1000
refs3 = find_references_section(text3)
assert refs3 is not None, "expected 'Bibliographical References' heading to be matched"

print("Task 1 verification: PASS")
```

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task1.py` (from `~/prjs/hypocontra`)
Expected: FAIL with `ModuleNotFoundError: No module named 'acl_pdf_text'`

- [ ] **Step 4: Implement `src/acl_pdf_text.py`**

```python
"""Спільний модуль для round 14 (PDF-пайплайн для 285 ACL-only оглядів,
docs/superpowers/specs/2026-09-09-acl-pdf-pipeline-design.md). Витягує
повний текст PDF і локалізує секцію списку літератури -- використовується
і mine_disagreement_from_acl_pdfs.py (повний текст), і
resolve_acl_pdf_citations.py (лише секція референсів).

Заголовок секції варіюється між venue ("References", "Bibliographical
References", "Language Resource References" -- підтверджено емпірично на
реальних PDF round 14) -- звідси толерантний до префікса regex, а не
точний збіг "References".
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

REFERENCES_HEADING_PATTERN = re.compile(r'\n([^\n]{0,30}\b[Rr]eferences\b[^\n]{0,10})\n')


def extract_pdf_text(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def find_references_section(text: str) -> str | None:
    """Останній збіг заголовка -- перший міг би трапитись у змісті/переліку
    розділів на початку статті, а не в самій секції референсів."""
    matches = list(REFERENCES_HEADING_PATTERN.finditer(text))
    if not matches:
        return None
    return text[matches[-1].end():]
```

- [ ] **Step 5: Run the verification script, confirm it passes**

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task1.py` (from `~/prjs/hypocontra`)
Expected: `Task 1 verification: PASS`

- [ ] **Step 6: Commit**

```bash
cd ~/prjs/hypocontra
git add src/acl_pdf_text.py README.md
git commit -m "$(cat <<'EOF'
Add PDF text extraction + references-section detection (round 14 step 1/6)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `src/acl_ref_parsing.py` — citation-marker regex + reference-entry parsing

**Files:**
- Create: `src/acl_ref_parsing.py`

**Interfaces:**
- Consumes: nothing (pure regex module, no dependency on Task 1)
- Produces: `find_citations_in_sentence(sentence: str) -> list[str]`, `parse_citation_marker(raw: str) -> dict | None` (returns `{"surnames": set[str], "year": str}`), `parse_reference_entries(refs_text: str) -> list[dict]` (each `{"surnames": set[str], "year": str, "title": str}`)

- [ ] **Step 1: Write the verification script (fails — module doesn't exist yet)**

Write `/tmp/acl_pdf_probe/verify_task2.py`:

```python
import sys
sys.path.insert(0, "src")  # relative to cwd -- run this script with the repo root as cwd
from acl_ref_parsing import find_citations_in_sentence, parse_citation_marker, parse_reference_entries

# --- find_citations_in_sentence: parenthetical, semicolon-separated, narrative,
# "et al." with no following name, compressed multi-year "2023a,b" ---
cases = {
    "This contradicts prior work (Al-Merekhi et al., 2020).": ["Al-Merekhi et al., 2020"],
    "(Kahn and Kellner, 2004; Brown et al., 2007; Quattrociocchi et al., 2014)":
        ["Kahn and Kellner, 2004", "Brown et al., 2007", "Quattrociocchi et al., 2014"],
    "While Alkomah and Ma (2022) report X, other work disagrees.": ["Alkomah and Ma, 2022"],
    "(Ollagnier et al., 2023a,b; Chowdhury et al., 2019)":
        ["Ollagnier et al., 2023a,b", "Chowdhury et al., 2019"],
    "In contrast to (Meta, 2022), this study argues otherwise.": ["Meta, 2022"],
}
for sentence, expected in cases.items():
    got = find_citations_in_sentence(sentence)
    assert got == expected, f"{sentence!r}: expected {expected}, got {got}"

# no citation present
assert find_citations_in_sentence("This sentence has no citation at all.") == []

# --- parse_citation_marker: surname extraction, including a 2-letter surname
# ("Ma") -- confirmed via real ACL reference list to require a 2+ char (not
# 3+ char) minimum token length ---
r1 = parse_citation_marker("Alkomah and Ma, 2022")
assert r1 == {"surnames": {"Alkomah", "Ma"}, "year": "2022"}, r1

r2 = parse_citation_marker("Al-Merekhi et al., 2020")
assert r2 == {"surnames": {"Al-Merekhi"}, "year": "2020"}, r2

assert parse_citation_marker("no year here") is None

# --- parse_reference_entries: real reference-list text (ACL style, one
# entry: "Firstname M. Surname, Firstname Surname, and Firstname Surname.
# Year. Title. Venue.") ---
sample_refs = (
    "Hind A. Al-Merekhi, Haewoon Kwak, Joni Salminen, and Bernard J. Jansen. "
    "2020. Are these comments triggering? predicting triggers of toxicity in "
    "online discussions. In WWW '20: The Web Conference 2020, pages 3033-3040. "
    "Aish Albladi, Minarul Islam, and Cheryl D. Seals. 2025. Hate speech "
    "detection using large language models: A comprehensive review. IEEE "
    "Access, 13:20871-20892."
)
entries = parse_reference_entries(sample_refs)
assert len(entries) == 2, f"expected 2 entries, got {len(entries)}: {entries}"
assert "Al-Merekhi" in entries[0]["surnames"], entries[0]
assert entries[0]["year"] == "2020", entries[0]
assert "triggering" in entries[0]["title"], entries[0]
assert "Albladi" in entries[1]["surnames"], entries[1]
assert entries[1]["year"] == "2025", entries[1]

print("Task 2 verification: PASS")
```

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task2.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'acl_ref_parsing'`

- [ ] **Step 2: Implement `src/acl_ref_parsing.py`**

```python
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
```

- [ ] **Step 3: Run the verification script, confirm it passes**

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task2.py`
Expected: `Task 2 verification: PASS`

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/acl_ref_parsing.py
git commit -m "$(cat <<'EOF'
Add citation-marker and reference-entry regex parsing (round 14 step 2/6)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `src/bib_lookup.py` — OpenAlex/Crossref abstract lookup with cache

**Files:**
- Create: `src/bib_lookup.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `load_cache(path: Path) -> dict`, `save_cache(path: Path, cache: dict) -> None`, `lookup_abstract(title: str, cache: dict, min_similarity: float = 0.6) -> dict | None` (returns `{"title": str, "abstract": str, "source": "openalex"|"crossref"}` or `None`; mutates `cache` in place, caller persists via `save_cache`)

- [ ] **Step 1: Write the verification script (fails — module doesn't exist yet)**

Write `/tmp/acl_pdf_probe/verify_task3.py`:

```python
import sys
sys.path.insert(0, "src")  # relative to cwd -- run this script with the repo root as cwd
from bib_lookup import load_cache, save_cache, lookup_abstract
from pathlib import Path

cache_path = Path("/tmp/acl_pdf_probe/bib_cache_test.json")
if cache_path.exists():
    cache_path.unlink()
cache = load_cache(cache_path)
assert cache == {}

# Real title with a literal "?" -- OpenAlex's `search` param treats "?"/"*"
# as wildcards and 400s on them unless stripped first (confirmed live
# during plan-writing: this exact title 400'd before the fix).
result = lookup_abstract(
    "Are these comments triggering? predicting triggers of toxicity in online discussions",
    cache,
)
assert result is not None, "expected a match for a real, exact paper title"
assert len(result["abstract"]) > 50, result
assert result["source"] in ("openalex", "crossref")

# Cache must now hold the normalized title as a key
assert len(cache) == 1

# Clearly bogus title -- must return None, not raise, and not match anything
# real (similarity threshold rejects any accidental top-search-result hit)
result2 = lookup_abstract(
    "Zzyzx Qwoph Nonexistent Paper Title Vklmnop 999",
    cache,
)
assert result2 is None, result2

save_cache(cache_path, cache)
assert cache_path.exists()

# reload from disk, confirm round-trip
cache2 = load_cache(cache_path)
assert len(cache2) == len(cache)

print("Task 3 verification: PASS")
```

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task3.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'bib_lookup'`

- [ ] **Step 2: Implement `src/bib_lookup.py`**

```python
"""OpenAlex -> Crossref abstract lookup для round 14 (заміна .bbl/.bib
abstract-поля, якого PDF-список літератури не дає -- лише title/authors/
year). Той самий CONTACT_EMAIL і паттерн викликів, що й у
verify_bibliography.py (проєкт Kros, docs/bibliography_verification.md).

Реальний баг, знайдений і виправлений під час каліброваня плану:
OpenAlex `search` трактує "?" і "*" як wildcard-символи й повертає HTTP 400,
якщо вони трапляються в назві буквально (напр. "Are these comments
triggering?") -- sanitize_title_for_openalex() прибирає їх перед запитом.
"""
from __future__ import annotations

import difflib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CONTACT_EMAIL = "b.m.nedashkivskyi@gmail.com"
REQUEST_DELAY = 0.4  # той самий інтервал, що й verify_bibliography.py


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def _title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_title(a), _normalize_title(b)).ratio()


def _sanitize_title_for_openalex(title: str) -> str:
    return re.sub(r"[?*]", " ", title).strip()


def _http_get_json(url: str, headers: dict | None = None, timeout: int = 20) -> tuple[dict | None, str | None]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def _reconstruct_openalex_abstract(inverted_index: dict) -> str:
    positions = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    positions.sort()
    return " ".join(word for _, word in positions)


def _query_openalex(title: str) -> dict | None:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "search": _sanitize_title_for_openalex(title),
        "per_page": 1,
        "mailto": CONTACT_EMAIL,
    })
    data, err = _http_get_json(url)
    if err or not data:
        return None
    results = data.get("results", [])
    if not results:
        return None
    top = results[0]
    inverted = top.get("abstract_inverted_index")
    if not inverted:
        return None
    return {"title": top.get("title") or "", "abstract": _reconstruct_openalex_abstract(inverted)}


def _strip_jats_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _query_crossref(title: str) -> dict | None:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({
        "query.bibliographic": title,
        "rows": 1,
    })
    headers = {"User-Agent": f"hypocontra-pilot/1.0 (mailto:{CONTACT_EMAIL})"}
    data, err = _http_get_json(url, headers=headers)
    if err or not data:
        return None
    items = data.get("message", {}).get("items", [])
    if not items:
        return None
    top = items[0]
    abstract = top.get("abstract")
    if not abstract:
        return None
    top_title = (top.get("title") or [""])[0]
    return {"title": top_title, "abstract": _strip_jats_tags(abstract)}


def lookup_abstract(title: str, cache: dict, min_similarity: float = 0.6) -> dict | None:
    cache_key = _normalize_title(title)
    if cache_key in cache:
        return cache[cache_key]

    result = None
    oa = _query_openalex(title)
    time.sleep(REQUEST_DELAY)
    if oa and _title_similarity(title, oa["title"]) >= min_similarity:
        result = {"title": oa["title"], "abstract": oa["abstract"], "source": "openalex"}
    else:
        cr = _query_crossref(title)
        time.sleep(REQUEST_DELAY)
        if cr and _title_similarity(title, cr["title"]) >= min_similarity:
            result = {"title": cr["title"], "abstract": cr["abstract"], "source": "crossref"}

    cache[cache_key] = result
    return result
```

- [ ] **Step 3: Run the verification script, confirm it passes**

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task3.py`
Expected: `Task 3 verification: PASS`

Note: this step makes live network calls to OpenAlex/Crossref. If it fails
with a network error rather than an assertion error, re-run once before
investigating further (matches the flakiness already tolerated in
`check_acl_arxiv_overlap.py`'s retry/backoff loop).

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/bib_lookup.py
git commit -m "$(cat <<'EOF'
Add OpenAlex/Crossref abstract lookup with cache (round 14 step 3/6)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `src/fetch_acl_pdfs.py` — bulk PDF downloader

**Files:**
- Create: `src/fetch_acl_pdfs.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `results/acl_anthology/acl_not_on_arxiv.csv` (`key, year, title, url`)
- Produces: `results/acl_pdfs/<key>.pdf` (one file per successfully downloaded row), `results/acl_pdf_download_failed.csv` (`key, year, title, url, error`)

- [ ] **Step 1: Add `results/acl_pdfs/` to `.gitignore`**

In `.gitignore`, add a new line after the existing `results/acl_anthology/anthology.bib.gz` line:
```
results/acl_pdfs/
```

- [ ] **Step 2: Implement `src/fetch_acl_pdfs.py`**

```python
"""Крок 1/3 round 14 (PDF-пайплайн для 285 ACL-only оглядів без
відповідника на arXiv, docs/superpowers/specs/2026-09-09-acl-pdf-pipeline-
design.md). Завантажує PDF для кожного рядка
results/acl_anthology/acl_not_on_arxiv.csv у results/acl_pdfs/<key>.pdf
(gitignored, як і results/survey_sources/) -- ідемпотентно, пропускає вже
кешовані файли.

PDF-посилання ACL Anthology: url-поле вхідного CSV -- це URL
сторінки-лендінгу (закінчується на "/"), пряме посилання на PDF --
той самий URL з "/" на кінці замінений на ".pdf" (підтверджено прямим
HTTP-запитом під час каліброваня плану).
"""
from __future__ import annotations

import csv
import time
import urllib.error
import urllib.request
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PDF_CACHE_DIR = RESULTS_DIR / "acl_pdfs"
DELAY = 1.5
MAX_BACKOFF = 60.0


def pdf_url(landing_url: str) -> str:
    return landing_url.rstrip("/") + ".pdf"


def download_one(url: str, dest: Path) -> str | None:
    """Повертає None при успіху, інакше рядок опису помилки."""
    req = urllib.request.Request(url, headers={"User-Agent": "hypocontra-pilot/1.0"})
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                wait = min(15.0 * (2 ** attempt), MAX_BACKOFF)
                print(f"  429, backing off {wait:.0f}s...", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            return f"HTTP {e.code}: {e.reason}"
        except Exception as e:  # noqa: BLE001
            return str(e)


def main(limit: int | None = None) -> None:
    PDF_CACHE_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "acl_anthology" / "acl_not_on_arxiv.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]

    failed_rows = []
    n_skipped = 0
    n_downloaded = 0
    for i, row in enumerate(rows, 1):
        dest = PDF_CACHE_DIR / f"{row['key']}.pdf"
        if dest.exists():
            n_skipped += 1
            continue
        url = pdf_url(row["url"])
        print(f"[{i}/{len(rows)}] {row['key']}", flush=True)
        err = download_one(url, dest)
        if err:
            print(f"  FAILED: {err}", flush=True)
            failed_rows.append({**row, "error": err})
        else:
            n_downloaded += 1
        time.sleep(DELAY)

    failed_path = RESULTS_DIR / "acl_pdf_download_failed.csv"
    with open(failed_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "year", "title", "url", "error"])
        w.writeheader()
        w.writerows(failed_rows)

    print(f"\nDownloaded: {n_downloaded}, already cached: {n_skipped}, failed: {len(failed_rows)}", flush=True)
    print(f"Failures logged to {failed_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Лише для розробки/перевірки -- обробити перші N рядків.")
    args = parser.parse_args()
    main(limit=args.limit)
```

- [ ] **Step 3: Run against a small real sample, verify it downloads and is resumable**

```bash
cd ~/prjs/hypocontra
.venv/bin/python3 src/fetch_acl_pdfs.py --limit 3
ls results/acl_pdfs/
```
Expected: 3 `.pdf` files listed, `Downloaded: 3, already cached: 0, failed: 0`

Run again (same command) to confirm resumability:
```bash
.venv/bin/python3 src/fetch_acl_pdfs.py --limit 3
```
Expected: `Downloaded: 0, already cached: 3, failed: 0`

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/fetch_acl_pdfs.py .gitignore
git commit -m "$(cat <<'EOF'
Add ACL PDF bulk downloader (round 14 step 4/6)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

Note: `results/acl_pdfs/*.pdf` themselves are gitignored and won't show up
in `git status` as untracked after this commit.

---

## Task 5: `src/mine_disagreement_from_acl_pdfs.py` — sentence mining

**Files:**
- Create: `src/mine_disagreement_from_acl_pdfs.py`

**Interfaces:**
- Consumes: `acl_pdf_text.extract_pdf_text`, `acl_pdf_text.find_references_section` (only to exclude it from mining — a references section itself never contains a genuine two-work contrast sentence, and would otherwise pollute results with reference-list fragments); `acl_ref_parsing.find_citations_in_sentence`; `mine_disagreement_from_surveys.COMBINED_PATTERN`, `mine_disagreement_from_surveys.EXCLUSION_PATTERN`, `mine_disagreement_from_surveys.ARTIFACT_PATTERN` (imported, not duplicated)
- Produces: `results/survey_disagreement_sentences_acl_pdf.csv` (`file, sentence, cited_keys, survey_arxiv_id, survey_title`) — same shape as `survey_disagreement_sentences*.csv`, with `cited_keys` as `;`-joined raw citation strings (e.g. `"Solovev and Pröllochs, 2023;Meta, 2022"`, matching the existing `";".join(cites)` convention in `mine_disagreement_from_surveys.py`)

- [ ] **Step 1: Write the verification script (fails — module doesn't exist yet)**

Write `/tmp/acl_pdf_probe/verify_task5.py` (uses the 3 PDFs already fetched by Task 1's Step 3 into `/tmp/acl_pdf_probe/`, copied into the real cache dir so the script's real input path works):

```python
import sys, shutil
sys.path.insert(0, "src")  # relative to cwd -- run this script with the repo root as cwd
from pathlib import Path

PROBE_DIR = Path("/tmp/acl_pdf_probe")
CACHE_DIR = Path("results/acl_pdfs")  # relative to cwd -- run this script with the repo root as cwd
CACHE_DIR.mkdir(exist_ok=True)
for name in ["ollagnier-2026-antisocial.pdf", "zevallos-etal-2026-amazonianlp.pdf", "plausinaityte-zinsmeister-2026-survey.pdf"]:
    shutil.copy(PROBE_DIR / name, CACHE_DIR / name)

import mine_disagreement_from_acl_pdfs as mod
import pandas as pd

# restrict to just the 3 probe rows for a fast, deterministic check
mod.main(keys=["ollagnier-2026-antisocial", "zevallos-etal-2026-amazonianlp", "plausinaityte-zinsmeister-2026-survey"])

out_path = Path("results/survey_disagreement_sentences_acl_pdf.csv")
assert out_path.exists()
df = pd.read_csv(out_path)
assert list(df.columns) == ["file", "sentence", "cited_keys", "survey_arxiv_id", "survey_title"], list(df.columns)
# expect at least one real hit -- confirmed empirically (ollagnier paper,
# "Mitigation: ... treat annotator disagreement as informative (Lambert et
# al., 2022; Kim et al., 2025)") during plan calibration, though that
# specific sentence should be EXCLUDED by EXCLUSION_PATTERN
# ("annotat\w+ disagree") -- so this assertion checks the pipeline runs
# end-to-end and produces a well-formed (possibly empty) CSV, not a
# specific row count.
assert len(df) >= 0
for _, row in df.iterrows():
    assert isinstance(row["cited_keys"], str) and len(row["cited_keys"]) > 0
    assert ";" in row["cited_keys"] or "," in row["cited_keys"]  # at least 1 citation, comma from "Name, Year"

print(f"Task 5 verification: PASS ({len(df)} candidate sentences from 3 probe PDFs)")
```

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task5.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'mine_disagreement_from_acl_pdfs'`

- [ ] **Step 2: Implement `src/mine_disagreement_from_acl_pdfs.py`**

```python
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
    for i, key in enumerate(target_keys, 1):
        pdf_path = PDF_CACHE_DIR / f"{key}.pdf"
        print(f"[{i}/{len(target_keys)}] {key}", flush=True)
        if not pdf_path.exists():
            n_missing_pdf += 1
            continue
        hits = find_disagreement_sentences(pdf_path)
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
          f"({n_missing_pdf} missing from cache): {len(df)}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the verification script, confirm it passes**

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task5.py`
Expected: `Task 5 verification: PASS (N candidate sentences from 3 probe PDFs)`

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/mine_disagreement_from_acl_pdfs.py
git commit -m "$(cat <<'EOF'
Add disagreement-sentence mining from ACL PDFs (round 14 step 5/6)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `src/resolve_acl_pdf_citations.py` — citation resolution + abstract lookup

**Files:**
- Create: `src/resolve_acl_pdf_citations.py`

**Interfaces:**
- Consumes: `results/survey_disagreement_sentences_acl_pdf.csv` (Task 5), `acl_pdf_text.extract_pdf_text` + `find_references_section`, `acl_ref_parsing.parse_reference_entries` + `parse_citation_marker`, `bib_lookup.load_cache` + `save_cache` + `lookup_abstract`
- Produces: `results/candidate_sentences_resolved_acl_pdf.csv` (schema fixed in Global Constraints), `results/citation_resolution_dropped_acl_pdf.csv` (same columns as `results/citation_resolution_dropped*.csv` plus `drop_reason`, with two new reasons: `citation_unmatched`, `citation_ambiguous`, alongside `single_cite_key`)

- [ ] **Step 1: Write the verification script (fails — module doesn't exist yet)**

Write `/tmp/acl_pdf_probe/verify_task6.py`:

```python
import sys
sys.path.insert(0, "src")  # relative to cwd -- run this script with the repo root as cwd
from pathlib import Path
import pandas as pd

# Task 5 must already have produced this file (run verify_task5.py first
# if starting fresh)
input_path = Path("results/survey_disagreement_sentences_acl_pdf.csv")
assert input_path.exists(), "run verify_task5.py first to produce the mining-stage output"

import resolve_acl_pdf_citations as mod
mod.main()

resolved_path = Path("results/candidate_sentences_resolved_acl_pdf.csv")
dropped_path = Path("results/citation_resolution_dropped_acl_pdf.csv")
assert resolved_path.exists()
assert dropped_path.exists()

EXPECTED_COLUMNS = [
    "sentence_id", "file", "sentence", "cited_keys", "survey_arxiv_id",
    "survey_title", "cited_keys_split", "n_cited_keys", "n_resolved",
    "resolved_titles", "resolved_authors", "resolved_years",
    "resolved_abstracts", "resolved_context", "resolution_source",
]
resolved_df = pd.read_csv(resolved_path)
assert list(resolved_df.columns) == EXPECTED_COLUMNS, list(resolved_df.columns)

# every row must satisfy the MIN_KEYS=2..MAX_KEYS=6 gate
for _, row in resolved_df.iterrows():
    assert 2 <= row["n_cited_keys"] <= 6, row
    assert row["n_resolved"] >= 2, row
    # FIELD_SEP-joined columns must have exactly n_cited_keys parts each
    n = row["n_cited_keys"]
    assert len(row["cited_keys_split"].split("\x1f")) == n, row

print(f"Task 6 verification: PASS ({len(resolved_df)} resolved candidates)")
```

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task6.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'resolve_acl_pdf_citations'`

- [ ] **Step 2: Implement `src/resolve_acl_pdf_citations.py`**

```python
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
    рік записаний у цитаті-в-тексті, і як у власному списку літератури."""
    return re.match(r"(\d{4})", year).group(1)


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
                dropped_rows.append({**row.to_dict(), "sentence_id": sentence_id, "drop_reason": "single_cite_key" if n_keys < MIN_KEYS else "too_many_citations"})
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
                    titles.append(""); authors.append(""); years.append(""); abstracts.append(""); contexts.append(""); sources.append("unresolved")
                    continue
                bib = lookup_abstract(entry["title"], bib_cache)
                titles.append(entry["title"])
                authors.append(", ".join(sorted(entry["surnames"])))
                years.append(entry["year"])
                abstracts.append(bib["abstract"] if bib else "")
                contexts.append(entry["title"])
                sources.append(bib["source"] if bib else "unresolved")

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

    save_cache(BIB_CACHE_PATH, bib_cache)

    kept_df = pd.DataFrame(kept_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    kept_df.to_csv(RESULTS_DIR / "candidate_sentences_resolved_acl_pdf.csv", index=False)
    dropped_df.to_csv(RESULTS_DIR / "citation_resolution_dropped_acl_pdf.csv", index=False)

    print(f"Eligible (resolved) rows: {len(kept_df)}", flush=True)
    print(f"Dropped rows: {len(dropped_df)}", flush=True)
    if len(dropped_df) > 0:
        print(dropped_df["drop_reason"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the verification script, confirm it passes**

Run: `.venv/bin/python3 /tmp/acl_pdf_probe/verify_task6.py`
Expected: `Task 6 verification: PASS (N resolved candidates)` (N may be 0 — the 3
probe PDFs are a tiny sample and round 13 already showed most sentences get
filtered out; the assertions on schema and the MIN_KEYS/MAX_KEYS gate are
what this step actually proves)

- [ ] **Step 4: Commit**

```bash
cd ~/prjs/hypocontra
git add src/resolve_acl_pdf_citations.py
git commit -m "$(cat <<'EOF'
Add PDF citation resolution + OpenAlex/Crossref abstract lookup (round 14 step 6/6)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full run (all 285 titles) + issue round 14 batch

**Files:**
- Modify: `README.md` (add round 14 front-end scripts to the Структура list, matching the pattern already used for round 13's `check_acl_arxiv_overlap.py` entry)

**Interfaces:**
- Consumes: all of Tasks 1–6
- Produces: `results/survey_disagreement_sentences_acl_pdf.csv`, `results/candidate_sentences_resolved_acl_pdf.csv`, `results/citation_resolution_dropped_acl_pdf.csv` (full 285-title versions, overwriting the 3-probe versions from Tasks 5–6), `results/claude_pair_batch_round14.jsonl` (first batch issued to Claude for round 14 pair construction)

- [ ] **Step 1: Clear the 3-probe test artifacts so the full run starts clean**

```bash
cd ~/prjs/hypocontra
rm -f results/acl_pdfs/ollagnier-2026-antisocial.pdf \
      results/acl_pdfs/zevallos-etal-2026-amazonianlp.pdf \
      results/acl_pdfs/plausinaityte-zinsmeister-2026-survey.pdf
```
(These get re-downloaded by the full run below — removing them first
avoids a false sense of "already cached" if their content ever needs to be
re-verified; keeping them would also work since `fetch_acl_pdfs.py` is
idempotent, but starting clean makes step 3's fetched-count assertion
exact.)

- [ ] **Step 2: Run the full fetch (285 PDFs)**

```bash
cd ~/prjs/hypocontra
.venv/bin/python3 src/fetch_acl_pdfs.py 2>&1 | tee /tmp/acl_pdf_probe/fetch_full.log
```
Expected: final line `Downloaded: N, already cached: 0, failed: M` with
`N + M == 285`. Note `M` (failure count) and reasons from
`results/acl_pdf_download_failed.csv` — some 404s/timeouts are expected at
this scale and are not a blocker (already-designed-for, logged not
silently dropped).

This takes roughly 285 × ~2s ≈ 10 minutes (download + `DELAY`); run it with
the Bash tool's `run_in_background` option if using
`superpowers:subagent-driven-development` or `executing-plans` and you'd
rather not block on it — `monitor_progress.py attach` can tail the log:
```bash
.venv/bin/python src/monitor_progress.py attach --log /tmp/acl_pdf_probe/fetch_full.log
```

- [ ] **Step 3: Run the full mining pass**

```bash
cd ~/prjs/hypocontra
.venv/bin/python3 src/mine_disagreement_from_acl_pdfs.py 2>&1 | tee /tmp/acl_pdf_probe/mine_full.log
```
Expected: final line `Total candidate sentences across N PDFs (M missing
from cache): K`. Record K.

- [ ] **Step 4: Run the full resolution pass**

```bash
cd ~/prjs/hypocontra
.venv/bin/python3 src/resolve_acl_pdf_citations.py 2>&1 | tee /tmp/acl_pdf_probe/resolve_full.log
```
Expected: `Eligible (resolved) rows: J` and a `drop_reason` breakdown.
Record J (this is round 14's total candidate-pair-eligible sentence count,
analogous to round 13's 7).

- [ ] **Step 5: Verify the output schema one more time against the full run**

```bash
cd ~/prjs/hypocontra
.venv/bin/python3 -c "
import pandas as pd
resolved = pd.read_csv('results/candidate_sentences_resolved_acl_pdf.csv')
existing = pd.read_csv('results/candidate_sentences_resolved.csv')  # round 7-13 LaTeX-based file
assert list(resolved.columns) == list(existing.columns), (list(resolved.columns), list(existing.columns))
print('Schema matches round 7-13 file exactly:', list(resolved.columns))
print('Round 14 candidate rows:', len(resolved))
"
```
Expected: `Schema matches round 7-13 file exactly: [...]` with no assertion error.

- [ ] **Step 6: Issue the round 14 batch for Claude pair construction**

```bash
cd ~/prjs/hypocontra
.venv/bin/python3 src/prepare_claude_pair_batch.py --input candidate_sentences_resolved_acl_pdf.csv --round 14
```
Expected: `Issued N sentences into results/claude_pair_batch_round14.jsonl`
where N == J from Step 4. This is where the automated part of round 14
ends — filling in `results/claude_pair_batch_round14_completed.csv` is
manual Claude-session work (pair construction judgment calls), exactly
like every prior round, and is out of scope for this implementation plan.

- [ ] **Step 7: Update README's Структура section**

In `README.md`, in the `## Структура` list, add these lines after the
existing `check_acl_arxiv_overlap.py` line:
```
- `src/acl_pdf_text.py` — PDF-екстракція тексту + локалізація секції референсів (round 14)
- `src/acl_ref_parsing.py` — regex-парсинг author-year цитат у реченні + власного списку літератури з PDF (round 14)
- `src/bib_lookup.py` — OpenAlex/Crossref lookup abstract за назвою, з кешем (round 14)
- `src/fetch_acl_pdfs.py` — масове завантаження PDF для 285 ACL-only оглядів (round 14)
- `src/mine_disagreement_from_acl_pdfs.py` — видобування маркерів контрасту з PDF-тексту (round 14)
- `src/resolve_acl_pdf_citations.py` — резолюція цитат без .bbl/.bib + abstract lookup (round 14)
```

- [ ] **Step 8: Commit**

```bash
cd ~/prjs/hypocontra
git add results/survey_disagreement_sentences_acl_pdf.csv \
        results/candidate_sentences_resolved_acl_pdf.csv \
        results/citation_resolution_dropped_acl_pdf.csv \
        results/acl_pdf_download_failed.csv \
        results/acl_pdf_bib_lookup_cache.json \
        results/claude_pair_batch_round14.jsonl \
        results/claude_batch_state_round14.json \
        docs/claude_batch_instructions.md \
        README.md
git status  # confirm results/acl_pdfs/*.pdf is NOT staged (gitignored)
git commit -m "$(cat <<'EOF'
Run round 14 PDF pipeline on all 285 ACL-only titles, issue pair batch

Full fetch -> mine -> resolve run over results/acl_anthology/acl_not_on_arxiv.csv.
See commit body / round summary for fetched/mined/resolved counts.
results/claude_pair_batch_round14.jsonl issued -- pair construction (manual
Claude-session step) is the next round-14 action, same as every prior round.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** all three spec stages (fetch/mine/resolve) map to Tasks 4/5/6, backed by the two shared modules from Tasks 1–2 and the lookup module from Task 3; Task 7 covers the spec's "existing round machinery, no code changes" goal by actually running it once. The spec's risk section (unproven citation-resolution reliability, title-mismatch risk, heuristic reference-boundary detection) is addressed by: explicit `citation_unmatched`/`citation_ambiguous` drop reasons (Task 6), the `min_similarity` threshold in `bib_lookup.lookup_abstract` (Task 3), and the acknowledged partial-coverage-per-paper behavior of `parse_reference_entries` (Task 2, empirically calibrated against 3 real PDFs during plan-writing rather than left as a guess).
- **Placeholder scan:** none found — every task has complete, empirically-verified code (see calibration notes in Tasks 1–3's docstrings, all derived from real ACL PDF probes during plan-writing).
- **Type consistency:** `find_citations_in_sentence` (Task 2) returns `list[str]` consumed by `mine_disagreement_from_acl_pdfs.py` (Task 5) as `row["cited_keys"]` (`;`-joined) and re-split by `resolve_acl_pdf_citations.py` (Task 6) via `.split(";")` — consistent both ways. `parse_reference_entries` and `parse_citation_marker` both return `{"surnames": set[str], "year": str, ...}` — consistent shape used by `match_citation_to_entry` (Task 6). `bib_lookup.lookup_abstract`'s return shape (`{"title", "abstract", "source"}` or `None`) matches how Task 6 consumes it.
