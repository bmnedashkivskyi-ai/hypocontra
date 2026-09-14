"""Крок 1 масштабування round 7 (§4.4 методології HypoContra, наступна ітерація
після round 3-6, docs/pilot_final_summary.md).

Вузьким місцем пілоту round 3-6 була НЕ якість regex-кандидатів із
mine_disagreement_from_surveys.py (16/17 = 94% генуїнних контрадикцій), а
РУЧНЕ перетворення речення-кандидата на пару гіпотез: резолюція \\cite-ключів
до реальних робіт і читання їхніх тез. Цей скрипт автоматизує саме резолюцію
-- повністю офлайн, з .bbl/.bib файлів, які вже лежать у кожній завантаженій
survey_sources/<id>/ теці (жодного нового мережевого запиту).

Два реальні баги вхідних даних, підтверджені прямою перевіркою
survey_disagreement_sentences_round4_dedup.csv, враховані явно (не мовчки):

1. survey_arxiv_id пошкоджено float-коерцією у 5/42 записів (напр. "2306.1053"
   замість реальної теки "2306.10530") -- виправляється дописуванням нулів до
   5-значної дробової частини (стандартний пост-2007 формат YYMM.NNNNN) із
   фолбеком на префіксний пошук по каталогу survey_sources/.
2. cited_keys розділені ДВОМА різними символами: ";" між окремими \\cite{}
   командами в одному реченні, "," всередині одного \\cite{a,b,c}. Наївний
   спліт лише по ";" дає 139 "одноключових" рядків замість реальних 123 --
   тут використано re.split(r"[;,]", ...).

Додатково: кілька рядків (n_keys=68, 38, 34, 32, 26, 25) виявились артефактами
LaTeX-таблиць/tikz-дерев (\\begin{figure*}, \\tikzstyle), які мав відсікати
ARTIFACT_PATTERN з mine_disagreement_from_surveys.py, але в цьому конкретному
dedup-файлі не відсікав -- застосовано тут повторно як явний другий фільтр,
плюс обмеження 2 <= n_keys <= 6 (більше -- це майже завжди перелік у
related work, а не справжній двобічний контраст).

Резолюція title з .bbl -- ЕВРИСТИЧНА й свідомо неповна (regex, не повний
LaTeX-парсер): для ACL-стилю (\\newblock \\href{url}{Title}) і для
IEEEtran-стилю (title у "..."-лапках) беруться різні шаблони; коли жоден не
спрацював, у "resolved_titles" лишається порожньо, але повний текст запису
все одно зберігається в "resolved_context" -- побудова пари (крок 2) читає
його як фолбек.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SRC_CACHE_DIR = RESULTS_DIR / "survey_sources"

MIN_KEYS = 2
MAX_KEYS = 6
CONTEXT_TRUNCATE = 800
FIELD_SEP = "\x1f"  # ASCII Unit Separator -- див. коментар у resolve_row()

# Скопійовано з mine_disagreement_from_surveys.py -- той самий фільтр
# артефактів LaTeX-таблиць/рисунків, що мав відсікти ці рядки на етапі
# майнінгу, але в round4_dedup не відсік.
ARTIFACT_PATTERN = re.compile(
    r"\\begin\{(table|figure|tikz)|\\midrule|\\tikzstyle|\\multirow|\\multicolumn",
    re.IGNORECASE,
)

BIBITEM_PATTERN = re.compile(
    r"\\bibitem(?:\[.*?\])?\{([^}]+)\}(.*?)(?=\\bibitem(?:\[.*?\])?\{|\\end\{thebibliography\}|\Z)",
    re.DOTALL,
)
HREF_TITLE_PATTERN = re.compile(r"\\href\s*\{[^}]*\}\s*\{([^}]+)\}")
NEWBLOCK_PATTERN = re.compile(r"\\newblock\s*(.*?)(?=\\newblock|\Z)", re.DOTALL)
QUOTED_TITLE_PATTERN = re.compile(r"``\s*(.*?)\s*''", re.DOTALL)
BIB_ENTRY_START_PATTERN = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,")


def normalize_survey_id(raw_id: str, cache_dir: Path) -> str | None:
    """Виправляє float-пошкоджені arXiv ID (§ баг 1 у докстрінгу вище)."""
    if (cache_dir / raw_id).is_dir():
        return raw_id
    if "." in raw_id:
        prefix, frac = raw_id.split(".", 1)
        padded = f"{prefix}.{frac.ljust(5, '0')}"
        if (cache_dir / padded).is_dir():
            return padded
    matches = sorted(p.name for p in cache_dir.iterdir() if p.is_dir() and p.name.startswith(raw_id))
    if matches:
        return matches[0]
    return None


def split_cited_keys(raw: str) -> list[str]:
    """§ баг 2 у докстрінгу вище -- ";" між \\cite{}, "," всередині \\cite{a,b}."""
    seen: dict[str, None] = {}
    for token in re.split(r"[;,]", raw):
        key = token.strip()
        if key:
            seen[key] = None
    return list(seen)


def clean_latex(text: str) -> str:
    text = re.sub(r"\\newblock", " ", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\s*\{[^}]*\}\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def guess_bbl_title(entry_text: str) -> str:
    href_match = HREF_TITLE_PATTERN.search(entry_text)
    if href_match:
        return clean_latex(href_match.group(1))
    quoted_match = QUOTED_TITLE_PATTERN.search(entry_text)
    if quoted_match:
        return clean_latex(quoted_match.group(1))
    newblock_match = NEWBLOCK_PATTERN.search(entry_text)
    if newblock_match:
        candidate = clean_latex(newblock_match.group(1))
        if candidate:
            return candidate[:300]
    return ""


def parse_bbl(path: Path) -> dict[str, dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    refs: dict[str, dict] = {}
    for match in BIBITEM_PATTERN.finditer(text):
        key = match.group(1).strip()
        entry_text = match.group(2)
        refs[key] = {
            "title": guess_bbl_title(entry_text),
            "authors": "",
            "year": "",
            "abstract": "",
            "context": clean_latex(entry_text)[:CONTEXT_TRUNCATE],
            "source": "bbl",
        }
    return refs


def extract_balanced(text: str, open_pos: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """Повертає (вміст без зовнішніх дужок, позицію одразу після close_ch)."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return text[open_pos + 1:i], i + 1
    return text[open_pos + 1:], len(text)


def parse_bib_entry_fields(entry_body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for fm in re.finditer(r"(\w+)\s*=\s*", entry_body):
        field_name = fm.group(1).lower()
        val_start = fm.end()
        if val_start >= len(entry_body):
            continue
        ch = entry_body[val_start]
        if ch == "{":
            value, _ = extract_balanced(entry_body, val_start, "{", "}")
        elif ch == '"':
            end = entry_body.find('"', val_start + 1)
            value = entry_body[val_start + 1:end] if end != -1 else ""
        else:
            m = re.match(r"([^,}]+)", entry_body[val_start:])
            value = m.group(1).strip() if m else ""
        fields[field_name] = clean_latex(value)
    return fields


def parse_bib(path: Path) -> dict[str, dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    refs: dict[str, dict] = {}
    for m in BIB_ENTRY_START_PATTERN.finditer(text):
        key = m.group(2).strip()
        brace_pos = text.find("{", m.start())
        if brace_pos == -1:
            continue
        body, _ = extract_balanced(text, brace_pos, "{", "}")
        fields = parse_bib_entry_fields(body)
        refs[key] = {
            "title": fields.get("title", ""),
            "authors": fields.get("author", ""),
            "year": fields.get("year", ""),
            "abstract": fields.get("abstract", "")[:CONTEXT_TRUNCATE],
            "context": clean_latex(body)[:CONTEXT_TRUNCATE],
            "source": "bib",
        }
    return refs


def build_survey_ref_index(survey_dir: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for bbl_path in survey_dir.rglob("*.bbl"):
        index.update({k: v for k, v in parse_bbl(bbl_path).items() if k not in index})
    for bib_path in survey_dir.rglob("*.bib"):
        for k, v in parse_bib(bib_path).items():
            if k not in index or (v.get("abstract") and not index[k].get("abstract")):
                index[k] = v
    return index


def resolve_row(row: pd.Series, ref_index: dict[str, dict]) -> tuple[dict | None, str | None]:
    if ARTIFACT_PATTERN.search(row["sentence"]):
        return None, "likely_latex_artifact"

    keys = split_cited_keys(row["cited_keys"])
    n_keys = len(keys)
    if n_keys < MIN_KEYS:
        return None, "single_cite_key"
    if n_keys > MAX_KEYS:
        return None, "likely_latex_artifact"

    resolved = [ref_index.get(k) for k in keys]
    n_resolved = sum(1 for r in resolved if r is not None)
    if n_resolved < MIN_KEYS:
        return None, "all_keys_unresolved" if n_resolved == 0 else "insufficient_resolved_lt_2"

    # УВАГА: роздільник для об'єднання полів кількох цитувань в один CSV-осередок.
    # Раніше тут був ";" -- виявлено емпірично (round 9, s0329 та ще 9 рядків):
    # бібліографічний текст (title/authors/context) сам часто містить ";"
    # (напр. "Chen et al., 2023b; Madaan et al., 2023"), що зсуває
    # .split(";") на прийомній стороні (prepare_claude_pair_batch.py) і
    # приписує контекст/анотацію НЕ ТОМУ цитованому ключу. FIELD_SEP -- ASCII
    # Unit Separator (0x1F), який ніколи не трапляється у звичайному тексті.
    def joined(field: str) -> str:
        return FIELD_SEP.join((r.get(field, "") if r else "") for r in resolved)

    return {
        "cited_keys_split": FIELD_SEP.join(keys),
        "n_cited_keys": n_keys,
        "n_resolved": n_resolved,
        "resolved_titles": joined("title"),
        "resolved_authors": joined("authors"),
        "resolved_years": joined("year"),
        "resolved_abstracts": joined("abstract"),
        "resolved_context": joined("context"),
        "resolution_source": FIELD_SEP.join((r.get("source", "unresolved") if r else "unresolved") for r in resolved),
    }, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, nargs="+",
        default=["survey_disagreement_sentences_round4_dedup.csv"],
        help="Одне чи кілька імен CSV у results/ (об'єднуються перед резолюцією).",
    )
    parser.add_argument("--output-suffix", type=str, default="")
    args = parser.parse_args()

    df = pd.concat(
        [pd.read_csv(RESULTS_DIR / name, dtype={"survey_arxiv_id": str}) for name in args.input],
        ignore_index=True,
    )
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["survey_arxiv_id", "sentence"]).reset_index(drop=True)
    if before_dedup != len(df):
        print(f"Dropped {before_dedup - len(df)} exact (survey_id, sentence) duplicates across input files", flush=True)

    n_repaired = 0
    normalized_ids = []
    for raw_id in df["survey_arxiv_id"]:
        norm = normalize_survey_id(raw_id, SRC_CACHE_DIR)
        if norm is None:
            normalized_ids.append(raw_id)
        else:
            if norm != raw_id:
                n_repaired += 1
            normalized_ids.append(norm)
    df["survey_arxiv_id"] = normalized_ids

    kept_rows = []
    dropped_rows = []
    sentence_counter = 0

    for survey_id, group in df.groupby("survey_arxiv_id", sort=False):
        survey_dir = SRC_CACHE_DIR / survey_id
        ref_index = build_survey_ref_index(survey_dir) if survey_dir.is_dir() else {}
        for _, row in group.iterrows():
            sentence_id = f"s{sentence_counter:04d}"
            sentence_counter += 1
            resolved, drop_reason = resolve_row(row, ref_index)
            if resolved is None:
                dropped_rows.append({**row.to_dict(), "sentence_id": sentence_id, "drop_reason": drop_reason})
            else:
                kept_rows.append({
                    "sentence_id": sentence_id,
                    "file": row["file"],
                    "sentence": row["sentence"],
                    "cited_keys": row["cited_keys"],
                    "survey_arxiv_id": survey_id,
                    "survey_title": row["survey_title"],
                    **resolved,
                })

    kept_df = pd.DataFrame(kept_rows)
    dropped_df = pd.DataFrame(dropped_rows)

    suffix = args.output_suffix
    kept_df.to_csv(RESULTS_DIR / f"candidate_sentences_resolved{suffix}.csv", index=False)
    dropped_df.to_csv(RESULTS_DIR / f"citation_resolution_dropped{suffix}.csv", index=False)

    print(f"Repaired survey_arxiv_id for {n_repaired} rows (float-truncation fix)", flush=True)
    print(f"Total input rows: {len(df)}", flush=True)
    print(f"Eligible (resolved) rows: {len(kept_df)}", flush=True)
    print(f"Dropped rows: {len(dropped_df)}", flush=True)
    if len(dropped_df) > 0:
        print(dropped_df["drop_reason"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
