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
import html
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
    """Crossref abstract-и часто мають JATS-теги (<jats:p>...) І escaped
    HTML-entities (&amp;, &lt;) одночасно -- обидва прибираються тут
    (round 15, tech-debt fix #4; раніше лише теги, без unescape)."""
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


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
