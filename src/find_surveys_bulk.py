"""Round 9+: масштабний пошук оглядів замість точкових keyword-запитів
(find_more_surveys.py). Попередні два прогони показали: (1) вузькі теми-
запити (bias/fairness/...) дали 0 генуїнно нових кандидатів (мета-огляди
про генерацію оглядів / симуляцію опитувань); (2) розширені теми
(attention/in-context learning/hallucination/causal inference/...) дали
~0.2 генуїнної пари на огляд. Щоб наблизитись до цільових 300-500 пар
(§4.4 статті) за такого темпу, потрібні СОТНІ оглядів, не десятки -- тому
тут один широкий запит з пагінацією (усі cs.CL статті з "survey"/
"systematic review"/"literature review" У НАЗВІ), а не десяток вузьких тем.

arXiv API агресивно рейт-лімітить (перевірено емпірично -- 429 навіть після
20с очікування після попередніх прогонів у цій сесії). Тому тут експоненційний
backoff на 429 (не миттєвий фейл) і консервативніша пауза між сторінками
(5с), ніж у find_more_surveys.py (3.5с).
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

PAGE_SIZE = 100
# Round 18 (2026-09-17): 5.0s held for ~1.5 days across ~13.5k requests still
# triggered a persistent 406 block that outlasted a full retry-and-resume
# cycle (see PersistentBlockError) -- raised well above arXiv's stated 3s
# minimum on the theory that it's sustained volume/duration, not the
# per-request rate, that tripped their abuse detection.
PAGE_DELAY = 20.0
MAX_RETRY_BACKOFF = 300.0

# Ідентичний фільтр хибних сенсів "survey", що й у find_more_surveys.py
# (два огляди мали розійтись -- тримаємо єдиним джерелом правди тут,
# скопійовано навмисно замість імпорту, щоб цей скрипт лишався самостійним
# і незалежним від змін у find_more_surveys.py).
EXCLUDE_PATTERN = re.compile(
    r"survey generation|generat\w+ (academic )?survey|survey (automation|writing|writer|benchmark|evaluator)"
    r"|survey response|survey simulation|simulate\w* .*survey|synthetic survey|social survey"
    r"|questionnaire|opinion survey|persona-grounded|automatic survey|survey item",
    re.IGNORECASE,
)

# Перевірено round 12: голе ti:review у cs.CL (1136 результатів) -- майже
# суцільно "peer review" (LLM-рецензування наукових статей, окремий модний
# піднапрямок), "product/app reviews" і "code review", НЕ огляди
# літератури; ті кілька, що є справжніми systematic review, вже покриті
# запитом "survey" (QUERY_MODES["survey"]). Бару "review" НЕ додано як
# QUERY_MODE через цю перевірену непродуктивність.
#
# Розширення на cs.LG/cs.AI (round 12): цей запит category-wise ширший за
# cs.CL, тому повертає багато оглядів, що взагалі не про NLP (vision, RL,
# теорія навчання). Щоб не розмивати заявлений домен статті (NLP/CS), тут
# додатково фільтруємо за релевантністю: title/summary мають містити хоча б
# один NLP-індикативний термін.
NLP_RELEVANCE_PATTERN = re.compile(
    r"language model|natural language|\bnlp\b|linguistic|\btext\b|translation|dialogue|"
    r"\bspeech\b|\bcorpus\b|corpora|multilingual|\bllm|large language|conversational ai|"
    r"question answering|summarization|sentiment|named entity|machine translation",
    re.IGNORECASE,
)


# Round 11: "survey"/"systematic review"/"literature review" (round 9-10)
# вичерпано повністю (1700/1700). Розширення на ширші категорії (cs.LG/cs.AI)
# дає 5007 результатів ti:survey -- зависоко для щільності релевантності
# (переважна більшість не про NLP-гіпотези), тож замість цього лишаємось у
# cs.CL і додаємо синонімічні фразові шаблони "overview of"/"review of"
# (1319 результатів, перевірено прямим запитом) -- вужче й точніше, ніж
# голі "overview"/"review"/"taxonomy"/"state of the art" (653-5007
# результатів, але з високим ризиком хибних спрацювань: "A New
# State-of-the-Art Model for X" -- це звичайна стаття, не огляд).
# Round 15: "a review" (кінець назви, напр. "Sentiment Analysis: A
# Review") -- на відміну від голого ti:review (round 12, 1136 результатів,
# майже все "peer review"), фразовий збіг "a review" безпечніший. Разом з
# "primer on"/"tutorial on" -- totalResults=1191 перевірено прямим запитом
# ДО повного прогону (як для "overview of"/"review of" у round 11);
# "a review" сама дає 1142 з 1191 -- більшість.
QUERY_MODES = {
    "survey": 'cat:cs.CL AND (ti:survey OR ti:"systematic review" OR ti:"literature review")',
    "overview": 'cat:cs.CL AND (ti:"overview of" OR ti:"review of")',
    "survey_broad": '(cat:cs.CL OR cat:cs.LG OR cat:cs.AI) AND (ti:survey OR ti:"systematic review" OR ti:"literature review")',
    "title_synonyms": 'cat:cs.CL AND (ti:"a review" OR ti:"primer on" OR ti:"tutorial on")',
}
# survey_broad вимагає додатково NLP_RELEVANCE_PATTERN у Python-фільтрі
# нижче (cs.LG/cs.AI без цього дають забагато off-topic оглядів).
REQUIRE_NLP_RELEVANCE = {"survey_broad"}


class PersistentBlockError(RuntimeError):
    """Raised by fetch_page when max_attempts is set and exceeded.

    Round 18 (2026-09-17): a query that hit 406 kept hitting 406 on every
    resumed run, immediately, for 4.5+ hours straight -- not the once-off
    transient block the original comment below assumed. That points to
    arXiv fingerprinting the specific query rather than a blanket IP-level
    block that just needs to be waited out. Unbounded retry then means one
    such query can stall an entire unattended run indefinitely.
    """


def fetch_page(start: int, max_results: int, query: str, max_attempts: int | None = None) -> ET.Element:
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "hypocontra-pilot/1.0"})

    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return ET.fromstring(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code == 406 or e.code >= 500:
                # 406 seen during round 18 (2026-09-16, ~13521/18172 in): arXiv's
                # anti-abuse layer escalated past plain 429 after ~1.5 days of
                # sustained querying, on an otherwise unremarkable query -- a
                # manual re-request of the same exact query later succeeded with
                # 200 once the block lifted, confirming it's transient like 429,
                # not a malformed-query error. Unhandled before this fix, it
                # killed the run outright.
                if max_attempts is not None and attempt >= max_attempts:
                    raise PersistentBlockError(f"HTTP {e.code} persisted past {max_attempts} attempts") from e
                wait = min(30.0 * (2 ** attempt), MAX_RETRY_BACKOFF)
                print(f"  HTTP {e.code}, backing off {wait:.0f}s (attempt {attempt + 1})...", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # Transient network failure (DNS hiccup, reset connection, etc.) --
            # not an HTTPError, so it bypassed retry entirely before this fix,
            # which killed unattended multi-hour runs (round 18) on the first blip.
            if max_attempts is not None and attempt >= max_attempts:
                raise PersistentBlockError(f"network error persisted past {max_attempts} attempts") from e
            wait = min(30.0 * (2 ** attempt), MAX_RETRY_BACKOFF)
            print(f"  network error ({e}), backing off {wait:.0f}s (attempt {attempt + 1})...", flush=True)
            time.sleep(wait)
            attempt += 1
            continue


def already_seen_ids() -> set[str]:
    def norm(id_: str) -> str:
        if "." in id_:
            p, f = id_.split(".", 1)
            return f"{p}.{f.ljust(5, '0')}"
        return id_

    seen: set[str] = set()
    sources = [
        ("results/survey_disagreement_sentences_round4_dedup.csv", "survey_arxiv_id"),
        ("results/additional_surveys_found_v1_81.csv", "arxiv_id"),
        ("results/additional_surveys_found.csv", "arxiv_id"),
        ("results/survey_batch2_truly_new.csv", "arxiv_id"),
        ("results/survey_bulk_search.csv", "arxiv_id"),
        ("results/survey_bulk_search_part2.csv", "arxiv_id"),
        ("results/survey_bulk_search_overview.csv", "arxiv_id"),
    ]
    for rel_path, col in sources:
        path = RESULTS_DIR.parent / rel_path
        if path.exists():
            df = pd.read_csv(path, dtype={col: str})
            seen.update(norm(x) for x in df[col])
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-total", type=int, default=1000, help="До якого офсету включно переглянути (абсолютний, не відносний до --start-offset).")
    parser.add_argument("--start-offset", type=int, default=0, help="З якого офсету в результатах arXiv продовжити (для наступних раундів пошуку).")
    parser.add_argument("--out", type=str, default="survey_bulk_search.csv")
    parser.add_argument("--query-mode", type=str, default="survey", choices=list(QUERY_MODES))
    args = parser.parse_args()

    query = QUERY_MODES[args.query_mode]
    print(f"Query: {query}", flush=True)

    seen = already_seen_ids()
    print(f"Already-seen survey IDs to exclude: {len(seen)}", flush=True)

    all_rows = []
    start = args.start_offset
    total_available = None

    while start < args.max_total:
        page_size = min(PAGE_SIZE, args.max_total - start)
        print(f"Fetching results {start}-{start + page_size}...", flush=True)
        root = fetch_page(start, page_size, query)

        if total_available is None:
            total_el = root.find("opensearch:totalResults", ARXIV_NS)
            total_available = int(total_el.text) if total_el is not None else None
            print(f"Total available in arXiv for this query: {total_available}", flush=True)

        entries = root.findall("atom:entry", ARXIV_NS)
        if not entries:
            print("No more entries returned, stopping.", flush=True)
            break

        for entry in entries:
            arxiv_id = entry.find("atom:id", ARXIV_NS).text.strip().split("/")[-1].split("v")[0]
            title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
            summary_el = entry.find("atom:summary", ARXIV_NS)
            summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
            published = entry.find("atom:published", ARXIV_NS).text[:10]

            if arxiv_id in seen:
                continue
            if EXCLUDE_PATTERN.search(title) or EXCLUDE_PATTERN.search(summary):
                continue
            if args.query_mode in REQUIRE_NLP_RELEVANCE:
                if not (NLP_RELEVANCE_PATTERN.search(title) or NLP_RELEVANCE_PATTERN.search(summary)):
                    continue
            all_rows.append({"arxiv_id": arxiv_id, "title": title, "published": published})
            seen.add(arxiv_id)

        start += page_size
        if total_available is not None and start >= total_available:
            break
        time.sleep(PAGE_DELAY)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / args.out, index=False)
    print(f"\nCollected {len(df)} genuinely new survey candidates -> {RESULTS_DIR / args.out}", flush=True)


if __name__ == "__main__":
    main()
