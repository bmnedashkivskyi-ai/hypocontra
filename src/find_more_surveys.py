"""Розширення джерела оглядів для round 3+ добору кандидатів (§4.2 v3).
Цілеспрямований пошук arXiv cs.CL за темами, де round 3 показав найвищий
вихід explicit-disagreement речень (вимірювально-методологічні теми:
bias, fairness, evaluation, robustness, interpretability, hallucination) --
на відміну від чисто архітектурних оглядів (0 збігів у round 3).

ІТЕРАЦІЯ 2 (round 7+): перший прогін цього скрипта (81 огляд) дав 0
генуїнно нових кандидатних речень при повторному майнінгу -- перевірено
прямо. Причина, підтверджена ручним переглядом усіх 81 знайдених назв:
слово "survey" у arXiv cs.CL за 2024-2026 роки систематично матчиться на
ДВА хибних сенси, не пов'язаних з "оглядом літератури":
  (а) мета-статті про LLM-автоматизацію НАПИСАННЯ/ГЕНЕРАЦІЇ оглядів
      (AutoSurvey, SurveyX, SurGE, SurveyBench, SurveyEval, SGSimEval,
      SurveyReview, "Evaluation Sheet for Deep Research") -- це про
      інструмент, не про предметний огляд NLP-гіпотез;
  (б) статті про LLM, що СИМУЛЮЮТЬ людські ОПИТУВАННЯ (questionnaire/
      opinion "survey", а не "survey" = "огляд літератури") -- "Survey
      Responses", "Survey Simulations", "Simulate Social Surveys",
      "Synthetic Survey Responses".
EXCLUDE_PATTERN нижче відсікає обидва сенси явно (не мовчки). QUERIES
розширено темами з підтвердженою продуктивністю в round 3-6/7 (attention
interpretability, in-context learning) -- саме звідти походять реальні
знайдені суперечності (Jain/Wiegreffe, Razeghi/Han).
"""
from __future__ import annotations

import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

QUERIES = [
    "attention interpretability survey",
    "in-context learning survey",
    "hallucination detection survey",
    "annotator disagreement survey",
    "reproducibility survey natural language",
    "spurious correlations survey NLP",
    "benchmark evaluation survey language models",
    "causal inference survey natural language",
    "dataset artifacts survey NLP",
    "model comparison survey natural language",
]

# § докстрінг вище -- два хибні сенси слова "survey" у 2024-2026 arXiv cs.CL.
EXCLUDE_PATTERN = re.compile(
    r"survey generation|generat\w+ (academic )?survey|survey (automation|writing|writer|benchmark|evaluator)"
    r"|survey response|survey simulation|simulate\w* .*survey|synthetic survey|social survey"
    r"|questionnaire|opinion survey|persona-grounded|automatic survey",
    re.IGNORECASE,
)


def search(query: str, max_results: int = 15) -> list[dict]:
    params = urllib.parse.urlencode({
        "search_query": f'cat:cs.CL AND abs:"survey" AND abs:"{query.split()[0]}"',
        "start": 0, "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    url = f"http://export.arxiv.org/api/query?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "hypocontra-pilot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_text = resp.read().decode("utf-8")
    root = ET.fromstring(xml_text)
    rows = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id = entry.find("atom:id", ARXIV_NS).text.strip().split("/")[-1].split("v")[0]
        title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ARXIV_NS).text.strip().replace("\n", " ")
        # тримаємось лише справжніх оглядів -- назва або абстракт мають явно це сигналізувати
        is_review_shaped = "survey" in title.lower() or "systematic review" in title.lower() or "literature review" in title.lower()
        is_false_sense = EXCLUDE_PATTERN.search(title) or EXCLUDE_PATTERN.search(summary)
        if is_review_shaped and not is_false_sense:
            rows.append({"arxiv_id": arxiv_id, "title": title, "query": query})
    return rows


def main():
    all_rows = []
    seen_ids = set()
    for q in QUERIES:
        print(f"Searching: {q!r}", flush=True)
        try:
            rows = search(q)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            continue
        new_rows = [r for r in rows if r["arxiv_id"] not in seen_ids]
        for r in new_rows:
            seen_ids.add(r["arxiv_id"])
        all_rows.extend(new_rows)
        print(f"  {len(new_rows)} new survey candidates")
        time.sleep(3.5)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "additional_surveys_found.csv", index=False)
    print(f"\nTotal new survey candidates: {len(df)}")
    print(df[["arxiv_id", "title"]].to_string())


if __name__ == "__main__":
    main()
