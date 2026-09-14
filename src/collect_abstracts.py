"""Крок 4.1 методології HypoContra: збір кандидатних абстрактів з arXiv
cs.CL, 2016-2026 (той самий часовий діапазон, що й огляд пов'язаних робіt).

ACL Anthology НЕ використана як друге джерело в цьому пілоті (звужено
відносно §4.1 методології) -- arXiv API публічний, добре документований
і не потребує окремого парсингу; ACL Anthology вимагає окремого
XML/BibTeX парсингу її offline-дампу. Це задокументоване звуження
пілотного обсягу, не приховане припущення.
"""
from __future__ import annotations

import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
BATCH_SIZE = 100
YEARS = list(range(2016, 2027))  # 2016-2026 включно, як у §4.1 методології
PER_YEAR_TARGET = 150  # ~150 абстрактів/рік * 11 років ~= 1650, рівномірне охоплення діапазону


def fetch_batch(start: int, batch_size: int = BATCH_SIZE, year: int | None = None) -> str:
    search_query = "cat:cs.CL"
    if year is not None:
        # arXiv date-range синтаксис: включає submittedDate у межах [YYYYMMDD0000 TO YYYYMMDD2359]
        search_query += f" AND submittedDate:[{year}01010000 TO {year}12312359]"
    query = urllib.parse.urlencode({
        "search_query": search_query,
        "start": start,
        "max_results": batch_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"http://export.arxiv.org/api/query?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "hypocontra-pilot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    rows = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id = entry.find("atom:id", ARXIV_NS).text.strip().split("/")[-1]
        title = entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ARXIV_NS).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ARXIV_NS).text.strip()
        year = int(published[:4])
        authors = [a.find("atom:name", ARXIV_NS).text for a in entry.findall("atom:author", ARXIV_NS)]
        categories = [c.get("term") for c in entry.findall("atom:category", ARXIV_NS)]
        rows.append({
            "arxiv_id": arxiv_id, "title": title, "abstract": summary,
            "year": year, "published": published, "authors": "; ".join(authors),
            "categories": ";".join(categories),
        })
    return rows


def main():
    all_rows = []
    t0 = time.time()
    for year in YEARS:
        year_rows = []
        start = 0
        while len(year_rows) < PER_YEAR_TARGET:
            print(f"Fetching year={year} start={start} ...", flush=True)
            xml_text = fetch_batch(start, year=year)
            rows = parse_entries(xml_text)
            if not rows:
                print(f"  no more entries for {year}, stopping this year.")
                break
            year_rows.extend(rows)
            start += BATCH_SIZE
            time.sleep(3.5)  # ввічливий rate-limit -- arXiv просить 1 запит/3с
        all_rows.extend(year_rows)
        print(f"  year {year}: {len(year_rows)} abstracts collected")

    df = pd.DataFrame(all_rows)
    df = df[(df["year"] >= 2016) & (df["year"] <= 2026)]
    df = df.drop_duplicates(subset="arxiv_id")
    df.to_csv(RESULTS_DIR / "arxiv_cscl_abstracts.csv", index=False)
    print(f"\nTotal: {len(df)} abstracts, years {df['year'].min()}-{df['year'].max()}, "
          f"{time.time()-t0:.1f}s")
    print(df["year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
