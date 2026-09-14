"""Розвідка джерела ACL Anthology (§docs/next_scaling_directions.md, п.1).
Для 306 ACL-оглядів, яких немає серед уже сфетчених arXiv-назв (нормалізоване
рядкове порівняння, див. one-off перевірку в сесії), перевіряє пряме
title-query до arXiv API: (а) якщо огляд ВСЕ Ж є на arXiv (просто інша
формула заголовка чи не спрацював normalize-збіг) -- фіксує arXiv ID для
повторного використання наявного LaTeX-пайплайна; (б) якщо немає -- це
кандидат на майбутній PDF-based пайплайн (не реалізований в цій сесії).
"""
from __future__ import annotations

import csv
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
DELAY = 4.0
MAX_BACKOFF = 300.0


def search_title(title: str) -> str | None:
    q = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": 1})
    url = f"https://export.arxiv.org/api/query?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "hypocontra-pilot/1.0"})
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                root = ET.fromstring(resp.read().decode("utf-8"))
            entry = root.find("atom:entry", ARXIV_NS)
            if entry is None:
                return None
            arxiv_id = entry.find("atom:id", ARXIV_NS).text.strip().split("/")[-1].split("v")[0]
            return arxiv_id
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(30.0 * (2 ** attempt), MAX_BACKOFF)
                print(f"  429, backing off {wait:.0f}s...", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            print(f"  HTTPError {e.code} for {title!r}", flush=True)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"  error for {title!r}: {e}", flush=True)
            return None


def clean_title(t: str) -> str:
    return t.replace("{", "").replace("}", "")


def main() -> None:
    with open(RESULTS_DIR / "acl_anthology" / "acl_unmatched_candidates.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    found_rows = []
    not_found_rows = []
    for i, row in enumerate(rows, 1):
        title = clean_title(row["title"])
        print(f"[{i}/{len(rows)}] {title[:70]}", flush=True)
        arxiv_id = search_title(title)
        if arxiv_id:
            found_rows.append({**row, "arxiv_id": arxiv_id})
        else:
            not_found_rows.append(row)
        time.sleep(DELAY)

    out_dir = RESULTS_DIR / "acl_anthology"
    with open(out_dir / "acl_found_on_arxiv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "year", "title", "url", "arxiv_id"])
        w.writeheader()
        w.writerows(found_rows)
    with open(out_dir / "acl_not_on_arxiv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "year", "title", "url"])
        w.writeheader()
        w.writerows(not_found_rows)

    print(f"\nFound on arXiv (reusable via existing pipeline): {len(found_rows)}", flush=True)
    print(f"NOT on arXiv (would need PDF-based pipeline): {len(not_found_rows)}", flush=True)


if __name__ == "__main__":
    main()
