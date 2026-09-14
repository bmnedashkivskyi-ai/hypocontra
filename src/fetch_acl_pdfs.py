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
    # Теоретичний ризик, не спостережений на практиці (acl_not_on_arxiv.csv
    # локально курований, не user-supplied): urlopen на будь-яку схему/хост
    # без перевірки міг би, наприклад, піти на file://. Явний allowlist
    # (round 15, tech-debt fix #8).
    if not url.startswith("https://aclanthology.org/"):
        return f"rejected: unexpected URL scheme/host ({url})"
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
            print(f"[{i}/{len(rows)}] {row['key']} (cached, skip)", flush=True)
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
