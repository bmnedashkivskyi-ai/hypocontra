"""Baseline generalization check (article Section 5.3): build training pairs
from the Alamri & Stevenson (2016) biomedical contradiction corpus
(https://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/,
CC BY-NC-SA 2.0 UK). Downloads corpus.xml once to a gitignored cache, then
constructs all claim-pairs within each shared QUESTION group: differing
ASSERTION (YS vs NO) -> Contradiction, matching ASSERTION -> NotContradiction.

The source scheme is binary (YS/NO), so no pair can ever be labeled
"Apparent" -- this is an inherent limitation of the training data, not a bug
in this script, and is called out in docs/generalization_report.md.
"""
from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CACHE_DIR = RESULTS_DIR / "alamri_stevenson"
CORPUS_PATH = CACHE_DIR / "corpus.xml"
CORPUS_URL = "https://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/corpus.xml"


def download_corpus() -> None:
    if CORPUS_PATH.exists():
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {CORPUS_URL} -> {CORPUS_PATH}", flush=True)
    request = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "hypocontra-research (b.m.nedashkivskyi@gmail.com)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        CORPUS_PATH.write_bytes(response.read())


def build_pairs() -> pd.DataFrame:
    tree = ET.parse(CORPUS_PATH)
    root = tree.getroot()

    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for review in root.findall("REVIEW"):
        review_pmid = review.get("REVIEW_PMID")
        for claim in review.findall("CLAIM"):
            question = claim.get("QUESTION")
            assertion = claim.get("ASSERTION")
            pmid = claim.get("PMID")
            text = (claim.text or "").strip()
            groups[question].append((assertion, pmid, text))

    rows = []
    pair_idx = 0
    for question, claims in groups.items():
        for (assertion_a, pmid_a, text_a), (assertion_b, pmid_b, text_b) in combinations(claims, 2):
            label_3way = "Contradiction" if assertion_a != assertion_b else "NotContradiction"
            rows.append({
                "pair_id": f"as_{pair_idx:06d}",
                "question": question,
                "hypothesis_a": text_a,
                "pmid_a": pmid_a,
                "hypothesis_b": text_b,
                "pmid_b": pmid_b,
                "label_3way": label_3way,
            })
            pair_idx += 1

    return pd.DataFrame(rows)


def main() -> None:
    download_corpus()
    pairs = build_pairs()
    out_path = RESULTS_DIR / "alamri_stevenson_training_pairs.csv"
    pairs.to_csv(out_path, index=False)

    print(f"n={len(pairs)} training pairs -> {out_path}")
    print(pairs["label_3way"].value_counts().to_string())


if __name__ == "__main__":
    main()
