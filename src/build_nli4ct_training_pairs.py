"""Baseline generalization check (article Section 5.3), second corpus: build
training pairs from NLI4CT (SemEval-2023 Task 7 / SemEval-2024 Task 2,
https://github.com/ai-systems/nli4ct, CC-BY-4.0 -- see README.md there).
Downloads training_data.zip once to a gitignored cache, then constructs
hypothesis_a/hypothesis_b pairs from each (Statement, evidence) instance:
Statement -> hypothesis_a; evidence sentences resolved from the referenced
Clinical Trial Report section(s) via Primary/Secondary_evidence_index ->
hypothesis_b. Label "Entailment" -> NotContradiction, "Contradiction" ->
Contradiction.

Same structural limitation as Alamri & Stevenson (docs/s53_alternative_generalization_corpora.md):
the source scheme is binary, so no pair can ever be labeled "Apparent" --
this is a second, independent, more recent (2023-2024) crossdomain data
point for Section 5.3, not a fix for that gap.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CACHE_DIR = RESULTS_DIR / "nli4ct"
ZIP_PATH = CACHE_DIR / "training_data.zip"
EXTRACT_DIR = CACHE_DIR / "training_data"
ZIP_URL = "https://raw.githubusercontent.com/ai-systems/nli4ct/main/training_data.zip"

LABEL_MAP = {"Entailment": "NotContradiction", "Contradiction": "Contradiction"}


def download_corpus() -> None:
    if ZIP_PATH.exists():
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ZIP_URL} -> {ZIP_PATH}", flush=True)
    request = urllib.request.Request(ZIP_URL, headers={"User-Agent": "hypocontra-research (b.m.nedashkivskyi@gmail.com)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        ZIP_PATH.write_bytes(response.read())
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(CACHE_DIR)


def load_ct_json_dir() -> Path:
    # the zip nests one more "training_data" directory inside itself
    candidates = list(CACHE_DIR.glob("**/CT json"))
    if not candidates:
        raise FileNotFoundError(f"'CT json' directory not found under {CACHE_DIR}")
    return candidates[0]


def resolve_evidence(ct_dir: Path, ct_id: str, section_id: str, indices: list[int]) -> str:
    ct = json.loads((ct_dir / f"{ct_id}.json").read_text())
    sentences = ct[section_id]
    return " ".join(sentences[i] for i in indices if 0 <= i < len(sentences))


def build_pairs(ct_dir: Path, instances: dict, split: str) -> list[dict]:
    rows = []
    for instance_id, inst in instances.items():
        label_3way = LABEL_MAP[inst["Label"]]
        primary_evidence = resolve_evidence(
            ct_dir, inst["Primary_id"], inst["Section_id"], inst["Primary_evidence_index"]
        )
        if inst["Type"] == "Comparison":
            secondary_evidence = resolve_evidence(
                ct_dir, inst["Secondary_id"], inst["Section_id"], inst["Secondary_evidence_index"]
            )
            hypothesis_b = f"Primary trial: {primary_evidence} Secondary trial: {secondary_evidence}"
        else:
            hypothesis_b = primary_evidence

        rows.append({
            "pair_id": f"nli4ct_{instance_id}",
            "split": split,
            "type": inst["Type"],
            "section_id": inst["Section_id"],
            "primary_id": inst["Primary_id"],
            "secondary_id": inst.get("Secondary_id", ""),
            "hypothesis_a": inst["Statement"],
            "hypothesis_b": hypothesis_b,
            "label_3way": label_3way,
        })
    return rows


def main() -> None:
    download_corpus()
    ct_dir = load_ct_json_dir()

    train_path = next(CACHE_DIR.glob("**/train.json"))
    dev_path = next(CACHE_DIR.glob("**/dev.json"))
    train_instances = json.loads(train_path.read_text())
    dev_instances = json.loads(dev_path.read_text())

    rows = build_pairs(ct_dir, train_instances, "train") + build_pairs(ct_dir, dev_instances, "dev")
    pairs = pd.DataFrame(rows)

    out_path = RESULTS_DIR / "nli4ct_training_pairs.csv"
    pairs.to_csv(out_path, index=False)

    print(f"n={len(pairs)} pairs -> {out_path}")
    print(pairs.groupby("split")["label_3way"].value_counts().to_string())


if __name__ == "__main__":
    main()
