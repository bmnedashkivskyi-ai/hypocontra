"""Крок 4 масштабування round 7. Анотатор B -- локальна Gemma
(google/gemma-4-26b-a4b), повністю скриптовано, на відміну від
Claude-частини пайплайну (кроки 2-3, ручні через живу сесію). Це єдина
частина цього раунду, яку сторонній дослідник міг би відтворити самостійно
без Claude Code сесії -- за умови доступу до того самого чи еквівалентного
локального інференс-сервера.

Ендпоінт: http://172.28.224.1:1234 -- WSL2 gateway IP до Windows-хоста, НЕ
127.0.0.1 (окремий мережевий namespace всередині WSL2). Ця адреса -- дефолтний
маршрут WSL2 (`ip route show default`) і може змінюватись між перезапусками
WSL -- якщо виклики почнуть провалюватись з ConnectionError, перевір її
повторно, перш ніж підозрювати сам сервер.

Модель -- reasoning-модель: відповідь містить "reasoning_content" (з'їдає
частину max_tokens ДО "content"). Емпірично перевірено (не гіпотетично):
на реалістичному промпті taxonomy+пара max_tokens=1200 дає ~460 токенів
reasoning + ~55 токенів валідного JSON-контенту, finish_reason="stop" --
1200 достатньо в типовому випадку, але скрипт все одно ескалує на випадок
довшого reasoning для складніших пар.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

GEMMA_ENDPOINT = "http://172.28.224.1:1234/v1/chat/completions"
MODEL_NAME = "google/gemma-4-26b-a4b"
DEFAULT_MAX_TOKENS = 1200
MAX_TOKENS_ESCALATION = [1200, 2400, 4000]
REQUEST_TIMEOUT = 300
MAX_CONN_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

VALID_LABELS = {
    "type1_direct_negation",
    "type2_quantitative_conflict",
    "type3_causal_conflict",
    "type4_apparent_contextual",
    "not_contradiction",
}

SYSTEM_PROMPT = """You are annotating pairs of scientific hypotheses for conceptual contradiction, following this taxonomy:

- type1_direct_negation: Hypothesis A asserts X, hypothesis B asserts NOT-X under the same conditions and entities.
- type2_quantitative_conflict: divergent numeric estimates of the same effect/metric beyond error bars/confidence intervals.
- type3_causal_conflict: opposite direction of causal relationship between the same variables.
- type4_apparent_contextual: differences explained by dataset/architecture/hyperparameter/evaluation-condition differences; the hypotheses are actually compatible.
- not_contradiction: the pair does not genuinely conflict.

Respond with ONLY a JSON object, no markdown fencing, no extra text: {"label": "<one of the 5 labels exactly as written above>", "rationale": "<1-2 sentences>"}"""


def build_messages(row: pd.Series) -> list[dict]:
    user_content = (
        f"topic: {row['topic']}\n"
        f"hypothesis_a: {row['hypothesis_a']}\n"
        f"source_a: {row['source_a']}\n"
        f"hypothesis_b: {row['hypothesis_b']}\n"
        f"source_b: {row['source_b']}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def call_gemma(messages: list[dict], max_tokens: int, temperature: float) -> dict:
    payload = {
        "model": MODEL_NAME,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    last_exc: Exception | None = None
    for attempt in range(MAX_CONN_RETRIES):
        try:
            resp = requests.post(GEMMA_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"    connection issue ({e}), retrying in {wait:.0f}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Gemma endpoint unreachable after {MAX_CONN_RETRIES} attempts") from last_exc


def annotate_pair(row: pd.Series, temperature: float) -> dict:
    messages = build_messages(row)
    for max_tokens in MAX_TOKENS_ESCALATION:
        response = call_gemma(messages, max_tokens, temperature)
        choice = response["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        if content:
            return {
                "pair_id": row["pair_id"],
                "raw_response": response,
                "content": content,
                "finish_reason": finish_reason,
                "max_tokens_used": max_tokens,
            }
        if finish_reason != "length":
            # Порожній контент з іншої причини, ніж вичерпаний ліміт -- ескалація не допоможе.
            return {
                "pair_id": row["pair_id"],
                "raw_response": response,
                "content": "",
                "finish_reason": finish_reason,
                "max_tokens_used": max_tokens,
            }
        print(f"    empty content at max_tokens={max_tokens} (finish_reason=length), escalating...", flush=True)
    return {
        "pair_id": row["pair_id"],
        "raw_response": response,
        "content": "",
        "finish_reason": "length_exhausted",
        "max_tokens_used": MAX_TOKENS_ESCALATION[-1],
    }


def regex_fallback_label(text: str) -> str | None:
    """Рятує мітку з відповіді, обірваної на довжині (finish_reason=length)
    ще до закриття JSON -- rationale обрізаний, але поле label зазвичай
    іде першим і встигає завершитись лапкою. Emпірично виявлено round 9
    (2/30 пар): повний json.loads() провалювався, хоча мітка була ціла."""
    m = re.search(r'"label"\s*:\s*"(\w+)"', text)
    if m and m.group(1) in VALID_LABELS:
        return m.group(1)
    return None


def parse_label_response(content: str) -> tuple[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        label = regex_fallback_label(text)
        return (label, f"[truncated response, rationale lost] {content[:300]}") if label else ("PARSE_ERROR", content)
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        label = regex_fallback_label(text)
        return (label, f"[truncated response, rationale lost] {content[:300]}") if label else ("PARSE_ERROR", content)
    label = str(obj.get("label", "")).strip()
    rationale = str(obj.get("rationale", "")).strip()
    if label not in VALID_LABELS:
        return "PARSE_ERROR", content
    return label, rationale


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[rec["pair_id"]] = rec
    return cache


def append_to_cache(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=7)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Перевикликати вже кешовані pair_id.")
    parser.add_argument("--rebuild-csv-from-cache", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else RESULTS_DIR / f"pilot_round{args.round}_curated_sample.csv"
    df = pd.read_csv(input_path, dtype=str)

    cache_path = RESULTS_DIR / f"gemma_raw_responses_round{args.round}.jsonl"
    cache = load_cache(cache_path)

    if not args.rebuild_csv_from_cache:
        for i, (_, row) in enumerate(df.iterrows(), 1):
            if row["pair_id"] in cache and not args.force:
                continue
            print(f"[{i}/{len(df)}] annotating {row['pair_id']} ({row['topic']})...", flush=True)
            record = annotate_pair(row, args.temperature)
            record["timestamp"] = time.time()
            append_to_cache(cache_path, record)
            cache[row["pair_id"]] = record

    output_rows = []
    for _, row in df.iterrows():
        rec = cache.get(row["pair_id"])
        if rec is None:
            output_rows.append({"pair_id": row["pair_id"], "label": "MISSING", "rationale": "not yet annotated"})
            continue
        label, rationale = parse_label_response(rec["content"]) if rec["content"] else ("PARSE_ERROR", f"empty_response:{rec['finish_reason']}")
        output_rows.append({"pair_id": row["pair_id"], "label": label, "rationale": rationale})

    out_path = RESULTS_DIR / f"annotator_B_round{args.round}_labels.csv"
    pd.DataFrame(output_rows, columns=["pair_id", "label", "rationale"]).to_csv(out_path, index=False)

    n_parse_error = sum(1 for r in output_rows if r["label"] == "PARSE_ERROR")
    n_missing = sum(1 for r in output_rows if r["label"] == "MISSING")
    print(f"\nWrote {len(output_rows)} labels -> {out_path}", flush=True)
    print(f"PARSE_ERROR: {n_parse_error}, MISSING: {n_missing}", flush=True)


if __name__ == "__main__":
    main()
