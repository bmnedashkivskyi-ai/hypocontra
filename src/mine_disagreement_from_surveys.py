"""Крок 4.2 методології HypoContra, ТРЕТЯ ітерація добору кандидатів
(після round 1: топічна TF-IDF -- 93-98% не суперечність; round 2: точний
збіг Task-фрази -- 0/55 навіть тип 4 від обох анотаторів).

НОВА СТРАТЕГІЯ: замість автоматичної кластеризації довільних абстрактів
за темою/задачею -- пряме наближення до методології прототипу Alamri і
Stevenson (2016, §2.2 методології HypoContra): вони брали кандидатів З
ГОТОВИХ СИСТЕМАТИЧНИХ ОГЛЯДІВ, де людина-автор огляду вже встановила
компарабельність джерел. Тут аналог -- NLP-огляди (survey papers) з arXiv
cs.CL: завантажується LaTeX-джерело кожного огляду, і в повному тексті
шукаються речення з ЯВНИМ лінгвістичним сигналом контрасту/незгоди
("in contrast to", "unlike X", "contrary to", "however, X found", "while
X reported ... Y found" тощо) ПОРУЧ із \\cite-командою -- тобто випадки,
де автор огляду САМ явно протиставляє два процитовані джерела.

Це набагато сильніший (людина-огляду вже встановила компарабельність),
хоч і менш масштабований (потребує повного тексту, не лише абстрактів)
критерій компарабельності, ніж будь-яка автоматична кластеризація за
темою чи назвою задачі round 1/2.
"""
from __future__ import annotations

import re
import tarfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SRC_CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "survey_sources"
SRC_CACHE_DIR.mkdir(exist_ok=True)

# Обрано за очікуваною ймовірністю explicit disagreement-дискусій:
# вимірювальні/методологічні теми (bias, evaluation, human-AI decision
# making) частіше явно обговорюють суперечливі знахідки, ніж чисто
# архітектурні огляди.
SURVEY_ARXIV_IDS = {
    "1812.08951": "Analysis Methods in Neural Language Processing: A Survey",
    "2112.14168": "A Survey on Gender Bias in Natural Language Processing",
    "2112.11471": "Towards a Science of Human-AI Decision Making: A Survey of Empirical Studies",
    "2012.12305": "Confronting Abusive Language Online: A Survey from the Ethical and Human Rights Perspective",
    "1901.01122": "Machine Translation: A Literature Review",
    "1712.05191": "Relation Extraction: A Survey",
}

# Лінгвістичні маркери контрасту/незгоди -- englitude regex, case-insensitive
DISAGREEMENT_PATTERNS = [
    r"in contrast to",
    r"unlike \\?cite",
    r"contrary to",
    r"however,?\s+\\?cite",
    r"while \\?cite\w*\{[^}]+\}[^.]{0,150}\\?cite\w*\{[^}]+\}",
    r"disagree",
    r"conflicting (results|findings|evidence)",
    r"mixed (results|findings|evidence)",
    r"contradict",
    r"inconsistent (results|findings)",
    r"different from \\?cite",
    r"in disagreement with",
    # Round 16 (docs/next_scaling_directions.md candidate #1): DISAGREEMENT_PATTERNS
    # hadn't been revisited since round 3, and acts on the 3700+ surveys already
    # downloaded into results/survey_sources/ -- an independent lever from finding
    # new survey sources (all of which are now exhausted, see that doc). Risk
    # (documented there too): these markers don't all contain "contradict", so a
    # sentence can hit COMBINED_PATTERN via one of these without also hitting the
    # existing EXCLUSION_PATTERNS' §8 contradiction-detection-system guard unless
    # that sentence *also* separately contains a contradiction/detect phrase --
    # verified still true for the known false-sense categories (see the round-16
    # plan's Task 1 verification script), but a real run may surface new ones.
    r"on the other hand",
    r"conversely",
    r"runs counter to",
    r"at odds with",
    r"diverges from",
    r"challenges the claim that",
    r"counter to",
]
COMBINED_PATTERN = re.compile("|".join(DISAGREEMENT_PATTERNS), re.IGNORECASE)
CITE_PATTERN = re.compile(r"\\(?:cite|citep|citet|citealp)\w*\{([^}]+)\}")

# Виявлено емпірично в round 4 (docs/pilot_results_round4.md): третина
# найпродуктивніших збігів regex-маркерів контрасту виявились хибними за
# СЕНСОМ -- той самий лексичний маркер ("however", "disagree",
# "contradict") спрацьовує у ТРЬОХ різних значеннях: (а) суперечність
# наукових ГІПОТЕЗ між статтями -- потрібний нам сенс §3 таксономії;
# (б) суб'єктивна незгода ЛЮДЕЙ-АНОТАТОРІВ при розмітці даних;
# (в) внутрішня неузгодженість/самосуперечність ОДНІЄЇ моделі (self-
# consistency). Ці два виключено явним негативним фільтром.
EXCLUSION_PATTERNS = [
    r"inter-?rater",
    r"\brater[s]?\b",
    r"annotat\w+ disagree",
    r"disagree\w* (among|between) (annotators|raters|crowd|judges|human)",
    r"crowd-?sourc\w*",
    r"subjectiv\w* (task|label|annotation|assessment)",
    r"self-?contradict\w*",
    r"internal(ly)? (in)?consisten\w*",
    r"internal contradict\w*",
    r"contradicts? itself",
    r"contradicts? (its|their) own",
    r"own (response|output|generation|answer|thought|reasoning)s?",
    r"in (its|their) (own )?(thought|reasoning|chain.of.thought)",
    r"corrects? itself",
    r"do(es)? not contradict earlier",
    r"single-agent baseline",
    # Третій сенс шуму (round 5, docs/pilot_results_round5.md): огляд ПРО
    # системи/методи ДЕТЕКЦІЇ суперечності як дослідницьку задачу --
    # "contradiction"/"contradictory" тут ОБ'ЄКТ дослідження цитованої
    # роботи (вона будує детектор), а не ознака того, що сама робота
    # суперечить іншій. Синтаксична евристика (не справжня семантика,
    # але наближення): дієслово вивчення/побудови методу + фраза про
    # (performance|ability|behavio(u)r) ПОРУЧ із detect/identify/handle
    # + contradict, або явне "contradiction detection" як досліджувана
    # здатність/задача.
    r"(performance|ability|behaviou?r) (of|in)[^.]{0,60}(detect|identif|encounter|handl|address|mitigat|resolv)\w*[^.]{0,30}contradict",
    r"(develop|propose|present|design|introduce|build)[^.]{0,60}(model|method|approach|framework|network|classifier|system)[^.]{0,80}(detect|identif)\w*[^.]{0,20}contradict",
    r"contradiction detection\b",
    r"contradiction (ability|task|capability)",
    r"in detecting the contradiction",
    r"in identifying contradictory",
]
EXCLUSION_PATTERN = re.compile("|".join(EXCLUSION_PATTERNS), re.IGNORECASE)

# Технічний (не семантичний) фільтр: LaTeX-таблиці/рисунки, які груба
# сентенс-токенізація за крапками помилково розбиває на "речення" --
# ніколи не були прозовим текстом автора, отже не можуть містити
# genuine contrastive claim.
ARTIFACT_PATTERN = re.compile(
    r"\\begin\{(table|figure|tikz)|\\midrule|\\tikzstyle|\\multirow|\\multicolumn",
    re.IGNORECASE,
)


def download_and_extract(arxiv_id: str) -> Path | None:
    target_dir = SRC_CACHE_DIR / arxiv_id
    if target_dir.exists():
        return target_dir
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    tmp_file = SRC_CACHE_DIR / f"{arxiv_id}.tar.gz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hypocontra-pilot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            tmp_file.write_bytes(resp.read())
        target_dir.mkdir(exist_ok=True)
        with tarfile.open(tmp_file) as tar:
            tar.extractall(target_dir, filter="data")
        return target_dir
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED to download/extract {arxiv_id}: {e}")
        return None


def find_disagreement_sentences(tex_dir: Path) -> list[dict]:
    hits = []
    for tex_file in tex_dir.rglob("*.tex"):
        try:
            text = tex_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        # Розбиваємо на псевдо-речення за крапкою (наближено -- LaTeX не
        # завжди зручний для точної токенізації речень)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            if (COMBINED_PATTERN.search(sent) and not EXCLUSION_PATTERN.search(sent)
                    and not ARTIFACT_PATTERN.search(sent)):
                cites = CITE_PATTERN.findall(sent)
                if cites:  # тільки речення, де є цитування -- потрібні пари джерел
                    clean_sent = re.sub(r"\s+", " ", sent).strip()[:500]
                    hits.append({
                        "file": tex_file.name,
                        "sentence": clean_sent,
                        "cited_keys": ";".join(cites),
                    })
    return hits


def main(survey_dict: dict[str, str] | None = None, out_suffix: str = ""):
    surveys = survey_dict if survey_dict is not None else SURVEY_ARXIV_IDS
    all_hits = []
    n_failed = 0
    for i, (arxiv_id, title) in enumerate(surveys.items(), 1):
        print(f"[{i}/{len(surveys)}] === {arxiv_id}: {title} ===", flush=True)
        tex_dir = download_and_extract(arxiv_id)
        if tex_dir is None:
            n_failed += 1
            continue
        hits = find_disagreement_sentences(tex_dir)
        print(f"  {len(hits)} candidate disagreement sentences found")
        for h in hits:
            h["survey_arxiv_id"] = arxiv_id
            h["survey_title"] = title
        all_hits.extend(hits)
        time.sleep(2)

    df = pd.DataFrame(all_hits)
    out_name = f"survey_disagreement_sentences{out_suffix}.csv"
    df.to_csv(RESULTS_DIR / out_name, index=False)
    print(f"\nTotal disagreement-signal sentences across {len(surveys)} surveys "
          f"({n_failed} failed downloads): {len(df)}")
    if len(df) > 0:
        print(df["survey_arxiv_id"].value_counts().to_string())


if __name__ == "__main__":
    main()
