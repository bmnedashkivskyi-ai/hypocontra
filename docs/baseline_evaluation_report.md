# HypoContra baseline NLI-model evaluation report

Спираючись на `docs/superpowers/specs/2026-09-10-baseline-nli-evaluation-design.md` (стаття §5, пункт "в"). Ground truth: n=59 round-7+ пар, де Claude (Annotator A) і Gemma (Annotator B) погодились (`label_A == label_B`).

Міжанотаторська згода (Cohen's κ), n=85 (round 7-17, усі валідні мітки обох анотаторів): 3-way κ=0.607 — ВИЩЕ порогу §4.4 статті (0.6); 5-way κ=0.556 (вже задокументовано раніше) — нижче порогу. Згортання до 3 класів відновлює міжанотаторську згоду на рівні порогу.

**Важливо: наведені нижче три числа обчислені на РІЗНИХ популяціях і не є прямо порівнюваними як три оцінки на одному й тому самому тестовому наборі.** Rule-based (0.310) і RoBERTa (0.575) обчислені на n=59 (agreement-only ground truth, Decision 1); LLM few-shot (0.690) обчислений на ширшому й іншому n=85 (усі валідні мітки round 7+, Decision 4 — щоб уникнути циркулярності, оскільки Gemma сама є одним з анотаторів корпусу).

## §5.3 (крос-доменна генералізація) виконано окремо

Fine-tune на біомедичному корпусі Alamri & Stevenson (2016), zero-shot оцінка на HypoContra — виконано в окремому Bounded-циклі, не в межах цього плану. Результат: zero-shot transfer macro F1=0.3776 (проти 0.575 in-domain RoBERTa) — див. `docs/generalization_report.md`.

## 1. Rule-based baseline (нижня межа складності)

### 3-way (основна метрика)

| Клас | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Apparent | 0.000 | 0.000 | 0.000 | 9 |
| Contradiction | 0.692 | 0.281 | 0.400 | 32 |
| NotContradiction | 0.370 | 0.944 | 0.531 | 18 |
| **MACRO AVG** | **0.354** | **0.409** | **0.310** | — |

### 5-way (додатково)

| Клас | Precision | Recall | F1 | n |
|---|---|---|---|---|
| not_contradiction | n/a | n/a | n/a | n/a |
| type1_direct_negation | n/a | n/a | n/a | n/a |
| type2_quantitative_conflict | n/a | n/a | n/a | n/a |
| type3_causal_conflict | n/a | n/a | n/a | n/a |
| type4_apparent_contextual | n/a | n/a | n/a | n/a |
| **MACRO AVG** | n/a | n/a | n/a | n/a |

5-класова задача не виконувалась для цієї baseline (design decision 3) — не плутати з n/a через нульову підтримку класу.

## 2. Fine-tuned RoBERTa (roberta-large-mnli, 5-fold stratified CV, лише 3-way)

Модель стартувала з уже натренованої MNLI-класифікаційної голови, а не зі свіжої: `roberta-large-mnli` вже має рівно 3 вихідні мітки (CONTRADICTION/NEUTRAL/ENTAILMENT), тому виклик `AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)` є no-op для розміру голови й зберігає передтреновані MNLI-ваги замість переініціалізації з нуля (`ignore_mismatched_sizes=True` доданий захисно на випадок, якщо кількість міток колись відрізнятиметься, а не тому що це реально сталося тут). Це дає асиметричний "теплий старт": CONTRADICTION/NEUTRAL приблизно узгоджуються з 2 із 3 класів HypoContra (CONTRADICTION↔Contradiction, NEUTRAL↔Apparent — §3 таксономії сама називає type4 "Neutral*"), але ENTAILMENT не є тим самим поняттям, що й "NotContradiction", тож третій клас узгоджений гірше. Це легітимний вибір baseline (§5.1 explicitly хоче MNLI-pretrained RoBERTa), але по-класові цифри варто читати з урахуванням цього асиметричного старту, а не як симетричну baseline зі свіжою головою.

Цей прогін НЕ був повністю seeded: детермінованим було лише розбиття на фолди (`StratifiedKFold(random_state=42)`), тоді як перемішування батчів у циклі тренування (`torch.randperm`, щоепохи) не мало власного сіда і ніде не викликався `torch.manual_seed()`. Тому наведені нижче цифри (macro F1=0.575) відображають один конкретний unseeded прогін, а не гарантовано відтворюваний результат; `torch.manual_seed(42)` додано до `src/baseline_finetune_roberta.py` для майбутніх прогонів (цей прогін не перезапускався і наведені цифри не змінились).

### 3-way (основна метрика)

| Клас | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Apparent | 0.375 | 0.333 | 0.353 | 9 |
| Contradiction | 0.722 | 0.812 | 0.765 | 32 |
| NotContradiction | 0.667 | 0.556 | 0.606 | 18 |
| **MACRO AVG** | **0.588** | **0.567** | **0.575** | — |

Фінальний train loss по фолдах:

| Fold | Final train loss |
|---|---|
| 0 | 0.0606 |
| 1 | 0.0700 |
| 2 | 0.1124 |
| 3 | 0.0760 |
| 4 | 0.1046 |

Усі 5 фолдів збігались (converged) монотонно: початковий train loss у діапазоні 1.18-1.47 спадав до фінального loss у діапазоні 0.06-0.11, без розбіжності (divergence) чи NaN у жодному фолді.

### 5-way (додатково)

| Клас | Precision | Recall | F1 | n |
|---|---|---|---|---|
| not_contradiction | n/a | n/a | n/a | n/a |
| type1_direct_negation | n/a | n/a | n/a | n/a |
| type2_quantitative_conflict | n/a | n/a | n/a | n/a |
| type3_causal_conflict | n/a | n/a | n/a | n/a |
| type4_apparent_contextual | n/a | n/a | n/a | n/a |
| **MACRO AVG** | n/a | n/a | n/a | n/a |

5-класова задача не виконувалась для цієї baseline (design decision 3) — не плутати з n/a через нульову підтримку класу.

## 3. LLM few-shot (Gemma) — методологічне застереження: НЕ незалежна оцінка

Ця baseline повторно використовує вже зібрані мітки Gemma (Annotator B) проти Claude як референсу, на всіх n=85 round-7+ парах (не на n=59 ground-truth наборі — щоб уникнути циркулярності, див. `results/baseline_llm_fewshot_metrics_NOTE.md`). Ці цифри вимірюють узгодженість із судженням Claude, не незалежну точність.

### 3-way (основна метрика)

| Клас | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Apparent | 0.900 | 0.360 | 0.514 | 25 |
| Contradiction | 1.000 | 0.857 | 0.923 | 42 |
| NotContradiction | 0.462 | 1.000 | 0.632 | 18 |
| **MACRO AVG** | **0.787** | **0.739** | **0.690** | — |

### 5-way (додатково — реальні обчислені цифри, на відміну від інших двох baseline)

| Клас | Precision | Recall | F1 | n |
|---|---|---|---|---|
| not_contradiction | 0.462 | 1.000 | 0.632 | 18 |
| type1_direct_negation | 0.889 | 0.889 | 0.889 | 36 |
| type2_quantitative_conflict | 0.000 | 0.000 | 0.000 | 1 |
| type3_causal_conflict | 0.000 | 0.000 | 0.000 | 5 |
| type4_apparent_contextual | 0.900 | 0.360 | 0.514 | 25 |
| **MACRO AVG** | **0.450** | **0.450** | **0.407** | — |

type2 (n=1) і type3 (n=5) мають надто малу підтримку для змістовної по-класової оцінки: їхній F1=0.000 — це справжній результат (Gemma не передбачила жодного прикладу цих класів), а не помилка чи сфабрикований нуль, і саме він тягне вниз macro-середнє (0.407) — це варто трактувати як шум малого n, а не як провал моделі.