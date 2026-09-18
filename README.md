# HypoContra: пілотна фаза + масштабування round 7-18 + baseline-оцінка (завершено)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22744801.svg)](https://doi.org/10.5281/zenodo.22744801)

Перше реальне виконання методології `~/test-article/02-HypoContra-taksonomiia-korpus-superechnostei/hypocontra-taksonomiia-korpus-superechnostei.md` (§4) — таксономії й методології побудови корпусу для виявлення концептуальних суперечностей між науковими гіпотезами в NLP-літературі. Оригінальна стаття — methodology-only ("на момент написання корпус ще не зібрано"); цей проєкт — перша реальна реалізація.

**Підсумковий звіт пілоту: `docs/pilot_final_summary.md`** — почніть звідси для round 1-6. Round 7-17 (масштабування, завершено) описані нижче.

## Головний результат

Шість раундів пілоту випробували два структурно різні методи добору кандидатних пар гіпотез:

- **Автоматична кластеризація** (топічна TF-IDF чи точний збіг назви задачі, раунди 1-2, n=97 пар): **1/97 (1%)** пар — реальна контрадикція.
- **Курована стратегія через огляди-джерела незгоди** (видобування явних маркерів контрасту з повного тексту NLP-оглядів, раунди 3-6, n=17 пар): **16/17 (94%)** пар — обидва незалежні AI-анотатори визнали реальною контрадикцією, включно з кількома добре відомими реальними науковими суперечками (Gonen&Goldberg vs Bolukbasi; Serrano&Smith vs Wiegreffe&Pinter).

**Критичне обмеження пілоту:** методологія §4.4 специфікує розмітку двома НЕЗАЛЕЖНИМИ ЛЮДЬМИ з Cohen's κ. Людські анотатори недоступні в цьому пілоті — замінено на два незалежні AI-проходи (координуюча сесія + окремий ізольований subagent). Це AI-AI згода, принципово інша (слабша) заявка про валідність, ніж людська.

## Round 7-17: масштабування (5-10 вересня 2026, завершено)

Куровану стратегію автоматизовано й наростили з 17 до **103 пар**:

- **Резолюція цитат офлайн** (`src/resolve_survey_citations.py`) — `\cite`-ключі з речень-кандидатів розкриваються проти вже завантажених `.bbl`/`.bib`, без нових мережевих запитів.
- **Побудова пар** — Claude (жива сесія, без API-ключа) читає речення+контекст і формулює `hypothesis_a`/`hypothesis_b`, або позначає `skipped`.
- **Анотатор B змінено**: замість другого Claude-проходу — локальна модель **Gemma** (`google/gemma-4-26b-a4b`, OpenAI-сумісний ендпоінт), повністю скриптована анотація (`src/annotate_with_gemma.py`). Сильніша заявка про незалежність, ніж AI-AI тієї самої моделі, хоч і досі не людська.
- **Сліпа розмітка Анотатора A** — окремий прохід Claude, що бачить лише готову пару (без контексту побудови), для часткової незалежності від власної побудови пари.
- Джерело кандидатів пройшло три хвилі розширення в межах `src/find_surveys_bulk.py` (`--query-mode`): `survey` (cs.CL "survey"/"systematic review"/"literature review", 1700 оглядів, вичерпано), `overview` (cs.CL "overview of"/"review of", 1319 оглядів, вичерпано), `survey_broad` (cs.CL+cs.LG+cs.AI з фільтром NLP-релевантності, 928 оглядів).

**Результат по раундах:** round 7 (4 пари, κ=0.600), round 8 (1 пара), round 9 (30 пар, κ=0.592), round 10 (17 пар, κ=0.562), round 11 (15 пар, κ=0.513), round 12 (3 пари, κ=0.000 — n замалий для сенсу), round 13 (1 пара, κ вироджений — n=1), round 14 (1 пара, κ вироджений — n=1, анотатори РОЗІЙШЛИСЯ: type3_causal_conflict vs type4_apparent_contextual, обидва підтвердили якусь контрадикцію, деталі нижче), round 15 (0 пар — title-синоніми "a review"/"primer on"/"tutorial on" вичерпано, напрямок закрито), round 16 (11 пар, κ=0.389 на n=11 — розширення DISAGREEMENT_PATTERNS на 7 нових маркерів контрасту, найнижча κ серед раундів: більшість розбіжностей анотаторів — на межі `type4_apparent_contextual` vs `not_contradiction`, очікувано для широких/слабких маркерів типу "on the other hand"), round 16.5/17 (citation-graph розширення, 3 пари, κ=0.000 на n=3 — n замалий для сенсу; 2/3 анотатори зійшлись на `not_contradiction`, третю Claude визначив як `type4_apparent_contextual`, Gemma — `not_contradiction`). Об'єднано round 7-17: **κ=0.556 на n=85** — нижче порогу §4.4 (0.6), майже незмінено відносно round 7-16 (n=3 замало, щоб суттєво зсунути об'єднану κ). Разом із раундами 3-6 — **103 куровані пари**.

**Масштабування зупинено свідомо після round 12**, не через технічне вичерпання: розширення на cs.LG/cs.AI (query-mode `survey_broad`) дало 928 нових оглядів, але лише **3 з 180 придатних кандидатів** (23 побудовані пари, 20 відкинуто) виявились справді про NLP — решта про медичну сегментацію зображень, GAN-стеганографію, стиснення нейромереж, теорію пам'яті гіпокампу тощо. Фільтр "NLP-термін десь в анотації" виявився недостатнім — предметна категорія (cs.CL) значно надійніший індикатор релевантності, ніж лексичний збіг. Голе `ti:review` (без "of") у cs.CL також перевірено й відкинуто — 1136 результатів, майже суцільно "peer review"/"product review", не огляди літератури.

**Round 13: ACL Anthology full-text перевірено (найпріоритетніший напрямок з `docs/next_scaling_directions.md`) — низька гранична віддача для arXiv-перетину.** З 606 ACL-заголовків, що виглядають як огляди (title-фільтр по `anthology.bib`), лише **23 мають прямий відповідник на arXiv** (title-query до arXiv API) — решта **284** доступні лише як ACL PDF, без LaTeX-джерела, і на момент round 13 вимагали б нового PDF-парсингу. З 23 arXiv-перетинних оглядів `mine_disagreement_from_surveys.py` видобув 9 речень-кандидатів із маркерами контрасту; офлайн-резолюція цитат відкинула 2 (лише один `\cite`-ключ, нема з чим протиставляти); з решти 7 Claude визнав 5 хибними спрацюваннями, 1 дублікатом уже наявної реальної суперечки (`duplicate_of_r9_003_same_survey_same_dispute` — той самий диспут, інша цитуюча стаття), і лише 1 — новою парою. **Висновок: ACL-огляди, що ВЖЕ індексовані на arXiv, здебільшого вже покриті cs.CL title-пошуком round 9-12 (звідси й дублікат); реальний невикористаний резерв на момент round 13 — 284 ACL-only огляди, що вимагали нового PDF-based кроку пайплайна, а не розширення в межах наявного LaTeX-пайплана. Цей крок реалізовано в round 14, див. нижче.**

**Round 14: PDF-based пайплайн для 284 ACL-only оглядів реалізовано й запущено.** Оскільки для цих оглядів немає LaTeX-джерела (лише PDF) і немає `.bbl`/`.bib`, побудовано окремий PDF-аналог `resolve_survey_citations.py`-пайплайна: екстракція тексту з PDF (`src/acl_pdf_text.py`), regex-парсинг author-year цитат у реченні й власного списку літератури (`src/acl_ref_parsing.py`), OpenAlex/Crossref lookup абстрактів за назвою з кешем (`src/bib_lookup.py`), масове завантаження PDF (`src/fetch_acl_pdfs.py`), видобування речень-кандидатів (`src/mine_disagreement_from_acl_pdfs.py`) і резолюція цитат без `.bbl`/`.bib` (`src/resolve_acl_pdf_citations.py`). Повний прогін: із 284 заголовків успішно завантажено **272 PDF** (12 не вдалося — HTTP 404, усі до 2013 року); з 272 PDF видобуто **24 речення-кандидати** з маркерами контрасту; офлайн-резолюція цитат залишила **14 придатних** (10 відкинуто: 4 `too_many_citations`, 4 `citation_unmatched`, 2 `citation_ambiguous`). Ці 14 речень видано як round-14 batch (`results/claude_pair_batch_round14.jsonl`) для побудови пар.

**Побудова пар (2026-09-09):** з 14 речень-кандидатів лише **1 виявилось генуїнним протиставленням** (`r14_000`, "position embeddings" як причина anisotropy-outlier dimension у mBERT vs monolingual моделі) — решта 13 відхилено як `not_genuine_contrast`: цитовані роботи в кожному з цих речень підтримують ОДНЕ й ТЕ САМЕ твердження спільно (агрегація), а не протиставлені одна одній (типовий хибний сенс маркерів контрасту "however"/"contradicts", вже документований у round 3-12). Анотатори РОЗІЙШЛИСЯ щодо єдиної пари: Claude (Анотатор A, сліпо) визначив `type3_causal_conflict`, Gemma (Анотатор B) — `type4_apparent_contextual`; обидва підтвердили якусь контрадикцію (`both_flag_contradiction=True`), розбіжність лише в підтипі — прикордонний випадок між "причинний конфлікт" і "відмінність через контекст/архітектуру", вартий згадки як приклад складності таксономії §3 на межевих кейсах.

**Знайдено й виправлено 3 реальні баги в процесі:** float-коерція `survey_arxiv_id` (втрата кінцевого нуля в ID на кшталт `2108.04840`→`2108.0484`), небезпечний роздільник `;` для об'єднання полів кількох цитувань в одному CSV-осередку (сам бібліографічний текст часто містить `;`, що зсувало прив'язку контексту до цитат — замінено на `\x1f`), обірвана JSON-відповідь Gemma без фолбеку на регулярний вираз. Також помічено: локальний Gemma-сервер періодично сповільнювався до тайм-ауту навіть на 180с — `REQUEST_TIMEOUT` зрештою піднято до 300с.

**Round 15, напрямок 1 (title-синоніми в cs.CL) — 0 нових пар.** Перевірено `docs/next_scaling_directions.md`, п.1: `ti:"a review"`/`ti:"primer on"`/`ti:"tutorial on"` (totalResults=1191, перевірено запитом ДО завантаження). Після виключення 2751 уже відомих ID — 74 генуїнно нових кандидати; **15 вручну відхилено** як хибний сенс "review"/"tutorial" (peer-review-процес наукових статей і user-generated review-як-дані для sentiment-аналізу — той самий клас проблеми, що round 12 задокументував для голого `ti:review`: терміни надто перевантажені в NLP-домені). З 59 огляджів, що лишились, майнінг видобув 20 речень-кандидатів (6 завантажень не вдалися — старі/пошкоджені arXiv ID); офлайн-резолюція цитат залишила 7 придатних (9 відкинуто `single_cite_key`, 1 `likely_latex_artifact`). **Побудова пар: 0 з 7** — усі відхилені як `not_genuine_contrast` (списки взаємодоповнюючих чи узгоджених робіт, не протиставлення). Напрямок вичерпано без додавання пар; перехід до п.2 (`docs/next_scaling_directions.md`).

**Round 16 (розширення DISAGREEMENT_PATTERNS, `docs/next_scaling_directions.md` п.1) — 11 нових пар.** Додано 7 нових маркерів контрасту (`"on the other hand"`, `"conversely"`, `"runs counter to"`, `"at odds with"`, `"diverges from"`, `"challenges the claim that"`, `"counter to"`) до `DISAGREEMENT_PATTERNS` у `src/mine_disagreement_from_surveys.py`; повторно проскановано всі **3720** уже завантажених survey-джерел (`src/rescan_cached_surveys.py`, без нових мережевих запитів) — **4063** сумарних спрацювань, **2107** уже бачених старими патернами, **1956** генуїнно нових (`src/filter_new_candidates.py`). Офлайн-резолюція цитат: 94 точні дублікати відкинуто, з решти 1862 — **609 придатних** (1253 відкинуто: 1170 `single_cite_key`, 42 `all_keys_unresolved`, 34 `likely_latex_artifact`, 7 `insufficient_resolved_lt_2`). Побудова пар виконана паралельно 16 subagent'ами (~40 речень кожен, superpowers dispatching-parallel-agents) — **13 пар** побудовано; координуюча сесія на етапі зведення відхилила 2: точний внутрішньораундовий дублікат (`s1115` -> `exact_duplicate_within_round16` дублікату `s1113`, той самий s-34/ehr-58 диспут двічі в одному огляді) і слабке "нова модель перевершує базову" обрамлення без справжньої контр-гіпотези з боку B (`s1117`). **Підсумок: 11 нових пар**, κ=0.389 на n=11 для самого раунду (найнижча серед раундів — очікувано для широких/слабких маркерів; більшість розбіжностей на межі `type4_apparent_contextual` vs `not_contradiction`). Перевірено на дублікат відомого диспуту Kaplan et al. 2020 vs Hoffmann et al. 2022 (масштабувальні закони LLM) проти вже наявної пари round 10 (`r10_002`, Kaplan vs Hernandez — інший диспут, про повторення даних) — підтверджено НЕ дублікат, обидві пари лишились окремими.

**Round 17 (citation-graph розширення, `docs/next_scaling_directions.md` п.1) — 3 нові пари.** Витягнуто з `.bbl`/`.bib` уже завантажених оглядів **18172** кандидатних заголовки цитованих робіт, чиї власні назви виглядають як огляд (`src/extract_cited_survey_titles.py`, offline, без нових мережевих запитів). Через різко нерівномірний розподіл (медіана n_citing_surveys=1) і вартість перевірки проти arXiv (~5с/назву, повний прогін коштував би ~25 годин) оброблено лише топ-733 (поріг ≥20 цитуючих оглядів, рішення користувача) — **249** співпали з реальною статтею на arXiv, **22** виявились генуїнно новими для корпусу. Завантажено й змайнено 21/22 (1 невдача — arXiv віддав PDF замість source tarball) → **44** речення-кандидати; резолюція цитат відсіяла 3 точні дублікати → 41 → **19 придатних**. Побудова пар (крок 2, виконана напряму координуючою Claude-сесією без паралельних subagent'ів — обсяг малий): **3 нові пари** з 19 (16 відхилено як `not_genuine_contrast`/`insufficient_context`). κ=0.000 на n=3 (n замалий для сенсу, як і round 12/13). Фінальний рев'ю гілки виявив і задокументував: ключ дедуплікації заголовків у Task 1 не знімає LaTeX-дужки й кінцеву пунктуацію, тож поріг "≥20" наближений, не точний — 79 робіт із реальним сумарним лічильником ≥20 випали з обробленого топ-733 (справжній топ-рівень був би ~812); ці 79 лишаються серед ~17400 необроблених кандидатів для майбутнього раунду (деталі — `docs/next_scaling_directions.md`).

**Round 18 (продовження citation-graph розширення, `docs/next_scaling_directions.md` п.1) — 0 нових пар, напрямок закрито.** Довів обробку решти кандидатних заголовків round 17 (`n_citing_surveys` < 20) проти arXiv: **13521/18172** (арешт через персистентний блок arXiv HTTP 406 після трьох спроб на різних інтервалах очікування й швидкостях запиту — 4651 лишились необробленими, деталі `docs/round18_candidate_analysis.md`) → **1661** генуїнно нових огляди → **2060** сирих речень-кандидатів. Офлайн-резолюція цитат дала здорову конверсію **654/1942 (33.7%)** — порівнянно з round 16 — але аналіз даних показав: **92.2% (364/395)** оглядів, що дали придатні речення, виходять за межі заявленого §4.1 джерела даних (arXiv cs.CL) — комп'ютерний зір, RL, федеративне навчання, безпека ML, і навіть два буквально astro-ph огляди (рентгенівські каталоги скупчень галактик), підхоплені лише через слово "survey" в назві. Той самий доменний дрейф, що round 12 вже задокументував для прямого cs.LG/cs.AI-розширення (3/180 придатних), відтворився іншим шляхом — через citation graph, а не query-розширення. NLP-релевантний title-фільтр (ручний keep-list) звузив 654→**19**. Побудова пар на всіх 19: **0 пар** (13 `not_genuine_contrast`, 6 `insufficient_context`) — і не через доменний дрейф чи хибу фільтра (обидва спрацювали коректно), а через саму структуру речень citation-graph-джерела (перелік прикладів через контрастний маркер, не пряме протиставлення двох тверджень автором огляду). Разом із round 17 (3/19) — **3 пари з 38 виданих речень за весь напрямок**; `docs/next_scaling_directions.md` тепер явно рекомендує не продовжувати citation-graph розширення заради обсягу корпусу. Корпус лишається на **103 куррованих парах** (без змін від round 17). Побічний результат round 18: виправлено баг втрати категорійного префіксу для старостильних arXiv ID (`cs/0504061` → `0504061`) і посилено retry/backoff-логіку `find_surveys_bulk.py` проти персистентних блоків.

**Обмеження відтворюваності:** Gemma-частина повністю скриптована й відтворювана третьою особою (за наявності еквівалентного локального інференс-сервера). Claude-частина (побудова пар, сліпа розмітка A) — жива сесія без API-ключа, не персистентний скрипт.

## Baseline NLI-оцінка (§5 статті, п. "в" §9) — завершено 2026-09-10/11

Перше виконання протоколу §5: три базові моделі навчено й оцінено на курованому корпусі (103 пари). Повний звіт: `docs/baseline_evaluation_report.md`.

Оскільки об'єднана 5-way κ=0.556 (n=85) нижче порогу §4.4 — чистого gold standard немає. Розв'язано через дві різні популяції: ground truth для rule-based і RoBERTa — n=59 підмножина, де Claude (A) і Gemma (B) ПОГОДИЛИСЬ (`results/baseline_ground_truth_agreement.csv`); LLM few-shot оцінено на ширшому n=85 (усі валідні мітки round 7+), явно оцінюючи власні мітки Gemma (B) проти Claude (A) — це циркулярно (Gemma вже анотатор корпусу), і явно задокументовано, а не приховано, в `results/baseline_llm_fewshot_metrics_NOTE.md`.

**Результати (macro F1, 3-way — Contradiction/Apparent/NotContradiction):**

- **Rule-based** (заперечення/антонімія, лексичні патерни): **0.310** (n=59, `src/baseline_rule_based.py`)
- **Fine-tuned RoBERTa** (`roberta-large-mnli`, 5-fold stratified CV): **0.575** (n=59, `src/baseline_finetune_roberta.py`). Застереження: класифікаційна голова НЕ була переініціалізована — `roberta-large-mnli` вже має рівно 3 мітки, тож це MNLI warm-start, а не "свіжа голова", як спочатку планувалось (перевірено через `torch.equal`).
- **LLM few-shot** (Gemma, циркулярність задокументована): **0.690** (n=85, `src/baseline_llm_fewshot_report.py`, без нових викликів Gemma — переюзано вже наявні мітки)

Додатково обчислено: **3-way κ=0.607 (n=59)** — ВИЩЕ порогу §4.4 (0.6), проти вже відомого 5-way κ=0.556 (n=85) — нижче порогу. Згортання до 3 класів матеріально покращує міжанотаторську згоду.

### §5.3 Крос-доменна генералізація — завершено окремим Bounded-циклом, 2026-09-11

Fine-tune на біомедичному корпусі Alamri & Stevenson (2016, публічний, CC BY-NC-SA 2.0 UK, `https://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/`), zero-shot оцінка на HypoContra — без донавчання на самому HypoContra. Побудовано 1775 комбінаторних пар з бінарної YS/NO-схеми джерела (728 Contradiction / 1047 NotContradiction, `src/build_alamri_stevenson_training_pairs.py`); `roberta-large-mnli` доналаштовано на 1597 з них (dev macro F1=0.988 in-domain), потім оцінено zero-shot на n=59 HypoContra (`src/baseline_generalization_alamri_stevenson.py`).

**Zero-shot transfer macro F1 = 0.3776** — помітно нижче за in-domain RoBERTa (0.575). Перше емпіричне свідчення того, що детектори суперечностей погано переносяться між доменами (передбачений статтею ризик, §6 "Gap 4"). `Apparent`-recall точно 0 — очікувано: бінарна YS/NO-схема джерела структурно не має еквівалента цього класу, детектори не бачили жодного прикладу під час навчання. Повний звіт: `docs/generalization_report.md`.

### §5.3 друга крос-доменна точка — NLI4CT, завершено 2026-09-13

За підсумками `docs/s53_alternative_generalization_corpora.md` (research напряму розвитку) реалізовано другу, незалежну crossdomain-перевірку тим самим протоколом: NLI4CT (SemEval-2023 Task 7 / SemEval-2024 Task 2, публічний, `github.com/ai-systems/nli4ct`) — 1900 пар (train 1700 / dev 200, `src/build_nli4ct_training_pairs.py`), `hypothesis_a`=Statement, `hypothesis_b`=evidence-речення з Clinical Trial Report (`src/baseline_generalization_nli4ct.py`).

**Zero-shot transfer macro F1 = 0.2643** — нижче за Alamri&Stevenson (0.3776). Діагностовано через `superpowers:systematic-debugging` (не прийнято на віру): перший прогін (max_length=256, як в Alamri&Stevenson-скрипті) звалився у constant-prediction — модель передбачала `Contradiction` для всіх 59 пар HypoContra, dev macro F1 застряг на 0.3333. Перевірено й відкинуто truncation як єдину причину (hypothesis_b тут — медіана 178 токенів, до 1954, 38% пар довші за 256 токенів проти 0% в Alamri&Stevenson; підняття до max_length=512 саме по собі колапс НЕ усунуло в контрольному прогоні на повних даних) — справжня причина глибша: `train_loss` на епосі 0 стартує біля `ln(2)=0.693` (майже випадковий рівень) для NLI4CT проти 0.26 для Alamri&Stevenson — CT-report evidence-текст (списки, заголовки секцій) значно далі від pretrained MNLI-розподілу, ніж природні claim-речення. max_length=512 залишено (виправдана окрема правка — уникнення втрати інформації), офіційний прогін із цим фіксом зловив кращий чекпоінт на епосі 1 (dev F1=0.4715) через уже наявний best-epoch checkpointing. **Тренування нестабільне** — повторний прогін тим самим кодом/сідом не гарантовано відтворить саме це число через GPU-недетермінізм; сама нестабільність (на відміну від надійної збіжності Alamri&Stevenson, dev F1 0.96-0.99) — частина знахідки. Повний звіт із деталями діагностики: `docs/generalization_report_nli4ct.md`.

## Структура

- `src/collect_abstracts.py` — збір абстрактів arXiv cs.CL (round 1-2)
- `src/extract_candidate_pairs.py` — TF-IDF топічна кластеризація (round 1)
- `src/extract_candidate_pairs_strict.py` — точний збір Task-фрази + метричний фільтр (round 2)
- `src/find_more_surveys.py` — цілеспрямований пошук NLP-оглядів arXiv (round 4)
- `src/mine_disagreement_from_surveys.py` — видобування маркерів контрасту з повного тексту оглядів + ітеративна фільтрація трьох сенсів шуму (round 3-13)
- `src/find_surveys_bulk.py` — широкий arXiv-пошук з пагінацією, `--query-mode {survey,overview,survey_broad}` (round 9-12)
- `src/check_acl_arxiv_overlap.py` — перевірка ACL Anthology-заголовків проти arXiv title-query (round 13)
- `src/acl_pdf_text.py` — PDF-екстракція тексту + локалізація секції референсів (round 14)
- `src/acl_ref_parsing.py` — regex-парсинг author-year цитат у реченні + власного списку літератури з PDF (round 14)
- `src/bib_lookup.py` — OpenAlex/Crossref lookup abstract за назвою, з кешем (round 14)
- `src/fetch_acl_pdfs.py` — масове завантаження PDF для 284 ACL-only оглядів (round 14)
- `src/mine_disagreement_from_acl_pdfs.py` — видобування маркерів контрасту з PDF-тексту (round 14)
- `src/resolve_acl_pdf_citations.py` — резолюція цитат без .bbl/.bib + abstract lookup (round 14)
- `src/resolve_survey_citations.py` — офлайн-резолюція `\cite`-ключів проти `.bbl`/`.bib` (round 7-13)
- `src/prepare_claude_pair_batch.py` / `src/ingest_claude_pairs.py` — емісія батчу для побудови пар Claude-сесією + валідація результату (round 7-14)
- `src/prepare_claude_label_batch.py` / `src/ingest_claude_labels.py` — те саме для сліпої розмітки Анотатора A (round 7-14)
- `src/annotate_with_gemma.py` — скриптована анотація Анотатора B через локальну Gemma (round 7-14)
- `src/merge_and_compute_kappa.py` — злиття A/B міток у майстер-датасет + Cohen's κ (round 7-14)
- `src/build_baseline_eval_set.py` — побудова ground-truth файлів для baseline-оцінки (n=59 agreement, n=85 усі валідні)
- `src/baseline_metrics.py` — спільна macro P/R/F1 з явним `n/a` для класів без прикладів
- `src/baseline_rule_based.py` — rule-based baseline (заперечення/антонімія)
- `src/baseline_finetune_roberta.py` — fine-tuned RoBERTa baseline (5-fold stratified CV, in-domain)
- `src/baseline_llm_fewshot_report.py` — LLM few-shot baseline (Gemma vs Claude, без нових викликів)
- `src/evaluate_baselines.py` — зведений звіт трьох baseline (`docs/baseline_evaluation_report.md`)
- `src/build_alamri_stevenson_training_pairs.py` — побудова тренувальних пар з корпусу Alamri & Stevenson (§5.3)
- `src/baseline_generalization_alamri_stevenson.py` — fine-tune на Alamri & Stevenson + zero-shot оцінка на HypoContra (`docs/generalization_report.md`)
- `docs/pilot_results*.md` — детальні звіти кожного з 6 раундів пілоту
- `docs/pilot_final_summary.md` — **зведений підсумок пілоту (round 1-6), почніть тут**
- `docs/claude_batch_instructions.md` / `docs/claude_label_instructions.md` — інструкції для Claude-кроків побудови пар і розмітки
- `docs/next_scaling_directions.md` — пул напрямків подальшого масштабування, оновлено за підсумками round 16-17
- `results/hypocontra_pilot_master_dataset.csv` — консолідований датасет усіх пар × 2 анотатори (round 1-17, 103 куровані)

## Запуск (відтворення)

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas requests scikit-learn spacy pymupdf torch transformers
.venv/bin/python -m spacy download en_core_web_sm

# Round 1-2: автоматична кластеризація абстрактів
.venv/bin/python src/collect_abstracts.py
.venv/bin/python src/extract_candidate_pairs.py          # топічна (round 1)
.venv/bin/python src/extract_candidate_pairs_strict.py   # Task-фраза (round 2)

# Round 3-6: видобування з оглядів
.venv/bin/python src/find_more_surveys.py                 # пошук додаткових оглядів (round 4)
.venv/bin/python src/mine_disagreement_from_surveys.py     # видобування (усі раунди, параметризовано)

# Round 7-12: масштабування (потребує локального Gemma-сервера, OpenAI-сумісний ендпоінт)
.venv/bin/python src/find_surveys_bulk.py --query-mode survey --max-total 1700   # широкий пошук cs.CL
.venv/bin/python src/resolve_survey_citations.py --input <csv>   # резолюція цитат
.venv/bin/python src/prepare_claude_pair_batch.py --round N      # batch для Claude (побудова пар)
# ... Claude заповнює results/claude_pair_batch_round{N}_completed.csv вручну ...
.venv/bin/python src/ingest_claude_pairs.py --round N
.venv/bin/python src/prepare_claude_label_batch.py --round N     # batch для сліпої розмітки A
# ... Claude заповнює results/claude_label_batch_round{N}_completed.csv вручну ...
.venv/bin/python src/ingest_claude_labels.py --round N --annotator A
.venv/bin/python src/annotate_with_gemma.py --round N            # Анотатор B, повністю скриптовано
.venv/bin/python src/merge_and_compute_kappa.py --round N

# Baseline-оцінка (§5 статті) — потребує GPU для RoBERTa-кроків
.venv/bin/python src/build_baseline_eval_set.py
.venv/bin/python src/baseline_rule_based.py
.venv/bin/python src/baseline_finetune_roberta.py
.venv/bin/python src/baseline_llm_fewshot_report.py               # потребує локального Gemma-сервера
.venv/bin/python src/evaluate_baselines.py

# §5.3 крос-доменна генералізація
.venv/bin/python src/build_alamri_stevenson_training_pairs.py     # завантажує corpus.xml (кешується)
.venv/bin/python src/baseline_generalization_alamri_stevenson.py
```

## Наступні кроки (не виконано)

Стаття (§9) визначає три пункти. Стан:

1. **(а) Реальні люди-анотатори** на курованому пулі 103 пар — для генуїнної людської міжанотаторської згоди (Анотатор B тепер архітектурно інша модель, ніж Claude, але це досі не людина). **❌ Свідомо відхилено (2026-09-11), не відкладено** — автор вирішив не виконувати цей пункт у межах цього дослідницького циклу; AI-AI природа розмітки (κ=0.556 5-way / κ=0.607 3-way, n=85) лишається постійною характеристикою корпусу, задокументованою як таке в статті (§8, §9, §10).
2. **(б) Подальше масштабування до 300-500 пар** — усі перевірені напрямки (arXiv cs.CL-запити, ACL Anthology обидві підмножини, DISAGREEMENT_PATTERNS, citation-graph топ-733) дали спадну віддачу (103 пари разом, далеко від 300-500). **Свідомо зупинено**, не рекомендовано як пріоритет — деталі й невикористаний резерв (~17400 кандидатів нижчого рівня довіри) у `docs/next_scaling_directions.md`.
3. **(в) Навчання й оцінка baseline NLI-моделей (§5 статті)** — **✅ завершено** (rule-based F1=0.310, RoBERTa F1=0.575, LLM few-shot F1=0.690; див. секцію вище й `docs/baseline_evaluation_report.md`).

Додатково, поза оригінальним §9-списком:

4. **§5.3 крос-доменна генералізація** — **✅ завершено** (zero-shot transfer F1=0.378, `docs/generalization_report.md`).
5. **§7 перенесення міток у граф суперечностей** — стаття явно вказує, що це можна ставити емпірично лише після (в); тепер розблоковано, не розпочато.
6. Розширення на інші домени/мови для перевірки узагальнюваності таксономії.

Деталі пілоту (round 1-6) — `docs/pilot_final_summary.md`, розділ "Рекомендація для подальшої роботи". Round 7-17 і baseline-оцінка задокументовані вище й у самій статті (§4.4, §5, §8, §9, §10).
