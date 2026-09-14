# Інструкція побудови пар гіпотез (Claude, крок 2 round N)

Вхід: `results/claude_pair_batch_round9.jsonl` -- по одному об'єкту на речення:
`{sentence_id, survey_arxiv_id, survey_title, sentence, cited_refs: [{key, title, authors, year, abstract, context}, ...]}`.

Для КОЖНОГО речення:

1. Прочитай `sentence` і список `cited_refs`. Якщо процитовано більше 2 робіт,
   визнач, ЯКІ САМЕ дві дійсно протиставлені реченням (а не просто перелічені
   поруч) -- решту ігноруй.
2. Якщо жодні дві роботи НЕ протиставлені по суті (речення хибно спрацювало на
   маркер контрасту без реального протиставлення гіпотез) -- познач
   `status=skipped`, вкажи `skip_reason` (напр. "not_genuine_contrast",
   "same_finding_different_wording", "insufficient_context").
3. Якщо протиставлення реальне -- сформулюй `hypothesis_a` і `hypothesis_b` як
   САМОСТІЙНІ, зрозумілі без контексту речення твердження, спираючись на
   реальний зміст `title`/`abstract`/`context` кожної роботи (НЕ вигадуй деталей,
   яких немає в наданих полях; якщо `abstract`/`context` не дають достатньо
   інформації для впевненого формулювання -- це підстава для `skipped` з
   `skip_reason=insufficient_context`, а не для вигадки).
4. Признач `topic` -- коротка назва спільної теми/задачі (2-5 слів).
5. `source_a`/`source_b` -- цитатний ключ (`key`) відповідної роботи.

## Формат виводу

Заповни `results/claude_pair_batch_round9_completed.csv` з колонками:
`sentence_id, status, skip_reason, topic, hypothesis_a, source_a, hypothesis_b, source_b`

`status` є `pair_created` або `skipped`. Для `pair_created` заповнюються всі
колонки крім `skip_reason` (лишити порожнім); для `skipped` -- лише
`sentence_id, status, skip_reason`.

Один рядок вхідного JSONL -> один рядок вихідного CSV (навіть для skipped --
не пропускай речення мовчки, це порушить облік прозорості пайплайну).
