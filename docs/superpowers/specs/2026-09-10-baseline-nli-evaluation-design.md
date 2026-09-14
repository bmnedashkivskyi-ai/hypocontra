# Baseline NLI-model evaluation protocol for HypoContra (§5, item "в") — design

Status: approved by user 2026-09-10, ready for implementation plan.

## Context

The taxonomy article (`~/test-article/02-HypoContra-taksonomiia-korpus-superechnostei/hypocontra-taksonomiia-korpus-superechnostei.md`) specifies, in §5, a protocol for evaluating three baseline models against the HypoContra taxonomy (§3). §9 lists this as item (в) of three remaining pieces of work — the only one of the three never started at all (no code exists for it anywhere in `~/prjs/hypocontra/src/`).

Corpus state going in: 103 curated pairs across rounds 1-17 (combined round 7-17 Cohen's κ=0.556 on n=85, below the article's own §4.4 threshold of 0.6). Both annotators (Claude = Annotator A, local Gemma = Annotator B) are AI, not human, per the article's own disclosed §8 limitation — this spec does not change that; it evaluates baseline models against the corpus as it actually exists, with the corpus's own known quality caveats carried forward honestly into the evaluation report rather than hidden.

## Goals

- Implement all three §5.1 baselines (rule-based, fine-tuned RoBERTa, LLM few-shot) and the §5.2 metrics (macro P/R/F1, type4 reported separately) against the actual HypoContra corpus as it stands today.
- Resolve, in a way that is defensible and explicitly documented (not silently assumed), the methodological gaps the article's abstract spec leaves open: what counts as ground truth given sub-threshold inter-annotator agreement, how to handle near-empty classes (type2 has zero examples in the eligible ground-truth set, type3 has three), and how to avoid a circular evaluation of the Gemma few-shot baseline (Gemma is already one of the corpus's own two annotators).
- Produce one final human-readable report (`docs/baseline_evaluation_report.md`) that could plausibly replace the article's current placeholder text in §5 ("baseline NLI-моделі... НЕ навчено й НЕ оцінено") with real numbers — updating the article itself is a separate, later step, not part of this plan.

## Non-goals

- §5.3 (cross-domain generalization check: fine-tune on Alamri & Stevenson's 2016 biomedical corpus, zero-shot eval on HypoContra) is explicitly deferred — sourcing that external corpus is a separate unknown from this experiment and is out of scope for this plan. Note this clearly in the final report as unaddressed, not silently dropped.
- No new Gemma API calls. The LLM few-shot baseline reuses `label_B` values already collected during the corpus's own annotation rounds (see Decision 4 below) — it does not re-run `annotate_with_gemma.py` or call the model again.
- No changes to any existing round-based corpus-building script (`resolve_survey_citations.py`, `prepare_claude_pair_batch.py`, `annotate_with_gemma.py`, `merge_and_compute_kappa.py`, etc.) — this plan only reads their output files.
- No attempt to update the taxonomy article itself with the results — that's a follow-up step after this experiment produces real numbers, explicitly out of scope here.
- No human-annotator work (article's §9 item (а)) — separate, unrelated piece of work.

## Decisions (resolved during brainstorming, 2026-09-10)

These are the open methodological questions the article's abstract §5 spec doesn't answer, and what was decided:

**1. Ground truth for the rule-based and RoBERTa baselines: round 7+ pairs where `label_A == label_B`.**
`hypocontra_pilot_master_dataset.csv`, filtered to `method == "survey_mining_dual_llm_pipeline"` (round 7+ only — excludes pilot rounds 3-6, which used two Claude passes rather than Claude-vs-Gemma, a weaker independence claim) AND `label_A == label_B` (both valid labels, agreeing). This yields **n=59**. Rationale: using disagreement-laden labels as training/eval ground truth would bake noise directly into the baselines; restricting to round 7+ keeps the stronger "different architectures" independence claim; restricting to agreement rows is the honest floor for what this corpus can currently support as a gold standard, even though it shrinks n substantially from the full 103-pair corpus.

**2. Task framing: 3-way is primary, 5-way is a secondary report with explicit `n/a` for unsupported classes.**
Within the n=59 agreement set, class counts are: `not_contradiction`=18, `type1_direct_negation`=32, `type4_apparent_contextual`=9, `type3_causal_conflict`=0, `type2_quantitative_conflict`=0 (both effectively absent once restricted to round-7+ agreement — type3 had 3 examples in the *full* 167-row all-rounds agreement set, but 0 within round-7+ specifically; verify this exact number when building the eval set, per Task 1 below, and correct this document's numbers if the rebuild disagrees). §3's own taxonomy table already maps type1-3 → "Contradiction", type4 → "Neutral*" as an NLI-like collapsed label — reuse that exact mapping as the 3-way primary framing: `Contradiction` = {type1, type2, type3}, `Apparent` = {type4}, `NotContradiction` = {not_contradiction}. Every baseline's headline metric is 3-way macro P/R/F1. A secondary 5-class table is still reported (per §5.2's literal spec), with classes that have zero eligible examples marked `n/a` in every column — never a fabricated 0.0 or a silently dropped row.

**3. RoBERTa: `roberta-large-mnli`, fine-tuned via 5-fold stratified cross-validation on the 3-way task only.**
n=59 is too small for a single train/test split to give a stable estimate (a single held-out fold would be ~12 examples). Cross-validation is standard practice for this scale and was chosen over a fixed split for that reason. The model is **not** separately fine-tuned for the 5-class task — with type2/type3 having 0 eligible examples, stratified CV folds cannot be constructed for them at all, so 5-class RoBERTa results are `n/a` by construction, not attempted. GPU is available in this environment (confirmed via `nvidia-smi` during brainstorming); `torch`/`transformers` are not yet installed in `~/prjs/hypocontra/.venv` and must be added as a dependency.

**4. LLM few-shot baseline: reuse existing Gemma labels, evaluated against Claude on ALL round 7+ pairs — no new model calls.**
The article's §5.1 spec ("a modern instructed model, few-shot, taxonomy in the prompt, no fine-tuning") already describes exactly what `annotate_with_gemma.py` does for Annotator B — Gemma's `label_B` values already ARE few-shot taxonomy classifications. Evaluating Gemma against the Decision-1 ground truth (which is *defined* by `label_A == label_B`) would be circular — Gemma would score 100% by construction, since agreement with Claude is what makes a row eligible. Instead, this baseline reports macro P/R/F1 treating `label_A` (Claude) as the reference across **all** round-7+ pairs with valid labels from both annotators (n=85, not just the n=59 agreement subset) — this is a different, wider set than Decision 1's ground truth, used only for this one baseline. This is explicitly reported as "agreement with Claude's judgment," not "accuracy against independent gold labels" — it is, numerically, a recomputation of the same combined κ=0.556 (n=85) already on record, just expressed as per-class macro P/R/F1 instead of a single kappa statistic. The report must state this circularity/reuse explicitly, not present it as equivalent in kind to the other two baselines' evaluations.

**5. §5.3 (cross-domain transfer) deferred**, per Non-goals above.

## Architecture

Four independent scripts under `~/prjs/hypocontra/src/` (flat, matching the project's existing convention — no subdirectory), sharing one eval-set file, plus one report generator:

### 1. `src/build_baseline_eval_set.py`

- Reads `results/hypocontra_pilot_master_dataset.csv` (`dtype=str` throughout, matching project convention for any ID-bearing column) and every `results/pilot_round{N}_curated_sample.csv` for N in the round-7+ range actually present on disk (do not hardcode the round list — glob for `pilot_round*_curated_sample.csv` and filter by which round numbers appear in the master dataset's `method == "survey_mining_dual_llm_pipeline"` rows, so a future round's data is picked up automatically).
- Joins on `pair_id` to attach each labeled row's `topic`/`hypothesis_a`/`source_a`/`hypothesis_b`/`source_b` text (needed by the rule-based and RoBERTa baselines; the LLM few-shot baseline needs only the labels, not the text).
- Produces two output files:
  - `results/baseline_ground_truth_agreement.csv` — Decision 1's n=59 set: `pair_id, topic, hypothesis_a, source_a, hypothesis_b, source_b, label_5way, label_3way`.
  - `results/baseline_all_round7plus_labels.csv` — Decision 4's wider n=85 set: `pair_id, label_A_5way, label_A_3way, label_B_5way, label_B_3way` (no hypothesis text needed here).
- Prints both n counts and the full class breakdown for both files on every run, so a silent corpus-state drift (e.g. a future round changing these numbers) is visible immediately rather than discovered later in a report.

### 2. `src/baseline_rule_based.py`

- Reads `results/baseline_ground_truth_agreement.csv`.
- For each row, applies a lexical heuristic to `hypothesis_a`/`hypothesis_b`: a small fixed negation-marker list (`not`, `no`, `never`, `cannot`, `n't`, `neither`, `nor`) and a small fixed antonym-pair list of common comparative/technical terms (`increase`/`decrease`, `improve`/`worsen`, `more`/`less`, `better`/`worse`, `higher`/`lower`, `positive`/`negative`, `support`/`refute`, `confirm`/`contradict` — start with this list, do not attempt WordNet or any ML-based antonym lookup, matching the article's own "lower complexity bound" framing). Predicts `Contradiction` if a negation marker or antonym pair is detected as present in one hypothesis and absent/opposed in the other; else `NotContradiction`. Never predicts `Apparent` — this is a known, intentional limitation of a lexical-only baseline, not a bug.
- Writes `results/baseline_rule_based_predictions.csv`: `pair_id, predicted_3way`.

### 3. `src/baseline_finetune_roberta.py`

- Reads `results/baseline_ground_truth_agreement.csv`.
- Uses `transformers`' `roberta-large-mnli` checkpoint as the starting point (already MNLI-pretrained, matching §5.1's "RoBERTa, попередньо натренована на MNLI" spec and the ConjNLI architectural precedent it cites). Note: `roberta-large-mnli` already has exactly 3 output labels (CONTRADICTION/NEUTRAL/ENTAILMENT), matching this task's 3-way class count, so `AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)` keeps and further fine-tunes the pretrained MNLI classification head rather than reinitializing a fresh one from scratch — `ignore_mismatched_sizes=True` is included defensively in case the label count ever differs, but in this specific case it is a no-op. This gives the RoBERTa baseline an asymmetric MNLI warm start (see the evaluation report's RoBERTa section for the implications), not a blank-slate 3-way classifier.
- 5-fold stratified cross-validation on `label_3way` (n=59 → roughly 47 train / 12 eval per fold). Stratification target is `label_3way`; if any fold cannot be stratified due to a class's small count, use `sklearn.model_selection.StratifiedKFold`'s standard behavior and report the actual fold sizes achieved, not a hand-tuned workaround.
- For each fold: fine-tune on the training split (small number of epochs appropriate to n≈47 — start conservative, e.g. 3-5 epochs, and log train loss per epoch so overfitting is visible, not hidden), predict on the held-out fold, collect predictions.
- After all 5 folds, pool the out-of-fold predictions (every one of the 59 examples gets exactly one prediction, from the fold where it was held out) and compute metrics on the full pooled set — the standard way to report cross-validated performance.
- Writes `results/baseline_roberta_predictions.csv`: `pair_id, predicted_3way, fold`.
- Also writes `results/baseline_roberta_training_log.csv` with per-fold per-epoch loss, so the report can show whether training was stable given the tiny per-fold training set.

### 4. `src/baseline_llm_fewshot_report.py`

- Reads `results/baseline_all_round7plus_labels.csv` (Decision 4's n=85 set — no model calls, no network access).
- Computes macro P/R/F1 for `label_B_3way` against `label_A_3way` as reference (and the 5-class equivalent, with `n/a` for any class absent from this specific n=85 set — check and report which, don't assume it matches the n=59 set's absent classes).
- Writes `results/baseline_llm_fewshot_metrics.csv` with the computed metrics, and a plain-language note field in the same file (or an adjacent `.md` note) stating explicitly that this reuses existing Annotator B labels and measures agreement-with-Claude, not independent accuracy.

### 5. `src/evaluate_baselines.py`

- Reads `results/baseline_ground_truth_agreement.csv` (for true labels) plus `baseline_rule_based_predictions.csv` and `baseline_roberta_predictions.csv` (for those two baselines' predictions), and `baseline_llm_fewshot_metrics.csv` (already-computed, just pass through).
- Computes macro P/R/F1 (3-way primary, 5-way secondary with `n/a` handling) for the rule-based and RoBERTa baselines using the same metric function, so all three baselines' numbers are computed identically and are genuinely comparable.
- Writes the final consolidated report: `docs/baseline_evaluation_report.md`, in a structure mirroring §5.2's requirement — one table per baseline (3-way headline + 5-way detail with `Apparent`/type4 called out separately, per the article's explicit ask), plus a short prose section stating: the ground-truth construction (Decision 1), the circularity caveat for the LLM few-shot baseline (Decision 4), the RoBERTa training stability observations from the fold-level log, and an explicit "§5.3 not attempted" line.

## Error Handling / Edge Cases

- If `build_baseline_eval_set.py`'s class counts differ from this document's Decision 2 numbers (e.g. because round 18+ has landed by the time this plan is implemented), the script must print the actual counts prominently — do not hardcode expected counts anywhere in the pipeline; every downstream script must handle whatever counts actually exist, including a class dropping to zero.
- If a RoBERTa cross-validation fold has fewer than 2 examples of some class after stratified splitting, `evaluate_baselines.py` must report that fold's contribution as `n/a` for that class in the per-class breakdown rather than crash or silently zero-fill.
- `torch`/`transformers` installation: add to a `requirements`-equivalent already established in this project's convention (check whether `~/prjs/hypocontra` has a `requirements.txt`/`pyproject.toml` already, or whether dependencies have so far just been installed ad hoc into `.venv` — follow whatever precedent exists; if none, install directly into `.venv` matching how `pandas`/`requests` etc. were presumably added).

## Testing

No pytest, per this project's established convention (see round 15/16/17 plans). Verify each script's correctness the same way prior rounds did: run it against the real data, print counts, and sanity-check a handful of rows by hand (e.g., manually re-derive 2-3 rule-based predictions from the actual hypothesis text; manually confirm 2-3 RoBERTa fold assignments are stratified correctly; manually recompute the LLM few-shot metrics for one class from the raw label counts and compare to the script's output).

## Self-Review

**Placeholder scan:** no TBD/TODO. Decision 2's exact type3 count in the round-7+-only agreement set is flagged as "verify when building the eval set" rather than asserted with false confidence — this is intentional (the number was computed from a slightly different query during brainstorming than what Task 1 will run), not a placeholder for missing design work; the design does not depend on the exact number, only on the general fact that type2/type3 are near-empty and must be handled as `n/a`, which is already fully specified.

**Internal consistency:** Decision 1's n=59 set (hypothesis text, used for rule-based + RoBERTa) and Decision 4's n=85 set (labels only, used for LLM few-shot) are deliberately different populations for a stated reason (avoiding circularity) — this is called out explicitly in both the Decisions section and Architecture component 4, not left as an unexplained inconsistency.

**Scope check:** this spec covers §5.1-§5.2 (three baselines + metrics) only. §5.3 is explicitly out of scope (Non-goals). Updating the article itself with results is explicitly out of scope (Goals' last bullet clarifies the report is produced, not that the article is edited). This is focused enough for one implementation plan.

**Ambiguity check:** "GPU available" was confirmed via `nvidia-smi` during brainstorming, not assumed — if a fresh session picks up this plan later and the GPU is no longer available or torch/CUDA setup fails, that's a real blocker to flag at implementation time, not something this spec can pre-resolve.
