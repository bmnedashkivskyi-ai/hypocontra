# HypoContra Baseline Generalization Report (Section 5.3) -- NLI4CT

Fine-tuned roberta-large-mnli on NLI4CT (SemEval-2023 Task 7 / SemEval-2024 Task 2, results/nli4ct_training_pairs.csv), then evaluated zero-shot on HypoContra's own n=59 3-way ground-truth set (results/baseline_ground_truth_agreement.csv) -- no fine-tuning on HypoContra. Second, independent crossdomain data point for Section 5.3, alongside docs/generalization_report.md (Alamri & Stevenson, 2016); see docs/s53_alternative_generalization_corpora.md for why this corpus was added.

Training pairs: n_train=1700, n_dev=200 (NLI4CT's own SemEval train/dev split; dev held out for best-epoch selection only, not reported as a result). Best dev macro F1: 0.4715.

## Zero-shot transfer result on HypoContra (n=59)

**Macro F1: 0.2643** (macro precision=0.2441, macro recall=0.2882)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Apparent | 0.0000 | 0.0000 | 0.0000 | 9 |
| Contradiction | 0.4595 | 0.5312 | 0.4928 | 32 |
| NotContradiction | 0.2727 | 0.3333 | 0.3000 | 18 |

## Caveat: `Apparent` is structurally absent from training data

NLI4CT's source scheme is binary (`Label="Entailment"`/`"Contradiction"`, does a statement match or contradict the evidence from one or two Clinical Trial Reports) -- there is no equivalent of HypoContra's type4/`Apparent` (contextual, non-logical) class, the same structural gap as in the Alamri & Stevenson corpus. Weak zero-shot recall on `Apparent` reflects this gap in the training data's label space, not a flaw in the model or this script.

## Note: hypothesis_b is evidence text, not a second claim

Unlike Alamri & Stevenson (both hypothesis_a and hypothesis_b are claims from research abstracts), here hypothesis_a is the Statement and hypothesis_b is evidence sentences resolved from the referenced Clinical Trial Report section(s) (concatenated primary + secondary trial evidence for "Comparison"-type instances). This is a claim-vs-evidence NLI task, structurally closer to fact verification (e.g. SciFact) than to HypoContra's claim-vs-claim pairs -- a difference worth weighing when interpreting how this result compares to the Alamri & Stevenson crossdomain point.

## Caveat: training is unstable and run-to-run variance is real (not just eval noise)

Diagnosed via `superpowers:systematic-debugging` before accepting this result. First attempt (`max_length=256`, the same setting used for Alamri & Stevenson) produced a degenerate model that predicted `Contradiction` for all 59 HypoContra pairs, with `dev_macro_f1` flat at 0.3333 (a fixed point corresponding to "always predict the majority-ish class") for 3 of 4 epochs. Two things were verified before accepting the fix below:

- **Truncation was tested and ruled out as the sole cause.** `hypothesis_b` here (CT-report evidence) is far longer than Alamri & Stevenson's (median 178 vs. 33 tokens; 38% of NLI4CT pairs exceed 256 combined tokens vs. 0% for Alamri & Stevenson) -- `max_length` was raised to 512 (roberta-large-mnli's position-embedding limit) accordingly. A controlled re-run at `max_length=512` on the *full* dataset **still collapsed** to the same `dev_macro_f1=0.3333` degenerate point in one trial run, falsifying "truncation alone explains it." `max_length=512` was kept anyway (a genuine, independently-justified fix: needlessly discarding 38% of training/eval evidence is a real defect regardless of whether it explains the collapse), and is what produced the numbers reported above.
- **The task is intrinsically harder / more out-of-distribution for `roberta-large-mnli`'s pretrained NLI prior than Alamri & Stevenson.** Epoch-0 `train_loss` starts near `ln(2)=0.693` (0.70-0.72, i.e. near-chance) for NLI4CT, vs. 0.26 for Alamri & Stevenson (`results/generalization_alamri_stevenson_training_log.csv`) -- the pretrained head transfers to Alamri & Stevenson's natural claim-vs-claim sentence pairs far more readily than to NLI4CT's list-and-header-formatted CT-report excerpts as "premise" text. Alamri & Stevenson also converges to dev macro F1 0.96-0.99 within 4 epochs; NLI4CT's best epoch here reached dev macro F1 0.4715 (epoch 1) before regressing back toward the degenerate point by epoch 3 -- `train_with_dev_selection`'s best-epoch checkpointing (already part of the shared training loop, not added for this diagnosis) is what rescues a usable checkpoint from a training trajectory that is otherwise unstable and prone to collapsing.

**Practical implication:** treat the reported macro F1 (0.2643) as this run's outcome under the project's standard single-seeded-run protocol (`torch.manual_seed(42)`, same protocol as Alamri & Stevenson, for comparability), not as a precisely reproducible point estimate -- GPU (cuDNN/cuBLAS) non-determinism means re-running this exact script is not guaranteed to reproduce this exact number, and could plausibly land back in the degenerate all-`Contradiction` regime observed during diagnosis (macro F1 approx. 0.23) rather than the escaped one reported here. This instability is itself part of the finding: NLI4CT transfers to HypoContra markedly less reliably than Alamri & Stevenson does, on top of transferring less well on average.
