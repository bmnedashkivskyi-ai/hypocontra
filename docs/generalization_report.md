# HypoContra Baseline Generalization Report (Section 5.3)

Fine-tuned roberta-large-mnli on the Alamri & Stevenson (2016) biomedical contradiction corpus (results/alamri_stevenson_training_pairs.csv), then evaluated zero-shot on HypoContra's own n=59 3-way ground-truth set (results/baseline_ground_truth_agreement.csv) -- no fine-tuning on HypoContra.

Training pairs: n_train=1597, n_dev=178 (held out for best-epoch selection only, not reported as a result). Best dev macro F1: 0.9884.

## Zero-shot transfer result on HypoContra (n=59)

**Macro F1: 0.3776** (macro precision=0.3553, macro recall=0.4086)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Apparent | 0.0000 | 0.0000 | 0.0000 | 9 |
| Contradiction | 0.5952 | 0.7812 | 0.6757 | 32 |
| NotContradiction | 0.4706 | 0.4444 | 0.4571 | 18 |

## Caveat: `Apparent` is structurally absent from training data

Alamri & Stevenson's source scheme is binary (`ASSERTION="YS"`/`"NO"`, support vs. contradict a shared clinical question) -- there is no equivalent of HypoContra's type4/`Apparent` (contextual, non-logical) class. Every training pair is labeled either `Contradiction` or `NotContradiction`; the model never sees an `Apparent` training example. Weak zero-shot recall on `Apparent` reflects this gap in the training data's label space, not a flaw in the model or this script.
