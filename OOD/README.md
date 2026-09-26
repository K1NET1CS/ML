# Out-of-Distribution (OOD) Detection on MNIST

NOTION PAGE for details of all methods implemented
([Notion](https://app.notion.com/p/ML-OOD-3cfc5fbebbce80a3984bf0db5ebdb539?source=copy_link))

A systematic exploration of OOD detection methods implemented from scratch. The model is trained **only on MNIST digits 1–9** and must flag everything else as out-of-distribution — without ever seeing those examples during training.

**OOD test sources:**
- **Digit 0** — near-OOD, same handwriting style and dataset as training data
- **FashionMNIST** — far-OOD, different domain but real image structure
- **Gaussian Noise** — far-OOD, no structure

Digit 0 is the meaningful benchmark. Anything shares a domain with digits 1–9; a model that genuinely learns *what digits look like* should still flag 0 as anomalous.

---

## Methods & Results

| Notebook | Method | Digit 0 AUROC | FashionMNIST AUROC | Noise AUROC |
|---|---|---|---|---|
| `0_MyAttempts` | Noise as reject class (K+1) | poor | poor | good |
| `0_MyAttempts` | Euclidean feature distance | 0.975 | 0.871 | 0.909 |
| `1_MSP_Baseline` | Max Softmax Probability | good (Gaussian) | — | — |
| `1_MSP_Baseline` | MSP + Abnormality Module | better across board | — | — |
| `2_ODIN` | ODIN (T scaling + perturbation) | 0.965 | — | — |
| `3_Mahalanobis` | Mahalanobis distance | 0.814 | 0.9999 | 1.000 |
| `4_OutlierExposure` | Outlier Exposure + MSP | 0.969 | 1.000 | 1.000 |
| `5_EnergyBased` | Energy score (inference-time) | 0.990 | 0.981 | 1.000 |
| `5_EnergyBased` | Energy fine-tuning | 0.983 | 1.000 | 1.000 |

---

## Notebooks

### `0_MyAttempts.ipynb` — Initial Ideas
Two from-scratch approaches before reading any papers.

**Idea 1:** Add a 10th class, train on synthetic noise as the reject class. Works on noise but fails badly on digit 0 — the model learns "reject = random pixels," not "reject = unfamiliar structure."

**Idea 2:** Extract penultimate CNN features, compute per-class means on training data, flag test images as OOD if their Euclidean distance to the nearest class mean exceeds a threshold. Digit 0 AUROC of 0.975 — surprisingly strong for a post-hoc method with no retraining. Noise AUROC (0.909) is lower than digit 0 due to ReLU collapse: noise produces near-zero feature vectors that cluster together, causing the ROC curve to jump vertically rather than rise smoothly.

**Idea 2+:** Normalize features to unit length before distance computation (cosine distance). Fixes the ReLU collapse issue.

---

### `1_MSP_Baseline.ipynb` — MSP + Abnormality Module
Implements Hendrycks & Gimpel (2017).

**Model:** MLP with a shared trunk, classifier head, and reconstruction decoder — trained jointly with cross-entropy + MSE reconstruction loss. 97.8% test accuracy.

**MSP:** threshold on max softmax probability. Works on Gaussian noise but fails on Uniform noise and Blur — softmax is overconfident on anything digit-shaped, even if degraded.

**Abnormality Module:** small MLP trained on top of the frozen classifier. Takes features + softmax probs + per-pixel reconstruction error as input. Catches what MSP misses: if the decoder can't reconstruct the input, it's probably OOD.

---

### `2_ODIN.ipynb` — ODIN
Implements Liang et al. (2018).

Temperature scaling (T) flattens the softmax distribution, amplifying the gap between ID and OOD confidence. Input perturbation nudges each image in the direction that *increases* its softmax score — ID images respond more strongly than OOD images because their gradient norm is larger. The two tricks compound.

Key finding: ODIN consistently improves over MSP, especially on Gaussian noise. Perturbation direction is `+grad.sign()` (opposite of adversarial attack — you want to *increase* score, not decrease it).

---

### `3_Mahalanobis.ipynb` — Mahalanobis Distance
Implements Lee et al. (2018).

Formalizes the Euclidean distance idea from `0_MyAttempts`: fit per-class Gaussians on training features with a **shared covariance matrix** across classes, then score test images by their Mahalanobis distance to the nearest class mean. Shared covariance accounts for the fact that feature dimensions aren't equally important or independent.

**Near-OOD (digit 0) AUROC: 0.814** — lower than plain Euclidean distance (0.975). Two reasons:

- Non-zero perturbation (eps) *hurts* on near-OOD: digit 0 shares structure with training digits, so perturbation adds noise instead of clarifying the gap.
- 3-layer ensemble with uniform weights dilutes the strong signal in the penultimate layer with weak early-layer features. The paper trains weights via logistic regression — uniform is a poor substitute.

FashionMNIST (0.9999) and noise (1.0000) are near-perfect, as expected for far-OOD.

---

### `4_OutlierExposure.ipynb` — Outlier Exposure
Implements Hendrycks et al. (2019).

Fine-tunes a pre-trained classifier with an extra loss term: `CE(f(x_oe), Uniform)` on FashionMNIST train examples. This forces the model to output a flat, uncertain softmax on OOD data — not confident for any class. The model learns the concept of "unfamiliar," not just "looks like FashionMNIST."

**Results:** 0.969 on digit 0, 1.000 on FashionMNIST and noise. The OE loss generalizes — even though digit 0 looks nothing like clothing, the model correctly flags it as anomalous. Classification accuracy on digits 1–9 is preserved (~99%).

---

### `5_EnergyBased.ipynb` — Energy-based OOD Detection
Implements Liu et al. (2020).

**Energy score:** `E(x) = -T · log Σ exp(f_k(x)/T)`. Unlike softmax which normalizes by subtracting the max logit (biasing the score), energy uses the raw sum across all logits. ID images produce lower energy; OOD images produce higher energy. One-line change on a pre-trained model, no retraining needed.

**Inference-time (no retraining):** T=1 is optimal (larger T hurts, unlike ODIN). Digit 0 AUROC: 0.990 — the best single-number result in this repo. Beats MSP (0.981) on the same model.

**Energy fine-tuning:** adds two hinge loss terms. ID energy is penalized if it rises above `m_in`; OOD energy is penalized if it falls below `m_out`. Explicitly shapes the energy surface to create a gap between ID and OOD. Final: digit 0 AUROC 0.983, FashionMNIST and noise both 1.000.

---

## Key Takeaways

Digit 0 is near-OOD and the hardest test. Methods that only work on far-OOD (noise, FashionMNIST) aren't actually solving the problem.

Better architecture doesn't help — overconfidence is the bottleneck, not underfitting. The scoring function matters far more than the backbone.

Energy score is a strictly better drop-in replacement for MSP on a pretrained model. It uses the full logit distribution rather than just the peak, which avoids the normalization bias in softmax.

Outlier Exposure is the most practically effective method here — simple to implement, generalizes to unseen OOD distributions, and doesn't require any architectural changes.

Mahalanobis distance with uniform layer weights underperforms plain Euclidean distance on near-OOD. The paper's suggestion to learn layer weights via logistic regression is not a minor detail.

---

## References

- Hendrycks & Gimpel (2017) — [MSP Baseline](https://arxiv.org/abs/1610.02136)
- Liang et al. (2018) — [ODIN](https://arxiv.org/abs/1706.02690)
- Lee et al. (2018) — [Mahalanobis Distance](https://arxiv.org/abs/1807.03888)
- Hendrycks et al. (2019) — [Outlier Exposure](https://arxiv.org/abs/1812.04606)
- Liu et al. (2020) — [Energy-based OOD](https://arxiv.org/abs/2010.03759)
