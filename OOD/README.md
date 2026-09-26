# Out-of-Distribution (OOD) Detection on MNIST

[Notion page with detailed implementation notes](https://app.notion.com/p/ML-OOD-3cfc5fbebbce80a3984bf0db5ebdb539?source=copy_link)

These are my notes and implementations for various OOD detection methods built from scratch. The setup is simple: train a model only on MNIST digits 1 through 9. It needs to flag everything else as out-of-distribution without seeing those examples during training.

**OOD test sources:**
*   **Digit 0:** The near-OOD benchmark. It shares the same handwriting style and dataset as the training data.
*   **FashionMNIST:** Far-OOD. It has real image structure but comes from a completely different domain.
*   **Gaussian Noise:** Far-OOD with zero structure.

Digit 0 is the real test here. A model that actually learns the structural concept of digits 1 through 9 should still recognize 0 as an anomaly.

## Methods & Results

| Notebook | Method | Digit 0 AUROC | FashionMNIST AUROC | Noise AUROC |
|---|---|---|---|---|
| `0_MyAttempts` | Noise as reject class (K+1) | poor | poor | good |
| `0_MyAttempts` | Euclidean feature distance | 0.975 | 0.871 | 0.909 |
| `1_MSP_Baseline` | Max Softmax Probability | good (Gaussian) | N/A | N/A |
| `1_MSP_Baseline` | MSP + Abnormality Module | better across board | N/A | N/A |
| `2_ODIN` | ODIN (T scaling + perturbation) | 0.965 | N/A | N/A |
| `3_Mahalanobis` | Mahalanobis distance | 0.814 | 0.9999 | 1.000 |
| `4_OutlierExposure` | Outlier Exposure + MSP | 0.969 | 1.000 | 1.000 |
| `5_EnergyBased` | Energy score (inference-time) | 0.990 | 0.981 | 1.000 |
| `5_EnergyBased` | Energy fine-tuning | 0.983 | 1.000 | 1.000 |

## Notebook Breakdown

### `0_MyAttempts.ipynb`: Initial Ideas
I tried two scratch-built approaches before reading the papers.

**Idea 1:** Add a 10th class and train on synthetic noise as the reject class. This works well on noise but fails completely on digit 0. The model just learns that "reject" means random pixels, completely missing the concept of unfamiliar structures.

**Idea 2:** Extract penultimate CNN features, compute per-class means on the training data, and flag test images as OOD if their Euclidean distance to the nearest class mean crosses a certain threshold. This achieved a 0.975 AUROC on digit 0, which is surprisingly strong for a post-hoc method with no retraining. The noise AUROC (0.909) dropped lower than digit 0 because of ReLU collapse. Noise creates near-zero feature vectors that cluster together, making the ROC curve jump vertically instead of rising smoothly.

**Idea 2+:** Normalize features to unit length before computing distance (cosine distance). This easily fixes the ReLU collapse issue.

### `1_MSP_Baseline.ipynb`: MSP + Abnormality Module
Based on Hendrycks & Gimpel (2017).

**Model:** An MLP with a shared trunk, classifier head, and reconstruction decoder. Trained jointly with cross-entropy and MSE reconstruction loss to hit 97.8% test accuracy.

**MSP:** Thresholding on the maximum softmax probability. It handles Gaussian noise fine but fails hard on Uniform noise and Blur. Softmax stays overly confident on anything vaguely digit-shaped regardless of degradation.

**Abnormality Module:** A small MLP trained on top of the frozen classifier. It takes features, softmax probabilities, and per-pixel reconstruction error as input. This catches exactly what MSP misses: if the decoder fails to reconstruct the input, it is likely OOD.

### `2_ODIN.ipynb`: ODIN
Based on Liang et al. (2018).

Temperature scaling (T) flattens the softmax distribution and widens the gap between ID and OOD confidence. Input perturbation pushes each image in the direction that increases its softmax score. ID images react stronger than OOD images because their gradient norm is larger. Combined, these two tricks work incredibly well.

The main takeaway is that ODIN consistently beats MSP, especially on Gaussian noise. Keep in mind the perturbation direction is `+grad.sign()`, which is the opposite of an adversarial attack since the goal is to increase the score.

### `3_Mahalanobis.ipynb`: Mahalanobis Distance
Based on Lee et al. (2018).

This formalizes the Euclidean distance concept from my initial attempts. You fit per-class Gaussians on training features using a shared covariance matrix across all classes, then score test images by their Mahalanobis distance to the nearest class mean. The shared covariance handles the fact that feature dimensions are not equally important or independent.

The near-OOD (digit 0) AUROC sits at 0.814, which is worse than plain Euclidean distance. There are two main culprits:
*   A non-zero perturbation (eps) actively hurts on near-OOD data. Digit 0 shares structural traits with training digits, meaning perturbation just adds noise rather than highlighting the gap.
*   Using a 3-layer ensemble with uniform weights washes out the strong signal in the penultimate layer with weak early-layer features. The paper recommends training weights via logistic regression, and uniform weighting is clearly a poor substitute.

FashionMNIST (0.9999) and noise (1.0000) are near-perfect, as you would expect for far-OOD data.

### `4_OutlierExposure.ipynb`: Outlier Exposure
Based on Hendrycks et al. (2019).

This fine-tunes a pre-trained classifier using an extra loss term of `CE(f(x_oe), Uniform)` on FashionMNIST train examples. It forces the model to output a flat, highly uncertain softmax on OOD data instead of confidently guessing a class. The model actively learns the concept of unfamiliarity rather than just memorizing FashionMNIST.

It scored 0.969 on digit 0 and a perfect 1.000 on FashionMNIST and noise. The OE loss generalizes surprisingly well. Even though digit 0 looks nothing like clothing, the model correctly spots it as an anomaly. Classification accuracy on digits 1 through 9 stays intact at around 99%.

### `5_EnergyBased.ipynb`: Energy-based OOD Detection
Based on Liu et al. (2020).

**Energy score:** Calculated as `E(x) = -T * log Σ exp(f_k(x)/T)`. Softmax normalizes by subtracting the max logit, which biases the score. Energy uses the raw sum across all logits instead. ID images generate lower energy, while OOD images generate higher energy. It only requires a one-line change on a pre-trained model with zero retraining.

**Inference-time (no retraining):** Setting T=1 works best here, and larger T values actually hurt performance unlike ODIN. The digit 0 AUROC hit 0.990, making it the best single-number result in this entire repository. It easily beats MSP (0.981) on the exact same model.

**Energy fine-tuning:** This adds two hinge loss terms. It penalizes ID energy if it rises above a set margin `m_in` and penalizes OOD energy if it falls below `m_out`. This explicitly shapes the energy surface to force a gap between ID and OOD. The final digit 0 AUROC was 0.983, with FashionMNIST and noise both hitting 1.000.

## Key Takeaways

Digit 0 is the hardest test because it acts as near-OOD. Methods that exclusively succeed on far-OOD datasets like noise or FashionMNIST are missing the core problem.

A better architecture will not save you here. Overconfidence is the bottleneck, not underfitting. Your scoring function matters infinitely more than the backbone you choose.

The Energy score is a strictly better drop-in replacement for MSP on any pretrained model. By using the full logit distribution instead of just the peak, it entirely avoids the normalization bias found in softmax.

Outlier Exposure proved to be the most practical and effective method. It is simple to implement, generalizes perfectly to unseen OOD distributions, and requires zero architectural changes.

Finally, Mahalanobis distance with uniform layer weights performs worse than plain Euclidean distance on near-OOD tasks. The paper's recommendation to learn layer weights via logistic regression is a critical detail, not just an optional optimization.

## References

*   Hendrycks & Gimpel (2017): [MSP Baseline](https://arxiv.org/abs/1610.02136)
*   Liang et al. (2018): [ODIN](https://arxiv.org/abs/1706.02690)
*   Lee et al. (2018): [Mahalanobis Distance](https://arxiv.org/abs/1807.03888)
*   Hendrycks et al. (2019): [Outlier Exposure](https://arxiv.org/abs/1812.04606)
*   Liu et al. (2020): [Energy-based OOD](https://arxiv.org/abs/2010.03759)
