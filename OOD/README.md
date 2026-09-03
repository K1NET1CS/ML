# OOD : Out of Distribution Detection

## 0_MyAttempts
These are 3 methods I tried to capture Out of Distribution examples before diving into research papers on OOD

### Custom OOD Detection on MNIST

**Objective:** A from-scratch exploration training a CNN solely on MNIST digits 1–9, tasked with detecting unseen data (MNIST 0,FashionMNIST,Gaussian noise) as Out-of-Distribution (OOD) without prior exposure.

**Core Approach of best idea: L2-Normalized Feature Distance**
Instead of relying on softmax confidence or adding an explicit "reject" class, this method measures semantic distance in the model's feature space:
* **Feature Extraction:** Extract the 128-dimensional penultimate features from a standard 9-way CNN trained on digits 1–9.
* **L2 Normalization:** Normalize features to unit length. This compares directional similarity rather than magnitude, which is crucial for mitigating ReLU collapse on structureless inputs.
* **Scoring:** Compute the Euclidean distance to the nearest training class mean. Inputs exceeding the 95th percentile of In-Distribution distances are flagged as OOD.

**Key Findings:**
* **Reject Class Limitation:** Training a 10th class on synthetic noise detects noise perfectly but fails catastrophically on structured OOD data (like FashionMNIST or digit 0).
* **Near-OOD Success:** The normalized feature distance successfully flags digit 0 (which shares the exact domain and style as the training data) with a strong **AUROC of 0.975**.
* **The Noise Anomaly:** Structureless Gaussian noise activates very few filters, causing feature vectors to collapse near the origin. L2 normalization forces these vectors outward onto the unit hypersphere, restoring smooth distance-based separation.
