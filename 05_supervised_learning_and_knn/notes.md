# 05. Supervised Learning & K-Nearest Neighbors (KNN)

## Introduction to Supervised Learning
In supervised learning, every training example comes paired with its true label $(x_i, y_i)$. Our goal is to learn a mapping $f(x) \approx y$ that generalizes accurately to **new, unseen** inputs—not just the points it already saw.

While **regression** outputs a continuous number, **classification** draws decision boundaries separating distinct categories.

## How K-Nearest Neighbors (KNN) Works
KNN is built on one simple idea: **"Tell me who your nearest neighbors are, and I'll tell you who you are."**

When a brand-new, unlabeled point arrives:
1. Measure its **Euclidean distance** to every point in the training set:
   $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
2. Find the **$K$ closest training points**.
3. Hold a **majority vote** among those $K$ neighbors. Whichever class gets the most votes becomes the prediction.

KNN is called a **lazy learner** (instance-based algorithm) because it doesn't fit an equation during training—it stores the dataset and performs distance computations at prediction time.

## Choosing $K$ (Bias vs. Variance Tradeoff)
- **$K = 1$ (Too Small $\rightarrow$ Overfitting):** Sensitive to single noisy points. The decision boundary becomes jagged and memorizes noise.
- **$K$ Too Large (Underfitting):** Smooths out genuine local patterns. If $K$ equals the entire dataset size, every point gets assigned to whichever class is most common overall.
- **Practical Tip:** Pick an **odd $K$** (3, 5, 7, ...) in binary classification to avoid tie votes, and tune $K$ on a validation set.

## The #1 Beginner Mistake With KNN: Forgetting Feature Scaling
Suppose you classify loan applicants using **Age** (20 to 60) and **Annual Salary** (300,000 to 2,000,000). Because salary numbers are 10,000x larger, the squared distance formula completely ignores Age and only looks at Salary.

**Always standardize your features first** using training-set statistics:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Compute $\mu$ and $\sigma$ strictly on the **training set**, then apply those same values to the test set so no test information leaks into training.
