# 06. Comparing Different Classification Models (Parts 1, 2 & 3)

## Part 1: Why Benchmark Multiple Algorithms?
The **No Free Lunch Theorem** states that no single algorithm wins on every problem. A linear model may beat a complex tree on clean, linearly separable data, while an ensemble tree may crush a linear model on messy non-linear boundaries. Good ML practice means comparing several families under the exact same train/test split and cross-validation scheme.

## Part 2: Six Core Classifiers in Plain English
1. **Logistic Regression:** Despite the word "regression", it is a *classifier*. It passes a linear score $z = w^T x + b$ through the S-shaped **sigmoid function**:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   converting any number into a probability between 0 and 1. Fast, interpretable baseline.
2. **K-Nearest Neighbors (KNN):** Votes based on the closest training points. Great for irregular shapes; requires feature scaling.
3. **Decision Tree:** Plays "20 Questions" by splitting data along thresholds that maximize purity (Gini impurity or Entropy). Super easy to explain, but a single deep tree easily overfits.
4. **Random Forest:** Trains hundreds of diverse Decision Trees on random bootstrap subsets of rows and features, then aggregates their votes. Dramatically cuts variance and overfitting.
5. **Support Vector Machine (SVM):** Finds the decision boundary with the **widest possible safety margin** between classes. With an RBF kernel, it separates complex curved patterns effortlessly.
6. **Gaussian Naive Bayes:** Applies Bayes' Theorem while assuming features are conditionally independent. Extremely fast and surprisingly strong on high-dimensional or small datasets.

## Part 3: Evaluation Metrics (Why Accuracy Alone Can Lie)
Suppose a rare disease occurs in only 1% of patients. A trivial model that always predicts *"Healthy"* scores **99% accuracy** while failing to detect a single sick patient.

Always check the **Confusion Matrix** (True Positives $TP$, False Positives $FP$, True Negatives $TN$, False Negatives $FN$):
- **Precision** $= \frac{TP}{TP + FP}$: Out of everyone flagged Positive, how many truly were Positive? (Vital for spam filtering.)
- **Recall (Sensitivity)** $= \frac{TP}{TP + FN}$: Out of all *actual* Positives, how many did we catch? (Vital for medical screening.)
- **F1-Score** $= 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$: Harmonic mean balancing both.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve across all probability thresholds ($0.5$ = random guessing, $1.0$ = perfect separation).
- **Stratified K-Fold Cross-Validation:** Preserves class ratios across $K$ folds so your comparison isn't swayed by one lucky split.
