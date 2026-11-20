import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN (K=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    "Naive Bayes": GaussianNB()
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for name, model in models.items():
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "CV Accuracy": cv_acc,
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, linewidth=1.8, label=f"{name} ({roc_auc_score(y_test, y_prob):.3f})")

summary_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

ax1.barh(summary_df["Model"], summary_df["F1-Score"], color="#2563eb", alpha=0.85)
ax1.set_xlim(0.85, 1.01)
ax1.set_xlabel("F1-Score")
ax1.set_title("Classification Model Benchmark (F1-Score)")
ax1.invert_yaxis()
ax1.grid(True, axis="x", linestyle="--", alpha=0.4)

ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curves Comparison")
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("classification_benchmark.png", dpi=600, bbox_inches="tight")
plt.show()
