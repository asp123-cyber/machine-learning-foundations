import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_s = (X_train - mean) / std
X_test_s = (X_test - mean) / std


def knn_predict(X_ref, y_ref, X_query, k):
    predictions = []
    for point in X_query:
        distances = np.linalg.norm(X_ref - point, axis=1)
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_ref[nearest_idx]
        values, counts = np.unique(nearest_labels, return_counts=True)
        predictions.append(values[np.argmax(counts)])
    return np.array(predictions)


k_values = list(range(1, 22, 2))
accuracies = []

for k in k_values:
    preds = knn_predict(X_train_s, y_train, X_test_s, k)
    accuracies.append(np.mean(preds == y_test))

best_k = k_values[int(np.argmax(accuracies))]
print(f"Best K = {best_k} with Test Accuracy = {max(accuracies):.4f}")

xx, yy = np.meshgrid(
    np.linspace(X_train_s[:, 0].min() - 0.6, X_train_s[:, 0].max() + 0.6, 160),
    np.linspace(X_train_s[:, 1].min() - 0.6, X_train_s[:, 1].max() + 0.6, 160)
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn_predict(X_train_s, y_train, grid_points, best_k).reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(k_values, accuracies, marker="o", color="#2563eb", linewidth=2)
ax1.set_title("Choosing Optimal K")
ax1.set_xlabel("Number of Neighbors (K)")
ax1.set_ylabel("Test Accuracy")
ax1.set_xticks(k_values)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.contourf(xx, yy, Z, alpha=0.28, cmap="coolwarm")
ax2.scatter(X_test_s[:, 0], X_test_s[:, 1], c=y_test, cmap="coolwarm", edgecolors="black", s=45)
ax2.set_title(f"KNN Decision Boundary (K = {best_k})")
ax2.set_xlabel("Scaled Feature 1")
ax2.set_ylabel("Scaled Feature 2")

plt.tight_layout()
plt.savefig("knn_analysis.png", dpi=600, bbox_inches="tight")
plt.show()
