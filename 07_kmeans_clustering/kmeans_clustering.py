import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=360, centers=4, cluster_std=0.85, random_state=42)


def run_kmeans(data, k, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(len(data), size=k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    wcss = sum(np.sum((data[labels == i] - centroids[i]) ** 2) for i in range(k))
    return labels, centroids, wcss


k_range = range(1, 9)
wcss_list = []

for k in k_range:
    _, _, wcss = run_kmeans(X, k)
    wcss_list.append(wcss)

labels_opt, centroids_opt, wcss_opt = run_kmeans(X, k=4)
print(f"Converged at K=4 with WCSS = {wcss_opt:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(list(k_range), wcss_list, marker="o", color="#2563eb", linewidth=2)
ax1.axvline(4, color="#dc2626", linestyle="--", label="Elbow at K=4")
ax1.set_title("Elbow Method for Optimal K")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS (Inertia)")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.scatter(X[:, 0], X[:, 1], c=labels_opt, cmap="viridis", alpha=0.75, s=40)
ax2.scatter(centroids_opt[:, 0], centroids_opt[:, 1], c="red", marker="X", s=180, edgecolors="white", label="Centroids")
ax2.set_title("K-Means Clusters (K = 4)")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("kmeans_clustering.png", dpi=600, bbox_inches="tight")
plt.show()
