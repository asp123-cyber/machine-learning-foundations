import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

epochs = np.arange(1, 26)
loss = 2.5 * np.exp(-0.18 * epochs) + np.random.normal(0, 0.04, size=len(epochs))

area = np.random.uniform(800, 2600, 80)
price = 0.045 * area + np.random.normal(0, 12, 80) + 15

residuals = np.random.normal(0, 10, 500)
models = ["Linear Reg", "KNN", "Decision Tree", "Random Forest"]
scores = [0.76, 0.81, 0.79, 0.89]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].plot(epochs, loss, marker="o", linewidth=2, color="#2563eb")
axes[0, 0].set_title("Line Plot: Training Loss Decay")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].grid(True, linestyle="--", alpha=0.4)

axes[0, 1].scatter(area, price, alpha=0.75, edgecolors="white", s=55, color="#0d9488")
axes[0, 1].set_title("Scatter Plot: Area vs. Price")
axes[0, 1].set_xlabel("Area (sq. ft.)")
axes[0, 1].set_ylabel("Price (Lakhs)")
axes[0, 1].grid(True, linestyle="--", alpha=0.4)

axes[1, 0].hist(residuals, bins=25, color="#7c3aed", alpha=0.8, edgecolor="white")
axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1.2)
axes[1, 0].set_title("Histogram: Prediction Error Distribution")
axes[1, 0].set_xlabel("Residual Error")
axes[1, 0].set_ylabel("Frequency")

axes[1, 1].bar(models, scores, color=["#64748b", "#3b82f6", "#f59e0b", "#10b981"])
axes[1, 1].set_ylim(0, 1.05)
axes[1, 1].set_title("Bar Chart: Model Accuracy Comparison")
axes[1, 1].set_ylabel("Validation Score")
axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("matplotlib_cheatsheet.png", dpi=600, bbox_inches="tight")
plt.show()
