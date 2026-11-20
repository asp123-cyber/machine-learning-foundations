import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X_raw = np.linspace(1, 10, 60)
y = 3.5 * X_raw + 7.0 + np.random.normal(0, 2.2, size=X_raw.shape)

X_design = np.column_stack([np.ones_like(X_raw), X_raw])
theta_normal = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y)
b_exact, w_exact = theta_normal[0], theta_normal[1]

w_gd, b_gd = 0.0, 0.0
learning_rate = 0.01
epochs = 1200
n = len(X_raw)
loss_history = []

for _ in range(epochs):
    y_pred = w_gd * X_raw + b_gd
    error = y_pred - y
    loss_history.append(np.mean(error ** 2))
    dw = (2 / n) * np.dot(error, X_raw)
    db = (2 / n) * np.sum(error)
    w_gd -= learning_rate * dw
    b_gd -= learning_rate * db

y_hat = w_gd * X_raw + b_gd
rmse = np.sqrt(np.mean((y - y_hat) ** 2))
ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"Normal Equation  -> w = {w_exact:.4f}, b = {b_exact:.4f}")
print(f"Gradient Descent -> w = {w_gd:.4f}, b = {b_gd:.4f}")
print(f"RMSE = {rmse:.4f} | R^2 = {r2:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.scatter(X_raw, y, color="#2563eb", alpha=0.75, label="Observed Data")
ax1.plot(X_raw, y_hat, color="#dc2626", linewidth=2.2, label=f"Fit: y = {w_gd:.2f}x + {b_gd:.2f}")
ax1.set_title("Linear Regression Fit")
ax1.set_xlabel("Feature x")
ax1.set_ylabel("Target y")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.plot(loss_history, color="#0d9488", linewidth=2)
ax2.set_title("Gradient Descent Convergence")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("MSE Loss")
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("linear_regression_fit.png", dpi=600, bbox_inches="tight")
plt.show()
