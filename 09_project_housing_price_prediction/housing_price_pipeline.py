import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(script_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)


def load_housing_data():
    try:
        dataset = fetch_california_housing(as_frame=True)
        df = dataset.frame.copy()
        df.rename(columns={"MedHouseVal": "HousePrice"}, inplace=True)
        return df, "California Housing Dataset"
    except Exception:
        rng = np.random.default_rng(42)
        n = 2500
        med_inc = rng.gamma(shape=4.0, scale=1.0, size=n)
        house_age = rng.integers(1, 52, size=n).astype(float)
        ave_rooms = np.clip(med_inc * 0.9 + rng.normal(2.2, 0.9, n), 2.0, 12.0)
        ave_bedrms = np.clip(ave_rooms * 0.2 + rng.normal(0.1, 0.08, n), 0.5, 4.0)
        population = rng.integers(200, 4000, size=n).astype(float)
        ave_occup = np.clip(rng.normal(3.0, 0.7, n), 1.2, 7.0)
        latitude = rng.uniform(32.5, 41.5, n)
        longitude = rng.uniform(-124.0, -114.5, n)
        price = np.clip(
            0.42 * med_inc + 0.012 * house_age + 0.12 * ave_rooms - 0.15 * ave_occup
            + rng.normal(0.3, 0.45, n),
            0.4,
            5.0
        )
        df = pd.DataFrame({
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
            "HousePrice": price
        })
        return df, "Synthetic Housing Dataset (Offline Fallback)"


df, source_name = load_housing_data()
print(f"Loaded: {source_name} | Shape: {df.shape}")

df = df[df["HousePrice"] < 5.0].copy()
df["RoomsPerOccupant"] = df["AveRooms"] / np.clip(df["AveOccup"], 0.5, None)
df["BedroomsRatio"] = df["AveBedrms"] / np.clip(df["AveRooms"], 0.5, None)

corr = df.corr(numeric_only=True)
fig_corr, ax_c = plt.subplots(figsize=(9, 7))
im = ax_c.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax_c.set_xticks(range(len(corr.columns)))
ax_c.set_yticks(range(len(corr.columns)))
ax_c.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax_c.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax_c.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
fig_corr.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
ax_c.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"), dpi=600, bbox_inches="tight")
plt.close(fig_corr)

X = df.drop(columns=["HousePrice"])
y = df["HousePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
}

metrics_rows = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    metrics_rows.append({"Model": name, "RMSE ($100k)": rmse, "MAE ($100k)": mae, "R2 Score": r2})
    fitted_models[name] = (model, preds)

results_df = pd.DataFrame(metrics_rows).sort_values(by="RMSE ($100k)")
print("\nModel Evaluation Summary:\n")
print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_name = results_df.iloc[0]["Model"]
best_model, best_preds = fitted_models[best_name]
residuals = y_test - best_preds

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

axes[0].scatter(y_test, best_preds, alpha=0.35, s=18, color="#2563eb")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=1.8)
axes[0].set_title(f"{best_name}: Actual vs. Predicted")
axes[0].set_xlabel("Actual Price ($100k)")
axes[0].set_ylabel("Predicted Price ($100k)")
axes[0].grid(True, linestyle="--", alpha=0.4)

axes[1].hist(residuals, bins=35, color="#0d9488", alpha=0.82, edgecolor="white")
axes[1].axvline(0, color="black", linestyle="--", linewidth=1.3)
axes[1].set_title("Residual Error Distribution")
axes[1].set_xlabel("Actual - Predicted ($100k)")
axes[1].set_ylabel("Count")

rf_model = fitted_models["Random Forest"][0]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
axes[2].barh(importances.index, importances.values, color="#7c3aed", alpha=0.85)
axes[2].set_title("Random Forest Feature Importance")
axes[2].set_xlabel("Relative Importance")
axes[2].grid(True, axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "model_diagnostics.png"), dpi=600, bbox_inches="tight")
plt.show()
