# End-to-End Housing Price Prediction Pipeline

## Objective
Predict median residential housing prices from neighborhood-level socioeconomic and structural features while demonstrating a leakage-free, reproducible Machine Learning workflow.

## Step-by-Step Pipeline
1. **Exploratory Data Analysis (EDA):** Inspect feature distributions, filter out the artificial `$500,000` ceiling cap in California Housing records, and compute a full correlation heatmap to see which variables drive price (Median Income dominates).
2. **Feature Engineering:** Create domain-grounded ratios that raw counts conceal:
   - `RoomsPerOccupant` = Average Rooms / Average Occupancy
   - `BedroomsRatio` = Average Bedrooms / Average Rooms
3. **Leakage-Free Preprocessing:** Perform the 80/20 Train-Test split *before* scaling, wrapping `StandardScaler` inside a scikit-learn `Pipeline` so test statistics never leak into training.
4. **Model Comparison:** Benchmark **Linear Regression** (interpretable baseline) against **Ridge Regression** (L2 regularization) and **Random Forest Regressor** (non-linear ensemble) using RMSE, MAE, and $R^2$.
5. **Diagnostics (600 DPI Figures):**
   - `figures/correlation_heatmap.png` — Correlation structure across features and target.
   - `figures/model_diagnostics.png` — Actual vs. Predicted parity plot, Residual Error distribution, and Random Forest Feature Importances.

## How to Run
```bash
cd 09_project_housing_price_prediction
python housing_price_pipeline.py
```

*(Automatically downloads the California Housing dataset via scikit-learn, and falls back gracefully to a realistic synthetic housing dataset if executed offline.)*
