import numpy as np
import pandas as pd

raw_data = {
    "House_ID": [101, 102, 103, 104, 105, 106],
    "City": ["Indore", "Delhi", "Indore", "Delhi", "Pune", "Pune"],
    "Area_sqft": [1200, 1500, np.nan, 1800, 1100, 1650],
    "Bedrooms": [2, 3, 2, 4, 2, 3],
    "Price_Lakhs": [55.0, 125.0, 62.0, 150.0, 72.0, 105.0]
}

df = pd.DataFrame(raw_data)

print("Missing values before cleaning:\n", df.isna().sum())

df["Area_sqft"] = df["Area_sqft"].fillna(df["Area_sqft"].median())
df["Price_per_sqft"] = (df["Price_Lakhs"] * 100000) / df["Area_sqft"]

print("\nCleaned DataFrame:\n", df)

big_homes = df[df["Area_sqft"] >= 1400]
print("\nHomes >= 1400 sqft:\n", big_homes[["House_ID", "City", "Area_sqft", "Price_Lakhs"]])

city_summary = df.groupby("City").agg(
    Avg_Price_Lakhs=("Price_Lakhs", "mean"),
    Avg_Area=("Area_sqft", "mean"),
    Listings=("House_ID", "count")
)
print("\nCity-wise Summary:\n", city_summary.round(2))
