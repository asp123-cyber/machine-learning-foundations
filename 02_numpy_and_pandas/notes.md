# 02. Python for Data Analysis: NumPy & Pandas

## NumPy (Part 1): Why Not Just Use Normal Python Lists?
A standard Python list can hold `[42, "hello", True, 3.14]` all at once. Because every item can be a different type, Python has to check each item's type one by one during math operations. That makes `for` loops painfully slow on large datasets.

**NumPy (`ndarray`)** fixes this by packing numbers of the *same* data type side-by-side in memory and executing optimized C routines across the entire block at once.

- **Vectorization:** Instead of looping through a list to multiply every number by 2, write `arr * 2`. It is cleaner, shorter, and often 50x–100x faster.
- **Array Creation Essentials:** `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, and `np.random`.

## NumPy (Part 2): Shapes, Indexing, Axes & Broadcasting
- **Shape & Reshaping:** Every array has a `.shape` tuple, e.g., `(100, 4)` for 100 rows and 4 columns. Use `.reshape()` to reorganize dimensions without copying data.
- **Boolean Masking:** Filter numbers in a single line: `prices[prices > 300]` immediately returns only values above 300.
- **Understanding `axis`:**
  - `axis=0` collapses **down the rows** (giving you one result per column).
  - `axis=1` collapses **across the columns** (giving you one result per row).
- **Broadcasting:** If you add a 1D array of 3 numbers to a `(100, 3)` matrix, NumPy automatically stretches that 1D row across all 100 rows without duplicating memory.
- **Matrix Multiplication (`@`):** Computes dot products and linear transformations directly (`X @ weights`).

## Pandas: Excel on Steroids
NumPy arrays only have numbered positions (`row 0, col 2`). Real-world tables have column names like `"Area_sqft"`, `"City"`, and `"Price"`, mixed with text, dates, and missing values (`NaN`).

Pandas introduces two core structures:
- **Series:** A single labeled 1D column.
- **DataFrame:** A full 2D table made of multiple Series sharing an index.

### The First 5 Commands to Run on Any New Dataset
1. `df.head()` — Peek at the first 5 rows so you know what the data actually looks like.
2. `df.shape` — Check how many rows and columns you have.
3. `df.info()` — Check whether numeric columns were accidentally read as text and see missing counts.
4. `df.describe()` — Get a quick statistical health check (mean, min, max, quartiles) to catch impossible values like `Age = -5`.
5. `df.isna().sum()` — Count missing values per column so you can decide whether to fill (`fillna`) or drop (`dropna`) them.

### Everyday Pandas Operations
- **Selecting Columns:** `df["City"]` (Series) or `df[["City", "Price_Lakhs"]]` (DataFrame).
- **Filtering Rows:** `df[df["Area_sqft"] >= 1400]`.
- **Label vs. Position Selection:** `.loc[row_label, col_name]` uses names; `.iloc[row_idx, col_idx]` uses integer positions.
- **Grouping & Aggregation:** `df.groupby("City")["Price_Lakhs"].mean()` splits the table by city, computes each average, and combines the results cleanly.
