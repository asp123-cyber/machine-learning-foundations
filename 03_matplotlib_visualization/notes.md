# 03. Python for Data Visualization: Matplotlib (Part 1 & 2)

## Why Plot Before Modeling?
Summary statistics can hide critical patterns. Two completely different datasets can share the *exact same* mean, variance, and correlation—one might follow a clean straight line while the other curves sharply or contains a single extreme outlier dragging the fit sideways. Visualizing your data lets you spot outliers, skew, and clusters immediately.

## Part 1: Picking the Right Chart
- **Line Plot (`ax.plot`):** Tracks how a value changes across an ordered sequence (time, temperature, training epochs).
- **Scatter Plot (`ax.scatter`):** Checks the relationship between two continuous numeric features (e.g., Area vs. House Price).
- **Histogram (`ax.hist`):** Reveals the shape and spread of a single numeric column (symmetric bell curve vs. right-skewed tail).
- **Bar Chart (`ax.bar`):** Compares totals or averages across distinct categories (e.g., validation scores across models).

## Part 2: Subplots, Styling & Publication Exports
Instead of relying on global `plt.plot()` calls, use the object-oriented `fig, ax = plt.subplots()` interface:
- `fig` represents the entire canvas (the page).
- `ax` represents an individual coordinate panel on that page.

### Quick Checklist for Clean Scientific Plots
- Always label both axes with units (e.g., `Area (sq. ft.)`).
- Use semi-transparent markers (`alpha=0.7`) on dense scatter plots so overlapping points stay visible.
- Add light dashed gridlines (`linestyle="--", alpha=0.4`) to make values easier to read without distracting from the data.
- Call `plt.tight_layout()` before saving so titles and tick labels never overlap.
- Export high-resolution figures with `dpi=600` and `bbox_inches="tight"`.
