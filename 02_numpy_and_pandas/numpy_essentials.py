import numpy as np

np.random.seed(42)

prices = np.array([250, 310, 400, 290, 520], dtype=float)
discounted = prices * 0.90
normalized = (prices - prices.mean()) / prices.std()
high_prices = prices[prices > 300]

print("Original prices:", prices)
print("10% discounted:", discounted)
print("Z-score normalized:", np.round(normalized, 3))
print("Prices > 300:", high_prices)

scores = np.array([
    [85, 90, 78],
    [92, 88, 95],
    [70, 75, 80],
    [88, 84, 91]
])

subject_means = scores.mean(axis=0)
student_means = scores.mean(axis=1)
curved_scores = scores + np.array([5, 0, 2])

print("\nMatrix shape:", scores.shape)
print("Average per subject (axis=0):", subject_means)
print("Average per student (axis=1):", np.round(student_means, 2))
print("Broadcasted curved scores:\n", curved_scores)

weights = np.array([0.3, 0.3, 0.4])
final_grades = scores @ weights
print("Weighted final grades (dot product):", np.round(final_grades, 2))
