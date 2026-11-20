# 04. Linear Regression (Part 1 & 2)

## Part 1: The Big Idea & Cost Function
Imagine plotting house sizes on the $x$-axis and their selling prices on the $y$-axis. The dots drift upward. **Linear Regression** finds the single straight line that passes as closely as possible through the cloud of points:

$$\hat{y} = w x + b$$

When you have multiple features (size, bedrooms, age, location), it becomes **Multiple Linear Regression**:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

### What Does "Best Line" Actually Mean?
For every data point, the vertical gap between the real value $y_i$ and our line's guess $\hat{y}_i$ is called the **residual** ($e_i = y_i - \hat{y}_i$).

We don't just add the gaps together—otherwise a $+10$ error and a $-10$ error would cancel out and pretend the error is $0$. Instead, we **square every error** (which also punishes large mistakes more heavily) and take the average. That gives us the **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

The best-fit line is the choice of $w$ and $b$ that minimizes MSE.

## Part 2: Finding the Best Line & Evaluating It
There are two classic ways to minimize MSE:

1. **Ordinary Least Squares (The Normal Equation):** Solves for the exact optimal weights in one shot using linear algebra:
   $$\theta = (X^T X)^{-1} X^T y$$
   Exact and fast on small-to-medium datasets, but matrix inversion becomes expensive when you have tens of thousands of features.
2. **Gradient Descent (Walking Downhill Blindfolded):** Start with a random guess for $w$ and $b$. Compute the slope (gradient) of MSE with respect to $w$ and $b$, step slightly downhill using a **learning rate ($\alpha$)**, and repeat until MSE stops dropping:
   $$w \leftarrow w - \alpha \frac{\partial \text{MSE}}{\partial w}, \qquad b \leftarrow b - \alpha \frac{\partial \text{MSE}}{\partial b}$$
   - Learning rate too tiny $\rightarrow$ crawls slowly toward the minimum.
   - Learning rate too large $\rightarrow$ overshoots the valley and diverges.

### How Do We Grade a Regression Model?
- **MAE (Mean Absolute Error):** Average absolute error in original units; robust to extreme outliers.
- **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}}$. Gives the typical error in the *same units* as your target (e.g., "off by $\pm 4.2$ Lakhs").
- **$R^2$ Score (Coefficient of Determination):** Ranges up to $1.0$. An $R^2$ of $0.85$ means the model explains 85% of the variance in $y$.
