"""
Run GP Regression (Algorithm 2.1) - Regression task and 2D toy dataset with covariance visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gp_regression import (
    squared_exponential_kernel,
    algorithm_2_1,
    predict_with_covariance
)


def run_regression_dataset():
    """Part 1: Use a dataset for regression, compute predictive mean and variance."""
    print("=" * 60)
    print("Part 1: GP Regression on Dataset")
    print("=" * 60)
    
    # Use Friedman1 dataset (regression)
    X, y = make_friedman1(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Hyperparameters
    sigma_n = 0.1
    length_scale = 1.0
    sigma_f = 1.0
    
    # Predict on test set
    means = []
    variances = []
    for i in range(len(X_test)):
        f_bar, V_f, log_ml = algorithm_2_1(
            X_train, y_train, X_test[i:i+1],
            squared_exponential_kernel,
            sigma_n=sigma_n,
            length_scale=length_scale,
            sigma_f=sigma_f
        )
        means.append(float(np.squeeze(f_bar)))
        variances.append(float(np.squeeze(V_f)))
    
    means = np.array(means)
    variances = np.array(variances)
    stds = np.sqrt(np.maximum(variances, 1e-10))
    
    # Metrics
    mse = np.mean((means - y_test)**2)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Mean absolute error: {np.mean(np.abs(means - y_test)):.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Predictions vs actual
    axes[0].scatter(y_test, means, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('GP Regression: Predicted vs Actual')
    axes[0].grid(True, alpha=0.3)
    
    # Predictions with uncertainty
    axes[1].errorbar(range(len(y_test)), means, yerr=2*stds, fmt='o', capsize=3, alpha=0.7)
    axes[1].plot(range(len(y_test)), y_test, 'rx', markersize=8, label='Actual')
    axes[1].set_xlabel('Test sample index')
    axes[1].set_ylabel('Target value')
    axes[1].set_title('Predictions with ±2σ uncertainty')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved regression_results.png")
    
    # Single test input example
    print("\nExample: Single test input prediction")
    x_star = X_test[0:1]
    f_bar, V_f, log_ml = algorithm_2_1(
        X_train, y_train, x_star,
        squared_exponential_kernel,
        sigma_n=sigma_n, length_scale=length_scale, sigma_f=sigma_f
    )
    print(f"  Test input: {x_star[0]}")
    print(f"  Predictive mean: {float(np.squeeze(f_bar)):.4f}")
    print(f"  Predictive variance: {float(np.squeeze(V_f)):.4f}")
    print(f"  Actual value: {y_test[0]:.4f}")


def run_toy_2d_dataset():
    """Part 2: Toy 2D dataset and display covariance for 2D test input."""
    print("\n" + "=" * 60)
    print("Part 2: Toy 2D Dataset - Covariance Visualization")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create toy 2D training data: y = sin(||x||) + noise
    n_train = 30
    X_train = np.random.uniform(-3, 3, (n_train, 2))
    y_train = np.sin(np.linalg.norm(X_train, axis=1)) + 0.1 * np.random.randn(n_train)
    
    # Two test points in 2D (for 2x2 covariance matrix)
    X_star = np.array([[1.0, 1.0], [-1.5, 0.5]])
    
    sigma_n = 0.1
    length_scale = 1.0
    sigma_f = 1.0
    
    f_bar, cov_f = predict_with_covariance(
        X_train, y_train, X_star,
        squared_exponential_kernel,
        sigma_n=sigma_n, length_scale=length_scale, sigma_f=sigma_f
    )
    
    print("\n2D Test inputs:")
    print(f"  x*_1 = {X_star[0]}")
    print(f"  x*_2 = {X_star[1]}")
    print("\nPredictive means:")
    print(f"  E[f(x*_1)] = {f_bar[0]:.4f}")
    print(f"  E[f(x*_2)] = {f_bar[1]:.4f}")
    print("\nPredictive covariance matrix (2x2):")
    print("  cov(f(x*_1), f(x*_2)) =")
    print(cov_f)
    print(f"\n  Variance at x*_1: {cov_f[0,0]:.4f}")
    print(f"  Variance at x*_2: {cov_f[1,1]:.4f}")
    print(f"  Covariance between x*_1 and x*_2: {cov_f[0,1]:.4f}")
    
    # Visualization
    fig = plt.figure(figsize=(14, 5))
    
    # Subplot 1: Training data and posterior mean surface
    ax1 = fig.add_subplot(131)
    
    # Create grid for contour
    xx = np.linspace(-4, 4, 40)
    yy = np.linspace(-4, 4, 40)
    XX, YY = np.meshgrid(xx, yy)
    X_grid = np.c_[XX.ravel(), YY.ravel()]
    
    f_grid, _ = predict_with_covariance(
        X_train, y_train, X_grid,
        squared_exponential_kernel,
        sigma_n=sigma_n, length_scale=length_scale, sigma_f=sigma_f
    )
    Z = f_grid.reshape(XX.shape)
    
    contour = ax1.contourf(XX, YY, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='Predictive mean')
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=80, edgecolors='black', 
                cmap='coolwarm', linewidth=1)
    ax1.scatter(X_star[0, 0], X_star[0, 1], c='red', s=200, marker='*', 
                edgecolors='black', linewidth=2, label='Test point 1')
    ax1.scatter(X_star[1, 0], X_star[1, 1], c='blue', s=200, marker='*', 
                edgecolors='black', linewidth=2, label='Test point 2')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('GP Posterior Mean (2D Toy Data)')
    ax1.legend()
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    
    # Subplot 2: Covariance matrix heatmap
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(cov_f, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["x*₁", "x*₂"])
    ax2.set_yticklabels(["x*₁", "x*₂"])
    ax2.set_title('Predictive Covariance Matrix\ncov(f(x*₁), f(x*₂))')
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{cov_f[i,j]:.3f}', ha='center', va='center', fontsize=12)
    plt.colorbar(im, ax=ax2, label='Covariance')
    
    # Subplot 3: Posterior covariance as function of x' (like Fig 2.4b in R&W)
    ax3 = fig.add_subplot(133)
    
    # Fix one test point, vary the other along a line
    x_prime_fixed = X_star[0]  # Fixed at test point 1
    n_pts = 100
    x_line = np.linspace(-4, 4, n_pts)
    X_line = np.column_stack([x_line, np.ones(n_pts) * x_prime_fixed[1]])
    
    _, cov_line = predict_with_covariance(
        X_train, y_train, np.vstack([np.tile(x_prime_fixed, (n_pts, 1)), X_line]),
        squared_exponential_kernel,
        sigma_n=sigma_n, length_scale=length_scale, sigma_f=sigma_f
    )
    
    # Compute cov(f(x'), f(x*)) for x' along a line
    cov_vals = np.zeros(n_pts)
    for i in range(n_pts):
        X_pair = np.array([x_prime_fixed, X_line[i]])
        try:
            _, c = predict_with_covariance(
                X_train, y_train, X_pair,
                squared_exponential_kernel,
                sigma_n=sigma_n, length_scale=length_scale, sigma_f=sigma_f
            )
            cov_vals[i] = c[0, 1]
        except (np.linalg.LinAlgError, FloatingPointError):
            cov_vals[i] = np.nan
    
    valid = ~np.isnan(cov_vals)
    ax3.plot(x_line[valid], cov_vals[valid], 'b-', lw=2)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.axvline(x_prime_fixed[0], color='red', linestyle=':', alpha=0.7)
    ax3.scatter(X_train[:, 0], np.zeros(n_train), c='green', s=30, alpha=0.5, label='Train x₁')
    ax3.set_xlabel("x'₁ (varying test point)")
    ax3.set_ylabel("cov(f(x'), f(x*))")
    ax3.set_title("Posterior covariance (x* fixed at red line)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('toy_2d_covariance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved toy_2d_covariance.png")


if __name__ == "__main__":
    run_regression_dataset()
    run_toy_2d_dataset()
    print("\n" + "=" * 60)
    print("GP Regression complete!")
    print("=" * 60)
