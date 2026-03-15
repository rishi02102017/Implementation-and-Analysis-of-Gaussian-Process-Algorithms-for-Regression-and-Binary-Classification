"""
Run GP Binary Classification (Algorithms 3.1 & 3.2) with uncertainty analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gp_classification import (
    squared_exponential_kernel,
    algorithm_3_1,
    predict_gpc
)


def run_binary_classification():
    """Run GP classification on a binary dataset with uncertainty explanation."""
    print("=" * 60)
    print("GP Binary Classification (Algorithms 3.1 & 3.2)")
    print("=" * 60)
    
    # Use breast cancer dataset (binary classification)
    data = load_breast_cancer()
    X, y_raw = data.data, data.target
    # Convert to {-1, +1}
    y = 2 * y_raw - 1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y_raw
    )
    
    # Scale and use subset for speed (GP is O(n^3))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Use subset if too large
    max_train = 200
    if len(X_train) > max_train:
        idx = np.random.RandomState(42).choice(len(X_train), max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    length_scale = 1.0
    sigma_f = 1.0
    
    print("\nTraining GP classifier (Laplace approximation, Newton method)...")
    pi_bar, f_bar, f_var = predict_gpc(
        X_train, y_train, X_test,
        squared_exponential_kernel,
        length_scale=length_scale, sigma_f=sigma_f,
        use_probit=True
    )
    
    # Metrics
    y_pred = 2 * (pi_bar >= 0.5).astype(int) - 1
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Uncertainty analysis
    print("\n" + "-" * 40)
    print("Uncertainty in Predictions")
    print("-" * 40)
    
    # High uncertainty: predictions near 0.5
    uncertain_mask = (pi_bar > 0.4) & (pi_bar < 0.6)
    confident_mask = (pi_bar < 0.2) | (pi_bar > 0.8)
    
    print(f"\n1. Prediction confidence:")
    print(f"   - Uncertain predictions (0.4 < p < 0.6): {np.sum(uncertain_mask)} / {len(y_test)}")
    print(f"   - Confident predictions (p<0.2 or p>0.8): {np.sum(confident_mask)} / {len(y_test)}")
    
    if np.sum(uncertain_mask) > 0:
        print(f"\n2. Latent variance for uncertain vs confident:")
        print(f"   - Mean f_var (uncertain): {np.mean(f_var[uncertain_mask]):.4f}")
        print(f"   - Mean f_var (confident): {np.mean(f_var[confident_mask]):.4f}")
        print("   (Higher variance → more uncertainty)")
    
    print("\n3. Why uncertainty occurs:")
    print("   - Near decision boundary: f_bar ≈ 0, so π̄ ≈ 0.5")
    print("   - Far from training data: high f_var (prior dominates)")
    print("   - Conflicting evidence: overlapping classes increase variance")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Predictions vs actual (colored by uncertainty)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y_test, pi_bar, c=f_var, cmap='viridis', alpha=0.7)
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Actual class (-1 or +1)')
    ax1.set_ylabel('Predictive probability π̄ (class +1)')
    ax1.set_title('Predictions colored by latent variance (uncertainty)')
    plt.colorbar(scatter, ax=ax1, label='Latent variance V[f*]')
    
    # Plot 2: Calibration - predicted prob vs fraction positive
    ax2 = axes[0, 1]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    frac_pos = []
    for i in range(len(bins)-1):
        mask = (pi_bar >= bins[i]) & (pi_bar < bins[i+1])
        if np.sum(mask) > 0:
            frac_pos.append(np.mean(y_test[mask] == 1))
        else:
            frac_pos.append(np.nan)
    ax2.plot(bin_centers, frac_pos, 'o-', label='Calibration')
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Fraction of positive class')
    ax2.set_title('Calibration curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: f_var vs |f_bar| (uncertainty vs distance from boundary)
    ax3 = axes[1, 0]
    ax3.scatter(np.abs(f_bar), f_var, c=pi_bar, cmap='RdYlBu_r', alpha=0.7)
    ax3.set_xlabel('|f̄*| (distance from decision boundary)')
    ax3.set_ylabel('V[f*] (latent variance)')
    ax3.set_title('Uncertainty vs distance from boundary')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 2D projection if we reduce dimensions (PCA for viz)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_test_2d = pca.fit_transform(X_test)
    
    ax4 = axes[1, 1]
    scatter = ax4.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=pi_bar, 
                          cmap='RdYlBu_r', s=f_var*50+10, alpha=0.7)
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('Test points (color=π̄, size∝uncertainty)')
    plt.colorbar(scatter, ax=ax4, label='π̄')
    
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved classification_results.png")


def run_toy_2d_classification():
    """Run on a simple 2D toy dataset for clearer visualization."""
    print("\n" + "=" * 60)
    print("Toy 2D Binary Classification")
    print("=" * 60)
    
    np.random.seed(42)
    X, y_raw = make_blobs(n_samples=100, centers=2, n_features=2, 
                          cluster_std=1.2, random_state=42)
    y = 2 * (y_raw == 0).astype(int) - 1  # Class 0 -> -1, Class 1 -> +1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    pi_bar, f_bar, f_var = predict_gpc(
        X_train, y_train, X_test,
        squared_exponential_kernel,
        length_scale=1.0, sigma_f=1.0,
        use_probit=True
    )
    
    # Decision boundary and uncertainty visualization
    xx = np.linspace(X_train[:, 0].min()-0.5, X_train[:, 0].max()+0.5, 80)
    yy = np.linspace(X_train[:, 1].min()-0.5, X_train[:, 1].max()+0.5, 80)
    XX, YY = np.meshgrid(xx, yy)
    X_grid = np.c_[XX.ravel(), YY.ravel()]
    
    pi_grid, f_grid, var_grid = predict_gpc(
        X_train, y_train, X_grid,
        squared_exponential_kernel,
        length_scale=1.0, sigma_f=1.0,
        use_probit=True
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Predictive probability
    ax1 = axes[0]
    Z = pi_grid.reshape(XX.shape)
    contour = ax1.contourf(XX, YY, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
    ax1.contour(XX, YY, Z, levels=[0.5], colors='black', linewidths=2)
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu_r', 
                s=80, edgecolors='black', linewidth=1)
    ax1.set_title('Predictive probability π̄(x)')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    plt.colorbar(contour, ax=ax1, label='π̄')
    
    # Right: Uncertainty (latent variance)
    ax2 = axes[1]
    Z_var = var_grid.reshape(XX.shape)
    contour2 = ax2.contourf(XX, YY, Z_var, levels=20, cmap='viridis', alpha=0.8)
    ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu_r', 
                s=80, edgecolors='black', linewidth=1)
    ax2.set_title('Predictive uncertainty V[f*]')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    plt.colorbar(contour2, ax=ax2, label='V[f*]')
    
    plt.tight_layout()
    plt.savefig('classification_toy_2d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved classification_toy_2d.png")


if __name__ == "__main__":
    run_binary_classification()
    run_toy_2d_classification()
    print("\n" + "=" * 60)
    print("GP Classification complete!")
    print("=" * 60)
