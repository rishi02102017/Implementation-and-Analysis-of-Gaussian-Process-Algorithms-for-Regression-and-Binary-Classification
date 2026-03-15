"""
Gaussian Process Regression - Algorithm 2.1 from Rasmussen & Williams (2006)
Gaussian Processes for Machine Learning
"""

import numpy as np
from typing import Tuple, Optional


def squared_exponential_kernel(X1: np.ndarray, X2: np.ndarray, 
                               length_scale: float = 1.0, 
                               sigma_f: float = 1.0) -> np.ndarray:
    """
    Squared exponential (RBF) covariance function.
    k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * length_scale^2))
    """
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    
    sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
              np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 * sq_dist / length_scale**2)


def algorithm_2_1(X: np.ndarray, y: np.ndarray, x_star: np.ndarray,
                   kernel_fn, sigma_n: float = 0.1,
                   length_scale: float = 1.0, sigma_f: float = 1.0
                   ) -> Tuple[float, float, float]:
    """
    Algorithm 2.1: Predictions and log marginal likelihood for GP regression.
    
    Input:
        X: Training inputs (n x D)
        y: Training targets (n,)
        x_star: Test input(s) - can be (1, D) or (m, D) for m test points
        kernel_fn: Kernel function
        sigma_n: Noise level (std dev of observation noise)
    
    Returns:
        f_bar: Predictive mean
        V_f: Predictive variance (or covariance matrix for multiple test points)
        log_marginal_likelihood: log p(y|X)
    """
    # Ensure proper shapes
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.ravel()
    if x_star.ndim == 1:
        x_star = x_star.reshape(1, -1)
    
    n = X.shape[0]
    
    # Line 2: L := cholesky(K + sigma_n^2 * I)
    K = kernel_fn(X, X, length_scale=length_scale, sigma_f=sigma_f)
    K_noise = K + (sigma_n**2) * np.eye(n) + 1e-8 * np.eye(n)  # Small jitter for stability
    L = np.linalg.cholesky(K_noise)
    
    # Line 3: alpha := L' \ (L \ y)  [solve L @ L.T @ alpha = y]
    # First solve L @ temp = y, then L.T @ alpha = temp
    temp = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, temp)
    
    # Line 4: f_bar := k_star^T @ alpha (predictive mean)
    k_star = kernel_fn(X, x_star, length_scale=length_scale, sigma_f=sigma_f)
    f_bar = k_star.T @ alpha
    
    # Line 5-6: v := L \ k_star, V[f*] := k(x*, x*) - v^T @ v (predictive variance)
    v = np.linalg.solve(L, k_star)
    k_star_star = kernel_fn(x_star, x_star, length_scale=length_scale, sigma_f=sigma_f)
    V_f = k_star_star - v.T @ v
    
    # Line 7: log p(y|X)
    log_marginal = -0.5 * (y.T @ alpha) - np.sum(np.log(np.diag(L))) - (n/2) * np.log(2*np.pi)
    
    return f_bar, V_f, log_marginal


def predict_with_covariance(X: np.ndarray, y: np.ndarray, X_star: np.ndarray,
                            kernel_fn, sigma_n: float = 0.1,
                            length_scale: float = 1.0, sigma_f: float = 1.0
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute predictive mean and full covariance matrix for multiple test points.
    For 2 test points, returns 2x2 covariance matrix showing cov(f(x*_1), f(x*_2)).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X_star.ndim == 1:
        X_star = X_star.reshape(-1, 1)
    
    n = X.shape[0]
    m = X_star.shape[0]
    
    K = kernel_fn(X, X, length_scale=length_scale, sigma_f=sigma_f)
    K_noise = K + (sigma_n**2) * np.eye(n) + 1e-8 * np.eye(n)  # Small jitter for stability
    L = np.linalg.cholesky(K_noise)
    
    temp = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, temp)
    
    k_star = kernel_fn(X, X_star, length_scale=length_scale, sigma_f=sigma_f)
    f_bar = k_star.T @ alpha
    
    v = np.linalg.solve(L, k_star)  # (n x m)
    k_star_star = kernel_fn(X_star, X_star, length_scale=length_scale, sigma_f=sigma_f)
    cov_f = k_star_star - v.T @ v
    
    return f_bar.ravel(), cov_f
