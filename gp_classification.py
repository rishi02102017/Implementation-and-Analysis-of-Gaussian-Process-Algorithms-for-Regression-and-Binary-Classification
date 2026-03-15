"""
Gaussian Process Binary Classification - Algorithms 3.1 and 3.2 from Rasmussen & Williams (2006)
Gaussian Processes for Machine Learning
Uses Laplace approximation with Newton's method for mode-finding.
"""

import numpy as np
from typing import Tuple, Callable


def squared_exponential_kernel(X1: np.ndarray, X2: np.ndarray,
                               length_scale: float = 1.0,
                               sigma_f: float = 1.0) -> np.ndarray:
    """Squared exponential (RBF) covariance function."""
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
              np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 * sq_dist / length_scale**2)


# --- Logistic likelihood (sigmoid) ---
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function: 1 / (1 + exp(-z))"""
    z = np.clip(z, -500, 500)  # Avoid overflow
    return 1.0 / (1.0 + np.exp(-z))


# Gradient: d/df log(sigma(y_i*f_i)) = y_i * sigma(-y_i*f_i) = y_i * (1 - sigma(y_i*f_i))
# When y_i=+1: 1 - sigma(f_i). When y_i=-1: -sigma(f_i) = -(1 - sigma(-f_i)).
# So: y_i * (1 - sigma(y_i*f_i)) = y_i * sigma(-y_i*f_i)
def grad_log_likelihood_logistic_correct(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient: d/df_i log p(y_i|f_i) = y_i * sigma(-y_i*f_i) = y_i * (1 - sigma(y_i*f_i))"""
    z = y * f
    return y * (1 - sigmoid(z))


def hessian_log_likelihood_logistic_correct(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    """W_ii = -d^2/df_i^2 log p(y_i|f_i) = sigma(y_i*f_i) * (1 - sigma(y_i*f_i)) = pi_i*(1-pi_i)"""
    z = y * f
    pi = sigmoid(z)
    W = np.clip(pi * (1 - pi), 1e-6, 1.0)  # Clip for numerical stability
    return np.diag(W)


def algorithm_3_1(K: np.ndarray, y: np.ndarray, max_iter: int = 100,
                  tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    Algorithm 3.1: Mode-finding for binary Laplace GPC using Newton's method.
    
    Input:
        K: Covariance matrix (n x n), should include jitter for stability
        y: Targets in {-1, +1}
    
    Returns:
        f_hat: Posterior mode (n,)
        log_q: Approximate log marginal likelihood
    """
    n = len(y)
    f = np.zeros(n)
    
    for _ in range(max_iter):
        # W := -nabla nabla log p(y|f)
        W = hessian_log_likelihood_logistic_correct(f, y)
        W_sqrt = np.sqrt(np.diag(W))
        
        # L := cholesky(I + W^(1/2) K W^(1/2))
        B = np.eye(n) + np.outer(W_sqrt, W_sqrt) * K  # W^(1/2) K W^(1/2)
        # Actually: (W^1/2)_ij = sqrt(W_ii) delta_ij, so W^1/2 K W^1/2 has (i,j) = sqrt(W_ii)*K_ij*sqrt(W_jj)
        W_sqrt_diag = np.diag(W_sqrt)
        B = np.eye(n) + W_sqrt_diag @ K @ W_sqrt_diag
        
        try:
            L = np.linalg.cholesky(B)
        except np.linalg.LinAlgError:
            # Add jitter if needed
            B += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(B)
        
        # b := W*f + nabla log p(y|f)
        grad_log = grad_log_likelihood_logistic_correct(f, y)
        b = W @ f + grad_log
        
        # a := b - W^(1/2) L' \ (L \ (W^(1/2) K b))
        Kb = K @ b
        W_sqrt_Kb = W_sqrt_diag @ Kb
        sol1 = np.linalg.solve(L, W_sqrt_Kb)
        sol2 = np.linalg.solve(L.T, sol1)
        a = b - W_sqrt_diag @ sol2
        
        # f_new := K @ a
        f_new = K @ a
        
        # Check convergence
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    
    # Log marginal likelihood
    log_q = -0.5 * (a @ f) + np.sum(np.log(sigmoid(y * f))) - np.sum(np.log(np.diag(L)))
    
    return f, log_q


def averaged_predictive_probability(f_mean: float, f_var: float, 
                                    use_probit: bool = False) -> float:
    """
    Compute pi_bar = integral sigma(z) N(z|f_mean, f_var) dz (eq. 3.25).
    For probit (cumulative Gaussian), this is analytic.
    For logistic, we use the approximation: pi_bar ≈ lambda(kappa * f_mean) 
    with kappa^2 = (1 + pi*V/8)^(-1) (MacKay 1992).
    """
    if use_probit:
        from scipy.stats import norm
        # For probit: analytic. pi_bar = Phi(f_mean / sqrt(1 + f_var))
        return float(norm.cdf(f_mean / np.sqrt(1 + f_var)))
    else:
        # MacKay approximation for logistic
        kappa_sq = 1.0 / (1.0 + np.pi * f_var / 8)
        kappa = np.sqrt(kappa_sq)
        return float(sigmoid(kappa * f_mean))


def algorithm_3_2(f_hat: np.ndarray, X: np.ndarray, y: np.ndarray, x_star: np.ndarray,
                 K: np.ndarray, kernel_fn, length_scale: float = 1.0, sigma_f: float = 1.0,
                 use_probit: bool = True) -> float:
    """
    Algorithm 3.2: Predictions for binary Laplace GPC.
    
    Input:
        f_hat: Posterior mode from Algorithm 3.1
        X: Training inputs
        y: Targets in {-1, +1}
        x_star: Test input(s)
        K: Training covariance matrix
    
    Returns:
        pi_bar: Averaged predictive class probability for class +1
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if x_star.ndim == 1:
        x_star = x_star.reshape(1, -1)
    
    # W at mode
    W = hessian_log_likelihood_logistic_correct(f_hat, y)
    W_sqrt = np.sqrt(np.diag(W))
    W_sqrt_diag = np.diag(W_sqrt)
    
    # L := cholesky(I + W^1/2 K W^1/2)
    B = np.eye(len(y)) + W_sqrt_diag @ K @ W_sqrt_diag
    L = np.linalg.cholesky(B)
    
    # k_star
    k_star = kernel_fn(X, x_star, length_scale=length_scale, sigma_f=sigma_f)
    
    # f_bar = k_star^T @ nabla log p(y|f_hat)
    grad_log = grad_log_likelihood_logistic_correct(f_hat, y)
    f_bar = (k_star.T @ grad_log).ravel()
    
    # v := L \ (W^1/2 k_star)
    v = np.linalg.solve(L, W_sqrt_diag @ k_star)
    
    # V[f*] = k(x*,x*) - v^T v
    k_star_star = kernel_fn(x_star, x_star, length_scale=length_scale, sigma_f=sigma_f)
    f_var = k_star_star - v.T @ v
    f_var = np.diag(f_var) if f_var.ndim == 2 else np.array([float(f_var)])
    
    # pi_bar = integral sigma(z) N(z|f_bar, V[f*]) dz
    pi_bars = []
    for i in range(len(f_bar)):
        vb = float(np.maximum(f_var[i], 1e-10))
        pi_bars.append(averaged_predictive_probability(f_bar[i], vb, use_probit=use_probit))
    
    return np.array(pi_bars) if len(pi_bars) > 1 else pi_bars[0]


def predict_gpc(X: np.ndarray, y: np.ndarray, X_star: np.ndarray,
                kernel_fn, length_scale: float = 1.0, sigma_f: float = 1.0,
                use_probit: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full GP classification: run Algorithm 3.1 then 3.2 for each test point.
    Returns (pi_bar, f_bar, f_var) for uncertainty analysis.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X_star.ndim == 1:
        X_star = X_star.reshape(-1, 1)
    
    K = kernel_fn(X, X, length_scale=length_scale, sigma_f=sigma_f)
    K = K + 1e-6 * np.eye(len(X))  # Jitter for numerical stability
    f_hat, log_q = algorithm_3_1(K, y)
    
    # Compute predictions for all test points
    W = hessian_log_likelihood_logistic_correct(f_hat, y)
    W_sqrt = np.sqrt(np.diag(W))
    W_sqrt_diag = np.diag(W_sqrt)
    B = np.eye(len(y)) + W_sqrt_diag @ K @ W_sqrt_diag
    L = np.linalg.cholesky(B)
    
    k_star = kernel_fn(X, X_star, length_scale=length_scale, sigma_f=sigma_f)
    grad_log = grad_log_likelihood_logistic_correct(f_hat, y)
    f_bar = (k_star.T @ grad_log).ravel()
    
    v = np.linalg.solve(L, W_sqrt_diag @ k_star)
    k_star_star = kernel_fn(X_star, X_star, length_scale=length_scale, sigma_f=sigma_f)
    f_var = np.diag(k_star_star - v.T @ v)
    f_var = np.maximum(f_var, 1e-10)
    
    pi_bars = np.array([averaged_predictive_probability(f_bar[i], f_var[i], use_probit) 
                        for i in range(len(f_bar))])
    
    return pi_bars, f_bar, f_var
