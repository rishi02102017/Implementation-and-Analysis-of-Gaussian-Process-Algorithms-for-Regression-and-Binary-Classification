"""
Save test data to CSV files for submission.
Assignment requires: "Along with the source code and test data you also need to submit"
"""

import numpy as np
from sklearn.datasets import make_friedman1, make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create data directory
import os
os.makedirs("data", exist_ok=True)

# --- Regression: Friedman1 ---
X, y = make_friedman1(n_samples=100, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.savetxt("data/regression_train_X.csv", X_train, delimiter=",")
np.savetxt("data/regression_train_y.csv", y_train, delimiter=",")
np.savetxt("data/regression_test_X.csv", X_test, delimiter=",")
np.savetxt("data/regression_test_y.csv", y_test, delimiter=",")

# --- Regression: 2D toy ---
np.random.seed(42)
n_train = 30
X_toy = np.random.uniform(-3, 3, (n_train, 2))
y_toy = np.sin(np.linalg.norm(X_toy, axis=1)) + 0.1 * np.random.randn(n_train)
np.savetxt("data/toy_2d_train_X.csv", X_toy, delimiter=",")
np.savetxt("data/toy_2d_train_y.csv", y_toy, delimiter=",")
# 2 test points for covariance
X_toy_test = np.array([[1.0, 1.0], [-1.5, 0.5]])
np.savetxt("data/toy_2d_test_X.csv", X_toy_test, delimiter=",")

# --- Classification: Breast cancer ---
data = load_breast_cancer()
X, y_raw = data.data, data.target
y = 2 * y_raw - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_raw)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Use subset for GP (O(n^3))
np.random.seed(42)
idx = np.random.choice(len(X_train), min(200, len(X_train)), replace=False)
X_train_sub = X_train[idx]
y_train_sub = y_train[idx]
np.savetxt("data/classification_train_X.csv", X_train_sub, delimiter=",")
np.savetxt("data/classification_train_y.csv", y_train_sub, delimiter=",")
np.savetxt("data/classification_test_X.csv", X_test, delimiter=",")
np.savetxt("data/classification_test_y.csv", y_test, delimiter=",")

# --- Classification: 2D toy ---
X, y_raw = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.2, random_state=42)
y = 2 * (y_raw == 0).astype(int) - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
np.savetxt("data/classification_toy_train_X.csv", X_train, delimiter=",")
np.savetxt("data/classification_toy_train_y.csv", y_train, delimiter=",")
np.savetxt("data/classification_toy_test_X.csv", X_test, delimiter=",")
np.savetxt("data/classification_toy_test_y.csv", y_test, delimiter=",")

print("Test data saved to data/")
print("  - regression_train_X.csv, regression_train_y.csv")
print("  - regression_test_X.csv, regression_test_y.csv")
print("  - toy_2d_train_X.csv, toy_2d_train_y.csv, toy_2d_test_X.csv")
print("  - classification_train_X.csv, classification_train_y.csv")
print("  - classification_test_X.csv, classification_test_y.csv")
print("  - classification_toy_* (2D toy for classification)")
