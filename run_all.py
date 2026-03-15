#!/usr/bin/env python3
"""Run all AML Assignment 2 experiments."""

import run_regression
import run_classification

if __name__ == "__main__":
    print("Running GP Regression...")
    run_regression.run_regression_dataset()
    run_regression.run_toy_2d_dataset()

    print("\nRunning GP Classification...")
    run_classification.run_binary_classification()
    run_classification.run_toy_2d_classification()

    print("\nAll experiments complete. See report.pdf for details.")
