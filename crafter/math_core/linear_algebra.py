import numpy as np


def solve_normal_equation(X, y):
    """
    Solve the least-squares problem for linear regression.

    Uses np.linalg.lstsq (SVD-based) rather than forming and inverting
    X.T @ X explicitly, which would square the condition number and lose
    precision on ill-conditioned designs.

    Parameters:
    - X (np.ndarray): The input feature matrix, with an intercept column if necessary.
    - y (np.ndarray): The target variable vector.

    Returns:
    - np.ndarray: The coefficients vector including the intercept term if X includes an intercept.
    """
    coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coefficients
