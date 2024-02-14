import numpy as np


def solve_normal_equation(X, y):
    """
    Solve the normal equation for linear regression.

    Parameters:
    - X (np.ndarray): The input feature matrix, with an intercept column if necessary.
    - y (np.ndarray): The target variable vector.

    Returns:
    - np.ndarray: The coefficients vector including the intercept term if X includes an intercept.
    """
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
