import numpy as np


def mpe(y_true, y_pred, offset=1e-6):
    """
    Calculate the Mean Prediction Error (MPE) with an offset for zeros in y_true.
    Issues a ValueError for empty input lists.

    Parameters:
    - y_true: array-like of true target values.
    - y_pred: array-like of predicted values.
    - offset: small value to add to y_true elements that are exactly zero to avoid division by zero.

    Returns:
    - mpe: Mean Prediction Error.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input lists cannot be empty.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Apply offset to zero elements in y_true
    y_true_adjusted = np.where(y_true == 0, y_true + offset, y_true)

    # Calculate the percentage differences between true and predicted values
    percentage_errors = ((y_pred - y_true_adjusted) / y_true_adjusted) * 100

    # Calculate the mean of these percentage differences
    mpe = np.mean(percentage_errors)

    return mpe


def mse(y_true, y_pred):
    """
    Calculate the mean squared error (MSE) between true and predicted values.

    Parameters:
    - y_true: array-like of true target values.
    - y_pred: array-like of predicted values.

    Returns:
    - mse: Mean Squared Error.
    """
    # Convert inputs to numpy arrays to ensure compatibility with numpy operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate squared differences
    squared_diffs = (y_true - y_pred) ** 2

    # Calculate mean of squared differences
    mse = np.mean(squared_diffs)

    return mse
