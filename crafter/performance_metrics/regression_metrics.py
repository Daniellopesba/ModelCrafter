import numpy as np


class MetricCalculator:
    """
    Base class for calculating regression metrics, with support for a baseline comparison mode.
    In baseline mode, `y_pred` is expected to be a single value, which is used to generate an array
    of equal length to `y_true` for comparison.

    Attributes:
        y_true (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted values or a single predicted value in baseline mode.
        baseline (bool): Indicates if baseline mode is enabled.
        baseline_value (float): The value to use as a constant baseline when baseline mode is enabled.

    Methods:
        calculate(): Abstract method for calculating the specific metric. Must be overridden by subclasses.

    Usage Example:
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = 2.5  # For baseline mode
        >>> baseline_value = 2.5
        >>> mse_calculator = MSE(y_true, y_pred, baseline=True, baseline_value=baseline_value)
        >>> mse = mse_calculator.calculate()
        >>> print(f"MSE (Baseline): {mse}")
    """

    def __init__(self, y_true, y_pred, baseline=False, baseline_value=None):
        if len(y_true) == 0:
            raise ValueError("y_true cannot be empty.")
        if baseline and not isinstance(y_pred, (int, float)):
            raise ValueError("In baseline mode, y_pred must be a single scalar value.")

        self.y_true = np.array(y_true)
        self.baseline = baseline
        self.baseline_value = baseline_value

        if self.baseline:
            # Generate an array of y_pred with the same length as y_true using the single scalar value
            self.y_pred = np.full_like(self.y_true, y_pred)
        else:
            if isinstance(y_pred, (list, np.ndarray)) and len(y_pred) == 0:
                raise ValueError("y_pred cannot be empty in non-baseline mode.")
            self.y_pred = np.array(y_pred)

    def calculate(self):
        # Placeholder for the abstract method
        raise NotImplementedError("Subclasses should implement this method.")

class MPE(MetricCalculator):
    """
    Calculates the Mean Percentage Error (MPE) between true and predicted values.
    Optionally adjusts for zeros in true values to avoid division by zero.
    """

    def __init__(self, y_true, y_pred, offset=1e-6, **kwargs):
        """
        Initializes the MPE calculator with true and predicted values, and an offset for zero adjustment.

        Parameters:
            y_true (array-like): True target values.
            y_pred (array-like): Predicted values.
            offset (float, optional): Offset added to zero elements in y_true to prevent division by zero.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(y_true, y_pred, **kwargs)
        self.offset = offset

    def calculate(self):
        """
        Calculates and returns the Mean Percentage Error.

        Returns:
            float: The MPE value.
        """
        # Adjust true values to avoid division by zero
        y_true_adjusted = np.where(self.y_true == 0, self.y_true + self.offset, self.y_true)
        percentage_errors = ((self.y_pred - y_true_adjusted) / y_true_adjusted) * 100
        return np.mean(percentage_errors)


class MSE(MetricCalculator):
    """
    Calculates the Mean Squared Error (MSE) between true and predicted values.
    """

    def calculate(self):
        """
        Calculates and returns the Mean Squared Error.

        Returns:
            float: The MSE value.
        """
        squared_diffs = (self.y_true - self.y_pred) ** 2
        return np.mean(squared_diffs)


class R2(MetricCalculator):
    """
    Calculates the R2 score, a measure of how well observed outcomes are replicated by the model.
    """

    def calculate(self):
        """
        Calculates and returns the R2 score.

        Returns:
            float: The R2 score.
        """
        total_variance = np.var(self.y_true)
        unexplained_variance = np.mean((self.y_true - self.y_pred) ** 2)
        return 1 - (unexplained_variance / total_variance)
