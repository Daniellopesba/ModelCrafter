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
        value (float): The calculated metric value.

    Methods:
        calculate(): Abstract method for calculating the specific metric. Must be overridden by subclasses.
        set_value(): Sets the `value` attribute by calculating the metric.
    """

    def __init__(self, y_true, y_pred, baseline=False, baseline_value=None):
        if len(y_true) == 0:
            raise ValueError("y_true cannot be empty.")
        if baseline and not isinstance(y_pred, (int, float)):
            raise ValueError("In baseline mode, y_pred must be a single scalar value.")

        self.y_true = np.array(y_true)
        self.baseline = baseline
        self.baseline_value = baseline_value
        self.value = None  # Initialize value attribute

        if self.baseline:
            self.y_pred = np.full_like(self.y_true, baseline_value)
        else:
            self.y_pred = np.array(y_pred)

        self.set_value()  # Automatically calculate and set the metric value upon initialization

    def calculate(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def set_value(self):
        self.value = self.calculate()

    # Comparison magic methods
    def __lt__(self, other):
        if not isinstance(other, MetricCalculator):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, MetricCalculator):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other):
        if not isinstance(other, MetricCalculator):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, MetricCalculator):
            return NotImplemented
        return self.value >= other.value


class MSE(MetricCalculator):
    def calculate(self):
        squared_diffs = (self.y_true - self.y_pred) ** 2
        return np.mean(squared_diffs)


class MPE(MetricCalculator):
    def __init__(self, y_true, y_pred, offset=1e-6, **kwargs):
        super().__init__(y_true, y_pred, **kwargs)
        self.offset = offset

    def calculate(self):
        y_true_adjusted = np.where(
            self.y_true == 0, self.y_true + self.offset, self.y_true
        )
        percentage_errors = ((self.y_pred - y_true_adjusted) / y_true_adjusted) * 100
        return np.mean(percentage_errors)


class R2(MetricCalculator):
    def calculate(self):
        total_variance = np.var(self.y_true)
        unexplained_variance = np.mean((self.y_true - self.y_pred) ** 2)
        return 1 - (unexplained_variance / total_variance)
