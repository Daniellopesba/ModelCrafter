from abc import ABC, abstractmethod
import numpy as np


class AbstractLeastSquaresRegression(ABC):
    """
    Abstract class for least squares regression.
    """

    def __init__(self):
        self.a = 0  # Slope
        self.b = 0  # Intercept

    @abstractmethod
    def fit(self, x, y):
        pass

    def predict(self, x):
        """
        Predict the y values using the fitted model.

        Args:
            x (list of float): The list of x-coordinates.

        Returns:
            list of float: The predicted y-coordinates.
        """
        if self.a is None or self.b is None:
            raise ValueError("Model is not fitted yet.")
        return [self.a * x_i + self.b for x_i in x]

    def coefficients(self):
        """
        Get the coefficients of the regression line.

        Returns:
            tuple: A tuple containing the slope and intercept (a, b).
        """
        return self.a, self.b

class AnalyticalLeastSquaresRegression(AbstractLeastSquaresRegression):
    """
    Analytical method implementation of least squares regression.
    """

    def fit(self, x, y):
        if len(x) != len(y):
            raise ValueError("The lists x and y must have the same length.")

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        denominator = n * sum_x2 - sum_x**2
        if denominator == 0:
            raise ValueError("The x values are singular and cannot be used for a fit.")

        self.a = (n * sum_xy - sum_x * sum_y) / denominator
        self.b = (sum_y - self.a * sum_x) / n

        return self


class MatrixLeastSquaresRegression(AbstractLeastSquaresRegression):
    """
    Matrix-based method implementation of least squares regression.
    """

    def fit(self, x, y):
        # Transform lists into numpy arrays
        X = np.vstack((np.ones(len(x)), x)).T
        Y = np.array(y)

        # Check if the matrix X'X is singular
        if np.linalg.matrix_rank(np.dot(X.T, X)) < min(X.shape):
            raise ValueError("The x values are singular and cannot be used for a fit.")

        # Calculate coefficients using the normal equation
        # beta = (X'X)^-1 X'Y
        beta = np.linalg.pinv(X.T @ X) @ X.T @ Y

        # Assign coefficients
        self.a = beta[1]
        self.b = beta[0]

        return self
