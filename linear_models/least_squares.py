from abc import ABC, abstractmethod
import numpy as np
from memory_profiler import profile  # noqa


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
    Solves the least squares regression problem using the Normal Equation approach.

    The Normal Equation is an analytical solution to the ordinary least squares method for
    linear regression.

    The equation is beta = (X^TX)^(-1)X^TY, where:
    - X is the matrix of input features and a column of ones ,
    - Y is the vector of target values ,
    - beta is the vector of calculated regression coefficients.

    This method assumes that (X^TX) is non-singular and invertible, which can be computationally
    intensive and may lead to numerical instability if the matrix is poorly conditioned.

    Performance Notes:
    - Transposing a matrix is a constant-time operation for numpy arrays.
    - Matrix multiplication is O(n^2.807) using the Strassen Algorithm, which numpy optimizes for.
    - Inverting a matrix is generally O(n^3) and can be the bottleneck if (X^TX) is large.
    - Reshaping an array is typically a constant-time operation unless a copy is needed.

    :param X: numpy.ndarray, shape (n_samples, n_features)
        Matrix of input features, where n_samples is the number of samples and
        n_features is the number of features.

    :param Y: numpy.ndarray, shape (n_samples,) or (n_samples, n_targets)
        Vector of target values. If Y is two-dimensional, its shape should be (n_samples, n_targets).

    :return: numpy.ndarray
        Vector of regression coefficients (beta).
    """

    # @profile
    def fit(self, x, y):
        # Add a column of ones to X to account for the intercept
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        # Step 1.1: Compute X^T (transpose of X)
        Xt = (
            x.T
        )  # Transposing is a cheap operation in terms of computational complexity.

        # Step 1.2: Compute X^T X
        XtX = Xt @ x  # Matrix multiplication, potentially computationally expensive.

        # Step 2: Compute the inverse of X^T X
        # This step can be a performance bottleneck and may fail for singular or near-singular matrices.
        XtX_inv = np.linalg.inv(XtX)

        # Ensure Y is a column vector
        # This is a reshape operation, which is very efficient in numpy as it returns a new view on the array, if possible.
        Yt = y.reshape(-1, 1) if len(y.shape) == 1 else y

        # Step 3: Compute (X^T X)^{-1} X^T Y
        # This matrix multiplication step is the final step in calculating the regression coefficients.
        beta = XtX_inv @ Xt @ Yt

        # Assign coefficients
        self.b = beta[0, 0]  # The intercept term
        self.a = beta[1:].flatten()  # Make sure 'a' is a flat array

        return self
