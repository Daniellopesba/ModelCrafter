from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    def __init__(self, fit_intercept=True):
        """
        Initialize the BaseModel with an option to fit an intercept.

        Parameters:
        - fit_intercept (bool): If True, an intercept term is added to the model. Defaults to True.
        """
        self.fit_intercept = fit_intercept
        self.coefficients = None  # Will be set by the fit method in subclasses

    def _ensure_numpy_array(self, X):
        """
        Converts input to a numpy array if it's a pandas DataFrame or Series.

        Parameters:
        - X (pd.DataFrame, pd.Series, np.ndarray): The input features or target variable.

        Returns:
        - np.ndarray: The converted numpy array from the input.

        Raises:
        - ValueError: If the input is neither a pandas DataFrame, Series, nor a numpy ndarray.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy()
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError(
                "Input must be a pandas DataFrame, Series, or a numpy ndarray."
            )

    def _add_intercept(self, X_np):
        """
        Adds an intercept column to the numpy array if fit_intercept is True.

        Parameters:
        - X_np (np.ndarray): The numpy array of input features.

        Returns:
        - np.ndarray: The numpy array with an intercept column added, if required.
        """
        if self.fit_intercept:
            intercept = np.ones((X_np.shape[0], 1))
            return np.concatenate((intercept, X_np), axis=1)
        return X_np

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the data. This method must be implemented by subclasses.

        The implementation should set the model's parameters based on the training data.

        Parameters:
        - X (pd.DataFrame, pd.Series, np.ndarray): The input features.
        - y (pd.DataFrame, pd.Series, np.ndarray): The target variable.

        Returns:
        - self: Returns an instance of self.

        Notes:
        - This method should process the inputs, fit the model to the data, and set any attributes necessary for the model's representation.
        """
        X_np = self._ensure_numpy_array(X)
        y_np = self._ensure_numpy_array(y).reshape(-1, 1)  # noqa
        X_np = self._add_intercept(X_np)  # noqa
        # Placeholder for model fitting logic to be implemented in subclass
        return self

    @abstractmethod
    def predict(self, X):
        """
        Predict target values using the fitted model. This method must be implemented by subclasses.

        Parameters:
        - X (pd.DataFrame, pd.Series, np.ndarray): The input features for which to predict target values.

        Returns:
        - predictions (np.ndarray): Predicted values for the input features.

        Notes:
        - This method should process the inputs and use the model's parameters to predict and return the target values.
        - Example return statement (to be implemented in subclass): return np.dot(X_np, self.coefficients)
        """
        X_np = self._ensure_numpy_array(X)
        X_np = self._add_intercept(X_np)
        # The actual return statement will depend on the subclass implementation.
        # Example placeholder: return np.dot(X_np, self.coefficients)

    @abstractmethod
    def summary(self):
        """
        Provide a summary of the model. This method must be implemented by subclasses.

        Returns:
        - summary (str, dict, or similar): A summary of the model, including details such as model parameters, performance metrics, and any other relevant information.

        Notes:
        - The implementation can vary greatly depending on the model's nature. It might include statistics like R-squared, coefficients, p-values, etc., for statistical models, or architecture details for neural networks.
        """
        pass
