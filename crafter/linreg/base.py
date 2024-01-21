import numpy as np
import pandas as pd

class BaseLinearRegressionModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.feature_names = None
        self.X_np = None
        self.y_np = None

    def process_X(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        if not all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
            raise ValueError("All columns in X must be numeric.")
        X_np = X.to_numpy()
        if self.fit_intercept:
            X_np = self.add_intercept(X_np)
        return X_np

    def process_y(self, y):
        if not isinstance(y, pd.Series) and not isinstance(y, pd.DataFrame):
            raise ValueError("y must be a pandas Series or DataFrame.")
        return y.to_numpy().reshape(-1, 1)

    def add_intercept(self, X_np):
        intercept = np.ones((X_np.shape[0], 1))
        return np.concatenate((intercept, X_np), axis=1)

    def fit(self, X, y):
        self.X_np = self.process_X(X)
        self.y_np = self.process_y(y)
        # Fit the model (to be implemented in derived classes)
        pass

    def predict(self, X):
        X_np = self.process_X(X)
        return np.dot(X_np, self.coefficients)

    def get_coefficients(self):
        return self.coefficients

    def model_summary(self):
        pass  # To be implemented in derived classes
