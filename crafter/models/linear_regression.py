import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
from statsmodels.stats.diagnostic import het_breuschpagan
from crafter.models.base import BaseModel
from crafter.math_core.linear_algebra import solve_normal_equation


class LinearRegression(BaseModel):
    def __init__(self, fit_intercept=True):
        """
        Linear Regression model that uses Ordinary Least Squares to estimate the coefficients.
        Parameters:
        - fit_intercept (bool): If True (default), calculates the intercept for this model. If False, no intercept will be used.
        """
        super().__init__(fit_intercept=fit_intercept)
        self.X_fit = None
        self.y_fit = None
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        """
        Fit the linear model to the data using Ordinary Least Squares.
        Parameters:
        - X (pd.DataFrame, pd.Series, np.ndarray): The input features.
        - y (pd.DataFrame, pd.Series, np.ndarray): The target variable.
        Returns:
        - self: Returns an instance of self with the model parameters set.
        """
        super().fit(X, y)  # Validates and prepares the data
        # Extract feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = (
                ["Intercept"] + X.columns.tolist()
                if self.fit_intercept
                else X.columns.tolist()
            )
        else:
            # Fallback for non-DataFrame inputs
            num_features = X.shape[1] if self.fit_intercept else X.shape[1] - 1
            self.feature_names = (
                ["Intercept"] + [f"X{i}" for i in range(1, num_features + 1)]
                if self.fit_intercept
                else [f"X{i}" for i in range(1, num_features + 2)]
            )

        X_np, y_np = self._ensure_numpy_array(X), self._ensure_numpy_array(y).reshape(
            -1, 1
        )
        if self.fit_intercept:
            X_np = self._add_intercept(X_np)

        # Use the solve_normal_equation function from the linear_algebra module
        self.coefficients = solve_normal_equation(X_np, y_np)

        self.X_fit = X_np
        self.y_fit = y_np

        return self

    def predict(self, X):
        """
        Predict target values using the linear model.
        Parameters:
        - X (pd.DataFrame, pd.Series, np.ndarray): The input features.
        Returns:
        - predictions (np.ndarray): Predicted values for the input features.
        """
        if self.coefficients is None:
            raise ValueError(
                "This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        X_np = self._ensure_numpy_array(X)
        # If fit_intercept is True, ensure X_np has an intercept term if not already included
        if self.fit_intercept and self.X_fit.shape[1] != X_np.shape[1]:
            X_np = self._add_intercept(X_np)

        return X_np.dot(self.coefficients)

    def summary(self):
        # Ensure X_fit and y_fit are available
        if self.X_fit is None or self.y_fit is None or self.coefficients is None:
            raise ValueError("The model must be fitted before generating a summary.")

        # Assuming self.coefficients is populated and includes intercept if fit_intercept is True
        n, _ = self.X_fit.shape
        residuals = self.y_fit - self.predict(self.X_fit)

        mse = np.mean(residuals**2)
        se = np.sqrt(mse * np.linalg.inv(self.X_fit.T @ self.X_fit).diagonal())
        t_stats = self.coefficients.flatten() / se
        p_values = [
            2 * (1 - stats.t.cdf(np.abs(t), df=n - len(self.coefficients)))
            for t in t_stats
        ]
        confidence_intervals = [
            (coef - 1.96 * std_err, coef + 1.96 * std_err)
            for coef, std_err in zip(self.coefficients.flatten(), se)
        ]

        # Header
        header = f"{'Feature':<20} | {'Coefficients':<15} | {'Std Err':<10} | {'t':<10} | {'P>|t|':<10} | {'[0.025':<10}| {'0.975]':<10}"
        print(header)
        print("=" * len(header))

        # Data Rows
        for i, name in enumerate(self.feature_names):
            coef_str = (
                f"{self.coefficients[i, 0]:.4f}" if i < len(self.coefficients) else ""
            )
            row = f"{name:<20} | {coef_str:<15} | {se[i]:<10.4f} | {t_stats[i]:<10.4f} | {p_values[i]:<10.4f} | [{confidence_intervals[i][0]:.4f}, {confidence_intervals[i][1]:.4f}]"
            print(row)

    def residuals(self):
        """
        Calculate residuals from the model using the fitted X and y.
        """
        if self.X_fit is None or self.y_fit is None:
            raise ValueError("The model must be fitted before calculating residuals.")
        predictions = self.predict(self.X_fit)
        residuals = self.y_fit - predictions
        return residuals

    def check_residuals(self):
        """
        Perform tests to check the normality and homoscedasticity of residuals
        using the fitted X and y.
        """
        if self.X_fit is None or self.y_fit is None:
            raise ValueError("The model must be fitted before checking residuals.")
        residuals = self.residuals().flatten()

        # Normality test using Shapiro-Wilk
        stat, p_value = stats.shapiro(residuals)
        print(f"Shapiro-Wilk Test: Stat={stat}, P-value={p_value}")
        if p_value > 0.05:
            print("Residuals seem to follow a normal distribution.")
        else:
            warnings.warn(
                "Residuals may not follow a normal distribution.", UserWarning
            )

        # Homoscedasticity test using Breusch-Pagan
        X_np = self._ensure_numpy_array(self.X_fit)
        if self.fit_intercept:
            X_np = self._add_intercept(X_np)
        bp_stat, bp_p_value, _, _ = het_breuschpagan(residuals, X_np)
        print(f"Breusch-Pagan Test: Stat={bp_stat}, P-value={bp_p_value}")
        if bp_p_value > 0.05:
            print("The residuals seem to be homoscedastic.")
        else:
            warnings.warn(
                "The residuals may not be homoscedastic (constant variance).",
                UserWarning,
            )

    def plot_residuals(self):
        """
        Plot histograms and Q-Q plots of residuals using the fitted X and y.
        """
        if self.X_fit is None or self.y_fit is None:
            raise ValueError("The model must be fitted before plotting residuals.")
        residuals = self.residuals().flatten()

        # Histogram of residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(x=0, color="r", linestyle="--")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Histogram of Residuals")
        plt.show()

        # Q-Q plot of residuals
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")
        plt.show()
