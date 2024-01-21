from scipy.stats import shapiro, norm
from .base import BaseLinearRegressionModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

class OLS(BaseLinearRegressionModel):
    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept)

    def fit(self, X, y):
        # Process X and y and store them in self.X_np and self.y_np
        super().fit(X, y)  # This calls BaseLinearRegressionModel.fit
        self.feature_names = list(X.columns) + (["Intercept"] if self.fit_intercept else [])
        try:
            # Coefficients calculation using the Normal Equation: beta = (X'X)^(-1)X'y
            self.coefficients = np.linalg.inv(self.X_np.T @ self.X_np) @ self.X_np.T @ self.y_np
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Matrix inversion failed. X may be singular.")

        # Post-fit calculations (standard errors, Z-scores, p-values)
        self._calculate_statistics(self.X_np, self.y_np)

    def _calculate_statistics(self, X_np, y_np):
        residuals = y_np - X_np @ self.coefficients
        mse = (residuals ** 2).mean()

        self.std_err = np.sqrt(np.diag(np.linalg.pinv(X_np.T @ X_np) * mse))

        # Ensure z_scores and p_values calculations are correct
        self.z_scores = np.array([coef / se if se != 0 else 0 for coef, se in zip(self.coefficients, self.std_err)])
        self.p_values = np.array([2 * (1 - norm.cdf(np.abs(z))) for z in self.z_scores])


    def model_summary(self):
        # Ensure coefficients and std_err are 1-dimensional
        coefficients = np.ravel(self.coefficients)
        std_err = np.ravel(self.std_err)

        # Calculating 95% CI
        ci_lower = np.ravel(coefficients - 1.96 * std_err)
        ci_upper = np.ravel(coefficients + 1.96 * std_err)

        summary = pd.DataFrame({
            'Feature': self.feature_names,
            "Coefficient": coefficients,
            "Std Error": std_err,
            "Z Score": np.ravel(self.z_scores),
            "P Value": np.ravel(self.p_values),
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper
        })
        # Formatting for better readability
        summary = summary.round(4)  # Round to 4 decimal places for clarity

        summary['Note'] = summary['Z Score'].apply(lambda x: 'Low Z-score' if abs(x) < 2 else '')

        # Apply conditional formatting
        return summary

    @staticmethod
    def check_residuals_normality(residuals):
        stat, p_value = shapiro(residuals)
        return p_value >= 0.05

    def plot_residuals(self):
        if self.X_np is None or self.y_np is None:
            raise ValueError("Model must be fitted before plotting residuals.")

        predictions = self.X_np @ self.coefficients
        residuals = self.y_np.flatten() - predictions.flatten()

        # Histogram of residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        plt.show()

        # Q-Q plot of residuals
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.show()
