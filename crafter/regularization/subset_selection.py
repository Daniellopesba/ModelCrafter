from itertools import combinations
from tqdm import tqdm
import numpy as np
import pandas as pd
from crafter.models.linear_regression import LinearRegression
from crafter.performance_metrics.regression_metrics import MSE as mean_squared_error
import matplotlib.pyplot as plt


class SubsetSelection:
    """
    This class handles the fitting of linear regression models for all possible subsets of a given set of predictors,
    and identifies the subset that minimizes the Residual Sum of Squares (RSS).

    Attributes:
    - X (pd.DataFrame): DataFrame of predictor variables.
    - y (pd.Series or np.array): Target variable.
    - best_model (LinearRegression): Best model found.
    - best_features (tuple): Best subset of features.
    - best_rss (float): RSS of the best model.
    - model_evaluations (list): Records of model evaluations.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_model = None
        self.best_features = None
        self.best_rss = np.inf
        self.model_evaluations = []

    def fit(self):
        """Fits models for all subsets of predictors and updates attributes with the best subset found."""
        n_features = self.X.shape[1]
        total_models = sum(
            1 for _ in range(1, n_features + 1) for _ in combinations(self.X.columns, _)
        )
        print(f"Total models to fit: {total_models}")

        with tqdm(total=total_models, desc="Fitting Models", leave=True) as pbar:
            for k in range(1, n_features + 1):
                for subset in combinations(self.X.columns, k):
                    self.evaluate_subset(subset, pbar)

    def evaluate_subset(self, subset, pbar):
        """Evaluates a single subset of predictors, fitting a model and updating the best model if necessary.

        Parameters:
        - subset (tuple): The subset of predictors to evaluate.
        - pbar (tqdm.std.tqdm): Progress bar instance for visual feedback.
        """
        try:
            X_subset = self.X[list(subset)]
            model = LinearRegression(fit_intercept=True)
            model.fit(X_subset, self.y)
            y_pred = model.predict(X_subset)
            rss = mean_squared_error(self.y, y_pred) * len(self.y)

            self.model_evaluations.append(
                {"subset_size": len(subset), "features": subset, "rss": rss}
            )

            if rss < self.best_rss:
                self.best_rss = rss
                self.best_features = subset
                self.best_model = model
        except Exception as e:
            print(f"Error fitting model with features {subset}: {e}")
        finally:
            pbar.update(1)


class AnalyticalInsights:
    """
    Generates and stores analytical insights from model evaluations, such as RSS values for all subsets.

    Attributes:
    - model_evaluations (list): List of dictionaries containing model evaluation metrics.
    """

    def __init__(self, model_evaluations):
        self.model_evaluations = model_evaluations

    def generate_insights_df(self):
        """Generates a DataFrame from the model evaluations for further analysis."""
        return pd.DataFrame(self.model_evaluations)


class Visualization:
    """Provides methods for visualizing model evaluation results."""

    @staticmethod
    def plot_rss_by_subset_size(analysis_df):
        """Plots the RSS for all models by subset size."""
        plt.figure(figsize=(10, 6))
        for size, group in analysis_df.groupby("subset_size"):
            plt.scatter(
                [size] * len(group), group["rss"], alpha=0.5, label=f"Size {size}"
            )

        plt.xlabel("Subset Size")
        plt.ylabel("RSS")
        plt.title("RSS of All Subset Models by Size")
        plt.legend()
        plt.show()
