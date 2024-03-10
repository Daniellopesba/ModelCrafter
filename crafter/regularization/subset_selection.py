from itertools import combinations
from tqdm import tqdm
import numpy as np
import pandas as pd
from crafter.models.linear_regression import LinearRegression
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
    - model_evaluations (List): Records of model evaluations.
    """

    def __init__(self, X_train, y_train, X_test, y_test, metric):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.best_model = None
        self.best_features = None
        self.best_metric_value = np.inf
        self.model_evaluations = []

    def fit(self):
        """Fits models for all subsets of predictors and updates attributes with the best subset found."""
        n_features = self.X_train.shape[1]
        total_models = sum(
            1
            for _ in range(1, n_features + 1)
            for _ in combinations(self.X_train.columns, _)
        )
        print(f"Total models to fit: {total_models}")

        with tqdm(total=total_models, desc="Fitting Models", leave=True) as pbar:
            for k in range(1, n_features + 1):
                for subset in combinations(self.X_train.columns, k):
                    self.evaluate_subset(subset, pbar)

        model_evaluations_df = pd.DataFrame(self.model_evaluations)
        best_model_settings = model_evaluations_df[
            model_evaluations_df["Metric Value"]
            == model_evaluations_df["Metric Value"].min()
        ]
        self.best_metric_value = best_model_settings["Metric Value"].iloc[0]
        self.best_features = best_model_settings["features"].iloc[0]

        # Refitting the best model on the entire training set using the best subset of features
        self.best_model = LinearRegression(fit_intercept=True).fit(
            self.X_train[self.best_features], self.y_train
        )

    def evaluate_subset(self, subset, pbar):
        """Evaluates a single subset of predictors, fitting a model and updating the best model if necessary.

        Parameters:
        - subset (tuple): The subset of predictors to evaluate.
        - pbar (tqdm.std.tqdm): Progress bar instance for visual feedback.
        """
        try:
            X_subset_train = self.X_train[list(subset)]
            X_subset_test = self.X_test[list(subset)]
            model = LinearRegression(fit_intercept=True)
            model.fit(X_subset_train, self.y_train)
            y_pred = model.predict(X_subset_test)

            metric_instance = self.metric(
                self.y_test, y_pred
            )  # Creates an instance of the metric class
            metric_value = (
                metric_instance.calculate()
            )  # Now, this should be a numeric value

            self.model_evaluations.append(
                {
                    "subset_size": len(subset),
                    "features": list(subset),
                    "Metric Value": metric_value,  # This is now a numeric value
                }
            )

        except Exception as e:
            print(f"Error fitting model with features {subset}: {e}")
        finally:
            pbar.update(1)


class AnalyticalInsights:
    """
    Generates and stores analytical insights from model evaluations, such as metric values for all subsets.

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
    def plot_metric_by_subset_size(analysis_df, metric_name="Metric Value"):
        """
        Plots the specified metric for all models by subset size.

        Parameters:
        - analysis_df (pd.DataFrame): DataFrame containing model evaluation results.
        - metric_name (str): The name of the metric to plot. Defaults to "Metric Value".
        """
        plt.figure(figsize=(10, 6))
        for size, group in analysis_df.groupby("subset_size"):
            plt.scatter(
                [size] * len(group), group[metric_name], alpha=0.5, label=f"Size {size}"
            )

        plt.xlabel("Subset Size")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} of All Subset Models by Size")
        plt.legend()
        plt.show()
