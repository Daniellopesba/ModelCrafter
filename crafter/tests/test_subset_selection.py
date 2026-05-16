import unittest
import numpy as np
import pandas as pd
from crafter.regularization.subset_selection import SubsetSelection, AnalyticalInsights
from crafter.performance_metrics.regression_metrics import MSE
from crafter.models.linear_regression import LinearRegression


class TestSubsetSelection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n_train, n_test, n_features = 50, 20, 3
        true_coef = np.array([1.5, -2.0, 0.0])  # feature3 is irrelevant

        X_train = rng.normal(size=(n_train, n_features))
        y_train = X_train @ true_coef + rng.normal(scale=0.1, size=n_train)
        X_test = rng.normal(size=(n_test, n_features))
        y_test = X_test @ true_coef + rng.normal(scale=0.1, size=n_test)

        cols = ["feature1", "feature2", "feature3"]
        self.X_train = pd.DataFrame(X_train, columns=cols)
        self.y_train = pd.Series(y_train)
        self.X_test = pd.DataFrame(X_test, columns=cols)
        self.y_test = pd.Series(y_test)

    def test_fit_selects_a_subset(self):
        selector = SubsetSelection(
            self.X_train, self.y_train, self.X_test, self.y_test, MSE
        )
        selector.fit()

        self.assertIsInstance(selector.best_model, LinearRegression)
        self.assertIsNotNone(selector.best_features)
        self.assertGreater(len(selector.best_features), 0)
        self.assertTrue(np.isfinite(selector.best_metric_value))

    def test_fit_evaluates_all_non_empty_subsets(self):
        selector = SubsetSelection(
            self.X_train, self.y_train, self.X_test, self.y_test, MSE
        )
        selector.fit()

        n_features = self.X_train.shape[1]
        expected_count = 2**n_features - 1
        self.assertEqual(len(selector.model_evaluations), expected_count)

    def test_best_metric_value_is_minimum(self):
        selector = SubsetSelection(
            self.X_train, self.y_train, self.X_test, self.y_test, MSE
        )
        selector.fit()

        all_values = [m["Metric Value"] for m in selector.model_evaluations]
        self.assertAlmostEqual(selector.best_metric_value, min(all_values))


class TestAnalyticalInsights(unittest.TestCase):
    def test_generate_insights_df(self):
        model_evaluations = [
            {"subset_size": 1, "features": ["feature1"], "Metric Value": 10},
            {"subset_size": 2, "features": ["feature1", "feature2"], "Metric Value": 8},
        ]

        insights = AnalyticalInsights(model_evaluations)
        df = insights.generate_insights_df()

        self.assertEqual(len(df), 2)
        self.assertIn("subset_size", df.columns)
        self.assertIn("features", df.columns)
        self.assertIn("Metric Value", df.columns)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
