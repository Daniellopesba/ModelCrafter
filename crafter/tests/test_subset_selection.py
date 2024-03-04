import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from crafter.regularization.subset_selection import SubsetSelection, AnalyticalInsights


class TestSubsetSelection(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.X = pd.DataFrame(
            {
                "feature1": np.random.rand(10),
                "feature2": np.random.rand(10),
                "feature3": np.random.rand(10),
            }
        )
        self.y = pd.Series(np.random.rand(10))

    @patch("crafter.models.linear_regression.LinearRegression")
    def test_fit(self, MockLR):
        # Mock the LinearRegression to simulate model fitting and prediction
        mock_rss_values = np.linspace(
            1, 0.1, 10
        )  # Decreasing RSS values to simulate model improvement
        model_instance = MockLR.return_value
        model_instance.fit.return_value = None
        # Simulate predictions leading to decreasing RSS values
        model_instance.predict.side_effect = lambda X: self.y - (
            np.arange(len(X)) * 0.1
        )

        selector = SubsetSelection(self.X, self.y)
        selector.fit()

        # Assert that the best model and features are properly identified
        # The logic assumes that a lower RSS value indicates a better model
        self.assertIsNotNone(selector.best_model, "The best model should not be None.")
        self.assertIsNotNone(
            selector.best_features, "The best features should not be None."
        )
        self.assertLess(
            selector.best_rss, np.inf, "The best RSS should be less than infinity."
        )
        self.assertTrue(
            len(selector.best_features) > 0,
            "The best features set should not be empty.",
        )
        self.assertEqual(
            selector.best_rss,
            min(mock_rss_values),
            "The best RSS should be the minimum of the simulated RSS values.",
        )


class TestAnalyticalInsights(unittest.TestCase):
    def test_generate_insights_df(self):
        model_evaluations = [
            {"subset_size": 1, "features": ("feature1",), "rss": 10},
            {"subset_size": 2, "features": ("feature1", "feature2"), "rss": 8},
        ]

        insights = AnalyticalInsights(model_evaluations)
        df = insights.generate_insights_df()

        self.assertEqual(len(df), 2)
        self.assertTrue("subset_size" in df.columns)
        self.assertTrue("features" in df.columns)
        self.assertTrue("rss" in df.columns)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
