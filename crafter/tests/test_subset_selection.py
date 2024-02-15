import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from crafter.regularization.subset_selection import SubsetSelection, AnalyticalInsights
class TestSubsetSelection(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.X = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'feature3': np.random.rand(10)
        })
        self.y = pd.Series(np.random.rand(10))

    @patch('crafter.models.linear_regression.LinearRegression')
    def test_fit(self, MockLR):
        # Mock the LinearRegression to always return a model with predefined coefficients
        model_instance = MockLR.return_value
        model_instance.fit.return_value = None
        model_instance.predict.return_value = np.random.rand(10)

        selector = SubsetSelection(self.X, self.y)
        selector.fit()

        # Assert that at least one model was considered as the best model
        self.assertIsNotNone(selector.best_model)
        self.assertTrue(len(selector.best_features) > 0)
        self.assertTrue(selector.best_rss < np.inf)


class TestAnalyticalInsights(unittest.TestCase):
    def test_generate_insights_df(self):
        model_evaluations = [{'subset_size': 1, 'features': ('feature1',), 'rss': 10},
                             {'subset_size': 2, 'features': ('feature1', 'feature2'), 'rss': 8}]

        insights = AnalyticalInsights(model_evaluations)
        df = insights.generate_insights_df()

        self.assertEqual(len(df), 2)
        self.assertTrue('subset_size' in df.columns)
        self.assertTrue('features' in df.columns)
        self.assertTrue('rss' in df.columns)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
