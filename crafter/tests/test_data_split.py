import unittest
import pandas as pd
import numpy as np
from crafter.utils.data_split import (
    shuffle_data,
    split_randomly,
    split_conditionally,
    split_data,
)


class TestDataSplit(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame(
            {
                "feature1": np.arange(100),
                "feature2": np.arange(100, 200),
                "target": np.random.randint(0, 2, size=100),
            }
        )
        self.X = self.data.drop("target", axis=1)
        self.y = self.data["target"]

    def test_shuffle_data(self):
        X_shuffled, y_shuffled = shuffle_data(self.X, self.y, random_state=42)
        # Check if indices have been shuffled but remain consistent between X and y
        self.assertFalse(np.array_equal(self.X.index, X_shuffled.index))
        self.assertTrue(np.array_equal(X_shuffled.index, y_shuffled.index))

    def test_split_randomly(self):
        X_train, X_test, y_train, y_test = split_randomly(
            self.X, self.y, test_size=0.2, random_state=42
        )
        # Check if the split ratio is correct
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

    def test_split_conditionally(self):
        conditions = [{"column": "feature1", "operator": "<=", "value": 50}]
        X_train, X_test, y_train, y_test = split_conditionally(
            self.X, self.y, conditions
        )
        # Check if condition-based splitting is correct
        self.assertTrue((X_train["feature1"] <= 50).all())
        self.assertTrue((X_test["feature1"] > 50).all())

    def test_split_data(self):
        # Test random split using the high-level function
        X_train, X_test, y_train, y_test = split_data(
            data=self.data,
            target_column="target",
            split_method="random",
            test_size=0.2,
            random_state=42,
        )
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)

        # Test condition split using the high-level function
        conditions = [{"column": "feature1", "operator": "<=", "value": 50}]
        X_train, X_test, y_train, y_test = split_data(
            data=self.data,
            target_column="target",
            split_method="condition",
            conditions=conditions,
        )
        self.assertTrue((X_train["feature1"] <= 50).all())
        self.assertTrue((X_test["feature1"] > 50).all())

    def test_error_handling(self):
        # Test for invalid test_size
        with self.assertRaises(ValueError):
            split_randomly(self.X, self.y, test_size=1.5, random_state=42)

        # Test for missing condition in split_conditionally
        with self.assertRaises(ValueError):
            split_data(data=self.data, target_column="target", split_method="condition")


if __name__ == "__main__":
    unittest.main()
