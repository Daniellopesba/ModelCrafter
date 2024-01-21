import unittest
import pandas as pd
from utils.data_split import split_data


class TestDataSplitting(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset for testing
        self.data = pd.DataFrame(
            {
                "feature1": range(100),
                "feature2": range(100, 200),
                "target": range(200, 300),
            }
        )

    def test_random_split(self):
        """Test random data splitting."""
        X_train, X_test, y_train, y_test = split_data(
            data=self.data, target_column="target", split_method="random", test_size=0.2
        )
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

    def test_conditional_split(self):
        """Test conditional data splitting."""
        condition = {"column": "feature1", "threshold": 50}
        X_train, X_test, y_train, y_test = split_data(
            data=self.data,
            target_column="target",
            split_method="condition",
            condition=condition,
        )
        self.assertTrue((X_train["feature1"] <= 50).all())
        self.assertTrue((X_test["feature1"] > 50).all())

    def test_invalid_input(self):
        """Test splitting with invalid input."""
        with self.assertRaises(ValueError):
            split_data()  # No input provided

        with self.assertRaises(ValueError):
            split_data(data=self.data)  # No target column provided

        with self.assertRaises(ValueError):
            split_data(
                data=self.data, target_column="target", split_method="unknown"
            )  # Invalid split method


if __name__ == "__main__":
    unittest.main()
