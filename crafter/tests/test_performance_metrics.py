import unittest
from unittest.mock import patch  # noqa
import numpy as np
from crafter.performance_metrics.regression_metrics import MPE, MSE, R2


class TestRegressionMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = [100, 200, 300, 400]
        self.y_pred = [110, 190, 310, 390]
        self.y_true_with_zero = [0, 200, 300]
        self.y_pred_with_zero_adjustment = [10, 210, 310]

    def test_MPE_positive_error(self):
        expected_mpe = (
            100
            * (
                (110 - 100) / 100
                + (190 - 200) / 200
                + (310 - 300) / 300
                + (390 - 400) / 400
            )
            / 4
        )
        mpe_calculator = MPE(self.y_true, self.y_pred)
        result = mpe_calculator.calculate()
        self.assertAlmostEqual(expected_mpe, result, places=4)

    def test_MPE_zero_in_y_true(self):
        mpe_calculator = MPE(
            self.y_true_with_zero, self.y_pred_with_zero_adjustment, offset=1e-6
        )
        result = mpe_calculator.calculate()
        # The exact expected value might need adjustment based on the offset handling
        self.assertNotEqual(0, result)

    def test_MSE_calculation(self):
        expected_mse = np.mean(
            [(110 - 100) ** 2, (190 - 200) ** 2, (310 - 300) ** 2, (390 - 400) ** 2]
        )
        mse_calculator = MSE(self.y_true, self.y_pred)
        result = mse_calculator.calculate()
        self.assertAlmostEqual(expected_mse, result, places=4)

    def test_R2_calculation(self):
        r2_calculator = R2(self.y_true, self.y_pred)
        result = r2_calculator.calculate()
        # Calculating expected R2 score manually or comparing to a known value
        # This is a placeholder; the exact calculation depends on the true and predicted values
        expected_r2 = 1 - sum(
            (np.array(self.y_true) - np.array(self.y_pred)) ** 2
        ) / sum((np.array(self.y_true) - np.mean(self.y_true)) ** 2)
        self.assertAlmostEqual(expected_r2, result, places=4)

    def test_empty_lists(self):
        with self.assertRaises(ValueError):
            MPE([], []).calculate()
        with self.assertRaises(ValueError):
            MSE([], []).calculate()
        with self.assertRaises(ValueError):
            R2([], []).calculate()


if __name__ == "__main__":
    unittest.main()
