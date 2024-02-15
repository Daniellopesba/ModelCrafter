import unittest
from crafter.performance_metrics.regression_metrics import mpe


class TestMeanPredictionError(unittest.TestCase):
    def test_positive_error(self):
        y_true = [100, 200, 300]
        y_pred = [110, 210, 310]
        expected_mpe = (
            100 * ((110 - 100) / 100 + (210 - 200) / 200 + (310 - 300) / 300) / 3
        )
        result = mpe(y_true, y_pred)
        self.assertAlmostEqual(
            expected_mpe, result, places=4, msg="Failed on test with positive error."
        )

    def test_negative_error(self):
        y_true = [100, 200, 300]
        y_pred = [90, 190, 290]
        expected_mpe = (
            100 * ((90 - 100) / 100 + (190 - 200) / 200 + (290 - 300) / 300) / 3
        )
        result = mpe(y_true, y_pred)
        self.assertAlmostEqual(
            expected_mpe, result, places=4, msg="Failed on test with negative error."
        )

    def test_mixed_error(self):
        y_true = [100, 200, 300, 400]
        y_pred = [110, 190, 310, 390]
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
        result = mpe(y_true, y_pred)
        self.assertAlmostEqual(
            expected_mpe, result, places=4, msg="Failed on test with mixed errors."
        )

    def test_zero_in_y_true(self):
        y_true = [0, 200, 300]
        y_pred = [10, 210, 310]
        expected_mpe = (
            100 * ((10 - 1e-6) / 1e-6 + (210 - 200) / 200 + (310 - 300) / 300) / 3
        )
        result = mpe(y_true, y_pred)
        self.assertAlmostEqual(
            expected_mpe,
            result,
            places=-4,
            # Using a lower precision due to the large values involved
            msg="Failed on test with zero in y_true.",
        )

    def test_empty_lists(self):
        y_true = []
        y_pred = []
        with self.assertRaises(ValueError):
            mpe(y_true, y_pred)


if __name__ == "__main__":
    unittest.main()
