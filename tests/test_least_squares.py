import unittest
import numpy as np
from linear_models.least_squares import (
    AnalyticalLeastSquaresRegression,
    MatrixLeastSquaresRegression,
)


class TestLeastSquaresRegression(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1, 2, 3, 4, 5])
        self.y = np.array([2, 4, 6, 8, 10])

    def test_analytical_fit(self):
        model = AnalyticalLeastSquaresRegression()
        model.fit(self.x, self.y)
        beta = model.coefficients()
        self.assertEqual(beta[0, 0], 2)
        self.assertEqual(beta[1, 0], 0)

    def test_matrix_fit(self):
        model = MatrixLeastSquaresRegression()
        model.fit(self.x.reshape(-1, 1), self.y)
        beta = model.coefficients()  # Intercept first, then slope
        self.assertAlmostEqual(beta[1, 0], 2, places=5)  # Testing the slope
        self.assertAlmostEqual(beta[0, 0], 0, places=5)  # Testing the intercept

    def test_analytical_predict(self):
        model = AnalyticalLeastSquaresRegression()
        model.fit(self.x, self.y)
        predictions = model.predict(np.array([6, 7]))
        predictions_list = [arr[0] for arr in predictions][0]
        self.assertEqual(predictions_list, 12.0)

    # def test_matrix_predict(self):
    #     model = MatrixLeastSquaresRegression()
    #     model.fit(self.x.reshape(-1, 1), self.y)
    #     predictions = model.predict(np.array([[1, 6], [1, 7]]))  # Add a column of ones for intercept
    #     np.testing.assert_array_almost_equal(predictions, [12, 14], decimal=7)

    def test_singular_fit(self):
        models = [AnalyticalLeastSquaresRegression(), MatrixLeastSquaresRegression()]
        x_singular = np.array([1, 1, 1, 1, 1])
        y_singular = np.array([2, 2, 2, 2, 2])
        for model in models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    model.fit(x_singular, y_singular)

    def test_different_length_input(self):
        models = [AnalyticalLeastSquaresRegression(), MatrixLeastSquaresRegression()]
        x_different_length = np.array([1, 2, 3, 4])
        y_different_length = np.array([2, 3, 5, 7, 9])
        for model in models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    model.fit(x_different_length, y_different_length)


if __name__ == "__main__":
    unittest.main()
