import unittest
from linear_models.least_squares import (
    AnalyticalLeastSquaresRegression,
    MatrixLeastSquaresRegression,
)


class TestLeastSquaresRegression(unittest.TestCase):
    def setUp(self):
        self.x = [1, 2, 3, 4, 5]
        self.y = [2, 4, 6, 8, 10]

    def test_analytical_fit(self):
        model = AnalyticalLeastSquaresRegression()
        model.fit(self.x, self.y)
        a, b = model.coefficients()
        self.assertEqual(a, 2)
        self.assertEqual(b, 0)

    def test_matrix_fit(self):
        model = MatrixLeastSquaresRegression()
        model.fit(self.x, self.y)
        a, b = model.coefficients()
        self.assertAlmostEqual(a, 2, places=5)
        self.assertAlmostEqual(b, 0, places=5)

    def test_analytical_predict(self):
        model = AnalyticalLeastSquaresRegression()
        model.fit(self.x, self.y)
        predictions = model.predict([6, 7])
        self.assertEqual(predictions, [12, 14])

    def test_matrix_predict(self):
        model = MatrixLeastSquaresRegression()
        model.fit(self.x, self.y)
        predictions = model.predict([6, 7])
        self.assertAlmostEqual(predictions[0], 12, places=7)
        self.assertAlmostEqual(predictions[1], 14, places=7)

    def test_singular_fit(self):
        models = [AnalyticalLeastSquaresRegression(), MatrixLeastSquaresRegression()]
        x_singular = [1, 1, 1, 1, 1]
        y_singular = [2, 2, 2, 2, 2]
        for model in models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    model.fit(x_singular, y_singular)

    def test_different_length_input(self):
        models = [AnalyticalLeastSquaresRegression(), MatrixLeastSquaresRegression()]
        x_different_length = [1, 2, 3, 4]
        y_different_length = [2, 3, 5, 7, 9]
        for model in models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    model.fit(x_different_length, y_different_length)


if __name__ == "__main__":
    unittest.main()
