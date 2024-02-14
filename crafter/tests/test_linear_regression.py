import unittest
import numpy as np
from crafter.models.linear_regression import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = 2 * np.random.rand(100, 1)
        self.y = 4 + 3 * self.X + np.random.randn(100, 1)
        self.model = LinearRegression()

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertTrue(np.isclose(self.model.coefficients[1], 3, atol=0.5))
        self.assertTrue(np.isclose(self.model.coefficients[0], 4, atol=1))

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_residuals(self):
        self.model.fit(self.X, self.y)
        residuals = self.model.residuals()
        self.assertEqual(residuals.shape, self.y.shape)

    def test_residuals_normality_and_homoscedasticity(self):
        self.model.fit(self.X, self.y)
        # Directly calling check_residuals; specific assertions might depend on your implementation
        # This call should not raise any errors if implemented correctly
        self.model.check_residuals()

    def test_summary(self):
        self.model.fit(self.X, self.y)
        # Ensure calling summary doesn't raise errors; capturing stdout to assert on output could be an advanced step
        self.model.summary()


if __name__ == "__main__":
    unittest.main()
