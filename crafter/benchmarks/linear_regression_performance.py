import numpy as np
import time
from sklearn.linear_model import LinearRegression as SkLearnLinearRegression
from sklearn.metrics import mean_absolute_error
from crafter.models.linear_regression import LinearRegression


def generate_data(n_samples, n_features):
    """
    Generate random data for linear regression.

    Parameters:
    - n_samples (int): Number of samples.
    - n_features (int): Number of features.

    Returns:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    """
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    coeffs = np.random.rand(n_features, 1)
    y = X.dot(coeffs) + np.random.randn(n_samples, 1) * 0.5
    return X, y


def test_model(model, X, y, name):
    """
    Fit a model and measure execution time.

    Parameters:
    - model: The regression model instance.
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - name (str): Name of the model.

    Returns:
    - model: The fitted model.
    """
    start_time = time.time()
    model.fit(X, y)
    elapsed_time = time.time() - start_time
    print(f"{name} took {elapsed_time:.4f} seconds")
    return model


def compare_predictions(model1, model2, X, y, name1, name2):
    """
    Compare predictions of two models using MAE.

    Parameters:
    - model1: First regression model instance.
    - model2: Second regression model instance.
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - name1 (str): Name of the first model.
    - name2 (str): Name of the second model.
    """
    predictions1 = model1.predict(X)
    predictions2 = model2.predict(X)
    mae1 = mean_absolute_error(y, predictions1)
    mae2 = mean_absolute_error(y, predictions2)
    print(f"MAE for {name1}: {mae1:.4f}")
    print(f"MAE for {name2}: {mae2:.4f}")


def print_coefficients(model1, model2, name1, name2):
    """
    Print coefficients of two models side by side.

    Parameters:
    - model1: First regression model instance.
    - model2: Second regression model instance.
    - name1 (str): Name of the first model.
    - name2 (str): Name of the second model.
    """
    coeffs1 = (
        model1.coefficients.flatten()
        if hasattr(model1, "coefficients")
        else np.concatenate(([model1.intercept_], model1.coef_.flatten()))
    )
    coeffs2 = (
        model2.coefficients.flatten()
        if hasattr(model2, "coefficients")
        else np.concatenate(([model2.intercept_], model2.coef_.flatten()))
    )

    print(f"Coefficients comparison: {name1} vs {name2}")
    print(f"{'Index':<5} {name1:<20} {name2:<20}")
    max_len = max(len(coeffs1), len(coeffs2))
    for i in range(max_len):
        c1 = coeffs1[i] if i < len(coeffs1) else "N/A"
        c2 = coeffs2[i] if i < len(coeffs2) else "N/A"
        print(f"{i:<5} {c1:<20} {c2:<20}")


# Main execution
if __name__ == "__main__":
    sizes = [(100, 2), (1000, 10), (10000, 20)]

    for n_samples, n_features in sizes:
        print(f"\nTesting with {n_samples} samples and {n_features} features")
        X, y = generate_data(n_samples, n_features)

        # ModelCrafter's Linear Regression
        mc_model = test_model(LinearRegression(), X, y.ravel(), "ModelCrafter")

        # Scikit-learn's Linear Regression
        skl_model = test_model(SkLearnLinearRegression(), X, y, "ScikitLearn")

        # Compare predictions
        compare_predictions(mc_model, skl_model, X, y, "ModelCrafter", "ScikitLearn")

        # Compare coefficients
        # print_coefficients(skl_model, mc_model, 'ScikitLearn', 'ModelCrafter')
