import numpy as np
import time
from least_squares import AnalyticalLeastSquaresRegression, MatrixLeastSquaresRegression


def generate_data(n):
    """
    Generates a linear dataset with Gaussian noise.

    Args:
            n (int): Number of data points.

    Returns:
            tuple: Generated x and y values as lists.
    """
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + np.random.normal(size=n)
    return x.tolist(), y.tolist()


def time_fit(regression_class, x, y):
    """
    Times the fit method of a regression class.

    Args:
            regression_class: The class to instantiate and time.
            x (list of float): x values.
            y (list of float): y values.

    Returns:
            float: Time taken to fit the model.
    """
    start_time = time.perf_counter()
    regression_class().fit(x, y)
    end_time = time.perf_counter()
    return end_time - start_time


# Simulation parameters
n_points = 10000  # Input Size
n_runs = 100  # Number of runs for averaging

x, y = generate_data(n_points)

times_analytical = []
times_matrix = []

for _ in range(n_runs):
    times_analytical.append(time_fit(AnalyticalLeastSquaresRegression, x, y))
    times_matrix.append(time_fit(MatrixLeastSquaresRegression, x, y))

max_time_analytical = max(times_analytical)
avg_time_analytical = sum(times_analytical) / n_runs
std_time_analytical = np.std(times_analytical)
min_time_analytical = min(times_analytical)

max_time_matrix = max(times_matrix)
avg_time_matrix = sum(times_matrix) / n_runs
std_time_matrix = np.std(times_matrix)
min_time_matrix = min(times_matrix)

# Print results in a formatted table
print(
    "Method               | Average Time (ms)  | Std Dev Time (ms) | Min Time (ms) | Max Time (ms)"
)
print(
    f"{'Analytical':<20} | {avg_time_analytical * 1000:>17.3f}  | {std_time_analytical * 1000:>16.3f}  | {min_time_analytical * 1000:>13.3f} | {max_time_analytical * 1000:>13.3f}"
)
print(
    f"{'Matrix':<20} | {avg_time_matrix * 1000:>17.3f}  | {std_time_matrix * 1000:>16.3f}  | {min_time_matrix * 1000:>13.3f} | {max_time_matrix * 1000:>13.3f}"
)
