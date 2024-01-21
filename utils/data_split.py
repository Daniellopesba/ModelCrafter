import pandas as pd
import numpy as np


def shuffle_data(X, y, random_state):
    """Shuffles the data and target synchronously."""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X.iloc[indices], y.iloc[indices]


def split_randomly(X, y, test_size):
    """Splits the data randomly into train and test sets."""
    split_idx = int(len(X) * (1 - test_size))
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


def split_conditionally(X, y, condition):
    """Splits the data based on a specified condition."""
    train_condition = X[condition["column"]] <= condition["threshold"]
    test_condition = X[condition["column"]] > condition["threshold"]
    return X[train_condition], X[test_condition], y[train_condition], y[test_condition]


def split_data(
    data=None,
    X=None,
    y=None,
    target_column=None,
    split_method="random",
    condition=None,
    test_size=0.2,
    random_state=42,
):
    """Main function to split data into training and testing sets."""
    if data is not None:
        if target_column is None:
            raise ValueError(
                "Target column name must be specified when using combined data."
            )
        y = data[target_column]
        X = data.drop(target_column, axis=1)
    elif X is None or y is None:
        raise ValueError("Feature matrix X and target vector y must be provided.")

    X, y = shuffle_data(X, y, random_state)

    if split_method == "random":
        return split_randomly(X, y, test_size)
    elif split_method == "condition":
        if (
            condition is None
            or "column" not in condition
            or "threshold" not in condition
        ):
            raise ValueError(
                "Condition must be specified correctly when using 'condition' split method."
            )
        return split_conditionally(X, y, condition)
    else:
        raise ValueError("Invalid split method. Choose 'random' or 'condition'.")
