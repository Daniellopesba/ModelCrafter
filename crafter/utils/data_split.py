import pandas as pd
import numpy as np


def shuffle_data(X, y, random_state=None):
    """Shuffles the data and target synchronously, with an optional random state."""
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X.iloc[indices], y.iloc[indices]


def split_randomly(X, y, test_size, random_state=None):
    """Splits the data randomly into train and test sets, with test size validation and optional random state."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X, y = shuffle_data(X, y, random_state)
    split_idx = int(len(X) * (1 - test_size))
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


def split_conditionally(X, y, conditions):
    """Splits the data based on specified conditions, now supporting multiple conditions."""
    if not isinstance(conditions, list):
        conditions = [conditions]  # Ensure conditions are iterable

    # Building the condition query in a way that pandas.eval() can process
    condition_queries = []
    for condition in conditions:
        column, operator, value = (
            condition["column"],
            condition["operator"],
            condition["value"],
        )
        if isinstance(value, str):
            # If value is a string, it should be enclosed in quotes for the eval expression
            condition_query = f'{column} {operator} "{value}"'
        else:
            condition_query = f"{column} {operator} {value}"
        condition_queries.append(condition_query)

    condition_query = " & ".join(condition_queries)

    train_condition = X.eval(condition_query)
    test_condition = ~train_condition

    return X[train_condition], X[test_condition], y[train_condition], y[test_condition]


def split_data(
    data=None,
    X=None,
    y=None,
    target_column=None,
    split_method="random",
    conditions=None,
    test_size=0.2,
    random_state=42,
):
    """Main function to split data into training and testing sets. Allows defining only y, assuming the remainder as X."""
    if data is not None:
        if target_column is not None:
            y = data[target_column]
            X = data.drop(target_column, axis=1)
        elif y is not None and isinstance(y, str):
            if y not in data.columns:
                raise ValueError(f"Specified target column '{y}' not found in data.")
            X = data.drop(y, axis=1)
            y = data[y]
        else:
            raise ValueError(
                "Target column name must be specified or y must be provided as a column name."
            )
    elif X is None or y is None:
        raise ValueError(
            "Feature matrix X and target vector y must be provided, or a combined DataFrame with a target column specified."
        )

    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("X must be a pandas DataFrame and y must be a pandas Series.")

    if split_method == "random":
        return split_randomly(X, y, test_size, random_state)
    elif split_method == "condition":
        if conditions is None:
            raise ValueError(
                "Conditions must be specified correctly when using 'condition' split method."
            )
        return split_conditionally(X, y, conditions)
    else:
        raise ValueError("Invalid split method. Choose 'random' or 'condition'.")
