"""
Decision Trees Classifier
-------------------------
1. calculate_entropy(rows)
2. split_data(rows, feature, value)
3. information_gain(left_rows, right_rows, current_uncertainty)
4. find_best_split(rows)
"""
from typing import Tuple, Union

import numpy as np
import pandas as pd


def split_data(
    data: pd.DataFrame, feature: int, value: Union[int, float], is_numeric: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into two subsets based on the value of a feature.

    Parameters
    ----------
    data : pd.DataFrame
        The data to split.
    feature : int
        The index of the feature to split on.
    value : int or float
        The value to split on.
    is_numeric : bool
        Whether the feature is numeric or not.

    Returns
    -------
    tuple
        A tuple containing the left and right subsets of the data.
    """
    # Get the column name of the feature to split on
    col_name = data.columns[feature]
    # If the feature is numeric
    if is_numeric:
        # Split the data using a boolean mask
        left_rows = data[data[col_name] <= value]
        right_rows = data[data[col_name] > value]
    else:
        # Split the data using a boolean mask
        left_rows = data[data[col_name] == value]
        right_rows = data[data[col_name] != value]

    return left_rows, right_rows


def calculate_entropy(data: pd.DataFrame) -> float:
    """
    Calculate the entropy of the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to calculate the entropy of.

    Returns
    -------
    float
        The entropy of the data.
    """
    # Get the labels from the last column of the data
    labels = data.iloc[:, -1]
    # Compute the counts of each unique label
    counts = labels.value_counts()
    # Compute the probabilities of each label
    probabilities = counts / len(labels)
    # Compute the entropy using the formula for entropy
    entropy = -(probabilities * np.log2(probabilities)).sum()
    return entropy


def information_gain(
    left_rows: pd.DataFrame, right_rows: pd.DataFrame, current_entropy: float
) -> float:
    """
    Calculate the information gain from splitting the data into left and right subsets.

    Parameters
    ----------
    left_rows : pd.DataFrame
        The left subset of the data.
    right_rows : pd.DataFrame
        The right subset of the data.
    current_entropy : float
        The current entropy of the data.

    Returns
    -------
    float
        The information gain from splitting the data.
    """
    # Compute the proportion of rows in the left subset
    p = len(left_rows) / (len(left_rows) + len(right_rows))
    # Compute the information gain from splitting the data
    # by subtracting the weighted entropies of the left and right subsets
    # from the current entropy of the data
    return (
        current_entropy
        - p * calculate_entropy(left_rows)
        - (1 - p) * calculate_entropy(right_rows)
    )


def find_best_split(
    data: pd.DataFrame, current_entropy: float
) -> Tuple[float, Tuple[int, Union[int, float]], bool]:
    """
    Find the best feature and value to split the data on to maximize information gain.

    Parameters
    ----------
    data : pd.DataFrame
        The data to split.
    current_entropy : float
        The current entropy of the data.

    Returns
    -------
    tuple
        A tuple containing the best information gain, the best feature and value to split on,
        and a boolean indicating if the feature is numeric.
    """
    # Initialize the best information gain and feature to split on
    best_gain = 0
    best_feature = None
    # Get the number of features in the data
    n_features = data.shape[1] - 1
    # Initialize a variable to store if the feature is numeric
    is_numeric = None

    # Iterate over each feature in the data
    for feature in range(n_features):
        # Get the values of the current feature
        feature_values = data.iloc[:, feature]
        # Check if the feature is numeric
        is_numeric = feature_values.dtype in [np.int64, np.float64]
        # If the feature is numeric
        if is_numeric:
            # Convert the values to a numpy array
            values = feature_values.to_numpy()
            # Compute the midpoints between adjacent values
            values = (values[:-1] + values[1:]) / 2
        else:
            # Get the unique values of the feature
            values = set(feature_values)
        # Iterate over each value of the feature
        for value in values:
            # Split the data based on the current feature and value
            left_rows, right_rows = split_data(
                data, feature, value, is_numeric=is_numeric
            )
            # If either split is empty, skip this value
            if len(left_rows) == 0 or len(right_rows) == 0:
                continue
            # Compute the information gain from this split
            gain = information_gain(left_rows, right_rows, current_entropy)

            # If this gain is better than the current best gain
            if gain > best_gain:
                # Update the best gain and feature to split on
                best_gain = gain
                best_feature = feature, value

    # Return the best information gain and feature to split on,
    # along with a boolean indicating if the feature is numeric.
    return best_gain, best_feature, is_numeric


class DecisionTreeClassifier:
    def __init__(self, data, max_depth):
        self.tree = None
        self.data = data
        self.max_depth = max_depth

    def _train(self, data):
        rows = data.values
        if len(np.unique(rows[:, -1])) == 1:
            return rows[0][-1]
        if self.max_depth == 0:
            return max(set(data.values[:, -1]), key=data.values[:, -1].tolist().count)
        current_entropy = calculate_entropy(data)
        best_gain, best_feature, is_numeric = find_best_split(data, current_entropy)
        left_rows, right_rows = split_data(
            data, best_feature[0], best_feature[1], is_numeric=is_numeric
        )
        left_trees = self._train(left_rows)
        right_trees = self._train(right_rows)
        self.max_depth -= 1
        return {
            "feature": best_feature[0],
            "value": best_feature[1],
            "left": left_trees,
            "right": right_trees,
        }

    def fit(self, data):
        self.tree = self._train(data)
        return self

    def _predict(self, tree, row):
        # If the tree is not a dictionary, it is a leaf node, and we return its value
        if not isinstance(tree, dict):
            return tree

        # Get the feature and value to split on from the tree
        feature = tree["feature"]
        value = tree["value"]

        # Check if the value is a float (i.e., the feature is numeric)
        if isinstance(value, float):
            # If the feature value for this row is less than or equal to the split value,
            # follow the left branch of the tree
            if row[feature] <= value:
                return self._predict(tree["left"], row)
            # Otherwise, follow the right branch of the tree
            else:
                return self._predict(tree["right"], row)
        # If the value is not a float (i.e., the feature is categorical)
        else:
            # If the feature value for this row is equal to the split value,
            # follow the left branch of the tree
            if row[feature] == value:
                return self._predict(tree["left"], row)
            # Otherwise, follow the right branch of the tree
            else:
                return self._predict(tree["right"], row)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for given data using the trained decision tree.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame containing the data to make predictions for.

        Returns
        -------
        np.ndarray
            A numpy array containing the predicted values.
        """
        # If there is only one column in the data
        if data.shape[1] == 1:
            # Convert the column to a list
            data = data[0].tolist()
            # Make a prediction using the trained decision tree
            return self._predict(self.tree, data)
        else:
            # Iterate over each row of the Pandas dataframe and make predictions
            predictions = [
                self._predict(self.tree, row.tolist()) for _, row in data.iterrows()
            ]
            return np.array(predictions)


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "loves_popcorn": ["yes", "yes", "no", "no", "yes", "yes", "no"],
            "loves_soda": ["yes", "no", "yes", "yes", "yes", "no", "no"],
            "age": [7, 12, 18, 35, 38, 50, 83],
            "loves_cool_as_ice": ["no", "no", "yes", "yes", "yes", "no", "no"],
        }
    )

    clf = DecisionTreeClassifier(df, 5)
    classifier = clf.fit(df)
    # split_data(df, 2, 36.5, is_numeric=True)

    X_test = pd.DataFrame(
        [
            ["yes", "yes", 12.5],
            ["no", "no", 16.5],
            ["yes", "yes", 16.5],
            ["yes", "no", 16.5],
            ["no", "yes", 20],
        ]
    )
    clf.predict(X_test)
