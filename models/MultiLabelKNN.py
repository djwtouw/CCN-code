import time
import numpy as np
from eccn.metrics import hamming_loss, macro_F1, micro_F1, zero_one_loss
from skmultilearn.adapt import MLkNN


class MultiLabelKNN(MLkNN):
    """
    Wrapper class for sk-multilearn's MLkNN. In order to function, line 165 in
    mlknn.py needs to be replaced with: self.knn_ =
    NearestNeighbors(n_neighbors=self.k).fit(X). This class overrides some
    member functions to improve their ease of use.
    """

    def fit(self, X, Y):
        super().fit(X, Y.astype(int))

        return self

    def predict(self, X):
        return np.asarray(super().predict(X).todense())

    def predict_proba(self, X):
        return np.asarray(super().predict_proba(X).todense())


def mlknn_fit(X, Y, k, s=1.0):
    """
    Fit a multi-label k-nearest neighbors classifier.

    :param X: 2D data matrix of shape (n_samples, n_features).
    :param Y: 2D label matrix of shape (n_samples, n_labels).
    :param k: Number of nearest neighbors.
    :param s: Smoothing parameter. Default is 1.0.
    :return: Fitted model.
    """
    t0 = time.time()
    model = MultiLabelKNN(k=k, s=s).fit(X, Y)
    t1 = time.time()

    # Add training time
    model.training_time = t1 - t0

    return model


def mlknn_cv(X, Y, folds, grid, metric):
    """
    Perform cross-validation to find the optimal value of the tuning parameter
    for multi-label k-nearest neighbors.

    :param X: 2D data matrix of shape (n_samples, n_features).
    :param Y: 2D label matrix of shape (n_samples, n_labels).
    :param folds: Folds used for cross-validation.
    :param grid: Data frame containing combinations of tuning parameters
     (column name: k).
    :param metric: Scoring metric used for cross-validation.
    :return: Optimal model.
    """
    # Number of ks in the grid
    n_k = grid.shape[0]

    # Initialize container for scores
    scores = np.empty(n_k)

    # Do the grid search
    for grid_index in range(grid.shape[0]):
        # Set hyperparameter
        k = grid["k"][grid_index]

        # Cross validation score
        score = 0

        # Loop through the folds
        for i in range(len(folds)):
            # Test and train indices
            test = folds[i]
            train = np.concatenate(folds[:i] + folds[(i + 1) :])

            # Fit the model on the train set
            model = mlknn_fit(X[train, :], Y[train, :], k)

            # Make predictions for the test set
            Y_hat = model.predict(X[test, :])

            # Fold weight for the score
            fold_weight = Y_hat.size / Y.size

            # Add score
            if metric == "hamming":
                score -= hamming_loss(Y[test, :], Y_hat) * fold_weight
            elif metric == "zero_one":
                score -= zero_one_loss(Y[test, :], Y_hat) * fold_weight
            elif metric == "macro_F1":
                score += macro_F1(Y[test, :], Y_hat) * fold_weight
            elif metric == "micro_F1":
                score += micro_F1(Y[test, :], Y_hat) * fold_weight
            else:
                raise ValueError(f"{metric} is not a valid metric")

        # Add the score to the grid with the tuning parameters
        scores[grid_index] = score

    # Select best value for k
    k = grid["k"][np.argmax(scores)]

    # Fit the model on the train set
    model = mlknn_fit(X, Y, k)

    return model
