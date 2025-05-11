import numpy as np
from eccn.metrics import (
    hamming_loss,
    macro_F1,
    micro_F1,
    negloglik,
    zero_one_loss,
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class AdaBoostMH:
    def __init__(self, ndt):
        self.ndt = ndt
        self.models = []

    def fit(self, X, Y):
        # Number of labels
        L = Y.shape[1]

        for i in range(L):
            self.models.append(
                AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=1),
                    algorithm="SAMME",
                    n_estimators=self.ndt,
                )
            )
            self.models[i].fit(X, Y[:, i])

        return self

    def predict_proba(self, X):
        # Number of labels
        L = len(self.models)

        # Result
        result = np.zeros((X.shape[0], L))

        for i in range(L):
            # Just in case: get class index as two columns are returned
            class1_idx = np.where(self.models[i].classes_ > 1 - 1e-6)[0][0]

            # Make a prediction
            result[:, i] = self.models[i].predict_proba(X)[:, class1_idx]

        return result

    def predict(self, X):
        # Number of labels
        L = len(self.models)

        # Result
        result = np.zeros((X.shape[0], L))

        for i in range(L):
            # Make a prediction
            result[:, i] = self.models[i].predict(X)

        return result


def adaboostmh_cv(X, Y, folds, grid, metric):
    """
    Perform cross-validation to find the optimal value of the tuning parameters
    for adaboost.MH.

    :param X: 2D data matrix of shape (n_samples, n_features).
    :param Y: 2D label matrix of shape (n_samples, n_labels).
    :param folds: Folds used for cross-validation.
    :param grid: Data frame containing combinations of tuning parameters
     (column name: ndt).
    :param metric: Scoring metric used for cross-validation.
    :return: Optimal model.
    """
    # Number of tuning parameter combinations
    n_settings = grid.shape[0]

    # Initialize container for scores
    scores = np.empty(n_settings)

    # Do the grid search
    for grid_index in range(grid.shape[0]):
        # Set hyperparameter
        ndt = grid["ndt"][grid_index]

        # Cross validation score
        score = 0

        # Loop through the folds
        for i in range(len(folds)):
            # Test and train indices
            test = folds[i]
            train = np.concatenate(folds[:i] + folds[(i + 1) :])

            # Fit the model on the train set
            model = AdaBoostMH(ndt=ndt).fit(X[train, :], Y[train, :])

            # Make predictions for the test set
            Y_hat = model.predict(X[test, :])
            Y_hat_proba = model.predict_proba(X[test, :])

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
            elif metric == "nll":
                score -= negloglik(Y[test, :], Y_hat_proba) * fold_weight
            else:
                raise ValueError(f"{metric} is not a valid metric")

        # Add the score to the grid with the tuning parameters
        scores[grid_index] = score

    # Select best value for tuning parameters
    ndt = grid["ndt"][np.argmax(scores)]

    # Fit the model on the train set
    model = AdaBoostMH(ndt=ndt).fit(X, Y)

    # Add optimal tuning parameter
    model.ndt = ndt

    return model
