import time
import numpy as np
from eccn.metrics import hamming_loss, macro_F1, micro_F1, zero_one_loss
from sklearn.linear_model import LogisticRegression
from skmultilearn.ensemble import RakelO


class RandomKLabelsets(RakelO):
    """
    Wrapper class for sk-multilearn's RakelO. This class overrides some member
    functions to improve their ease of use.
    """

    def predict(self, X):
        return np.asarray(super().predict(X).todense())

    def predict_proba(self, X):
        return np.asarray(super().predict_proba(X).todense())


def rakel_fit(X, Y, alpha):
    # The penalty parameter used by scikit-learn is defined in a different way
    # than the classifier chain network. This transforms alpha into one that
    # has an effect comparable to the one used in the network.
    C = X.shape[1] / (2 * alpha * X.shape[0])

    # Model count
    labelset_size = min(Y.shape[1] - 1, 3)

    # Fit model. The implementation of RAKEL makes one model more than the
    # model count, so for model_count, subtract 1 to get the desired number of
    # models
    model = RandomKLabelsets(
        LogisticRegression(C=C, max_iter=10000),
        base_classifier_require_dense=[True, True],
        labelset_size=labelset_size,
        model_count=2 * Y.shape[1] - 1,
    )
    t0 = time.time()
    model.fit(X, Y)
    t1 = time.time()

    # Add value of the regularization parameter to the result
    model.alpha = alpha

    # Add training time
    model.training_time = t1 - t0

    return model


def rakel_cv(X, Y, folds, grid, metric):
    """
    Perform cross-validation to find the optimal value of the tuning parameter
    for random k-labelsets.

    :param X: 2D data matrix of shape (n_samples, n_features).
    :param Y: 2D label matrix of shape (n_samples, n_labels).
    :param folds: Folds used for cross-validation.
    :param grid: Data frame containing combinations of tuning parameters
     (column name: alpha).
    :param metric: Scoring metric.
    :return: The model fitted using the optimal tuning parameter.
    """
    # Number of alphas in the grid
    n_alphas = grid.shape[0]

    # Initialize container for scores
    scores = np.empty(n_alphas)

    # Do the grid search
    for grid_index in range(grid.shape[0]):
        # Set hyperparameter
        alpha = grid["alpha"][grid_index]

        # Cross validation score
        score = 0

        # Loop through the folds
        for i in range(len(folds)):
            # Test and train indices
            test = folds[i]
            train = np.concatenate(folds[:i] + folds[(i + 1) :])

            # Fit the model on the train set
            model = rakel_fit(X[train, :], Y[train, :], alpha)

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

    # Select best value for alpha
    alpha = grid["alpha"][np.argmax(scores)]

    # Fit the model on the train set
    model = rakel_fit(X, Y, alpha)

    return model
