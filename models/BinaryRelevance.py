import time
import numpy as np
from eccn.metrics import (
    hamming_loss,
    macro_F1,
    micro_F1,
    negloglik,
    zero_one_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def br_fit(X, Y, alpha):
    """
    Fit a binary relevance classifier based on logistic regression.

    :param X: 2D data matrix of shape (n_samples, n_features).
    :param Y: 2D label matrix of shape (n_samples, n_labels).
    :param alpha: Value of the penalty parameter.
    :return: The fitted model.
    """
    # The penalty parameter used by scikit-learn is defined in a different way
    # than the classifier chain network. This transforms alpha into one that
    # has an effect comparable to the one used in the network.
    C = X.shape[1] / (2 * alpha * X.shape[0])

    # Fit model
    model = OneVsRestClassifier(LogisticRegression(C=C, max_iter=10000))
    t0 = time.time()
    model.fit(X, Y)
    t1 = time.time()

    # Add value of the regularization parameter to the result
    model.alpha = alpha

    # Add training time
    model.training_time = t1 - t0

    return model


def br_cv(X, Y, folds, grid, metric):
    """
    Perform cross-validation to find the optimal value of the tuning parameter
    for binary relevance.

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
            model = br_fit(X[train, :], Y[train, :], alpha)

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

    # Select best value for alpha
    alpha = grid["alpha"][np.argmax(scores)]

    # Fit the model on the train set
    model = br_fit(X, Y, alpha)

    return model
