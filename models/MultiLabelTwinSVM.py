import time
import numpy as np
from eccn.metrics import hamming_loss, macro_F1, micro_F1, zero_one_loss
from skmultilearn.adapt import MLTSVM


class MultiLabelTwinSVM(MLTSVM):
    """
    Wrapper class for sk-multilearn's MLTSVM. This class overrides some
    member functions to improve their ease of use.
    """

    def fit(self, X, Y):
        super().fit(np.asmatrix(X), Y)

        return self


def mltsvm_fit(
    X,
    Y,
    c_k,
    sor_omega=0.2,
    threshold=1e-6,
    lambda_param=1.0,
    max_iteration=500,
):
    """
    Fit a multi-label twin support vector machine.

    :param X: 2D data matrix of shape (n_samples, n_features).
    :param Y: 2D label matrix of shape (n_samples, n_labels).
    :param c_k: Empirical risk penalty parameter that determines the trade-off
    between the loss terms.
    :param sor_omega: The smoothing parameter. Default is 0.2 (based on
     footnote 3 in Chen et al. (2016)).
    :param threshold: Threshold above which a label should be assigned. Default
    is 1e-6.
    :param lambda_param: The regularization parameter. Default is 1.0.
    :param max_iteration: Maximum number of iterations to use in successive
     overrelaxation. Default is 500.
    :return: Fitted model.
    """
    model = MultiLabelTwinSVM(
        c_k=c_k,
        sor_omega=sor_omega,
        threshold=threshold,
        lambda_param=lambda_param,
        max_iteration=max_iteration,
    )
    t0 = time.time()
    model.fit(X, Y)
    t1 = time.time()

    # Add training time
    model.training_time = t1 - t0

    return model


def mltsvm_cv(X, Y, folds, grid, metric):
    """
    Perform cross-validation to find the optimal value of the tuning parameters
    for multi-label twin suport vector machines.

    :param X: 2D data matrix of shape (n_samples, n_features).
    :param Y: 2D label matrix of shape (n_samples, n_labels).
    :param folds: Folds used for cross-validation.
    :param grid: Data frame containing combinations of tuning parameters
     (column names: c_k, lambda_param).
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
        c_k = grid["c_k"][grid_index]
        lambda_param = grid["lambda_param"][grid_index]

        # Cross validation score
        score = 0

        # Loop through the folds
        for i in range(len(folds)):
            # Test and train indices
            test = folds[i]
            train = np.concatenate(folds[:i] + folds[(i + 1) :])

            # Fit the model on the train set
            model = mltsvm_fit(
                X[train, :], Y[train, :], c_k=c_k, lambda_param=lambda_param
            )

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

    # Select best value for tuning parameters
    c_k = grid["c_k"][np.argmax(scores)]
    lambda_param = grid["lambda_param"][np.argmax(scores)]

    # Fit the model on the train set
    model = mltsvm_fit(X, Y, c_k=c_k, lambda_param=lambda_param)

    return model
