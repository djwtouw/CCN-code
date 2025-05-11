import numpy as np
from eccn.metrics import (
    hamming_loss,
    micro_F1,
    negloglik,
    zero_one_loss,
)
from sklearn.linear_model import LogisticRegression


def logit(X, y, alpha):
    """
    Fit a logistic regression.

    Parameters
    ----------
    X : np.ndarray
        2D data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Binary outcomes of shape (n_samples,).
    alpha : double
        Value of the penalty parameter

    Returns
    -------
    The fitted model.
    """
    # The penalty parameter used by scikit-learn is defined in a different way
    # than the classifier chain network. This transforms alpha into one that
    # has an effect comparable to the one used in the network.
    C = X.shape[1] / (2 * alpha * X.shape[0])

    # Fit model
    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X, y)

    # Add value of the regularization parameter to the result
    model.alpha = alpha

    return model


def logit_cv(X, y, folds, grid, metric):
    """
    Perform cross-validation to find the optimal value of the tuning parameter
    for binary relevance.

    Parameters
    ----------
    X : np.ndarray
        2D data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Binary outcomes of shape (n_samples,).
    folds : list
        Folds used for cross-validation.
    grid : pd.dataframe
        Data frame containing combinations of tuning parameters (column name:
         alpha).
    metric: string
        Scoring metric.

    Returns
    -------
    The model fitted using the optimal tuning parameter.
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
            model = logit(X[train, :], y[train], alpha)

            # Make predictions for the test set
            y_hat = model.predict(X[test, :])
            y_hat_proba = model.predict_proba(X[test, :])[:, 1]

            # Fold weight for the score
            fold_weight = y_hat.size / y.size

            # Add score
            if metric == "hamming":
                score -= hamming_loss(y[test], y_hat) * fold_weight
            elif metric == "zero_one":
                score -= zero_one_loss(y[test], y_hat) * fold_weight
            elif metric in ["macro_F1", "micro_F1", "F1"]:
                score += micro_F1(y[test], y_hat) * fold_weight
            elif metric == "nll":
                score -= negloglik(y[test], y_hat_proba) * fold_weight
            else:
                raise ValueError(f"{metric} is not a valid metric")

        # Add the score to the grid with the tuning parameters
        scores[grid_index] = score

    # Select best value for alpha
    alpha = grid["alpha"][np.argmax(scores)]

    # Fit the model on the train set
    model = logit(X, y, alpha)

    return model
