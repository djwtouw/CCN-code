import matlab.engine
import numpy as np
import os

from eccn.metrics import hamming_loss, macro_F1, micro_F1, zero_one_loss


class LinearRBRL:
    def __init__(self, lambda1, lambda2, lambda3, n_iter=2000):
        # Check presence of Matlab source files
        if not os.path.exists("packages/rbrl"):
            msg = (
                "RBRL source files not found. Make sure to download them "
                "from https://github.com/GuoqiangWoodrowWu/RBRL and place "
                "them in packages/rbrl"
            )
            raise RuntimeError(msg)
        if not os.path.exists("packages/rbrl/train_linear_RBRL_APG.m"):
            msg = (
                "RBRL source files not found. Make sure that "
                "packages/rbrl/train_linear_RBRL_APG.m exists"
            )
            raise RuntimeError(msg)
        if not os.path.exists("packages/rbrl/Predict.m"):
            msg = (
                "RBRL source files not found. Make sure that "
                "packages/rbrl/Predict.m exists"
            )
            raise RuntimeError(msg)

        # Initialize engine
        self.__eng = matlab.engine.start_matlab()
        self.__eng.addpath("packages/rbrl")

        # Number of iterations
        self.n_iter = n_iter

        # Tuning parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Initialize other parameters
        self.W = None
        self.n_labels = None

        pass

    def fit(self, X, Y):
        # Get number of labels
        self.n_labels = Y.shape[1]

        # Make sure that Y is in [-1, 1]
        Y_internal = (Y - Y.min()) / (Y.max() - Y.min()) * 2 - 1

        # Additional columns that indicate a homogenous observation
        y_col0 = 2 * np.all(Y_internal == -1, 1) - 1
        y_col1 = 2 * np.all(Y_internal == 1, 1) - 1

        # Add columns
        Y_internal = np.c_[Y_internal, y_col0, y_col1]

        # Fit model
        self.W = self.__eng.train_linear_RBRL_APG(
            X,
            Y_internal,
            self.lambda1,
            self.lambda2,
            self.lambda3,
            self.n_iter,
        )

        # Throw error if things go wrong
        if np.isnan(self.W).any():
            raise ValueError("NaNs detected in RBRL estimate")

        return self

    def predict(self, X):
        # Get prediction
        pred = self.__eng.Predict(X, self.W)

        # Convert prediction
        py_pred = (np.array(pred) + 1) / 2

        return py_pred[:, : self.n_labels]

    def __del__(self):
        self.__eng.quit()


def linear_rbrl_cv(X, Y, folds, grid, metric):
    """
    Perform cross-validation to find the optimal value of the tuning parameters
    for LinearRBRL.

    :param X: 2D data matrix of shape (n_samples, n_features).
    :param Y: 2D label matrix of shape (n_samples, n_labels).
    :param folds: Folds used for cross-validation.
    :param grid: Data frame containing combinations of tuning parameters
     (column names: lambda1, lambda2, lambda3).
    :param metric: Scoring metric used for cross-validation.
    :return: Optimal model.
    """
    # Number of tuning parameter combinations
    n_settings = grid.shape[0]

    # Initialize container for scores
    scores = np.empty(n_settings)

    # Initialize model
    model = LinearRBRL(0, 0, 0)

    # Do the grid search
    for grid_index in range(grid.shape[0]):
        # Set hyperparameter
        lambda1 = grid["lambda1"][grid_index]
        lambda2 = grid["lambda2"][grid_index]
        lambda3 = grid["lambda3"][grid_index]

        model.lambda1 = lambda1
        model.lambda2 = lambda2
        model.lambda3 = lambda3

        # Cross validation score
        score = 0

        # Loop through the folds
        for i in range(len(folds)):
            # Test and train indices
            test = folds[i]
            train = np.concatenate(folds[:i] + folds[(i + 1) :])

            # Fit the model on the train set
            model.fit(X[train, :], Y[train, :])

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
    lambda1 = grid["lambda1"][np.argmax(scores)]
    lambda2 = grid["lambda2"][np.argmax(scores)]
    lambda3 = grid["lambda3"][np.argmax(scores)]

    model.lambda1 = lambda1
    model.lambda2 = lambda2
    model.lambda3 = lambda3

    # Fit the model on the train set
    model = model.fit(X, Y)

    return model
