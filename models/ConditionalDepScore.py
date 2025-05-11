import numpy as np
from eccn.metrics import hamming_loss, micro_F1, negloglik

from models.LogisticRegression import logit_cv
from utils.folds import get_folds


def cd_score(X, Y, folds, grid, metric):
    # Score for the constrained model (without other labels)
    score_constrained = 0
    scores_constrained = []

    # Score for the unconstrained model (with other labels)
    score_unconstrained = 0
    scores_unconstrained = []

    # Vector with fold weights
    weights = []

    # Loop through the folds
    for i in range(len(folds)):
        # Test and train indices
        test = folds[i]
        train = np.concatenate(folds[:i] + folds[(i + 1) :])

        # Select subsets of data
        X_test = X[test, :]
        X_train = X[train, :]
        Y_test = Y[test, :]
        Y_train = Y[train, :]

        # Folds for inner loop of cross validation
        folds_inner = get_folds(len(train), n_splits=5)

        # Loop over the labels
        for label in range(Y.shape[1]):
            # Select variables
            y_test = Y_test[:, label]
            y_train = Y_train[:, label]
            Z_test = np.delete(Y_test, label, axis=1)
            Z_test = np.c_[X_test, Z_test]
            Z_train = np.delete(Y_train, label, axis=1)
            Z_train = np.c_[X_train, Z_train]

            # Cross-validate constrained model
            m_constrained = logit_cv(
                X_train, y_train, folds_inner, grid, metric
            )
            y_hat_constrained = m_constrained.predict(X_test)
            y_proba_constrained = m_constrained.predict_proba(X_test)[:, 1]

            # Cross-validate unconstrained model
            m_unconstrained = logit_cv(
                Z_train, y_train, folds_inner, grid, metric
            )
            y_hat_unconstrained = m_unconstrained.predict(Z_test)
            y_proba_unconstrained = m_unconstrained.predict_proba(Z_test)[:, 1]

            # Initialize variables
            s_constrained, s_unconstrained = None, None

            # Compute fold scores
            if metric == "hamming":
                s_constrained = hamming_loss(y_test, y_hat_constrained)
                s_unconstrained = hamming_loss(y_test, y_hat_unconstrained)
            elif metric == "nll":
                s_constrained = negloglik(y_test, y_proba_constrained)
                s_unconstrained = negloglik(y_test, y_proba_unconstrained)
            elif metric in ["micro_F1", "macro_F1", "F1"]:
                s_constrained = micro_F1(y_test, y_hat_constrained)
                s_unconstrained = micro_F1(y_test, y_hat_unconstrained)

            # Compute fold weight
            weight = len(test) / (len(train) + len(test)) / Y.shape[1]

            # Add scores
            score_constrained += weight * s_constrained
            score_unconstrained += weight * s_unconstrained

            # Also keep track of the individual scores
            scores_constrained.append(s_constrained)
            scores_unconstrained.append(s_unconstrained)
            weights.append(weight)

    # Transform objects
    scores_constrained = np.array(scores_constrained)
    scores_unconstrained = np.array(scores_unconstrained)
    weights = np.array(weights)

    # Compute final score
    result = None

    if metric in ["hamming", "nll"]:
        result = score_constrained - score_unconstrained
        # result = ((scores_unconstrained <= scores_constrained) * weights)
        # .sum()
    elif metric in ["micro_F1", "macro_F1", "F1"]:
        result = score_unconstrained - score_constrained
        # result = ((scores_unconstrained >= scores_constrained) * weights)
        # .sum()

    return result
