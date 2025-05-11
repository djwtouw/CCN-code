import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from datetime import datetime
from eccn import CCN
from eccn.metrics import hamming_loss, negloglik
from matplotlib import rc

from models.AdaBoostMH import adaboostmh_cv
from models.ConditionalDepScore import cd_score
from utils.folds import get_folds
from utils.parameter_grid import expand_grid
from utils.save_load import save


def cond_ent_mat_jun2019(Y):
    result = np.zeros((Y.shape[1], Y.shape[1]))

    for i in range(Y.shape[1]):
        for j in range(Y.shape[1]):
            if i == j:
                continue

            # Compute frequency table
            M = np.zeros((2, 2))
            # i=1, j=1
            M[1, 1] = (Y[:, i] * Y[:, j]).sum()
            # i=1, j=0
            M[1, 0] = (Y[:, i] * (1 - Y[:, j])).sum()
            # i=0, j=1
            M[0, 1] = ((1 - Y[:, i]) * Y[:, j]).sum()
            # i=0, j=0
            M[0, 0] = ((1 - Y[:, i]) * (1 - Y[:, j])).sum()
            M /= Y.shape[0]

            # Compute conditional entropy of j given i
            cond_ent = 0
            for ii in [0, 1]:
                for jj in [0, 1]:
                    if M[ii, jj] == 0:
                        continue
                    cond_ent -= M[ii, jj] * np.log2(M[ii, jj] / M[ii, :].sum())

            # Element (i, j) is conditional entropy of j given i
            result[i, j] = cond_ent

    return result


def order_jun2019(CE_matrix):
    # Numer of labels
    L = CE_matrix.shape[1]

    # Array of labels
    labels = np.array([i for i in range(L)])

    # Initialize order
    result = []

    # While not all labels have been assigned
    while len(result) < L:
        # Get minimum
        o_i = np.argmin(CE_matrix.sum(axis=0))

        # Append
        result = result + [labels[o_i]]

        # Modify H
        CE_matrix = np.delete(np.delete(CE_matrix, o_i, axis=0), o_i, axis=1)

        # Modify labels
        labels = np.delete(labels, o_i)

    # Selected label order
    result = np.array(result)

    return result


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Sans Serif"],
    }
)

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

# %%

# Set seed
np.random.seed(1234)
random.seed(4321)

# Data name
data_type = "emotions"
cv_metric = "hamming"

# Load data
data = pd.read_csv("applications/data/emotions_preprocessed.csv")
data_X = data.iloc[:, :-6].values
data_Y = data.iloc[:, -6:].values

# %%

# Print info
print(f"Analyzing {data_type} using {cv_metric}")

# Compute conditional dependency score
cv_grid = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
cv_grid = expand_grid(cv_grid)
cd = cd_score(
    data_X, data_Y, get_folds(data_X.shape[0], 10), cv_grid, "hamming"
)
print(f"Conditional dependency score: {cd:.3f}")

# %%


def nested_cv(X, Y, metric):
    # Functions for evaluation
    metrics = dict(
        hamming=hamming_loss,
        nll=negloglik,
    )

    # Get grid with tuning parameters
    grid_ada = dict(ndt=[25, 50, 75, 100, 125])
    grid_ada = expand_grid(grid_ada)
    grid_ccn = dict(
        alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1], q=[1, 1.5, 2, 3, 5]
    )
    grid_ccn = expand_grid(grid_ccn)

    # Folds for outer loop
    folds_outer = get_folds(X.shape[0], 10)

    # Dataframe with scores
    result = pd.DataFrame(
        np.zeros((len(folds_outer), 2)), columns=["Ada", "CCN"]
    )

    # Nested cross validation
    for i in range(len(folds_outer)):
        # Print info
        print(f"{datetime.now().strftime('%H:%M:%S')} Outer fold {i + 1}")

        # Train and test indices
        train = folds_outer[:i] + folds_outer[(i + 1) :]
        train = [t_i for t in train for t_i in t]
        test = folds_outer[i]

        # Variables
        X_train = X[train, :]
        X_test = X[test, :]
        Y_train = Y[train, :]
        Y_test = Y[test, :]

        # Inner cross validation folds
        folds_inner = get_folds(X_train.shape[0], 5)

        # Determine label order
        H = cond_ent_mat_jun2019(Y_train)
        order = order_jun2019(H)

        # Set label order
        Y_train = Y_train[:, order]
        Y_test = Y_test[:, order]

        # Print update
        print(f"Label order determined: {order}")

        # Adaboost.MH: cross validate and obtain optimal model
        model_ada = adaboostmh_cv(
            X_train, Y_train, folds=folds_inner, grid=grid_ada, metric=metric
        )
        Y_test_hat_ada = model_ada.predict(X_test)
        Y_test_proba_ada = model_ada.predict_proba(X_test)
        if metric != "nll":
            result.loc[i, "Ada"] = metrics[metric](Y_test, Y_test_hat_ada)
        else:
            result.loc[i, "Ada"] = metrics[metric](Y_test, Y_test_proba_ada)

        # CCN: cross validate and obtain optimal model
        model_ccn = CCN().cv(
            X_train, Y_train, grid=grid_ccn, metric=metric, folds=folds_inner
        )
        Y_test_hat_ccn = model_ccn.predict(X_test)
        Y_test_proba_ccn = model_ccn.predict_proba(X_test)
        if metric != "nll":
            result.loc[i, "CCN"] = metrics[metric](Y_test, Y_test_hat_ccn)
        else:
            result.loc[i, "CCN"] = metrics[metric](Y_test, Y_test_proba_ccn)

        # Print scores
        print(f"Scores up to fold {i + 1}:")
        print(result.round(4).iloc[: (i + 1), :])
        print()

    return result


scores = nested_cv(data_X, data_Y, cv_metric)
save(scores, f"applications/results/{data_type}_{cv_metric}_scores.pkl")


# %%


def get_label_effects(X, Y, metric):
    # Inner cross validation folds
    folds = get_folds(X.shape[0], 5)

    # Determine label order
    H = cond_ent_mat_jun2019(Y)
    order = order_jun2019(H)

    # Set label order
    Y = Y[:, order]

    # Grid for cross validating CCN
    grid = dict(
        alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1], q=[1, 1.5, 2, 3, 5]
    )
    grid = expand_grid(grid)

    # CCN: cross validate and obtain optimal model
    model = CCN().cv(X, Y, grid=grid, metric=metric, folds=folds)

    return model, order


model, order = get_label_effects(data_X, data_Y, cv_metric)
save(model, f"applications/results/{data_type}_{cv_metric}_model.pkl")
save(order, f"applications/results/{data_type}_{cv_metric}_order.pkl")
