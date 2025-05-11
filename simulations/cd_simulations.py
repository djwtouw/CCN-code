"""
Author: Daniel Touw
"""

import copy
import loky
import random

import numpy as np
from eccn import CCN
from eccn.metrics import hamming_loss, micro_F1

from models.BinaryRelevance import br_cv
from models.ConditionalDepScore import cd_score
from utils.data import dgp
from utils.folds import get_folds
from utils.parameter_grid import expand_grid
from utils.save_load import save, load
from simulations.dgp_config import dgp_6strong


def random_dgp():
    # Initialize basic settings for the dgp
    result = copy.deepcopy(dgp_6strong)

    # Random assignment of effects
    for i in range(len(result["parameters"][0])):
        result["parameters"][0][i] = np.random.normal(scale=4)

        for j in range(len(result["parameters"][1][0])):
            result["parameters"][1][i][j] = np.random.normal(scale=4)

    # Random assignment of label interdependencies
    for c_i in result["parameters"][2]:
        for j in range(c_i.size):
            c_i[j] = np.random.normal(scale=4)

    return result


def validated_random_dgp():
    # Generate random DGP
    result = random_dgp()

    # Validate the data generating process by checking if the imbalance in the
    # labels is not too extreme. Start with generating some data
    _, Y = dgp(
        1000,
        result["parameters"],
        reverse=result["reverse"],
        realize_labels=result["realize_labels"],
    )

    # If any label is too imbalanced, draw random data generating processes
    # until one is created with the desired properties
    while np.any(np.abs(Y.mean(axis=0) - 0.5) >= 0.35):
        # Generate random DGP again
        result = random_dgp()

        # Generate data again
        _, Y = dgp(
            1000,
            result["parameters"],
            reverse=result["reverse"],
            realize_labels=result["realize_labels"],
        )

    return result


def gen_data(dgp_settings):
    # Generate training data
    X, Y = dgp(
        dgp_settings["n"],
        dgp_settings["parameters"],
        reverse=dgp_settings["reverse"],
        realize_labels=dgp_settings["realize_labels"],
    )

    # Generate testing data
    X_test, Y_test = dgp(
        1000,
        dgp_settings["parameters"],
        reverse=dgp_settings["reverse"],
        realize_labels=dgp_settings["realize_labels"],
    )

    return X, Y, X_test, Y_test


def simulation_iteration(folds, metric, i):
    # Print iteration number
    print(f"Iteration {i + 1}", end="\x1b[1K\r")

    # Generate a validated DGP
    dgp_settings = validated_random_dgp()

    # Prepare lists used for output
    out_X = []
    out_Y = []
    out_X_test = []
    out_Y_test = []
    out_BR = []
    out_CCN = []

    # Repeat the simulation a number of times for each DGP
    for _ in range(10):
        # Generate training and testing data
        X, Y, X_test, Y_test = gen_data(dgp_settings)

        # Binary relevance
        # Get grid with tuning parameters
        grid_br = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
        grid_br = expand_grid(grid_br)

        # Cross validate and obtain optimal model
        model_br = br_cv(X, Y, folds=folds, grid=grid_br, metric=metric)
        Y_hat_br = model_br.predict(X_test)
        if metric == "hamming":
            score_br = hamming_loss(Y_test, Y_hat_br)
        elif metric == "micro_F1":
            score_br = micro_F1(Y_test, Y_hat_br)

        # Classifier chain network
        # Get grid with tuning parameters
        grid_ccn = dict(
            alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1], q=[1, 1.5, 2, 3, 5]
        )
        grid_ccn = expand_grid(grid_ccn)

        # Cross validate and obtain optimal model
        model_ccn = CCN().cv(X, Y, grid=grid_ccn, metric=metric, folds=folds)
        Y_hat_ccn = model_ccn.predict(X_test)
        if metric == "hamming":
            score_ccn = hamming_loss(Y_test, Y_hat_ccn)
        elif metric == "micro_F1":
            score_ccn = micro_F1(Y_test, Y_hat_ccn)

        # Track important variables
        out_X.append(X)
        out_Y.append(Y)
        out_X_test.append(X_test)
        out_Y_test.append(Y_test)
        out_BR.append(score_br)
        out_CCN.append(score_ccn)

    # Gather details
    result = dict(
        X=out_X,
        Y=out_Y,
        X_test=out_X_test,
        Y_test=out_Y_test,
        BR=out_BR,
        CCN=out_CCN,
        dgp=dgp_settings,
    )

    return result


# Global settings:
# Metric used for evaluation
cv_metric = "hamming"
# Number of simulations
n_sim = 100

# %%

# Create random data generating processes and use these to generate random data
# sets. These data sets are then analyzed with binary relevance and the
# classifier chain network to obtain performance differences.


def simulate_excess_performance():
    print("Simulating excess performance")

    # There are multiple random processes each drawing from different seeds
    np.random.seed(49)
    random.seed(12)

    # Folds for k-fold cross validation
    cv_folds = get_folds(200, n_splits=5)

    # Number of cores to set to work
    num_cpu = loky.cpu_count(only_physical_cores=True)

    # Container for the futures
    results = []

    with loky.ProcessPoolExecutor(num_cpu) as executor:
        for i in range(n_sim):
            # Send function to thread
            results.append(
                executor.submit(
                    simulation_iteration,
                    cv_folds,
                    cv_metric,
                    i,
                )
            )

        # Collect results
        results = [r.result() for r in results]

        # Save results
        save(results, f"simulations/results/conditional_deps_{cv_metric}.pkl")

    # Print info
    print("Simulation completed")

    return None


# Run the simulation
simulate_excess_performance()

# %%

# Use the performance gap between models that do not use all other labels to
# model a label and models that do use all other labels to model a label. The
# gap indicates whether using additional labels improves predictions. This
# approach is independent of label order, as it uses the realizations of all
# labels in modeling the outcomes.


# Function used in parallel computations
def compute_cd_scores(data, folds, grid, metric, i):
    # Print info
    print(f"Iteration {i + 1}", end="\x1b[1K\r")

    # List of scores
    result = []

    # Compute conditional dependency scores
    for i in range(len(data["BR"])):
        s = cd_score(data["X"][i], data["Y"][i], folds, grid, metric)
        result.append(s)

    return result


def simulate_cd_scores():
    print("Simulating conditional dependency scores")

    # Get grid with tuning parameters
    cv_grid = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
    cv_grid = expand_grid(cv_grid)

    # Load results
    results = load(f"simulations/results/conditional_deps_{cv_metric}.pkl")

    # Folds used for nested cross-validation
    folds_outer = get_folds(results[0]["X"][0].shape[0], n_splits=10)

    # Number of parallel jobs
    num_cpu = loky.cpu_count(only_physical_cores=True)

    # Containers for the futures
    scores = []

    # Fit the models in parallel
    with loky.ProcessPoolExecutor(num_cpu) as executor:
        for i, result in enumerate(results):
            # Send function to thread
            scores.append(
                executor.submit(
                    compute_cd_scores,
                    result,
                    folds_outer,
                    cv_grid,
                    cv_metric,
                    i,
                )
            )

    # Collect results
    scores = [m.result() for m in scores]

    # Save results
    save(scores, f"simulations/results/cd_scores_{cv_metric}.pkl")

    # Print info
    print("Simulation completed")

    return None


# Run the simulation
simulate_cd_scores()
