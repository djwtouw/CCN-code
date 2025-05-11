"""
Script for simulations to compare multi-label classification methods. Data
generating processes are defined in dgp_config.py. Visualizations used to
analyze the results are made in simulation_study_visualizations.py.

Author: Daniel Touw
"""

import loky
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eccn import CCN
from eccn.metrics import (
    hamming_loss,
    macro_F1,
    micro_F1,
    negloglik,
    zero_one_loss,
)
from matplotlib import rc

from models.AdaBoostMH import adaboostmh_cv
from models.BinaryRelevance import br_cv
from models.ClassifierChain import cc_cv
from models.LinearRBRL import linear_rbrl_cv
from models.MultiLabelKNN import mlknn_cv
from models.MultiLabelTwinSVM import mltsvm_cv
from models.RandomKLabelsets import rakel_cv
from simulations.dgp_config import (
    dgp_3strong,
    dgp_3weak,
    dgp_6reverse,
    dgp_6cc,
    dgp_9strong,
)
from utils.data import dgp
from utils.folds import get_folds
from utils.parameter_grid import expand_grid
from utils.printing import pretty_label
from utils.save_load import save, load

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Computer Modern Sans Serif"],
    }
)

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)


def compute_probabilities(parameters, X):
    # Unpack parameters
    b, W, c = parameters

    # Number of labels
    num_labels = len(b)

    # Number of observations
    n = X.shape[0]

    # Initialize label matrix
    result = np.zeros((n, num_labels))

    # Compute label probabilities
    for i in range(num_labels):
        result[:, i] = b[i] + X @ W[i] + result[:, :i] @ c[i]
        result[:, i] = 1 / (1 + np.exp(-result[:, i]))

    return result


def simulation_iteration(dgp_settings, metric, methods, folds, i):
    # Print iteration number
    print(f"Iteration {i + 1}", end="\x1b[1K\r")

    # Generate training and testing data
    X_train, Y_train = dgp(
        dgp_settings["n"],
        dgp_settings["parameters"],
        reverse=dgp_settings["reverse"],
        realize_labels=dgp_settings["realize_labels"],
    )
    X_test, Y_test = dgp(
        1000,
        dgp_settings["parameters"],
        reverse=dgp_settings["reverse"],
        realize_labels=dgp_settings["realize_labels"],
    )

    # Dictionary of scoring functions
    metrics = dict(
        hamming=hamming_loss,
        zero_one=zero_one_loss,
        macro_F1=macro_F1,
        micro_F1=micro_F1,
        nll=negloglik,
    )

    # Dictionary with results
    scores = dict()

    # Binary relevance
    if "BR" in methods:
        # Get grid with tuning parameters
        grid_br = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
        grid_br = expand_grid(grid_br)

        # Cross validate and obtain optimal model
        model_br = br_cv(
            X_train, Y_train, folds=folds, grid=grid_br, metric=metric
        )
        Y_hat_br = model_br.predict(X_test)
        Y_hat_proba_br = model_br.predict_proba(X_test)

        # Compute scores
        if metric != "nll":
            score_br = metrics[metric](Y_test, Y_hat_br)
        elif metric == "nll":
            score_br = metrics[metric](Y_test, Y_hat_proba_br)

        # Score and tuning parameter(s)
        scores["BR"] = score_br
        scores["BR_alpha"] = model_br.alpha

    # Classifier chain
    if "CC" in methods:
        # Get grid with tuning parameters
        grid_cc = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
        grid_cc = expand_grid(grid_cc)

        # Cross validate and obtain optimal model
        model_cc = cc_cv(
            X_train, Y_train, folds=folds, grid=grid_cc, metric=metric
        )
        Y_hat_cc = model_cc.predict(X_test)
        Y_hat_proba_cc = model_cc.predict_proba(X_test)

        # Compute scores
        if metric != "nll":
            score_cc = metrics[metric](Y_test, Y_hat_cc)
        elif metric == "nll":
            score_cc = metrics[metric](Y_test, Y_hat_proba_cc)

        # Score and tuning parameter(s)
        scores["CC"] = score_cc
        scores["CC_alpha"] = model_cc.alpha

    # Multi-label k-nn
    if "MLKNN" in methods and metric != "nll":
        # Get grid with tuning parameters
        grid_mlknn = dict(
            k=[
                k
                for k in range(
                    5, int(np.ceil(np.sqrt(X_train.shape[0]))) + 1, 2
                )
            ]
        )
        grid_mlknn = expand_grid(grid_mlknn)

        # Cross validate and obtain optimal model
        model_mlknn = mlknn_cv(
            X_train, Y_train, folds=folds, grid=grid_mlknn, metric=metric
        )
        Y_hat_mlknn = model_mlknn.predict(X_test)

        # Compute score
        score_mlknn = metrics[metric](Y_test, Y_hat_mlknn)

        # Score and tuning parameter(s)
        scores["MLKNN"] = score_mlknn
        scores["MLKNN_k"] = model_mlknn.k

    # Multi-label twin SVM
    if "MLTSVM" in methods and metric != "nll":
        # Get grid with tuning parameters
        grid_mltsvm = dict(
            c_k=[0.75, 1.00, 1.25, 1.50, 1.75],
            lambda_param=[0.00, 0.50, 1.00, 1.50, 2.00],
        )
        grid_mltsvm = expand_grid(grid_mltsvm)

        # Cross validate and obtain optimal model
        model_mltsvm = mltsvm_cv(
            X_train, Y_train, folds=folds, grid=grid_mltsvm, metric=metric
        )
        Y_hat_mltsvm = model_mltsvm.predict(X_test)

        # Compute score
        score_mltsvm = metrics[metric](Y_test, Y_hat_mltsvm)

        # Score and tuning parameter(s)
        scores["MLTSVM"] = score_mltsvm
        scores["MLTSVM_ck"] = model_mltsvm.c_k
        scores["MLTSVM_alpha"] = model_mltsvm.lambda_param

    # (Random) k-labelsets
    if "RAKEL" in methods and metric != "nll":
        # Get grid with tuning parameters
        grid_rakel = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
        grid_rakel = expand_grid(grid_rakel)

        # Cross validate and obtain optimal model
        model_rakel = rakel_cv(
            X_train, Y_train, folds=folds, grid=grid_rakel, metric=metric
        )
        Y_hat_rakel = model_rakel.predict(X_test)

        # Compute score
        score_rakel = metrics[metric](Y_test, Y_hat_rakel)

        # Score and tuning parameter(s)
        scores["RAKEL"] = score_rakel
        scores["RAKEL_alpha"] = model_rakel.alpha

    # RBRL
    if "RBRL" in methods and metric != "nll":
        # Get grid with tuning parameters
        grid_rbrl = dict(
            lambda1=[1e-4, 1e-2, 1e0, 1e2],
            lambda2=[1e-4, 1e-2, 1e0, 1e2],
            lambda3=[1e-4, 1e-2, 1e0, 1e2],
        )
        grid_rbrl = expand_grid(grid_rbrl)

        # Cross validate and obtain optimal model
        model_rbrl = linear_rbrl_cv(
            X_train, Y_train, folds=folds, grid=grid_rbrl, metric=metric
        )
        Y_hat_rbrl = model_rbrl.predict(X_test)

        # Compute score
        score_rbrl = metrics[metric](Y_test, Y_hat_rbrl)

        # Score and tuning parameter(s)
        scores["RBRL"] = score_rbrl
        scores["RBRL_lambda1"] = model_rbrl.lambda1
        scores["RBRL_lambda2"] = model_rbrl.lambda2
        scores["RBRL_lambda3"] = model_rbrl.lambda3

    # AdaBoost.MH
    if "ADABOOSTMH" in methods:
        # Get grid with tuning parameters
        grid_adaboostmh = dict(ndt=[25, 50, 75, 100, 125])
        grid_adaboostmh = expand_grid(grid_adaboostmh)

        # Cross validate and obtain optimal model
        model_adaboostmh = adaboostmh_cv(
            X_train, Y_train, folds=folds, grid=grid_adaboostmh, metric=metric
        )
        Y_hat_adaboostmh = model_adaboostmh.predict(X_test)
        Y_hat_proba_adaboostmh = model_adaboostmh.predict_proba(X_test)

        # Compute scores
        if metric != "nll":
            score_adaboostmh = metrics[metric](Y_test, Y_hat_adaboostmh)
        elif metric == "nll":
            score_adaboostmh = metrics[metric](Y_test, Y_hat_proba_adaboostmh)

        # Score and tuning parameter(s)
        scores["ADABOOSTMH"] = score_adaboostmh
        scores["ADABOOSTMH_ndt"] = model_adaboostmh.ndt

    # Classifier chain network
    if "CCN" in methods:
        # Get grid with tuning parameters
        grid_ccn = dict(
            alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1], q=[1, 1.5, 2, 3, 5]
        )
        grid_ccn = expand_grid(grid_ccn)

        # Cross validate and obtain optimal model
        model_ccn = CCN().cv(
            X_train, Y_train, grid=grid_ccn, metric=metric, folds=folds
        )
        Y_hat_ccn = model_ccn.predict(X_test)
        Y_hat_proba_ccn = model_ccn.predict_proba(X_test)

        # Compute scores
        if metric != "nll":
            score_ccn = metrics[metric](Y_test, Y_hat_ccn)
        elif metric == "nll":
            score_ccn = metrics[metric](Y_test, Y_hat_proba_ccn)

        # Score and tuning parameter(s)
        scores["CCN"] = score_ccn
        scores["CCN_alpha"] = model_ccn.alpha
        scores["CCN_q"] = model_ccn.q

    return scores


# %%

# Folds for k-fold cross validation
cv_folds = get_folds(200, n_splits=5)

# Number of simulations
n_sim = 200

# Print core count
print(f"Using {loky.cpu_count(only_physical_cores=True)} cores")

# Set metrics
score_metrics = ["hamming", "zero_one", "macro_F1", "micro_F1", "nll"]

# Use these methods
use_methods = [
    "CCN",
    "BR",
    "CC",
    "ADABOOSTMH",
    "MLKNN",
    "MLTSVM",
    "RAKEL",
    "RBRL",
]

# Create directory for intermediate results
try:
    os.mkdir("simulations/results/intermediate")
    print("\nDirectory for intermediate results created successfully")
except FileExistsError:
    print("\nDirectory for intermediate results already exists")
except Exception as e:
    print(f"\nAn error occurred: {e}")

# Loop over the dgp types
for dgp_type in [
    dgp_3strong,
    dgp_3weak,
    dgp_6reverse,
    dgp_6cc,
    dgp_9strong,
]:
    # Print info
    print(f"\nData generating process: {dgp_type['name']}")

    # Check if results exist already
    if os.path.exists(f"simulations/results/{dgp_type['name']}.pkl"):
        print("Results exist already. Proceeding to next dgp")

        # Next
        continue

    # List of results
    final_results = []

    for score_metric in score_metrics:
        # Print info
        print(f"\nScoring metric: {pretty_label(score_metric)}")

        if os.path.exists(
            (
                f"simulations/results/intermediate/{dgp_type['name']}_"
                f"{score_metric}.pkl"
            )
        ):
            print("Results exist already. Proceeding to next metric")

            # Load results
            results = load(
                (
                    f"simulations/results/intermediate/{dgp_type['name']}_"
                    f"{score_metric}.pkl"
                )
            )

            # Append results
            final_results.append(results)

            # Next
            continue

        # Print time
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

        # Perform the simulations
        num_cpu = loky.cpu_count(only_physical_cores=True)

        # Container for the futures
        results = []

        with loky.ProcessPoolExecutor(num_cpu) as executor:
            for i in range(n_sim):
                results.append(
                    executor.submit(
                        simulation_iteration,
                        dgp_type,
                        score_metric,
                        use_methods,
                        cv_folds,
                        i,
                    )
                )

        # Collect results
        results = [r.result() for r in results]

        # Print ending time
        print(f"End time: {datetime.now().strftime('%H:%M:%S')}\n")

        # Get the results
        results = pd.DataFrame.from_dict(results)

        # Save the results
        save(
            results,
            (
                f"simulations/results/intermediate/{dgp_type['name']}_"
                f"{score_metric}.pkl"
            ),
        )

        # Append the results
        final_results.append(results)

        # Print intermediary results
        print(f"Metric: {pretty_label(score_metric)}")
        print("Average results:")
        print(
            results[
                [
                    method
                    for method in [
                        "CCN",
                        "BR",
                        "CC",
                        "ADABOOSTMH",
                        "MLKNN",
                        "MLTSVM",
                        "RAKEL",
                        "RBRL",
                    ]
                    if method in results.columns
                ]
            ].mean()
        )

    # Save results
    save(final_results, f"simulations/results/{dgp_type['name']}.pkl")

    # Remove intermediate results
    temp_files = os.listdir("./simulations/results/intermediate")
    rm_files = [f for f in temp_files if dgp_type["name"] in f]
    for file in rm_files:
        os.remove("./simulations/results/intermediate/" + file)
