import copy
import time
import os
from datetime import datetime
import numpy as np
import pandas as pd

from eccn import CCN

from models.BinaryRelevance import br_fit
from models.ClassifierChain import cc_fit
from models.MultiLabelKNN import mlknn_fit
from models.MultiLabelTwinSVM import mltsvm_fit
from models.RandomKLabelsets import rakel_fit
from models.AdaBoostMH import AdaBoostMH
from models.LinearRBRL import LinearRBRL
from utils.save_load import load, save
from utils.data import dgp


dgp_base = dict(
    n=200,
    reverse=False,
    contamination_type=0,
    contamination_prob=0,
    realize_labels=False,
)


def select_tuning_parameters(files):
    # Initialize result
    sim_res = []

    # Load simulation results and select those for Hamming loss
    for file in files:
        temp_res = load(f"simulations/results/{file}.pkl")[0]
        sim_res.append(temp_res)

    # Concatenate results
    sim_res = pd.concat(sim_res).reset_index(drop=True)

    # Get rid of the columns with the scores, first get the column names
    names = [s.split("_")[0] for s in sim_res.columns]
    methods = list(set(names))

    # Drop the columns
    sim_res.drop(methods, axis=1, inplace=True)

    # Initialize container for most used tuning parameters
    params = sim_res.mode(axis=0)

    for method in methods:
        # Find columns corresponding to tuning parameters for the current
        # method
        param_columns = [
            c for c in sim_res.columns if method == c.split("_")[0]
        ]

        # If only one tuning parameter is found, we already have the mode
        if len(param_columns) == 1:
            continue

        # Else, select only the columns with tuning parameters for the current
        # method
        sim_res_temp = sim_res[param_columns]

        # Get the unique rows and their counts
        sim_res_temp_u = np.unique(
            sim_res_temp.values, axis=0, return_counts=True
        )

        # Select the set of tuning parameters that occurs most often
        params_temp = sim_res_temp_u[0][np.argmax(sim_res_temp_u[1])]

        # Add parameters to the parameter vector
        params[param_columns] = params_temp

    return methods, params.iloc[0, :]


def random_dgp(m, L):
    # Initialize basic settings for the dgp
    result = copy.deepcopy(dgp_base)

    # Create containers for parameters based on m and L
    parameters_b = [0 for _ in range(L)]
    parameters_W = [np.zeros(m) for _ in range(L)]
    parameters_c = [np.zeros(i) for i in range(L)]

    # Add to the dgp configuration
    result["parameters"] = (parameters_b, parameters_W, parameters_c)

    # Random assignment of effects
    for i in range(L):
        result["parameters"][0][i] = np.random.normal(scale=4)

        for j in range(m):
            result["parameters"][1][i][j] = np.random.normal(scale=4)

    # Random assignment of label interdependencies
    for c_i in result["parameters"][2]:
        for j in range(c_i.size):
            c_i[j] = np.random.normal(scale=4)

    return result


def validated_random_dgp(m, L):
    # Generate random DGP
    result = random_dgp(m, L)

    # Validate the data generating process by checking if the imbalance in the
    # labels is not too extreme. Start with generating some data
    _, Y = dgp(
        1000,
        result["parameters"],
        reverse=result["reverse"],
        contamination_type=result["contamination_type"],
        contamination_prob=result["contamination_prob"],
        realize_labels=result["realize_labels"],
    )

    # If any label is too imbalanced, draw random data generating processes
    # until one is created with the desired properties
    while np.any(np.abs(Y.mean(axis=0) - 0.5) >= 0.35):
        # Generate random DGP again
        result = random_dgp(m, L)

        # Generate data again
        _, Y = dgp(
            1000,
            result["parameters"],
            reverse=result["reverse"],
            contamination_type=result["contamination_type"],
            contamination_prob=result["contamination_prob"],
            realize_labels=result["realize_labels"],
        )

    return result


def simulation_iteration(methods, tuning_params, m, L, n_rep):
    # Print info
    print(f"Dimensions: [m, L] = [{m}, {L}]")

    # Print time
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

    # Dictionary with results
    timings = dict()

    # Initialize results
    for method in methods:
        timings[method] = np.zeros(n_rep)

    # Repeat for n_rep times
    for i in range(n_rep):
        # Print iteration number
        print(f"Iteration {i + 1}", end="\x1b[1K\r")

        # Generate dgp parameters
        dgp_settings = validated_random_dgp(m, L)

        # Generate data
        X, Y = dgp(
            dgp_settings["n"],
            dgp_settings["parameters"],
            reverse=dgp_settings["reverse"],
            contamination_type=dgp_settings["contamination_type"],
            contamination_prob=dgp_settings["contamination_prob"],
            realize_labels=dgp_settings["realize_labels"],
        )

        # Binary relevance
        if "BR" in methods:
            # Train model
            model_br = br_fit(X, Y, alpha=tuning_params["BR_alpha"])

            # Training time
            timings["BR"][i] = model_br.training_time

            del model_br

        # Classifier chain
        if "CC" in methods:
            # Train model
            model_cc = cc_fit(X, Y, alpha=tuning_params["CC_alpha"])

            # Training time
            timings["CC"][i] = model_cc.training_time

            del model_cc

        # Multi-label k-nn
        if "MLKNN" in methods:
            # Train model
            model_mlknn = mlknn_fit(X, Y, k=int(tuning_params["MLKNN_k"]))

            # Training time
            timings["MLKNN"][i] = model_mlknn.training_time

            del model_mlknn

        # Multi-label twin SVM
        if "MLTSVM" in methods:
            # Train model
            model_mltsvm = mltsvm_fit(
                X,
                Y,
                c_k=tuning_params["MLTSVM_ck"],
                lambda_param=tuning_params["MLTSVM_alpha"],
            )

            # Training time
            timings["MLTSVM"][i] = model_mltsvm.training_time

            del model_mltsvm

        # (Random) k-labelsets
        if "RAKEL" in methods:
            # Train model
            model_rakel = rakel_fit(X, Y, alpha=tuning_params["RAKEL_alpha"])

            # Training time
            timings["RAKEL"][i] = model_rakel.training_time

            del model_rakel

        # AdaBoost.MH
        if "ADABOOSTMH" in methods:
            # Train model
            model_adaboostmh = AdaBoostMH(
                ndt=int(tuning_params["ADABOOSTMH_ndt"])
            )
            t0 = time.time()
            model_adaboostmh.fit(X, Y)
            t1 = time.time()

            # Training time
            timings["ADABOOSTMH"][i] = t1 - t0

            del model_adaboostmh

        # LinearRBRL
        if "RBRL" in methods:
            # Train model
            model_rbrl = LinearRBRL(
                lambda1=tuning_params["RBRL_lambda1"],
                lambda2=tuning_params["RBRL_lambda2"],
                lambda3=tuning_params["RBRL_lambda3"],
            )
            t0 = time.time()
            model_rbrl.fit(X, Y)
            t1 = time.time()

            # Training time
            timings["RBRL"][i] = t1 - t0

            del model_rbrl, t0, t1

        # Classifier chain network
        if "CCN" in methods:
            # Train model
            model_ccn = CCN(
                init="random",
                best_of=1,
                alpha=tuning_params["CCN_alpha"],
                q=tuning_params["CCN_q"],
            )
            t0 = time.time()
            model_ccn.fit(X, Y)
            t1 = time.time()

            # Training time
            timings["CCN"][i] = t1 - t0

            del model_ccn, t0, t1

    # Print ending time
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}\n")

    return timings


def ccn_n_params(m, L):
    return (m + 1) * L + L * (L - 1) // 2


# %%

# Get available methods and most frequent tuning parameters
methods, tuning_params = select_tuning_parameters(
    ["3strong", "3weak", "6reverse", "6cc", "9strong"]
)

# Number of repetitions
n = 25

# Sequences for m and L
m_seq = [5 + 5 * i for i in range(10 + 1)]
L_seq = [5 + 3 * i for i in range(5 + 1)]

# Base values
m_base = 5
L_base = 5

# CCN parameter information
print(f"Maximum parameters for L: {ccn_n_params(m_base, L_seq[-1])}")
print(f"Maximum parameters for m: {ccn_n_params(m_seq[-1], L_base)}\n")

# Subset of methods to be used
# methods = ["BR", "CC", "CCN"]

# Containers for results
results_m = []
results_L = []

# %%

# Do timing simulations for m
for m in m_seq:
    if os.path.exists(
        f"simulations/results/intermediate/timings_{m}_{L_base}.pkl"
    ):
        print("Results exist already. Proceeding to next m")

        # Load results
        res = load(
            f"simulations/results/intermediate/timings_{m}_{L_base}.pkl"
        )

        # Append results
        results_m.append(res)

        # Next
        continue

    # Perform timing simulation
    res = simulation_iteration(methods, tuning_params, m=m, L=L_base, n_rep=n)

    # Add m
    res["m"] = m

    # Turn into pandas dataframe
    res = pd.DataFrame.from_dict(res)

    # Save intermediate result
    save(res, f"simulations/results/intermediate/timings_{m}_{L_base}.pkl")

    # Add to all results
    results_m.append(res)

# Save timings
save(results_m, "simulations/results/timings_m.pkl")

# Remove intermediate results
temp_files = os.listdir("./simulations/results/intermediate")
rm_files = [f for f in temp_files if "timings_" in f]
for file in rm_files:
    os.remove("./simulations/results/intermediate/" + file)

# %%

# Do timing simulations for L
for L in L_seq:
    if os.path.exists(
        f"simulations/results/intermediate/timings_{m_base}_{L}.pkl"
    ):
        print("Results exist already. Proceeding to next L\n")

        # Load results
        res = load(
            f"simulations/results/intermediate/timings_{m_base}_{L}.pkl"
        )

        # Append results
        results_L.append(res)

        # Next
        continue

    # Perform timing simulation
    res = simulation_iteration(methods, tuning_params, m=m_base, L=L, n_rep=n)

    # Add m
    res["L"] = L

    # Turn into pandas dataframe
    res = pd.DataFrame.from_dict(res)

    # Save intermediate result
    save(res, f"simulations/results/intermediate/timings_{m_base}_{L}.pkl")

    # Add to all results
    results_L.append(res)

# Save timings
save(results_L, "simulations/results/timings_L.pkl")

# Remove intermediate results
temp_files = os.listdir("./simulations/results/intermediate")
rm_files = [f for f in temp_files if "timings_" in f]
for file in rm_files:
    os.remove("./simulations/results/intermediate/" + file)
