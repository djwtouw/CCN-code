"""
Script to test all multi-label classification methods used in the simulation
study. Includes a section with cross-validation to set the relevant tuning
parameters.

Author: Daniel Touw
"""

import numpy as np
from eccn import CCN

from models.BinaryRelevance import br_cv, br_fit
from models.ClassifierChain import cc_cv, cc_fit
from models.MultiLabelKNN import mlknn_cv, mlknn_fit
from models.MultiLabelTwinSVM import mltsvm_cv, mltsvm_fit
from models.RandomKLabelsets import rakel_cv, rakel_fit
from models.AdaBoostMH import AdaBoostMH, adaboostmh_cv
from models.LinearRBRL import LinearRBRL, linear_rbrl_cv
from simulations.dgp_config import dgp_3strong
from utils.data import dgp
from utils.folds import get_folds
from utils.parameter_grid import expand_grid

# %%

# Simple fitting functions, no tuning of parameters.
# Set seed
np.random.seed(123)

# Generate data
X_train, Y_train = dgp(n=200, parameters=dgp_3strong["parameters"])
X_test, Y_test = dgp(n=1000, parameters=dgp_3strong["parameters"])

# Binary relevance
model_br = br_fit(X_train, Y_train, alpha=1e-3)
Y_hat = model_br.predict(X_test)
print(f"Binary relevance accuracy:         {(Y_test == Y_hat).mean():.3f}")

# Classifier chain
model_cc = cc_fit(X_train, Y_train, alpha=1e-3)
Y_hat = model_cc.predict(X_test)
print(f"Classifier chain accuracy:         {(Y_test == Y_hat).mean():.3f}")

# Multi-label k-nn
model_mlknn = mlknn_fit(X_train, Y_train, k=3)
Y_hat = model_mlknn.predict(X_test)
print(f"Multi-label k-nn accuracy:         {(Y_test == Y_hat).mean():.3f}")

# Multi-label twin SVM
model_mltsvm = mltsvm_fit(X_train, Y_train, c_k=0.5)
Y_hat = model_mltsvm.predict(X_test)
print(f"Multi-label twin SVM accuracy:     {(Y_test == Y_hat).mean():.3f}")

# (Random) k-labelsets
model_rakel = rakel_fit(X_train, Y_train, alpha=1e-3)
Y_hat = model_rakel.predict(X_test)
print(f"Random k-labelsets accuracy:       {(Y_test == Y_hat).mean():.3f}")

# AdaBoost.MH
model_adaboostmh = AdaBoostMH(ndt=50).fit(X_train, Y_train)
Y_hat = model_adaboostmh.predict(X_test)
print(f"AdaBoost.MH accuracy:              {(Y_test == Y_hat).mean():.3f}")

# LinearRBRL
model_rbrl = LinearRBRL(lambda1=1.0, lambda2=0.01, lambda3=0.1).fit(
    X_train, Y_train
)
Y_hat = model_rbrl.predict(X_test)
print(f"LinearRBRL accuracy:               {(Y_test == Y_hat).mean():.3f}")

# Classifier chain network
model_ccn = CCN(init="auto", alpha=1e-3).fit(X_train, Y_train)
Y_hat = model_ccn.predict(X_test)
print(f"Classifier chain network accuracy: {(Y_test == Y_hat).mean():.3f}")

# %%

# Using cross validation to determine the optimal values of the tuning
# parameters.
# Get folds for cross validation
folds = get_folds(X_train.shape[0], n_splits=5)

# Choose metric
metric = "nll"
metric = "zero_one"
metric = "macro_F1"
metric = "micro_F1"
metric = "hamming"

# Binary relevance
# Get grid with tuning parameters
grid_br = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
grid_br = expand_grid(grid_br)

# Cross validate and obtain optimal model
model_br = br_cv(X_train, Y_train, folds=folds, grid=grid_br, metric=metric)
Y_hat = model_br.predict(X_test)
print(f"Binary relevance accuracy:         {(Y_test == Y_hat).mean():.3f}")

# Classifier chain
# Get grid with tuning parameters
grid_cc = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
grid_cc = expand_grid(grid_cc)

# Cross validate and obtain optimal model
model_cc = cc_cv(X_train, Y_train, folds=folds, grid=grid_cc, metric=metric)
Y_hat = model_cc.predict(X_test)
print(f"Classifier chain accuracy:         {(Y_test == Y_hat).mean():.3f}")

# Multi-label k-nn
# Get grid with tuning parameters
grid_klmnn = dict(
    k=[k for k in range(5, int(np.ceil(np.sqrt(X_train.shape[0]))) + 1, 2)]
)
grid_klmnn = expand_grid(grid_klmnn)

# Cross validate and obtain optimal model
model_mlknn = mlknn_cv(
    X_train, Y_train, folds=folds, grid=grid_klmnn, metric=metric
)
Y_hat = model_mlknn.predict(X_test)
print(f"Multi-label k-nn accuracy:         {(Y_test == Y_hat).mean():.3f}")

# Multi-label twin SVM
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
Y_hat = model_mltsvm.predict(X_test)
print(f"Multi-label twin SVM accuracy:     {(Y_test == Y_hat).mean():.3f}")

# (Random) k-labelsets
# Get grid with tuning parameters
grid_rakel = dict(alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1])
grid_rakel = expand_grid(grid_rakel)

# Cross validate and obtain optimal model
model_rakel = rakel_cv(
    X_train, Y_train, folds=folds, grid=grid_rakel, metric=metric
)
Y_hat = model_rakel.predict(X_test)
print(f"Random k-labelsets accuracy:       {(Y_test == Y_hat).mean():.3f}")

# AdaBoost.MH
# Get grid with tuning parameters
grid_adaboostmh = dict(ndt=[25, 50, 75, 100, 125])
grid_adaboostmh = expand_grid(grid_adaboostmh)

# Cross validate and obtain optimal model
model_adaboostmh = adaboostmh_cv(
    X_train, Y_train, folds=folds, grid=grid_adaboostmh, metric=metric
)
Y_hat = model_adaboostmh.predict(X_test)
print(f"AdaBoost.MH accuracy:              {(Y_test == Y_hat).mean():.3f}")

# LinearRBRL
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
Y_hat = model_rbrl.predict(X_test)
print(f"LinearRBRL accuracy:               {(Y_test == Y_hat).mean():.3f}")

# Classifier chain network
# Get grid with tuning parameters
grid_ccn = dict(
    alpha=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1], q=[1, 1.5, 2, 3, 5]
)
grid_ccn = expand_grid(grid_ccn)

# Cross validate and obtain optimal model
model_ccn = CCN().cv(
    X_train, Y_train, folds=folds, grid=grid_ccn, metric=metric
)
Y_hat = model_ccn.predict(X_test)
print(f"Classifier chain network accuracy: {(Y_test == Y_hat).mean():.3f}")
