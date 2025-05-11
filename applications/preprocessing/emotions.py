import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.io import arff

# %%

# Load data
arff_file = arff.loadarff("applications/data/emotions.arff")
data = pd.DataFrame(arff_file[0])

# Separate
data_X = data.iloc[:, :-6]
data_Y = data.iloc[:, -6:]
variables = data_X.columns.tolist()
labels = data_Y.columns.tolist()

# Transform into numpy arrays
data_X = data_X.values
data_Y = data_Y.values

# Standardize the data
data_X /= data_X.std(axis=0)

# PCA
pca = PCA()
data_X_pc = pca.fit_transform(data_X)

# Plot a figure of the explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance ratio")
plt.tight_layout()
plt.show()

# Select the desired ratio of the explained variance
evr_threshold = 0.9
n_pc = np.where(np.cumsum(pca.explained_variance_ratio_) > evr_threshold)[0][0]
print(
    f"Number of principal components to reach an explained variance ratio "
    f"of {evr_threshold:.2f} is {n_pc}"
)

# Select principal components
data_X_pc = data_X_pc[:, :n_pc]

# Create the final data set
data_pc = np.c_[data_X_pc, data_Y]
data_pc = pd.DataFrame(
    data_pc,
    columns=[f"PC{i + 1}" for i in range(n_pc)] + labels,
)

# Save data
data_pc.to_csv("applications/data/emotions_preprocessed.csv", index=False)
