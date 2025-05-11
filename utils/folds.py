import numpy as np


def get_folds(n, n_splits=5):
    # Get indices and shuffle them
    indices = list(range(n))
    np.random.shuffle(indices)

    # Initialize folds
    folds = []

    while n_splits > 0:
        # Get length of fold and append
        fold_len = len(indices) // n_splits
        folds.append(indices[:fold_len])

        # Get rid of used indices
        indices = indices[fold_len:]

        # Decrement number of splits
        n_splits -= 1

    return folds
