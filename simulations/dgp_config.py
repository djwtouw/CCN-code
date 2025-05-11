import numpy as np


# DGP for strong label dependencies
dgp_3strong = dict(
    parameters=(
        [1.0, 3.0, 0.5],
        [
            np.array([2.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5, 0.0, 0.0]),
        ],
        [np.array([]), np.array([-6.0]), np.array([2.0, -4.0])],
    ),
    n=200,
    reverse=False,
    realize_labels=False,
    name="3strong",
)

# DGP for weak label dependencies
dgp_3weak = dict(
    parameters=(
        [1.0, -2.5, -0.5],
        [
            np.array([2.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([-3.0, 0.0, 0.0]),
        ],
        [np.array([]), np.array([1.0]), np.array([2.5, -3.0])],
    ),
    n=200,
    reverse=False,
    realize_labels=False,
    name="3weak",
)

# DGP with 6 labels
dgp_6strong = dict(
    parameters=(
        [1.0, 3.0, 0.5, 0.0, 0.0, 0.0],
        [
            np.array([2.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([-3.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ],
        [
            np.array([]),
            np.array([-4.0]),
            np.array([-1.0, 0.0]),
            np.array([4.0, -2.0, -2.0]),
            np.array([0.0, -2.0, -6.0, 6.0]),
            np.array([0.0, 0.0, 6.0, 0.0, -6.0]),
        ],
    ),
    n=200,
    reverse=False,
    realize_labels=False,
)

# DGP for six label DGP with reversed label order
dgp_6reverse = dict(
    parameters=dgp_6strong["parameters"],
    n=200,
    reverse=True,
    realize_labels=False,
    name="6reverse",
)

# DGP for labels that are realized during the chain, following the modeling
# assumption of the classifier chain
dgp_6cc = dict(
    parameters=dgp_6strong["parameters"],
    n=200,
    reverse=False,
    realize_labels=True,
    name="6cc",
)

# DGP with 9 labels and few observations
dgp_9strong = dict(
    parameters=(
        [1.0, 3.0, 0.5, 0.0, 0.0, 0.0, 1.0, 3.0, 0.5],
        [
            np.array([2.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([-3.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5, 0.0, 0.0]),
        ],
        [
            np.array([]),
            np.array([-4.0]),
            np.array([-1.0, 0.0]),
            np.array([4.0, -2.0, -2.0]),
            np.array([0.0, -2.0, -6.0, 6.0]),
            np.array([0.0, 0.0, 6.0, 0.0, -6.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -4.0]),
        ],
    ),
    n=200,
    reverse=False,
    realize_labels=False,
    name="9strong",
)
