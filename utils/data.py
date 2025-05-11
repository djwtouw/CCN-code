import numpy as np


def dgp(
    n,
    parameters,
    reverse=False,
    realize_labels=False,
):
    """
    Data generating process for a multilabel data set.

    :param n: Number of observations.
    :param parameters: Tuple of parameters in the form (b, W, c). Each element
    has a length equal to the number of labels: b contains the intercepts, W
    contains the coefficient vectors with length equal to the number of
    variables, c contains the label dependency coefficients where c[i] has
    length i+1.
    :param reverse: Boolean to indicate whether the labels should be reversed.
    :param realize_labels: Boolean to indicate whether labels should be
    realized immediately, so that the actual outcome determines the effect on
    subsequent labels instead of the probability.
    :return: A tuple with the data and corresponding labels.
    """
    # Unpack parameters
    b, W, c = parameters

    # Number of variables for the data
    num_variables = len(W[0])

    # Mean vector
    mean = np.zeros(num_variables)

    # Covariance matrix
    cov = np.zeros((num_variables, num_variables))
    cov[:, :] = 0.4
    np.fill_diagonal(cov, 2.0)

    # Generate data
    data = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

    # Number of labels
    num_labels = len(b)

    # Initialize label matrix
    labels = np.zeros((n, num_labels))

    # Compute label probabilities
    for i in range(num_labels):
        labels[:, i] = b[i] + data @ W[i] + labels[:, :i] @ c[i]
        labels[:, i] = 1 / (1 + np.exp(-labels[:, i]))

        # Use probabilities to obtain binary outcomes
        if realize_labels:
            labels[:, i] = (labels[:, i] > np.random.rand(n)).astype(float)

    # Use probabilities to obtain binary outcomes
    if not realize_labels:
        labels = (labels > np.random.rand(*labels.shape)).astype(float)

    # Reverse the label order
    if reverse:
        labels = labels[:, ::-1]

    return data, labels
