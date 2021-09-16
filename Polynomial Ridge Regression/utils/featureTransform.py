import numpy as np
from itertools import combinations_with_replacement


def polynomial_features(X, degree):
    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC

    n_features, n_samples = np.shape(X)

    def index_combinations():  # (1, 2) => [(1),(2),(1,1),(1,2),(2,2)]

        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        ##comb = [(),((1),(2)),((1,1),(1,2),(2,2))]
        flat_combs = [item for sublist in combs for item in sublist]
        ##flat_combs = [(1),(2),(1,1),(1,2),(2,2)]
        return flat_combs

    # get combination by index
    combinations = index_combinations()

    n_output_features = len(combinations)

    X_new = np.empty((n_output_features, n_samples))

    # transform indexes into values
    for i, index_combs in enumerate(combinations):
        X_new[i, :] = np.prod(X[index_combs, :], axis=0)

        # if index_combs == (1,2,3) =>  X_new[:,i] = X[:,1] * X[:,2] * X[:,3]
    return X_new


# GRADED FUNCTION: mean_squared_error

def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred

    Arguments:
    y_true -- array of true values
    y_pred -- array of predicted values

    Returns:
    mse -- mean squared error
    """
    mse = np.sum(np.mean((y_pred - y_true) ** 2))

    return mse