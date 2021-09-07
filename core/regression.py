import numpy as np


def propagate(theta, b, X, Y):
    """
    As our parameters are initialized, we can do the "forward" and "backward" propagation steps for learning the parameters.
    Arguments:
    theta -- parameter, a scalar
    b -- bias, a scalar
    X -- features vector of size (number of examples, )
    Y -- results vector (number of examples, )

    Return:
    cost -- cost function for linear regression
    dt -- gradient of the loss with respect to theta, thus same shape as theta
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[0]

    # FORWARD PROPAGATION (FROM X TO COST)
    H = np.dot(X, theta) + b  # compute activation
    cost = np.sum(np.power((H - Y), 2)) / (2 * m)  # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dt = (np.dot((H - Y), X.T) / m)
    db = (np.sum((H - Y)) / m)

    # Small tests to make sure variables are correct type
    assert (dt.dtype == float)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dt": dt,
             "db": db}

    return grads, cost