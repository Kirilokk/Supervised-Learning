import numpy as np


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (number of features, 1)
    b -- bias, a scalar
    X -- data of shape (number of features, number of examples)
    Y -- results of shape (1, number of examples)

    Return:
    cost -- cost function for linear regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    # activation computing
    H = np.dot(w.T, X) + b

    # cost computing
    cost = np.sum(np.power((H - Y), 2)) / (2 * m)  # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)

    # gradient descent computing
    dw = (np.dot(X, (H - Y).T) / m)
    db = (np.sum((H - Y)) / m)

    # small tests to make sure variables are correct type
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


