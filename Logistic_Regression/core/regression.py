import numpy as np
from core.sigmoid_func import sigmoid

def propagate(w, b, X, Y):
    """
     Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (number of features, 1)
    b -- bias, a scalar
    X -- data of size (number of features, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    """

    m = X.shape[1]

    # FORWARD PROPAGATION  (FROM A TO COST)
    A = sigmoid((np.dot(w.transpose(), X) + b))                      # compute activation
    cost = -np.sum(Y @ np.log(A).T + (1 - Y) @ np.log(1 - A).T) / m  # compute cost function

    # BACKWARD PROPAGATION (TO FIND GRADIENT)
    dw = (np.dot(X, (A - Y).T) / m)
    db = np.sum(A - Y) / m


    # small tests to make sure variables are correct type
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

