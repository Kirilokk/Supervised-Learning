import numpy as np
from core.regression import propagate



def optimize(theta, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes theta and b by running a gradient descent algorithm

    Arguments:
    theta -- parameter, a scalar
    b -- bias, a scalar
    X -- features vector of shape (number of examples, )
    Y -- results vector of shape (number of examples, )
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights theta and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1 line of code)
        grads, cost = propagate(theta, b, X, Y)

        # Retrieve derivatives from grads
        dt = grads["dt"]
        db = grads["db"]

        # update rule
        theta = theta - (learning_rate * dt)
        b = b - (learning_rate * db)

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"theta": theta,
              "b": b}

    grads = {"dt": dt,
             "db": db}

    return params, grads, costs
