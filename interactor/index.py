import numpy as np
from data.dataPreparation import load_data
from core.optimize import optimize

def initialize_with_zeros():
    """
    This function initializes parameters theta and b as 0.

    Returns:
    theta -- initialized scalar parameter
    b -- initialized scalar (corresponds to the bias)
    """
    theta = 0
    b = 0

    # Small tests to make sure variables are correct type
    assert (isinstance(theta, int))
    assert (isinstance(b, int))

    return theta, b


def predict(theta, b, X):
    """
    Predict using learned linear regression parameters (theta, b)

    Arguments:
    theta -- parameter, a scalar
    b -- bias, a scalar
    X -- features vector of size (number of examples, )

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions for the examples in X
    """

    # Compute vector "Y_prediction" predicting the width of a kangoroo nasal
    Y_prediction = (X * theta) + b

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the linear regression model

    Arguments:
    X_train -- training set represented by a numpy array of shape (m_train, )
    Y_train -- training values represented by a numpy array (vector) of shape (m_train, )
    X_test -- test set represented by a numpy array of shape (m_test, )
    X_test -- test values represented by a numpy array (vector) of shape (m_test, )
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # initialize parameters with zeros
    theta, b = initialize_with_zeros()

    # Gradient descent
    parameters, grads, costs = optimize(theta, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    theta = parameters["theta"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(theta, b, X_test)
    Y_prediction_train = predict(theta, b, X_train)


    # Print train/test Errors
    print("Train RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_train - Y_train) ** 2))))
    print("Test RMSE: {} ".format(np.sqrt(np.mean((Y_prediction_test - Y_test) ** 2))))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "theta": theta,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

if __name__ == '__main__':
    train_set_x, test_set_x, train_set_y, test_set_y = load_data()
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=500, learning_rate=0.05, print_cost=True)

