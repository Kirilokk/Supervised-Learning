import numpy as np
from data.dataPreparation import load_data
from core.ridgeRegularization import l2_regularization
from utils.featureTransform import *


# GRADED CLASS: PolynomialRidgeRegression

class PolynomialRidgeRegression(object):
    """
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage.
    n_iterations: int
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, print_error=False):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.print_error = print_error

    def initialize_with_zeros(self, n_features):
        """
        This function creates a vector of zeros of shape (n_features, 1)

        Arguments:
        n_features -- amount of features
        """
        self.w = np.zeros((n_features, 1))

    def fit(self, X, Y):
        ### START CODE HERE ###
        # Generate polynomial features
        X = polynomial_features(X, self.degree)

        # Create array
        self.initialize_with_zeros(n_features=X.shape[0])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            # Calculate prediction
            H = self.w.transpose() @ X

            # Gradient of l2 loss w.r.t w
            grad_w = X @ (H - Y).transpose() + self.regularization.grad(self.w)

            # Update the weights
            self.w = self.w - (self.learning_rate * grad_w)

            if self.print_error and i % 1000 == 0:
                # Calculate l2 loss
                mse = mean_squared_error(Y, H)
                print("MSE after iteration %i: %f" % (i, mse))

    def predict(self, X):
        # Generate polynomial features
        X = polynomial_features(X, self.degree)

        # Calculate prediction)
        y_pred = self.w.transpose() @ X

        return y_pred

if __name__ == '__main__':

    # First of all, we should define a maximum possible polynomial degree (poly_degree),
    # learning rate (learning_rate), a number of iterations (num_iteration)
    # and regularization factor (reg_factor) for our model. Often reg_factor is chosen with help of cross-validation.

    poly_degree = 15
    learning_rate = 0.001
    n_iterations = 10000
    reg_factor = 0

    # init our model parameters
    model = PolynomialRidgeRegression(
        degree=poly_degree,
        reg_factor=reg_factor,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        print_error=True
    )


    train_set_x, test_set_x, train_set_y, test_set_y, full_feature_set_for_plot = load_data()
    # model training
    model.fit(train_set_x, train_set_y)

    # Making predictions
    y_predictions = model.predict(test_set_x)
    mse = mean_squared_error(test_set_y, y_predictions)
    print("Mean squared error on test set: %s (given by reg. factor: %s)" % (mse, reg_factor))