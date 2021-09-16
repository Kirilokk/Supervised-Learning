# GRADED CLASS: l2_regularization(Ridge)

import numpy as np

class l2_regularization():
    """ Regularization for Ridge Regression """

    def __init__(self, alpha):
        # Set alpha or lambda(our penalty)
        self.alpha = alpha

    def __call__(self, w):
        """
        Computes l2 regularization term

        Arguments:
        w -- weights

        Returns:
        term -- l2 regularization term
        """
        term = np.sum(self.alpha * (abs(w)) ** 2) / 2
        return term

    def grad(self, w):
        """
        Computes derivative of l2 regularization term

        Arguments:
        w -- weights

        Returns:
        vector -- derivative of l2 regularization term
        """
        derivative = (self.alpha) * w

        return derivative
