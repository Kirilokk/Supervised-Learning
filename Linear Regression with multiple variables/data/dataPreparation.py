import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Loading the data

def load_data():
    """
       First of all, we need to load our data and make it ready for training our linear regression model
       Return:
       train_set_x -- train dataset with house features for our model created by train_test_split() function
       train_set_y -- train dataset with house cost for our model created by train_test_split() function
       test_set_x  -- test dataset with house features created by train_test_split() function
       test_set_y  -- test dataset with house cost   created by train_test_split() function

    """
    boston = load_boston()

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(boston.data, boston.target, test_size=0.33,random_state=42)

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    # finding mean and standard deviation of data set
    all_set_x = np.concatenate([train_set_x.T, test_set_x.T], axis=1)

    mean = all_set_x.mean(axis=1, keepdims=True)
    std = all_set_x.std(axis=1, keepdims=True)

    # standardization process
    train_set_x = (train_set_x.T - mean) / std
    test_set_x = (test_set_x.T - mean) / std

    return train_set_x, train_set_y, test_set_x, test_set_y, boston

