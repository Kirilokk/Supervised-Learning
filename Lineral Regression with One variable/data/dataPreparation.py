import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Loading the data

def load_data():
    """
    First of all, we need to load our data and make it ready for training our linear regression model

    Return:
    train_set_x -- train dataset with kangaroo height for our model created by train_test_split() function
    train_set_y -- train dataset with kangaroo width for our model created by train_test_split() function
    test_set_x  -- test dataset with kangaroo height created by train_test_split() function
    test_set_y  -- test dataset with kangaroo width  created by train_test_split() function

    """
    data = np.genfromtxt('C:\\Users\\Kirilok\\Downloads\\Ml\\Lineral Regression with One variable\\data\\kangaroo.csv', delimiter=',')

    x = data[:, 0]
    y = data[:, 1]

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)

    # finding mean and standard deviation of data set
    mean = np.concatenate([train_set_x, test_set_x]).mean()
    std = np.concatenate([train_set_x, test_set_x]).std()

    # data standardization
    train_set_x = (train_set_x - mean) / std
    test_set_x = (test_set_x - mean) / std

    return train_set_x, test_set_x, train_set_y, test_set_y
