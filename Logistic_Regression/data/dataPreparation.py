import h5py
import numpy as np

# Loading the data

def load_data():
    """
       First of all, we need to load our data and make it ready for training our linear regression model
       Return:
       train_set_x -- train dataset with width*height*RGB pixel features
       train_set_y -- train dataset with image classification result for our model
       test_set_x  -- test  dataset with width*height*RGB pixel features
       test_set_y  -- test  dataset with image classification result for our model

    """

    train_dataset = h5py.File("C:\\Users\\Kirilok\\Downloads\\Ml\\Logistic_Regression\\data\\train_cats.h5", "r")

    train_set_x = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File("C:\\Users\\Kirilok\\Downloads\\Ml\\Logistic_Regression\\data\\test_cats.h5", "r")
    test_set_x = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # Images have (64(width), 64(height), 3(RGB)) dimensions, we need to have all the pixels as features
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T  # shape ((209)examples, (64)width, (64)height, (3)RGB) -> (64 * 64 * 3, 209)
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T  # shape ((209)examples, (64)width, (64)height, (3)RGB) -> (64 * 64 * 3, 209)

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    # Data Standardization
    train_set_x = train_set_x / 255.
    test_set_x = test_set_x / 255.

    return train_set_x, train_set_y, test_set_x, test_set_y, classes

