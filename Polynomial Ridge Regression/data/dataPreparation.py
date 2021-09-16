import h5py
import numpy as np

# Loading the data

def load_data():
    from sklearn.model_selection import train_test_split

    """
       First of all, we need to load our data and make it ready for training our polynomial regression model
       Return:
       train_set_x -- train dataset with days in range from 0 to 1 in 2016 year
       train_set_y -- train dataset with temperature in Celsius
       test_set_x  -- test  dataset with days in range from 0 to 1 in 2016 year
       test_set_y  -- test  dataset with temperature in Celsius

    """

    # Your dataset path
    data = np.genfromtxt("C:\\Users\\Kirilok\\Documents\\GitHub\\Supervised-Learning\\Polynomial Ridge Regression\\data\\time_temp_2016.tsv", delimiter='\t')

    x = data[:, 0]
    x = x.reshape((x.shape[0], 1))
    y = data[:, 1]

    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x.T, test_set_x.T, train_set_y, test_set_y, x.T
