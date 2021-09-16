from data.dataPreparation import  load_data
import matplotlib.pyplot as plt


from interactor.index import *
from data.dataVisualization import cmap

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

y_predictions = model.predict(test_set_x)

# Mean Squared Error(we want to get the least value)
mse = mean_squared_error(test_set_y, y_predictions)

# Making predictions
y_val = model.predict(full_feature_set_for_plot)


# Plot the results
m1 = plt.scatter(366 * train_set_x, train_set_y, color=cmap(0.9), s=10)
m2 = plt.scatter(366 * test_set_x, test_set_y, color=cmap(0.5), s=10)
plt.plot(366 * full_feature_set_for_plot.T, y_val.T, color='black', linewidth=2, label="Prediction")
plt.suptitle("Polynomial Ridge Regression")
plt.title("MSE: %.2f" % mse, fontsize=10)
plt.xlabel('Day')
plt.ylabel('Temperature in Celcius')
plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
plt.show()
