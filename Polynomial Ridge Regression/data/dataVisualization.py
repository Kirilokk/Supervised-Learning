from data.dataPreparation import  load_data
import matplotlib.pyplot as plt


train_set_x, test_set_x, train_set_y, test_set_y, full_feature_set_for_plot = load_data()

# To familiarize oneself with the data obtained, we will plot the Temperature as a function of Day.
# Since our Day feature was already normalized to 0-1 range, we will multiply it by 366 to restore
# the correct day of the year. Let's also add different colors to train and test samples to make it fancy.


# Color maps
cmap = plt.get_cmap('viridis')

# Plot the results
m1 = plt.scatter(366 * train_set_x, train_set_y, color=cmap(0.9), s=10)
m2 = plt.scatter(366 * test_set_x, test_set_y, color=cmap(0.5), s=10)
plt.xlabel('Day')
plt.ylabel('Temperature in Celcius')
plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
plt.show()