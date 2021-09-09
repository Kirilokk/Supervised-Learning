from dataPreparation import load_data
import matplotlib.pyplot as plt


train_set_x, train_set_y, test_set_x, test_set_y, visualization_set = load_data()


for index, feature_name in enumerate(visualization_set.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(visualization_set.data[:, index], visualization_set.target)
    plt.ylabel("Price", size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()

plt.show()