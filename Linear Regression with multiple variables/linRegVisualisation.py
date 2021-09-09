import matplotlib.pyplot as plt
import numpy as np
from data.dataPreparation import load_data
from interactor.index import model

train_set_x, train_set_y, test_set_x, test_set_y, visualization_set = load_data()
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=3000, learning_rate=0.05, print_cost=True)

# Training set
plt.figure(figsize=(4, 3))
plt.title("Training set")
plt.scatter(train_set_y, d["Y_prediction_train"])
plt.plot([0, 50], [0, 50], "--k")
plt.axis("tight")
plt.xlabel("True price ($1000s)")
plt.ylabel("Predicted price ($1000s)")
plt.tight_layout()

# Test set
plt.figure(figsize=(4, 3))
plt.title("Test set")
plt.scatter(test_set_y, d["Y_prediction_test"])
plt.plot([0, 50], [0, 50], "--k")
plt.axis("tight")
plt.xlabel("True price ($1000s)")
plt.ylabel("Predicted price ($1000s)")
plt.tight_layout()

plt.show()