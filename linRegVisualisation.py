import matplotlib.pyplot as plt
import numpy as np
from data.dataPreparation import load_data
from interactor.index import model


train_set_x, test_set_x, train_set_y, test_set_y = load_data()
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=500, learning_rate=0.05, print_cost=True)

# Training set
plt.figure(figsize=(4, 3))
plt.title("Training set")

plt.scatter(train_set_x, train_set_y)
x = np.array([min(train_set_x), max(train_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y)
plt.axis("tight")
plt.xlabel("Length")
plt.ylabel("Width");
plt.tight_layout()



# Test set
plt.figure(figsize=(4, 3))
plt.title("Test set")

plt.scatter(test_set_x, test_set_y)
x = np.array([min(test_set_x), max(test_set_x)])
theta = d["theta"]
b = d["b"]
y = theta * x + b
plt.plot(x, y)
plt.axis("tight")
plt.xlabel("Length")
plt.ylabel("Width");
plt.tight_layout()

plt.show()