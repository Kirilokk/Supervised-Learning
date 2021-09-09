from dataPreparation import load_data
import matplotlib.pyplot as plt


train_set_x, test_set_x, train_set_y, test_set_y = load_data()

# Train set
plt.figure(figsize=(4, 3))
plt.scatter(train_set_x, train_set_y)
plt.title("Training set")
plt.xlabel("Length")
plt.ylabel("Width")


# Test set
plt.figure(figsize=(4, 3))
plt.scatter(test_set_x, test_set_y)
plt.title("Test set")
plt.xlabel("Length")
plt.ylabel("Width")


plt.show()