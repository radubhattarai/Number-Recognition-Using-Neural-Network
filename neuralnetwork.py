import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

images = mnist.data.to_numpy() / 255.0
labels = mnist.target.astype(int)



# Network parameters
input_size = 784
hidden_size = 16
output_size = 10

np.random.seed(42)

W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))

W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))


# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)



# Prediction function
def predict(x):
    x = x.reshape(784, 1)

    # Forward pass
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)

    return np.argmax(a2)



# Random image test
index = np.random.randint(0, len(images))

x = images[index]
y_true = labels[index]

prediction = predict(x)

# Show result
plt.imshow(x.reshape(28, 28), cmap='gray')
plt.title(f"Actual: {y_true} | Predicted: {prediction}")
plt.show()