import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load and preprocess MNIST
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy() / 255.0
y = mnist.target.astype(int).values

X_train, y_train = X[:50000], y[:50000]
X_test, y_test = X[50000:60000], y[50000:60000]

# Network params
input_size, hidden_size, output_size = 784, 128, 10
learning_rate, epochs, batch_size = 0.1, 20, 32

np.random.seed(42)
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))


# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Forward and backward pass
def train_step(X, y):
    global W1, b1, W2, b2
    m = X.shape[1]
    
    # Forward
    z1 = np.dot(W1, X) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)
    
    # Backward
    dz2 = a2 - y
    dW2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    
    da1 = np.dot(W2.T, dz2)
    dz1 = da1 * (a1 * (1 - a1))
    dW1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    
    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

def predict(X):
    z1 = np.dot(W1, X) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)
    return np.argmax(a2, axis=0)

# Train
print("Training...")
for epoch in range(epochs):
    indices = np.random.permutation(X_train.shape[0])
    for i in range(0, X_train.shape[0], batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx].T
        y_batch = np.eye(10)[y_train[batch_idx]].T
        train_step(X_batch, y_batch)
    
    preds = predict(X_test.T)
    acc = np.mean(preds == y_test)
    print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.4f}")

# Test on random image
idx = np.random.randint(0, len(X_test))
img = X_test[idx].reshape(1, 784)
pred = predict(img.T)[0]
true = y_test[idx]

plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Actual: {true} | Predicted: {pred}")
plt.tight_layout()
plt.show()