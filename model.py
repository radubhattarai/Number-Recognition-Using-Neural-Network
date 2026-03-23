"""CNN model definition for MNIST handwritten digit recognition."""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model():
    """Build and return a CNN model for MNIST digit classification.

    Architecture:
        - Input: 28x28 grayscale images
        - Two convolutional blocks (Conv2D + MaxPooling2D) for feature extraction
        - Flatten + Dropout for regularization
        - Dense hidden layer with ReLU activation
        - Output layer with 10 neurons (softmax) for digits 0-9

    Returns:
        A compiled Keras Model.
    """
    model = models.Sequential([
        # Input layer: 28x28 pixel images with 1 colour channel (grayscale)
        layers.Input(shape=(28, 28, 1)),

        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten and regularise
        layers.Flatten(),
        layers.Dropout(0.5),

        # Hidden dense layer for pattern recognition
        layers.Dense(128, activation="relu"),

        # Output layer: 10 neurons, one per digit class (0-9)
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    cnn = build_cnn_model()
    cnn.summary()
