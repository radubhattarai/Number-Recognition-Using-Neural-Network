"""Train the CNN on the MNIST dataset and save the resulting model."""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import build_cnn_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 64
MODEL_PATH = "mnist_cnn.keras"


def load_and_preprocess():
    """Download (or load cached) MNIST and normalise pixel values to [0, 1].

    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as numpy arrays.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape to (N, 28, 28, 1) and normalise to [0, 1]
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    return x_train, y_train, x_test, y_test


def plot_history(history, save_path="training_history.png"):
    """Plot and save training / validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="train accuracy")
    ax1.plot(history.history["val_accuracy"], label="val accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="train loss")
    ax2.plot(history.history["val_loss"], label="val loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to '{save_path}'")


def main():
    print("Loading MNIST dataset …")
    x_train, y_train, x_test, y_test = load_and_preprocess()
    print(f"  Training samples : {len(x_train)}")
    print(f"  Test samples     : {len(x_test)}")

    print("\nBuilding CNN model …")
    model = build_cnn_model()
    model.summary()

    print("\nTraining …")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    print("\nEvaluating on test set …")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test accuracy : {test_acc * 100:.2f}%")
    print(f"  Test loss     : {test_loss:.4f}")

    print(f"\nSaving model to '{MODEL_PATH}' …")
    model.save(MODEL_PATH)

    plot_history(history)


if __name__ == "__main__":
    main()
