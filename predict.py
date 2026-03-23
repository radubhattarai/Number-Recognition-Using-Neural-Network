"""Run inference with the trained MNIST CNN model.

Usage
-----
# Predict on a single image file (PNG / JPEG, will be resized to 28x28):
    python predict.py path/to/image.png

# Predict on several MNIST test images and show a sample grid:
    python predict.py --demo
"""

import argparse
import sys

import numpy as np
import tensorflow as tf

MODEL_PATH = "mnist_cnn.keras"


def load_model(model_path: str = MODEL_PATH):
    """Load the saved Keras model from *model_path*."""
    try:
        model = tf.keras.models.load_model(model_path)
    except (OSError, IOError):
        print(
            f"Error: model file '{model_path}' not found.\n"
            "Run 'python train.py' first to train and save the model."
        )
        sys.exit(1)
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """Load an image from disk, convert to 28x28 grayscale, and normalise.

    Args:
        image_path: Path to a PNG or JPEG image file.

    Returns:
        A float32 numpy array of shape (1, 28, 28, 1) with values in [0, 1].
    """
    img = tf.keras.utils.load_img(
        image_path, color_mode="grayscale", target_size=(28, 28)
    )
    arr = tf.keras.utils.img_to_array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


def predict_digit(model, image_array: np.ndarray) -> tuple[int, float]:
    """Return the predicted digit and confidence for a pre-processed image.

    Args:
        model: A loaded Keras model.
        image_array: Float32 array of shape (1, 28, 28, 1).

    Returns:
        (digit, confidence) where digit is in 0-9 and confidence is in [0, 1].
    """
    probabilities = model.predict(image_array, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    return digit, confidence


def demo(model, num_samples: int = 16):
    """Predict on random MNIST test images and print a summary."""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    indices = np.random.choice(len(x_test), num_samples, replace=False)

    print(f"\n{'Index':>6}  {'True':>4}  {'Pred':>4}  {'Confidence':>10}  {'Match':>5}")
    print("-" * 42)
    correct = 0
    for idx in indices:
        img = x_test[idx : idx + 1]
        true_label = y_test[idx]
        digit, conf = predict_digit(model, img)
        match = "✓" if digit == true_label else "✗"
        if digit == true_label:
            correct += 1
        print(f"{idx:>6}  {true_label:>4}  {digit:>4}  {conf * 100:>9.2f}%  {match:>5}")

    print(f"\nAccuracy on sample: {correct}/{num_samples} ({correct / num_samples * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Predict handwritten digits using the trained CNN model."
    )
    parser.add_argument(
        "image", nargs="?", help="Path to an image file to predict."
    )
    parser.add_argument(
        "--model", default=MODEL_PATH, help="Path to the saved model file."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo predictions on random MNIST test images.",
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.demo:
        demo(model)
    elif args.image is None:
        parser.print_help()
        sys.exit(1)
    else:
        image_array = preprocess_image(args.image)
        digit, confidence = predict_digit(model, image_array)
        print(f"Predicted digit : {digit}")
        print(f"Confidence      : {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
