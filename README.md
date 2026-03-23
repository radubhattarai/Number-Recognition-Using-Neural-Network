# Number Recognition Using Neural Network

Classifies handwritten digits (0–9) by processing 28×28 pixel grayscale images using a
Convolutional Neural Network (CNN) trained on the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Architecture

| Layer | Details |
|---|---|
| Input | 28×28×1 grayscale image |
| Conv2D (32 filters, 3×3) + ReLU | Feature extraction |
| MaxPooling2D (2×2) | Spatial downsampling |
| Conv2D (64 filters, 3×3) + ReLU | Deeper feature extraction |
| MaxPooling2D (2×2) | Spatial downsampling |
| Flatten + Dropout (0.5) | Regularisation |
| Dense 128 + ReLU | Hidden layer for pattern recognition |
| Dense 10 + Softmax | Output layer — one neuron per digit (0–9) |

The model typically achieves **≥ 99% accuracy** on the MNIST test set.

## Requirements

- Python 3.10+
- TensorFlow ≥ 2.10

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python train.py
```

This downloads the MNIST dataset (if not already cached), trains the CNN for 10 epochs,
prints the final test accuracy, saves the model to `mnist_cnn.keras`, and writes a
`training_history.png` plot.

### Predict

```bash
# Predict the digit in a custom image file
python predict.py path/to/digit.png

# Run a quick demo on 16 random MNIST test images
python predict.py --demo
```

## Files

| File | Purpose |
|---|---|
| `model.py` | CNN model definition (`build_cnn_model`) |
| `train.py` | Download MNIST, train, evaluate, and save the model |
| `predict.py` | Load the saved model and predict on new images |
| `requirements.txt` | Python package dependencies |
