# Number Recognition Using Neural Network

This is a neural network created from scratch by Jack Carter but I discovered the core concepts from a cool math YouTuber [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk).

![MNIST Examples](https://camo.githubusercontent.com/96c0b673c785478dd8629e0c9631d017c1ec3b74629deed003249be69f73f271/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f662f66372f4d6e6973744578616d706c65734d6f6469666965642e706e67)

## Process / Core Idea

```
[ Image (28×28 pixels) ]
            ↓
[ Flatten → 784 values ]
            ↓
[ Neural Network ]
            ↓
[ 10 Output Probabilities ]
            ↓
[ Predicted Digit (0–9) ]
```

## Overview

This project implements a neural network from scratch to recognize handwritten digits (0-9) from the MNIST dataset using NumPy and backpropagation.

## Architecture

- **Input Layer**: 784 neurons (28×28 pixels)
- **Hidden Layer**: 128 neurons with sigmoid activation
- **Output Layer**: 10 neurons (digits 0-9) with softmax activation

## How It Works

1. **Forward Pass**: Pixels → Hidden layer (weighted sum + sigmoid) → Output layer (weighted sum + softmax)
2. **Backward Pass**: Calculate gradients using backpropagation
3. **Weight Update**: Adjust weights to minimize classification error
4. **Training**: Repeat for multiple epochs on MNIST training images

## Performance

- **Accuracy**: ~97% on test set
- **Training Data**: 50,000 images
- **Test Data**: 10,000 images

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn (for MNIST dataset)

Install dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage

Run the training and prediction:
```bash
python neuralnetwork.py
```

This will:
1. Load the MNIST dataset
2. Train the neural network
3. Print accuracy metrics
4. Display a random test image with actual and predicted labels

## Files

- `neuralnetwork.py` - Complete neural network implementation with training and testing

## Key Concepts

- **Sigmoid**: Non-linear activation function in hidden layer
- **Softmax**: Converts output to probability distribution
- **Backpropagation**: Calculates gradients to update weights
- **Gradient Descent**: Optimization algorithm to minimize loss
