```markdown
# Neural Network From Scratch Documentation

Welcome to the Neural Network From Scratch documentation.

## Overview

This library provides a simple, educational implementation of a neural network from scratch using only NumPy. It's designed to help understand the fundamentals of neural networks without the complexity of modern frameworks.

## Quick Links

- [Quick Start Guide](quickstart.md)
- [API Reference](api_reference.md)
- [Examples](examples.md)

## Installation

```bash
pip install git+https://github.com/yourusername/neural-network-scratch.git
```

## Features

- Multiple activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU)
- Multiple loss functions (BCE, CE, MSE)
- Smart weight initialization
- Easy-to-use API
- Comprehensive examples

## Quick Example

```python
from neural_network import NeuralNetwork

# Create network
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)

# Train
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]

for epoch in range(5000):
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        nn.backward(y, output, history)

# Predict
prediction = nn.predict([1, 0])
```
## License

MIT License
```

---
