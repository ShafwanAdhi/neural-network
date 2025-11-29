
# Quick Start Guide

This guide will help you get started with the Neural Network library.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/ShafwanAdhi/neural-network.git
```

## Basic Usage

### 1. Import the Library

```python
from neural_network import NeuralNetwork
```

### 2. Create a Network

Define your network architecture as a list:

```python
nn = NeuralNetwork([2, 4, 'sigmoid', 3, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)
```

This creates a network with:
- 2 input neurons
- 4 hidden neurons in 1st hidden layer with sigmoid activation
- 3 hidden neurons in 2nd hidden layer with sigmoid activation
- 1 output neuron with sigmoid activation

### 3. Training

```python
# Training data
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]

# Training loop
for epoch in range(5000):
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        nn.backward(y, output, history)
```

### 4. Prediction

```python
prediction = nn.predict([1, 0])
print(f"Prediction: {prediction}")
```

## Next Steps

- Check out the [Examples](examples.md) for more use cases
- Read the [API Reference](api_reference.md) for detailed documentation
