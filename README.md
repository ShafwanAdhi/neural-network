# Neural Network From Scratch

A simple, modular neural network implementation built from scratch using Python and NumPy. It is designed for experimentation, learning, and customization without relying on high-level machine learning libraries.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Creating a Network](#1-creating-a-network)
  - [Forward Propagation](#2-forward-propagation)
  - [Backward Propagation](#3-backward-propagation)
  - [Training](#4-training)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Weight Initialization](#weight-initialization)
- [Complete Examples](#complete-examples)
- [API Reference](#api-reference)

---

## Features

- Multiple Activation Functions: Sigmoid, Tanh, ReLU, Leaky ReLU
- Multiple Loss Functions: Binary Cross Entropy (BCE), Cross Entropy (CE), Mean Squared Error (MSE)
- Smart Weight Initialization: Xavier (for sigmoid/tanh) and He (for ReLU variants)
- Flexible Architecture: Define network architecture easily
- Clean API: No need to pass variables repeatedly
- Easy Training: Built-in training loop
- Data Splitting: Dividing the dataset into training and testing sets.
- Metrics Evaluation: Such as accuracy, precision, recall, F1-score

---

## Installation Instructions

### For Users (Install from GitHub)

```bash
# Install directly from GitHub
pip install git+https://github.com/yourusername/neural-network-scratch.git

# Or clone and install
git clone https://github.com/yourusername/neural-network-scratch.git
cd neural-network-scratch
pip install .

# Install with development dependencies
pip install -e .[dev]
```
### For Developers

```bash
# Clone repository
git clone https://github.com/ShafwanAdhi/neural-network.git
cd neural-network-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run examples
python examples/xor_example.py
```
---
## Usage After Installation

```python
# Import from anywhere
from neural_network import NeuralNetwork

# Create and train
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='bce', learning_rate=0.01)

# Use the network
output, history = nn.forward([0.5, 0.3])
nn.backward(1, output, history)
```
---
## Quick Start (example)

```python
from neural_network import NeuralNetwork

# 1. Create network that have 2 input -> 4 hidden (sigmoid) -> 1 output (sigmoid)
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)

# 2. Training data (XOR problem)
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]

X_test = [[0, 0], [0, 1]]
y_test = [0, 1]

# 3. Training
for epoch in range(5000):
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        nn.backward(y, output, history)

# 4. Prediction
prediction = nn.predict([1, 0])
print(f"Prediction: {prediction}")  # Output: ~1.0

#5. Evaluation
nn.metrics_evaluation(X_test, y_test)
```

---

## Usage Guide

### 1. Creating a Network

**Architecture Format:**
```python
[num_input, num_hidden, 'activation', num_output, 'activation']
```

**Examples:**

```python
# Binary Classification (2 input, 4 hidden with sigmoid, 1 output with sigmoid)
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='bce', learning_rate=0.01)

# Multi-class Classification (3 input, 8 hidden with relu, 5 output with sigmoid)
nn = NeuralNetwork([3, 8, 'relu', 5, 'sigmoid'], loss='ce', learning_rate=0.001)

# Regression (4 input, 10 hidden with tanh, 1 output without activation)
nn = NeuralNetwork([4, 10, 'tanh', 1], loss='mse', learning_rate=0.01)

# Deep Network (2 input, 8 hidden, 4 hidden, 1 output)
nn = NeuralNetwork([2, 8, 'relu', 4, 'relu', 1, 'sigmoid'], loss='bce', learning_rate=0.01)
```

**Parameters:**
- `layer_neuron`: List describing the network architecture
- `loss`: Loss function ('bce', 'ce', 'mse')
- `learning_rate`: Learning rate for training (default: 0.01)

---
### 2. Splitting Dataset
```python
x_train, y_train, x_test, y_test = nn.train_test_split(trains, labels)
```
**Returns:**
- `x_train`: feature data used to train the model
- `y_train`: label/target corresponding to x_train
- `x_test`: feature data used to test the performance of the model
- `y_test`: the label/target corresponding to x_test
---

### 3. Forward Propagation

```python
output, history = nn.forward([0.5, 0.3])
```

**Returns:**
- `output`: Prediction result from the network
- `history`: Dictionary containing `logits` and `activation` from each layer (for backprop)

---

### 4. Backward Propagation

```python
output, history = nn.forward(X)
nn.backward(label=y, y_pred=output, history=history)
```

This method will:
- Calculate gradients
- Update weights and bias automatically
---

### 5. Training

**Manual Training Loop:**
```python
for epoch in range(1000):
    total_loss = 0
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        loss = nn.function_list[nn.funct_list["loss"]][0](y, output)
        total_loss += loss
        nn.backward(y, output, history)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(X_train):.4f}")
```

**Or using Built-in Train Method:**
```python
# For single sample
nn.train(X=[0.5, 0.3], y=1, epochs=1000, verbose=True)
```
---
### 6. Evaluation
```python
nn.metrics_evaluation(x_test, y_test)
```
This method will:
- Forward testing's dataset
- Calculate precision, recall and f1score for each class
- Also calculate their macro and weighted average
---
## Activation Functions

| Function | Usage | Best For |
|----------|-------|----------|
| `sigmoid` | `[2, 4, 'sigmoid', 1]` | Binary classification, output layer |
| `tanh` | `[2, 4, 'tanh', 1]` | Hidden layers, centered data (-1 to 1) |
| `relu` | `[2, 4, 'relu', 1]` | Hidden layers, deep networks |
| `leakyRelu` | `[2, 4, 'leakyRelu', 1]` | Avoiding dead neurons |

**Without activation:**
```python
# Output layer without activation (for regression)
nn = NeuralNetwork([2, 4, 1], loss='mse')
```

---

## Loss Functions

| Loss | Code | Best For |
|------|------|----------|
| Binary Cross Entropy | `loss='bce'` | Binary classification (0 or 1) |
| Cross Entropy | `loss='ce'` | Multi-class classification |
| Mean Squared Error | `loss='mse'` | Regression problems |

**Examples:**
```python
# Binary classification
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='bce')

# Multi-class classification
nn = NeuralNetwork([10, 20, 'relu', 5, 'sigmoid'], loss='ce')

# Regression
nn = NeuralNetwork([3, 8, 'relu', 1], loss='mse')
```

---

## Weight Initialization

Weight initialization is done automatically based on activation function:

| Activation | Initialization | Formula |
|------------|----------------|---------|
| `sigmoid`, `tanh` | Xavier | `sqrt(6 / (fan_in + fan_out))` |
| `relu`, `leakyRelu` | He Normal | `sqrt(2 / fan_in)` |
| No activation | Random Uniform | `-0.7` to `0.7` |

Bias is initialized with **0** for all layers.

---

## Complete Examples

### XOR Problem (Binary Classification)

```python
from neural_network import NeuralNetwork

# 1. Create network
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)

# 2. Data
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]

# 3. Training
print("Training...")
for epoch in range(5000):
    total_loss = 0
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        loss = nn.function_list[nn.funct_list["loss"]][0](y, output)
        total_loss += loss
        nn.backward(y, output, history)
    
    if epoch % 1000 == 0:
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

# 4. Testing
print("\nTesting...")
for X, y in zip(X_train, y_train):
    pred = nn.predict(X)
    print(f"Input: {X}, True: {y}, Predicted: {pred:.4f}")
```

**Output:**
```
Training...
Epoch 0, Avg Loss: 0.2500
Epoch 1000, Avg Loss: 0.0234
Epoch 2000, Avg Loss: 0.0087
Epoch 3000, Avg Loss: 0.0045
Epoch 4000, Avg Loss: 0.0028

Testing...
Input: [0, 0], True: 0, Predicted: 0.0156
Input: [0, 1], True: 1, Predicted: 0.9823
Input: [1, 0], True: 1, Predicted: 0.9831
Input: [1, 1], True: 0, Predicted: 0.0189
```

---

### Regression Problem

```python
# Simple linear regression: y = 2x + 1
nn = NeuralNetwork([1, 5, 'relu', 1], loss='mse', learning_rate=0.01)

X_train = [[0], [1], [2], [3], [4]]
y_train = [1, 3, 5, 7, 9]

for epoch in range(1000):
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        nn.backward(y, output, history)

# Test
test_X = [[5], [6], [7]]
for X in test_X:
    pred = nn.predict(X)
    true_y = 2 * X[0] + 1
    print(f"X={X[0]}, True={true_y}, Predicted={pred:.2f}")
```

---

## API Reference

### Class: `NeuralNetwork`

#### Constructor
```python
NeuralNetwork(layer_neuron, loss='mse', learning_rate=0.01)
```
- **layer_neuron** (list): Network architecture
- **loss** (str): Loss function ('bce', 'ce', 'mse')
- **learning_rate** (float): Learning rate

#### Methods

##### `train_test_split(trains, labels)`
Performs splitting dataset.

**Parameters:**
- `trains`: feature data from the dataset
- `labels`: label/target corresponding to dataset's data
  
**Returns:**
- `x_train`: feature data used to train the model
- `y_train`: label/target corresponding to x_train
- `x_test`: feature data used to test the performance of the model
- `y_test`: the label/target corresponding to x_test

**Example:**
```python
x_train, y_train, x_test, y_test = nn.train_test_split(trains, labels)
```
---

##### `forward(input_data)`
Performs forward propagation.

**Parameters:**
- `input_data`: Input (can be single value or list)

**Returns:**
- `output`: Prediction result
- `history`: Dictionary with logits and activation

**Example:**
```python
output, history = nn.forward([0.5, 0.3])
```

---

##### `backward(label, y_pred, history)`
Performs backward propagation and updates weights/bias.

**Parameters:**
- `label`: True label
- `y_pred`: Predicted output from forward pass
- `history`: History from forward pass

**Example:**
```python
nn.backward(label=1, y_pred=output, history=history)
```

---

##### `predict(X)`
Prediction without returning history (faster for inference).

**Parameters:**
- `X`: Input data

**Returns:**
- `output`: Prediction result

**Example:**
```python
prediction = nn.predict([0.5, 0.3])
```

---

##### `train(X, y, epochs=1000, verbose=True)`
Simple training loop for single sample.

**Parameters:**
- `X`: Input data
- `y`: Label
- `epochs`: Number of epochs (default: 1000)
- `verbose`: Print loss or not (default: True)

**Example:**
```python
nn.train(X=[0.5, 0.3], y=1, epochs=5000, verbose=True)
```
---

##### `metrics_evaluation(x_test, y_test)`
metrics evaluation to evaluate model.

**Parameters:**
- `x_test`: feature data used to test the performance of the model
- `y_test`: the label/target corresponding to x_test

**Example:**
```python
nn.metrics_evaluation(x_test, y_test)
```
---

##### `set_learning_rate(new_lr)`
Update learning rate.

**Parameters:**
- `new_lr`: New learning rate

**Example:**
```python
nn.set_learning_rate(0.001)
```

---

##### `get_weights()`
Returns current weights.

**Returns:**
- `weights`: List of weights for each layer

**Example:**
```python
weights = nn.get_weights()
print(f"Layer 1 weights shape: {len(weights[0])}x{len(weights[0][0])}")
```

---

##### `get_bias()`
Returns current bias.

**Returns:**
- `bias`: List of bias for each layer

**Example:**
```python
bias = nn.get_bias()
```

---

## Tips and Best Practices

### 1. Choosing Learning Rate
```python
# Too large: Training won't converge
nn = NeuralNetwork([2, 4, 'sigmoid', 1], learning_rate=10.0)  # Bad

# Too small: Training is very slow
nn = NeuralNetwork([2, 4, 'sigmoid', 1], learning_rate=0.00001)  # Bad

# Good starting points:
nn = NeuralNetwork([2, 4, 'sigmoid', 1], learning_rate=0.1)  # Good for sigmoid/tanh
nn = NeuralNetwork([2, 4, 'relu', 1], learning_rate=0.01)    # Good for relu
```

### 2. Choosing Activation Function

```python
# Binary classification: sigmoid at output
nn = NeuralNetwork([2, 4, 'relu', 1, 'sigmoid'], loss='bce')

# Multi-class: sigmoid/softmax at output
nn = NeuralNetwork([10, 20, 'relu', 5, 'sigmoid'], loss='ce')

# Regression: no activation at output
nn = NeuralNetwork([3, 8, 'relu', 1], loss='mse')

# Deep networks: ReLU at hidden layers
nn = NeuralNetwork([2, 8, 'relu', 4, 'relu', 1, 'sigmoid'], loss='bce')
```

### 3. Monitoring Training

```python
# Track loss to detect overfitting
losses = []
for epoch in range(5000):
    total_loss = 0
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        loss = nn.function_list[nn.funct_list["loss"]][0](y, output)
        total_loss += loss
        nn.backward(y, output, history)
    
    avg_loss = total_loss / len(X_train)
    losses.append(avg_loss)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# Plot losses
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

### 4. Dynamic Learning Rate

```python
# Learning rate decay
for epoch in range(5000):
    # Training loop...
    
    # Decay learning rate every 1000 epochs
    if epoch % 1000 == 0 and epoch > 0:
        new_lr = nn.learning_rate * 0.5
        nn.set_learning_rate(new_lr)
        print(f"Learning rate updated to: {new_lr}")
```

---

## FAQ

**Q: How to create a network with more than 2 hidden layers?**

A: Just add the number of neurons and activation:
```python
# 3 hidden layers
nn = NeuralNetwork([2, 8, 'relu', 4, 'relu', 2, 'relu', 1, 'sigmoid'], 
                   loss='bce', learning_rate=0.01)
```

**Q: Can I use batch training?**

A: This implementation is for single samples. For batch training, you need to modify the forward/backward pass to handle matrix operations.

**Q: How to save/load the model?**

A: Use pickle:
```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(nn, f)

# Load
with open('model.pkl', 'rb') as f:
    nn = pickle.load(f)
```

---

## License

MIT License

## Contributing

Feel free to fork and create pull requests for improvements!
