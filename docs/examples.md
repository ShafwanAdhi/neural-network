# Examples

Collection of examples demonstrating the library usage.

## XOR Problem

Binary classification problem that requires non-linear decision boundary.

```python
from neural_network import NeuralNetwork

# Create network
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)

# Data
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]

# Training
for epoch in range(5000):
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        nn.backward(y, output, history)

# Testing
for X, y in zip(X_train, y_train):
    pred = nn.predict(X)
    print(f"Input: {X}, True: {y}, Predicted: {pred:.4f}")
```

## AND Gate

Simple binary logic gate.

```python
from neural_network import NeuralNetwork

nn = NeuralNetwork([2, 2, 'sigmoid', 1, 'sigmoid'], loss='bce', learning_rate=0.5)

X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 0, 0, 1]

for epoch in range(2000):
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        nn.backward(y, output, history)

for X, y in zip(X_train, y_train):
    pred = nn.predict(X)
    print(f"{X} AND = {y} (predicted: {pred:.2f})")
```

## Linear Regression

Regression problem: y = 2x + 1

```python
from neural_network import NeuralNetwork

nn = NeuralNetwork([1, 5, 'relu', 1], loss='mse', learning_rate=0.01)

X_train = [[0], [1], [2], [3], [4]]
y_train = [1, 3, 5, 7, 9]

for epoch in range(2000):
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        nn.backward(y, output, history)

# Test on new data
test_X = [[5], [6], [7], [10]]
for X in test_X:
    pred = nn.predict(X)
    true_y = 2 * X[0] + 1
    print(f"X={X[0]}, True={true_y}, Predicted={pred:.2f}")
```

## Multi-class Classification

```python
from neural_network import NeuralNetwork

# 3 classes classification
nn = NeuralNetwork([4, 8, 'relu', 3, 'sigmoid'], loss='ce', learning_rate=0.01)

# Your training data here
# X_train = [[...], [...], ...]
# y_train = [[1,0,0], [0,1,0], [0,0,1], ...]

for epoch in range(1000):
    for X, y in zip(X_train, y_train):
        output, history = nn.forward(X)
        nn.backward(y, output, history)
```

## Deep Network

Creating a deeper network with multiple hidden layers.

```python
from neural_network import NeuralNetwork

# 4 hidden layers
nn = NeuralNetwork(
    [10, 16, 'relu', 8, 'relu', 4, 'relu', 2, 'relu', 1, 'sigmoid'],
    loss='bce',
    learning_rate=0.01
)
```

## Saving and Loading Models

```python
from neural_network import NeuralNetwork
from neural_network.utils.helpers import save_model, load_model

# Train model
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)
# ... training code ...

# Save model
save_model(nn, 'trained_model.pkl')

# Load model later
nn_loaded = load_model('trained_model.pkl')
prediction = nn_loaded.predict([1, 0])
```

## Monitoring Training Progress

```python
from neural_network import NeuralNetwork

nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)

X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]

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
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# Plot losses
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()
```
