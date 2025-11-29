```markdown
# API Reference

Complete reference for the Neural Network library.

## NeuralNetwork Class

### Constructor

```python
NeuralNetwork(layer_neuron, loss='mse', learning_rate=0.01)
```

**Parameters:**
- `layer_neuron` (list): Network architecture description
- `loss` (str): Loss function - 'bce', 'ce', or 'mse'
- `learning_rate` (float): Learning rate for training

**Example:**
```python
nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='bce', learning_rate=0.01)
```

---

### Methods

#### forward(input_data)

Performs forward propagation through the network.

**Parameters:**
- `input_data`: Input data (can be a single value or list)

**Returns:**
- `output`: Network prediction
- `history`: Dictionary containing logits and activations

**Example:**
```python
output, history = nn.forward([0.5, 0.3])
```

---

#### backward(label, y_pred, history)

Performs backward propagation and updates weights and bias.

**Parameters:**
- `label`: True label
- `y_pred`: Predicted output from forward pass
- `history`: History dictionary from forward pass

**Example:**
```python
nn.backward(label=1, y_pred=output, history=history)
```

---

#### predict(X)

Makes a prediction without returning history (faster for inference).

**Parameters:**
- `X`: Input data

**Returns:**
- `output`: Network prediction

**Example:**
```python
prediction = nn.predict([0.5, 0.3])
```

---

#### train(X, y, epochs=1000, verbose=True)

Simple training loop for single sample.

**Parameters:**
- `X`: Input data
- `y`: Label
- `epochs` (int): Number of training epochs
- `verbose` (bool): Whether to print loss during training

**Example:**
```python
nn.train(X=[0.5, 0.3], y=1, epochs=5000, verbose=True)
```

---

#### set_learning_rate(new_lr)

Updates the learning rate.

**Parameters:**
- `new_lr` (float): New learning rate value

**Example:**
```python
nn.set_learning_rate(0.001)
```

---

#### get_weights()

Returns current network weights.

**Returns:**
- `weights` (list): List of weight matrices for each layer

**Example:**
```python
weights = nn.get_weights()
```

---

#### get_bias()

Returns current network bias values.

**Returns:**
- `bias` (list): List of bias vectors for each layer

**Example:**
```python
bias = nn.get_bias()
```

---

## Activation Functions

Available activation functions:

- `'sigmoid'` - Sigmoid activation
- `'tanh'` - Hyperbolic tangent
- `'relu'` - Rectified Linear Unit
- `'leakyRelu'` - Leaky ReLU

## Loss Functions

Available loss functions:

- `'bce'` - Binary Cross Entropy
- `'ce'` - Cross Entropy
- `'mse'` - Mean Squared Error

## Helper Functions

### save_model(model, filepath)

Saves the complete model to a file.

```python
from neural_network.utils.helpers import save_model

save_model(nn, 'model.pkl')
```

### load_model(filepath)

Loads a model from a file.

```python
from neural_network.utils.helpers import load_model

nn = load_model('model.pkl')
```

### print_network_info(model)

Prints detailed information about the network architecture.

```python
from neural_network.utils.helpers import print_network_info

print_network_info(nn)
```
```
