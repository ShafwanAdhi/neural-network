```python
"""
Unit tests for Neural Network
"""

import pytest
import numpy as np
from neural_network import NeuralNetwork


def test_network_creation():
    """Test if network is created properly"""
    nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.01)
    
    assert len(nn.weights) == 2
    assert len(nn.bias) == 2
    assert nn.learning_rate == 0.01
def test_forward_pass():
    """Test forward propagation"""
    nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.01)
    
    output, history = nn.forward([0.5, 0.3])
    
    assert isinstance(output, (int, float))
    assert 0 <= output <= 1
    assert "logits" in history
    assert "activation" in history


def test_backward_pass():
    """Test backward propagation"""
    nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.01)
    
    output, history = nn.forward([0.5, 0.3])
    initial_weights = [w[:] for w in nn.weights[0]]
    
    nn.backward(1, output, history)
    
    # Weights should change after backprop
    assert nn.weights[0] != initial_weights


def test_xor_training():
    """Test if network can learn XOR"""
    nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='mse', learning_rate=0.5)
    
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    
    # Train for a few epochs
    for epoch in range(1000):
        for X, y in zip(X_train, y_train):
            output, history = nn.forward(X)
            nn.backward(y, output, history)
    
    # Test predictions
    predictions = [nn.predict(X) for X in X_train]
    
    # Check if predictions are close to expected values
    assert predictions[0] < 0.2  # 0,0 -> 0
    assert predictions[1] > 0.8  # 0,1 -> 1
    assert predictions[2] > 0.8  # 1,0 -> 1
    assert predictions[3] < 0.2  # 1,1 -> 0


if __name__ == "__main__":
    pytest.main([__file__])
```
