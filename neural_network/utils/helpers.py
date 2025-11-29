```python
"""
Helper functions for Neural Network
"""

import pickle
import json


def save_model(model, filepath):
    """
    Save model to file using pickle
    
    Args:
        model: NeuralNetwork instance
        filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load model from file
    
    Args:
        filepath: Path to model file
        
    Returns:
        model: NeuralNetwork instance
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_weights(model, filepath):
    """
    Save only weights and bias to JSON file
    
    Args:
        model: NeuralNetwork instance
        filepath: Path to save file
    """
    data = {
        'weights': model.get_weights(),
        'bias': model.get_bias(),
        'architecture': model.layer_neuron,
        'loss': model.loss_type,
        'learning_rate': model.learning_rate
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Weights saved to {filepath}")


def print_network_info(model):
    """
    Print network architecture information
    
    Args:
        model: NeuralNetwork instance
    """
    print("=" * 50)
    print("Network Architecture")
    print("=" * 50)
    print(f"Architecture: {model.layer_neuron}")
    print(f"Loss Function: {model.loss_type}")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"\nNumber of layers: {len(model.weights)}")
    
    total_params = 0
    for i, (w, b) in enumerate(zip(model.weights, model.bias)):
        layer_params = len(w) * len(w[0]) + len(b)
        total_params += layer_params
        print(f"Layer {i+1}: {len(w[0])} -> {len(w)} neurons, {layer_params} parameters")
    
    print(f"\nTotal parameters: {total_params}")
    print("=" * 50)
```
