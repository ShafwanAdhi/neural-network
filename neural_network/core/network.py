"""
Core Neural Network Implementation
"""

import numpy as np
import copy
from neural_network.activations.functions import ActivationFunctions
from neural_network.losses.functions import LossFunctions
from neural_network.core.initializers import WeightInitializers


class NeuralNetwork:
    """
    Neural Network from scratch with optimized global variables
    
    Usage:
        nn = NeuralNetwork([2, 4, 'sigmoid', 1, 'sigmoid'], loss='bce', learning_rate=0.01)
        output, history = nn.forward([0.5, 0.3])
        nn.backward(label=1, y_pred=output, history=history)
    """
    
    def __init__(self, layer_neuron, loss='mse', learning_rate=0.01):
        """
        Initialize neural network
        
        Args:
            layer_neuron: List describing network architecture
            loss: Loss function ('bce', 'ce', 'mse')
            learning_rate: Learning rate for training
        """
        self.layer_neuron = layer_neuron
        self.loss_type = loss
        self.learning_rate = learning_rate
        
        # Function registry
        self.function_list = {
            'sigmoid': [ActivationFunctions.sigmoid, ActivationFunctions.der_sigmoid, 
                       WeightInitializers.xavier_initialization],
            'tanh': [ActivationFunctions.tanh, ActivationFunctions.der_tanh, 
                    WeightInitializers.xavier_initialization],
            'relu': [ActivationFunctions.relu, ActivationFunctions.der_relu, 
                    WeightInitializers.he_initialization_normal],
            'leakyRelu': [ActivationFunctions.leaky_relu, ActivationFunctions.der_leaky_relu, 
                         WeightInitializers.he_initialization_normal],
            'bce': [LossFunctions.bce_loss, LossFunctions.der_bce],
            'ce': [LossFunctions.ce_loss, LossFunctions.der_ce],
            'mse': [LossFunctions.mse_loss, LossFunctions.der_mse]
        }
        
        # Initialize network
        self.weights, self.bias, self.funct_list = self._create_nn()
        
    def _normalize_input(self):
        """Separate layer neurons and function list from input"""
        funct_list = {}
        clean_layers = []
        
        for i, num in enumerate(self.layer_neuron):
            if isinstance(num, str):
                funct_list[len(clean_layers) - 1] = num
            else:
                clean_layers.append(num)
        
        return clean_layers, funct_list
    
    def _create_weight(self, layer_neuron, funct_list):
        """Generate weights for all layers"""
        weights = []
        
        for i in range(len(layer_neuron) - 1):
            fan_in = layer_neuron[i]
            fan_out = layer_neuron[i + 1]
            layer_idx = i + 1
            
            new_layer = []
            for j in range(fan_out):
                new_layer2 = []
                for k in range(fan_in):
                    if layer_idx in funct_list:
                        new_layer2.append(
                            self.function_list[funct_list[layer_idx]][2](fan_in, fan_out)
                        )
                    else:
                        new_layer2.append(WeightInitializers.general_weight_generator())
                new_layer.append(new_layer2)
            weights.append(new_layer)
        
        return weights
    
    def _create_bias(self, layer_neuron):
        """Generate bias for all layers"""
        bias = []
        
        for i, n_neuron in enumerate(layer_neuron):
            if i == 0:
                continue
            layer_bias = [WeightInitializers.general_bias_generator() for _ in range(n_neuron)]
            bias.append(layer_bias)
        
        return bias
    
    def _create_nn(self):
        """Initialize network with weights and bias"""
        layer_neuron, funct_list = self._normalize_input()
        funct_list["loss"] = self.loss_type
        
        weights = self._create_weight(layer_neuron, funct_list)
        bias = self._create_bias(layer_neuron)
        
        return weights, bias, funct_list
    
    def forward(self, input_data):
        """
        Forward propagation
        
        Args:
            input_data: Input for network
            
        Returns:
            result: Network output
            history: Dictionary containing logits and activation
        """
        if isinstance(input_data, (int, float)):
            input_data = [input_data]
        
        arr_logits = []
        arr_activation = []
        
        result, arr_logits, arr_activation = self._forward_recursive(
            input_data, 0, arr_logits, arr_activation
        )
        
        arr_logits.insert(0, input_data)
        arr_activation.insert(0, input_data)
        
        history = {
            "logits": arr_logits,
            "activation": arr_activation
        }
        
        result = result[0] if len(result) == 1 else result
        return result, history
    
    def _forward_recursive(self, input_data, layer, arr_logits, arr_activation):
        """Recursive forward pass for one layer"""
        hasil_active = []
        hasil_logits = []
        layer_idx = layer + 1
        
        for i in range(len(self.weights[layer])):
            sum_val = 0
            for j in range(len(self.weights[layer][i])):
                sum_val += self.weights[layer][i][j] * input_data[j]
            sum_val += self.bias[layer][i]
            
            logits = sum_val
            
            if layer_idx in self.funct_list:
                activation = self.function_list[self.funct_list[layer_idx]][0](logits)
            else:
                activation = logits
            
            hasil_active.append(activation)
            hasil_logits.append(logits)
        
        arr_activation.append(hasil_active)
        arr_logits.append(hasil_logits)
        
        if layer == len(self.weights) - 1:
            return hasil_active, arr_logits, arr_activation
        else:
            return self._forward_recursive(
                hasil_active, layer + 1, arr_logits, arr_activation
            )
    
    def backward(self, label, y_pred, history):
        """
        Backward propagation and update weights & bias
        
        Args:
            label: True label
            y_pred: Predicted output
            history: History from forward pass
        """
        bp_z = self._backprop_bias(label, y_pred, history)
        # print(f"bp_z magnitudes: {[np.abs(np.array(layer)).max() for layer in bp_z]}")
        self._backprop_weights(history, bp_z)
    
    def _backprop_bias(self, label, y_pred, history):
        """Backpropagation for bias"""
        arr_logits = history["logits"]
        arr_activation = history["activation"]
        
        bp_z = copy.deepcopy(self.bias)
        
        for i in range(len(self.weights) - 1, -1, -1):
            layer_idx = i + 1
            
            if i == len(self.weights) - 1:
                loss_derivative = self.function_list[self.funct_list["loss"]][1](label, y_pred)
                
                for n_last_layer in range(len(self.weights[-1])):
                    if layer_idx not in self.funct_list:
                        activ_derivative = 1
                    elif self.funct_list[layer_idx] in ["relu", "leakyRelu"]:
                        activ_derivative = self.function_list[self.funct_list[layer_idx]][1](
                            arr_logits[layer_idx][n_last_layer]
                        )
                    else:
                        activ_derivative = self.function_list[self.funct_list[layer_idx]][1](
                            arr_activation[layer_idx][n_last_layer]
                        )
                    
                    bp_z[-1][n_last_layer] = loss_derivative * activ_derivative
                    self.bias[-1][n_last_layer] -= self.learning_rate * bp_z[-1][n_last_layer]
                continue
            
            for j in range(len(self.weights[i])):
                sum_bp = 0
                
                for jumlah_neuron_setelah in range(len(self.weights[i + 1])):
                    sum_bp += (self.weights[i + 1][jumlah_neuron_setelah][j] * 
                              bp_z[i + 1][jumlah_neuron_setelah])
                
                bp_z[i][j] = sum_bp
                
                if layer_idx in self.funct_list:
                    if self.funct_list[layer_idx] in ["relu", "leakyRelu"]:
                        activ_derivative = self.function_list[self.funct_list[layer_idx]][1](
                            arr_logits[layer_idx][j]
                        )
                    else:
                        activ_derivative = self.function_list[self.funct_list[layer_idx]][1](
                            arr_activation[layer_idx][j]
                        )
                    bp_z[i][j] *= activ_derivative
                
                self.bias[i][j] -= self.learning_rate * bp_z[i][j]
        
        return bp_z
    
    def _backprop_weights(self, history, bp_z):
        """Backpropagation for weights"""
        arr_activation = history["activation"]
        
        for i in range(len(self.weights) - 1, -1, -1):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    gradient = bp_z[i][j] * arr_activation[i][k]
                    self.weights[i][j][k] -= self.learning_rate * gradient
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Simple training loop
        
        Args:
            X: Input data
            y: Labels
            epochs: Number of epochs
            verbose: Print loss every epoch
        """
        for epoch in range(epochs):
            output, history = self.forward(X)
            loss = self.function_list[self.funct_list["loss"]][0](y, output)
            self.backward(y, output, history)
            
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        """Prediction without history"""
        output, _ = self.forward(X)
        return output
    
    def get_weights(self):
        """Return current weights"""
        return self.weights
    
    def get_bias(self):
        """Return current bias"""
        return self.bias
    
    def set_learning_rate(self, new_lr):
        """Update learning rate"""
        self.learning_rate = new_lr
