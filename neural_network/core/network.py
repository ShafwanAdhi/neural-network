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
            'softmax': [softmax, der_softmax, xavier_initialization],
            'bce': [LossFunctions.bce_loss, LossFunctions.der_bce],
            'ce': [LossFunctions.ce_loss, LossFunctions.der_ce],
            'mse': [LossFunctions.mse_loss, LossFunctions.der_mse],
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

    def metrics_evaluation(self, x_test, y_test):
        guess_right = 0
        for x, y in zip(x_test, y_test):
            y_pred, history = forward(x, funct_list)
            if np.argmax(y) == np.argmax(y_pred):
            guess_right += 1
        accuracy = guess_right/len(x_test)

        n_class = len(y_test[0])
        conf_matrics = []
        for i in range(n_class):
            matrics_temp = []
            for j in range(n_class):
            matrics_temp.append(0)
            conf_matrics.append(matrics_temp)

        for x, y in zip(x_test, y_test):
            y_pred, history = forward(x, funct_list)
            pred_idx = np.argmax(y_pred)
            label_idx = np.argmax(y)
            conf_matrics[label_idx][pred_idx] += 1

        #menghitung precision dan recall
        precision = [0] * n_class
        recall = [0] * n_class
        f1_score = [0] * n_class
        support = [0] * n_class

        for i in range(n_class):
            for j in range(n_class):
            precision[i] += conf_matrics[j][i]
            recall[i] += conf_matrics[i][j]
            support[i] += conf_matrics[i][j]
            precision[i] = conf_matrics[i][i]/precision[i]
            recall[i] =  conf_matrics[i][i]/recall[i]
            f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 else 2/(1/precision[i] + 1/recall[i])
        self._print_classification_report(precision, recall, f1_score, support)
    
    def _print_classification_report(self, precision, recall, f1_score, support):
        n_classes = len(precision)
        
        total_samples = sum(support)
        
        # Menghitung macro dan weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        
        weighted_precision = np.sum(np.array(precision) * np.array(support)) / total_samples
        weighted_recall = np.sum(np.array(recall) * np.array(support)) / total_samples
        weighted_f1 = np.sum(np.array(f1_score) * np.array(support)) / total_samples
        
        # Header tabel
        print(f"{'Class':<10}{'Precision':<10}{'Recall':<10}{'F1-score':<10}{'Support':<10}")
        print("-"*50)
        
        # Print per class
        for i in range(n_classes):
            print(f"{i:<10}{precision[i]:<10.3f}{recall[i]:<10.3f}{f1_score[i]:<10.3f}{support[i]:<10}")
        
        # Print averages
        print("-"*50)
        print(f"{'Macro avg':<10}{macro_precision:<10.3f}{macro_recall:<10.3f}{macro_f1:<10.3f}{total_samples:<10}")
        print(f"{'Weighted avg':<10}{weighted_precision:<10.3f}{weighted_recall:<10.3f}{weighted_f1:<10.3f}{total_samples:<10}")
        
    def train_test_split(self, train_data, label_data, train_size=0.8, test_size=0.2, shuffle=True):
        n_samples = len(train_data)
    
        # Menentukan indeks apabila shuffle
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        train_end = int(n_samples * train_size)
        
        train_idx = indices[:train_end]
        test_idx = indices[train_end:]
        
        x_train = train_data[train_idx]
        y_train = label_data[train_idx]
        x_test = train_data[test_idx]
        y_test = label_data[test_idx]
        
        return x_train, y_train, x_test, y_test
    
    def forward(self, input_data):

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
                if self.funct_list[layer_idx] != "softmax":
                    activation = self.function_list[self.funct_list[layer_idx]][0](logits)
                else:
                    activation = logits
            else:
                activation = logits
            
            hasil_active.append(activation)
            hasil_logits.append(logits)
        
        if layer_idx in self.funct_list:
            if self.funct_list[layer_idx] == "softmax":
                hasil_active = self.function_list[self.funct_list[layer_idx]][0](hasil_active)
        
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
                    elif self.funct_list[layer_idx] == "softmax":
                        bp_z[-1] = self.function_list[self.funct_list[layer_idx]][1](arr_activation[layer_idx], loss_derivative)
                        self.bias[-1] = [x - (self.learning_rate * y) for x,y in zip(self.bias[-1], bp_z[-1])]
                        continue
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
