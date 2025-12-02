"""
Activation Functions and Their Derivatives
"""

import numpy as np


class ActivationFunctions:
    """Class for storing all activation functions and derivatives"""
    
    @staticmethod
    def sigmoid(num):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-1 * num))
    
    @staticmethod
    def der_sigmoid(x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    @staticmethod
    def tanh(num):
        """Tanh activation function"""
        return (np.exp(num) - np.exp(-num)) / (np.exp(num) + np.exp(-num))
    
    @staticmethod
    def der_tanh(x):
        """Derivative of tanh"""
        return 1 - x ** 2
    
    @staticmethod
    def relu(num):
        """ReLU activation function"""
        return max(0, num)
    
    @staticmethod
    def der_relu(x):
        """Derivative of ReLU"""
        return 1 if x > 0 else 0
    
    @staticmethod
    def leaky_relu(num):
        """Leaky ReLU activation function"""
        return max(0.1 * num, num)
    
    @staticmethod
    def der_leaky_relu(x):
        """Derivative of Leaky ReLU"""
        return 1 if x > 0 else 0.1

    @staticmethod        
    def softmax(arr):
        exps = [math.exp(x) for x in arr]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def der_softmax(a, loss_der): #semuanya memiliki panjang idx yang sama
        bp_z = []
        for i in range(len(a)): #setiap c/zn
            sum = 0
            for j in range(len(a)): #menghitung c/zn
                if i == j:
                    del_z = a[i] * (1 - a[j])
                else:
                    del_z = -1 * a[i] *a[j]
                del_z *= loss_der[j]
                sum += del_z
            bp_z.append(sum)
        return bp_z
