"""
Loss Functions and Their Derivatives
"""

import numpy as np


class LossFunctions:
    """Class for storing all loss functions and derivatives"""
    
    @staticmethod
    def bce_loss(y_true, y_pred):
        """Binary Cross Entropy Loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    @staticmethod
    def der_bce(y_true, y_pred):
        """Derivative of Binary Cross Entropy"""
        return y_pred - y_true
    
    @staticmethod
    def ce_loss(arr_label, arr_pred):
        epsilon = 1e-15
        arr_pred = np.clip(arr_pred, epsilon, 1.0)
        loss = -np.sum(arr_label * np.log(arr_pred))
        return loss
    
    @staticmethod
    def der_ce(arr_label, arr_pred):
        return [-y / a for y, a in zip(arr_label, arr_pred)]
    
    @staticmethod
    def mse_loss(y_true, y_pred):
        """Mean Squared Error Loss"""
        loss = (y_true - y_pred) ** 2
        return loss
    
    @staticmethod
    def der_mse(y_true, y_pred):
        """Derivative of MSE"""
        return -2 * (y_true - y_pred)
