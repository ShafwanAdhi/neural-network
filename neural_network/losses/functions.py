```python
"""
Loss Functions and Their Derivatives
"""

import numpy as np


class LossFunctions:
    """Class for storing all loss functions and derivatives"""
    
    @staticmethod
    def bce_loss(y_true, y_pred):
        """Binary Cross Entropy Loss"""
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    @staticmethod
    def der_bce(y_true, y_pred):
        """Derivative of Binary Cross Entropy"""
        return y_pred - y_true
    
    @staticmethod
    def ce_loss(y_true, y_pred):
        """Cross Entropy Loss"""
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - np.sum(y_true * np.log(y_pred), axis=-1)
        return loss
    
    @staticmethod
    def der_ce(y_true, y_pred):
        """Derivative of Cross Entropy"""
        return y_pred - y_true
    
    @staticmethod
    def mse_loss(y_true, y_pred):
        """Mean Squared Error Loss"""
        loss = (y_true - y_pred) ** 2
        return loss
    
    @staticmethod
    def der_mse(y_true, y_pred):
        """Derivative of MSE"""
        return -2 * (y_true - y_pred)
```
