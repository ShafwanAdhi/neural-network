```python
"""
Weight Initialization Methods
"""

import random
import numpy as np


class WeightInitializers:
    """Class for storing all weight initialization methods"""
    
    @staticmethod
    def xavier_initialization(fan_in, fan_out):
        """Xavier initialization for sigmoid and tanh"""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return random.uniform(-limit, limit)
    
    @staticmethod
    def he_initialization_normal(fan_in, fan_out):
        """He initialization for relu and leaky relu"""
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std)
    
    @staticmethod
    def general_weight_generator():
        """General random weight initialization"""
        return random.uniform(-0.7, 0.7)
    
    @staticmethod
    def general_bias_generator():
        """General bias initialization"""
        return 0
```
