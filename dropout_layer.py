import numpy as np
from layer import Layer

class DropoutLayer(Layer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        
    def forward_propagation(self, input_data):
        self.input = input_data
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.input.shape)
        return self.input * self.mask
    
    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.mask  # Only backpropagate through the active neurons
