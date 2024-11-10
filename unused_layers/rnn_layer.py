import numpy as np
from layer import Layer

class RNNLayer(Layer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.Wx = np.random.randn(input_size, hidden_size)
        self.Wh = np.random.randn(hidden_size, hidden_size)
        self.Wy = np.random.randn(hidden_size, output_size)
        self.bh = np.zeros((hidden_size,))
        self.by = np.zeros((output_size,))
        
        self.h = np.zeros((hidden_size,))  # Hidden state
        
    def forward_propagation(self, input_data):
        # For each time step
        self.input = input_data
        self.h = np.tanh(np.dot(input_data, self.Wx) + np.dot(self.h, self.Wh) + self.bh)
        output = np.dot(self.h, self.Wy) + self.by
        return output
    
    def backward_propagation(self, output_error, learning_rate):
        # Implementing backpropagation through time (BPTT) is complex, so we will omit details for now
        return None
