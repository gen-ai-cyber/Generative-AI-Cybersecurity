import numpy as np
from layer import Layer

class ConvLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.kernels = np.random.randn(depth, kernel_size[0], kernel_size[1], input_shape[2])
        self.biases = np.random.randn(depth, 1)

    def forward_propagation(self, input_data):
        self.input = input_data
        input_height, input_width, _ = self.input_shape
        kernel_height, kernel_width = self.kernel_size
        
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        self.output = np.zeros((output_height, output_width, self.depth))
        
        for d in range(self.depth):  # For each filter
            for i in range(output_height):
                for j in range(output_width):
                    region = self.input[i:i+kernel_height, j:j+kernel_width, :]
                    self.output[i, j, d] = np.sum(region * self.kernels[d]) + self.biases[d]
        
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        kernel_height, kernel_width = self.kernel_size
        input_height, input_width, _ = self.input_shape
        
        d_kernels = np.zeros_like(self.kernels)
        d_biases = np.zeros_like(self.biases)
        
        for d in range(self.depth):  # For each filter
            for i in range(input_height - kernel_height + 1):
                for j in range(input_width - kernel_width + 1):
                    region = self.input[i:i+kernel_height, j:j+kernel_width, :]
                    d_kernels[d] += output_error[i, j, d] * region
                    d_biases[d] += output_error[i, j, d]
        
        self.kernels -= learning_rate * d_kernels
        self.biases -= learning_rate * d_biases
        
        return None  # Not implementing backprop to input for now