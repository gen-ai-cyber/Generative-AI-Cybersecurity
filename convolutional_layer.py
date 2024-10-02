import numpy as np
from layer import Layer

class ConvLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        # Initialize layer parameters
        self.input_shape = input_shape  # (channels, height, width)
        self.kernel_size = kernel_size  # (kernel_height, kernel_width)
        self.depth = depth  # Number of filters (depth of the output)

        # Initialize the kernels (filters) and biases
        # Shape of kernels: (depth, input_channels, kernel_height, kernel_width)
        self.kernels = np.random.randn(depth, input_shape[0], kernel_size[0], kernel_size[1])
        self.biases = np.random.randn(depth, 1)

    def forward_propagation(self, input_data):
        self.input = input_data  # Store input for backprop
        input_channels, input_height, input_width = self.input_shape
        kernel_height, kernel_width = self.kernel_size
        
        # Calculate output dimensions
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        self.output = np.zeros((self.depth, output_height, output_width))
        
        # Perform convolution
        for d in range(self.depth):  # For each filter
            for i in range(output_height):
                for j in range(output_width):
                    region = self.input[:, i:i+kernel_height, j:j+kernel_width]  # Select region
                    self.output[d, i, j] = np.sum(region * self.kernels[d]) + self.biases[d]
        
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        kernel_height, kernel_width = self.kernel_size
        input_channels, input_height, input_width = self.input_shape
        
        # Initialize gradients for kernels and biases
        d_kernels = np.zeros_like(self.kernels)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)  # To store the gradient wrt input (for backpropagation)
        
        # Backpropagate the error through the layer
        for d in range(self.depth):  # For each filter
            for i in range(input_height - kernel_height + 1):
                for j in range(input_width - kernel_width + 1):
                    region = self.input[:, i:i+kernel_height, j:j+kernel_width]
                    d_kernels[d] += output_error[d, i, j] * region  # Gradient wrt kernels
                    d_biases[d] += output_error[d, i, j]  # Gradient wrt biases
                    d_input[:, i:i+kernel_height, j:j+kernel_width] += output_error[d, i, j] * self.kernels[d]  # Gradient wrt input
        
        # Update parameters
        self.kernels -= learning_rate * d_kernels
        self.biases -= learning_rate * d_biases
        
        return d_input  # Return the gradient wrt input for backpropagation to previous layers
