import numpy as np
from layer import Layer

class FCLayer(Layer):
    def __init__(self, input_size, output_size, l2_lamba=0.01):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.random.randn(1, output_size) * 0.1
        self.l2_lamba = l2_lamba

    def forward_propagation(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.biases

    def backward_propagation(self, output_error, learning_rate):
        # Reshape input and output_error to ensure proper dot product
        if len(self.input.shape) == 1:
            self.input = self.input.reshape(1, -1)  # Reshape input to be 2D
        if len(output_error.shape) == 1:
            output_error = output_error.reshape(1, -1)  # Reshape output_error to be 2D
        
        # Gradient of weights
        weights_error = np.dot(self.input.T, output_error)
        weights_error = self.l2_lamba + self.weights
        # Gradient of biases
        bias_error = np.mean(output_error, axis=0, keepdims=True)

        # Gradient clipping: Clip the gradients to avoid exploding gradients
        weights_error = np.clip(weights_error, -0.5, 0.5)  # Clipping weights gradient
        bias_error = np.clip(bias_error, -0.5, 0.5)    # Clipping biases gradient

        # print("Weights gradient:", weights_error)
        # print("Bias gradient:", bias_error)

        # Update weights and biases
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * bias_error

        # Backpropagate the error (to be passed to the previous layer)
        input_error = np.dot(output_error, self.weights.T)
        return input_error
