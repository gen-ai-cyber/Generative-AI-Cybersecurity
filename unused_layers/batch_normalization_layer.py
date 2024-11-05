import numpy as np
from layer import Layer

class BatchNormalizationLayer(Layer):
    def __init__(self, input_shape, momentum=0.9, epsilon=1e-5):
        self.input_shape = input_shape
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros(input_shape)
        self.running_var = np.ones(input_shape)
        self.gamma = np.ones(input_shape)  # Scaling factor
        self.beta = np.zeros(input_shape)  # Shift factor
        self.training = True
        
    def forward_propagation(self, input_data):
        self.input = input_data
        if self.training:  # Use batch statistics during training
            self.mean = np.mean(input_data, axis=0)
            self.var = np.var(input_data, axis=0)
            self.normed_input = (input_data - self.mean) / np.sqrt(self.var + self.epsilon)
            self.output = self.gamma * self.normed_input + self.beta
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:  # Use running statistics during inference
            self.normed_input = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * self.normed_input + self.beta
            
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        N, D = self.input.shape
        
        # Gradients for gamma and beta
        d_gamma = np.sum(output_error * self.normed_input, axis=0)
        d_beta = np.sum(output_error, axis=0)
        
        # Backpropagate through normalization
        normed_error = output_error * self.gamma
        d_input = (1. / N) * (1 / np.sqrt(self.var + self.epsilon)) * (N * normed_error - np.sum(normed_error, axis=0) - self.normed_input * np.sum(normed_error * self.normed_input, axis=0))
        
        # Update parameters
        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta
        
        return d_input
