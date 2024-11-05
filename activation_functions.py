#imports
import numpy as np

#Activation functions
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def relu(x):
    return np.maximum(0, x);

def relu_prime(x):
    return np.where(x > 0, 1, 0);

def sigmoid(x):
    return 1 / (1 + np.exp(-x));

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig);