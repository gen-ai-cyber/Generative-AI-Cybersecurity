import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def binary_crossentropy(predictions, targets):
    # Ensure numerical stability by adding a small value (epsilon) to prevent log(0)
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    
    # Calculate binary cross-entropy loss
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return loss