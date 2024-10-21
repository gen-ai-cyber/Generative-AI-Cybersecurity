import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def binary_crossentropy(predictions, targets):
    # Ensure numerical stability by adding a small value (epsilon) to prevent log(0)
    epsilon = 1e-10
    predictions = np.array(predictions)
    targets = np.array(targets)
    targets = np.clip(targets, epsilon, 1. - epsilon)
    
    # Calculate binary cross-entropy loss
    loss = -np.mean((predictions * np.log(targets + epsilon)) + ((1 - predictions) * np.log(1 - targets + epsilon)))
    return loss

def binary_crossentropy_prime(predictions, targets):
    epslison = 1e-10
    predictions = np.array(predictions)
    targets = np.array(targets)
    targets = np.clip(targets, epslison, 1 - epslison)
    loss = -(predictions / targets) + ((1 - predictions) / (1 - targets))
    return loss