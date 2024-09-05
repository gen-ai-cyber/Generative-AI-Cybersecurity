import numpy as np
from network import Network
from fc_layer import FCLayer
from loss_functions import mse, mse_prime
from activation_layer import ActivationLayer
from activation_functions import tanh, tanh_prime

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

out = net.predict(x_train)
print(out)