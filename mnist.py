import numpy as np
from network import Network
from fc_layer import FCLayer
from loss_functions import mse, mse_prime
from activation_layer import ActivationLayer
from activation_functions import tanh, tanh_prime

from keras import datasets
from keras import utils

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = utils.np_utils.to_categorial(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
y_test = utils.np_utils.to_categorical(y_test)

net = Network()
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train[0:1000], epochs=35, learning_rate=0.1)

out = net.predict(x_test[0:3])
print("")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])