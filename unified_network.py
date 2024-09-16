import numpy as np
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from dropout_layer import DropoutLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime, sigmoid, sigmoid_prime
from vae import VAE
from gan import GAN

class UnifiedNetwork:
    def __init__(self, input_shape, phishing_email_shape):
        self.input_shape = input_shape
        self.phishing_email_shape = phishing_email_shape
        self.build()

    def build(self):
        # VAE Component: Encoder and Decoder for anomaly detection
        self.vae = VAE(input_size=self.input_shape, latent_size=10)

        # GAN Components: Generator and Discriminator
        self.generator = Network()
        self.generator.add(FCLayer(100, 128))  # Latent space to phishing email
        self.generator.add(ActivationLayer(relu, relu_prime))
        self.generator.add(FCLayer(128, self.phishing_email_shape))
        self.generator.add(ActivationLayer(tanh, tanh_prime))

        self.discriminator = Network()
        self.discriminator.add(FCLayer(self.phishing_email_shape, 256))  # Input is phishing email
        self.discriminator.add(ActivationLayer(relu, relu_prime))
        self.discriminator.add(FCLayer(256, 128))
        self.discriminator.add(ActivationLayer(relu, relu_prime))
        self.discriminator.add(FCLayer(128, 1))  # Real or Fake classification
        self.discriminator.add(ActivationLayer(sigmoid, sigmoid_prime))

        # Basic Neural Network for Classification with Dropout for regularization
        self.network = Network()
        self.network.add(FCLayer(input_size=self.input_shape, output_size=128))
        self.network.add(DropoutLayer(0.5))  # Dropout layer with 50% dropout rate
        self.network.add(ActivationLayer(relu, relu_prime))
        self.network.add(FCLayer(128, 64))
        self.network.add(DropoutLayer(0.5))  # Another Dropout layer after the hidden layer
        self.network.add(ActivationLayer(relu, relu_prime))
        self.network.add(FCLayer(64, 2))  # Binary classification layer
        self.network.add(ActivationLayer(sigmoid, sigmoid_prime))

    def forward(self, input_data, real_phishing_emails):
        # Forward pass through VAE for anomaly detection
        reconstruction, z_mean, z_log_var = self.vae.forward_propagation(input_data)

        # Forward pass through GAN for phishing email generation
        noise = np.random.randn(real_phishing_emails.shape[0], 100)
        fake_phishing_emails = self.generator.forward_propagation(noise)
        real_output = self.discriminator.forward_propagation(real_phishing_emails)
        fake_output = self.discriminator.forward_propagation(fake_phishing_emails)

        # Forward through basic classification network
        output = self.network.forward_propagation(input_data)
        return output, real_output, fake_output, reconstruction, z_mean, z_log_var

    def backward(self, total_loss, learning_rate):
        # Backpropagate through the whole network
        self.network.backward_propagation(total_loss, learning_rate)
        self.generator.backward_propagation(total_loss, learning_rate)
        self.discriminator.backward_propagation(total_loss, learning_rate)
        self.vae.backward_propagation(total_loss, learning_rate)
