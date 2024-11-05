import numpy as np
from gan import GAN
from vae import VAE
from network import Network
from fc_layer import FCLayer
from dropout_layer import DropoutLayer
from activation_layer import ActivationLayer
from loss_functions import binary_crossentropy, binary_crossentropy_prime
from activation_functions import relu, relu_prime, sigmoid, sigmoid_prime, tanh, tanh_prime

class UnifiedNetwork:
    def __init__(self, input_shape, phishing_email_shape):
        self.input_shape = input_shape
        self.phishing_email_shape = phishing_email_shape
        self.build()

    def build(self):
        # VAE Component: Encoder and Decoder for anomaly detection
        self.vae = VAE(input_size=self.input_shape, latent_size=10)

        # GAN Components: Generator and Discriminator for phishing emails
        self.generator = Network()
        self.generator.add(FCLayer(100, 256))  # Latent space to phishing email
        self.generator.add(ActivationLayer(relu, relu_prime))
        self.generator.add(DropoutLayer(0.3))
        self.generator.add(FCLayer(256, self.phishing_email_shape))
        self.generator.add(ActivationLayer(tanh, tanh_prime))

        self.discriminator = Network()
        self.discriminator.add(FCLayer(self.phishing_email_shape, 512))  # Discriminator input shape
        self.discriminator.add(ActivationLayer(relu, relu_prime))
        self.discriminator.add(DropoutLayer(0.3))
        self.discriminator.add(FCLayer(512, 256))
        self.discriminator.add(ActivationLayer(relu, relu_prime))
        self.discriminator.add(DropoutLayer(0.3))
        self.discriminator.add(FCLayer(256, 1))  # Output layer for real/fake classification
        self.discriminator.add(ActivationLayer(sigmoid, sigmoid_prime))

        self.gan = GAN(self.generator, self.discriminator)

        # Basic Neural Network for IDS and malware analysis
        self.network = Network()
        self.network.add(FCLayer(self.input_shape, 512))  # First fully connected layer
        self.network.add(ActivationLayer(relu, relu_prime))
        self.network.add(DropoutLayer(0.3))
        self.network.add(FCLayer(512, 256))  # Second fully connected layer
        self.network.add(ActivationLayer(relu, relu_prime))
        self.network.add(DropoutLayer(0.3))
        self.network.add(FCLayer(256, 64))  # Third fully connected layer
        self.network.add(ActivationLayer(relu, relu_prime))
        self.network.add(FCLayer(64, 2))  # Output layer for binary classification
        self.network.add(ActivationLayer(sigmoid, sigmoid_prime))

        # Set loss functions
        self.generator.use(binary_crossentropy, binary_crossentropy_prime)
        self.discriminator.use(binary_crossentropy, binary_crossentropy_prime)
        self.network.use(binary_crossentropy, binary_crossentropy_prime)

    def fit(self, x_train, phishing_emails, labels, epochs, learning_rate):
        for epoch in range(epochs):
            # Step 1: VAE for anomaly detection
            reconstruction, z_mean, z_log_var = self.vae.forward_propagation(x_train)
            reconstruction_loss = np.mean((x_train - reconstruction) ** 2)
            kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
            vae_loss = reconstruction_loss + kl_loss
            self.vae.backward_propagation(x_train, reconstruction, z_mean, z_log_var, learning_rate)

            # Step 2: Train GAN for phishing email generation
            noise = np.random.randn(phishing_emails.shape[0], 100)
            fake_phishing_emails = self.generator.predict(noise)
            real_output = self.discriminator.predict(phishing_emails)
            fake_output = self.discriminator.predict(fake_phishing_emails)
            gan_loss_real = binary_crossentropy(real_output, np.ones_like(real_output))
            gan_loss_fake = binary_crossentropy(fake_output, np.zeros_like(fake_output))
            gan_loss = gan_loss_real + gan_loss_fake
            self.generator.fit(noise, fake_phishing_emails, epochs=1, learning_rate=learning_rate)
            self.discriminator.fit(fake_phishing_emails, real_output, epochs=1, learning_rate=learning_rate)

            # Step 3: Train basic network for IDS and malware analysis
            self.network.fit(x_train, labels, epochs=1, learning_rate=learning_rate)

            # Combine losses with weights
            vae_loss_weight = 0.1
            gan_loss_weight = 0.1
            classification_loss_weight = 1.0
            total_loss = classification_loss_weight * np.mean(labels) + gan_loss_weight * gan_loss + vae_loss_weight * vae_loss

            # Check for NaN or Inf in total loss
            if np.isnan(total_loss) or np.isinf(total_loss):
                print("NaN or Inf detected in total loss!")
                break

            print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss:.4f}")

    def predict(self, input_data):
        return self.network.predict(input_data)

    def generate_synthetic_data(self, num_samples, noise_size=100):
        synthetic_data = self.gan.generate(num_samples=num_samples, noise_size=noise_size)
        return synthetic_data
