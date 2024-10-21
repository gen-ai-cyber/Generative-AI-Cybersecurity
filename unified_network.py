import numpy as np
from gan import GAN
from vae import VAE
from network import Network
from fc_layer import FCLayer
from convolutional_layer import ConvLayer 
from max_pooling_layer import MaxPoolingLayer
from dropout_layer import DropoutLayer 
from activation_layer import ActivationLayer
from loss_functions import binary_crossentropy, binary_crossentropy_prime
from activation_functions import relu, relu_prime, tanh, tanh_prime, sigmoid, sigmoid_prime

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
        self.generator.add(FCLayer(100, 256))  # Latent space to phishing email (increased to 256 units for complexity)
        self.generator.add(ActivationLayer(relu, relu_prime))
        self.generator.add(DropoutLayer(0.3))  # Dropout to prevent overfitting
        self.generator.add(FCLayer(256, self.phishing_email_shape))
        self.generator.add(ActivationLayer(tanh, tanh_prime))

        self.discriminator = Network()
        self.discriminator.add(FCLayer(self.phishing_email_shape, 512))  # Increased complexity
        self.discriminator.add(ActivationLayer(relu, relu_prime))
        self.discriminator.add(DropoutLayer(0.3))
        self.discriminator.add(FCLayer(512, 256))
        self.discriminator.add(ActivationLayer(relu, relu_prime))
        self.discriminator.add(DropoutLayer(0.3))
        self.discriminator.add(FCLayer(256, 1))  # Real or Fake classification
        self.discriminator.add(ActivationLayer(sigmoid, sigmoid_prime))

        # Basic Neural Network for Classification (with Convolutional Layers)
        self.network = Network()

        # Replace convolutional layers with fully connected layers
        self.network.add(FCLayer(self.input_shape, 512))  # First fully connected layer
        self.network.add(ActivationLayer(relu, relu_prime))
        self.network.add(DropoutLayer(0.3))  # Dropout to prevent overfitting
        
        self.network.add(FCLayer(512, 256))  # Second fully connected layer
        self.network.add(ActivationLayer(relu, relu_prime))
        self.network.add(DropoutLayer(0.3))
        
        self.network.add(FCLayer(256, 64))  # Third fully connected layer
        self.network.add(ActivationLayer(relu, relu_prime))
        
        self.network.add(FCLayer(64, 2))  # Output layer for binary classification
        self.network.add(ActivationLayer(sigmoid, sigmoid_prime))

        # Set the loss functions for the classifier network
        self.generator.use(binary_crossentropy, binary_crossentropy_prime)
        self.discriminator.use(binary_crossentropy, binary_crossentropy_prime)
        self.network.use(binary_crossentropy, binary_crossentropy_prime)

    def fit(self, x_train, phishing_emails, labels, epochs, learning_rate):
        """
        Train the Unified Network.
        """

        for epoch in range(epochs):
            # Step 1: Forward pass through VAE for anomaly detection
            reconstruction, z_mean, z_log_var = self.vae.forward_propagation(x_train)
            
            # VAE Loss: Reconstruction + KL divergence
            reconstruction_loss = np.mean((x_train - reconstruction) ** 2)
            kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
            vae_loss = reconstruction_loss + kl_loss
            
            # Backpropagate through VAE (pass z_mean, z_log_var, and learning_rate)
            self.vae.backward_propagation(x_train, reconstruction, z_mean, z_log_var, learning_rate)

            # Step 2: Train GAN
            noise = np.random.randn(phishing_emails.shape[0], 100)  # Random noise for GAN
            fake_phishing_emails = self.generator.predict(noise)  # GAN generator output
            real_output = self.discriminator.predict(phishing_emails)  # Real phishing emails
            fake_output = self.discriminator.predict(fake_phishing_emails)  # Fake phishing emails

            # GAN Loss: Real and Fake classification
            classification_loss = binary_crossentropy(real_output, labels)
            gan_loss_real = binary_crossentropy(real_output, np.ones_like(real_output))  # Real = 1
            gan_loss_fake = binary_crossentropy(fake_output, np.zeros_like(fake_output))  # Fake = 0
            gan_loss = gan_loss_real + gan_loss_fake

            # Backpropagate through GAN
            self.generator.fit(noise, fake_phishing_emails, epochs=1, learning_rate=learning_rate)
            self.discriminator.fit(fake_phishing_emails, real_output, epochs=1, learning_rate=learning_rate)

            # Step 3: Train the basic classification network
            self.network.fit(x_train, labels, epochs=1, learning_rate=learning_rate)

            # Scale the losses if necessary
            vae_loss_weight = 0.1
            gan_loss_weight = 0.1
            classification_loss_weight = 1.0

            # Combine the losses with weights
            total_loss = classification_loss_weight * classification_loss + \
                        gan_loss_weight * gan_loss + \
                        vae_loss_weight * vae_loss

            # Check for NaN or Inf in the total loss
            if np.isnan(total_loss) or np.isinf(total_loss):
                print("NaN or Inf detected in total loss!")
                break  # Stop the training if NaN or Inf is detected


            print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}")

    def predict(self, input_data):
        """
        Predict output using the classification network.
        """
        return self.network.predict(input_data)
