# imports
import numpy as np

class GAN:
    # Initialization of GAN object
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    # Train the GAN generator and discriminator
    def train(self, real_data, epochs, learning_rate, noise_size):
        for epoch in range(epochs):
            # Train Discriminator with real data
            real_output = self.discriminator.forward_propagation(real_data)
            real_loss = self.loss_function(real_output, 1)  # Label "1" for real

            # Generate fake data
            noise = np.random.randn(real_data.shape[0], noise_size)
            fake_data = self.generator.forward_propagation(noise)
            fake_output = self.discriminator.forward_propagation(fake_data)
            fake_loss = self.loss_function(fake_output, 0)  # Label "0" for fake

            # Update discriminator
            d_loss = real_loss + fake_loss
            self.discriminator.backward_propagation(d_loss, learning_rate)

            # Train Generator to fool the discriminator
            g_loss = self.loss_function(fake_output, 1)  # Generator wants discriminator to believe it's real
            self.generator.backward_propagation(g_loss, learning_rate)

            # Print loss at the end of each epoch
            print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

    def generate(self, num_samples, noise_size=100):
        # Generate random noise
        noise = np.random.randn(num_samples, noise_size)
        
        # Generate synthetic data using the generator's forward propagation
        synthetic_data = self.generator.forward_propagation(noise)
        
        return synthetic_data

    def loss_function(self, prediction, target):
        return np.mean((prediction - target) ** 2)
