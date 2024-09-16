import numpy as np

class VAE:
    def __init__(self, input_size, latent_size):
        self.input_size = input_size
        self.latent_size = latent_size

        # Encoder parameters
        self.encoder_w = np.random.randn(input_size, latent_size)
        self.encoder_b = np.zeros((latent_size,))

        # Decoder parameters
        self.decoder_w = np.random.randn(latent_size, input_size)
        self.decoder_b = np.zeros((input_size,))

    def forward_propagation(self, input_data):
        # Encoder
        z_mean = np.dot(input_data, self.encoder_w) + self.encoder_b
        z_log_var = np.dot(input_data, self.encoder_w) + self.encoder_b

        # Latent space sampling
        epsilon = np.random.randn(*z_mean.shape)
        z = z_mean + np.exp(0.5 * z_log_var) * epsilon

        # Decoder
        reconstruction = np.dot(z, self.decoder_w) + self.decoder_b
        return reconstruction, z_mean, z_log_var

    def backward_propagation(self, input_data, reconstruction, z_mean, z_log_var, learning_rate):
        # Compute reconstruction loss
        reconstruction_loss = np.mean((input_data - reconstruction) ** 2)

        # Compute KL-divergence loss
        kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))

        # Total loss
        loss = reconstruction_loss + kl_loss

        # Gradient updates (simplified for this example)
        grad_encoder_w = np.dot(input_data.T, (reconstruction - input_data))
        grad_decoder_w = np.dot(z_mean.T, (reconstruction - input_data))

        # Update weights
        self.encoder_w -= learning_rate * grad_encoder_w
        self.decoder_w -= learning_rate * grad_decoder_w

        return loss

    def generate_samples(self, n_samples):
        z = np.random.randn(n_samples, self.latent_size)
        generated_data = np.dot(z, self.decoder_w) + self.decoder_b
        return generated_data
