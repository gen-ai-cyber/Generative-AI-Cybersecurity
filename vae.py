import numpy as np

class VAE:
    def __init__(self, input_size, latent_size):
        self.input_size = input_size
        self.latent_size = latent_size

        # Encoder weights and biases
        self.encoder_w = np.random.randn(input_size, latent_size)
        self.encoder_b = np.zeros((1, latent_size))

        # Decoder weights and biases
        self.decoder_w = np.random.randn(latent_size, input_size)
        self.decoder_b = np.zeros((1, input_size))

    def forward_propagation(self, x):
        # Encoder: Linear transformation for the mean and variance
        self.z_mean = np.dot(x, self.encoder_w) + self.encoder_b
        self.z_log_var = np.dot(x, self.encoder_w) + self.encoder_b  # Variance in the latent space
        
        # Clipping to prevent overflow in exponentials
        self.z_log_var = np.clip(self.z_log_var, -10, 10)
        
        # Reparameterization trick: Sample latent variables
        self.epsilon = np.random.randn(*self.z_mean.shape)
        self.z = self.z_mean + np.exp(0.5 * self.z_log_var) * self.epsilon

        # Decoder: Reconstruct input from latent variable
        reconstruction = np.dot(self.z, self.decoder_w) + self.decoder_b
        return reconstruction, self.z_mean, self.z_log_var

    def backward_propagation(self, x, reconstruction, z_mean, z_log_var, learning_rate):
        # Clip values to prevent overflow
        z_log_var = np.clip(z_log_var, -10, 10)  # Prevents extreme values
        z_mean = np.clip(z_mean, -10, 10)

        # Compute the gradient of the reconstruction loss (MSE)
        reconstruction_error = reconstruction - x
        grad_decoder_w = np.dot(self.z.T, reconstruction_error) / x.shape[0]
        grad_decoder_b = np.mean(reconstruction_error, axis=0)

        # Backprop through decoder
        self.decoder_w -= learning_rate * grad_decoder_w
        self.decoder_b -= learning_rate * grad_decoder_b

        # Backprop through latent space (KL Divergence)
        kl_grad = (z_mean / np.exp(z_log_var))  # Gradient of KL divergence w.r.t z_mean and z_log_var

        # Gradient of encoder weights
        grad_encoder_w = np.dot(x.T, kl_grad) / x.shape[0]

        # Update encoder weights and biases
        self.encoder_w -= learning_rate * grad_encoder_w
        self.encoder_b -= learning_rate * np.mean(kl_grad, axis=0)
