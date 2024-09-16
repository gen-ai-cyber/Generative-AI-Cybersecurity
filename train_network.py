# Import the necessary modules
from unified_network import UnifiedNetwork
from gan import GAN
import numpy as np
from data_loader import load_phishing_emails, load_network_traffic
from loss_functions import binary_crossentropy

# Load your datasets (including labels)
network_traffic = load_network_traffic('datasets/network_traffic.csv')  # Path to your network traffic data
phishing_emails, labels = load_phishing_emails('datasets/phishing_emails.csv')  # Load phishing emails and labels

# Initialize the unified network
unified_network = UnifiedNetwork(input_shape=50, phishing_email_shape=300)

# Initialize the GAN object
gan = GAN(generator=unified_network.generator, discriminator=unified_network.discriminator)

# Training loop
epochs = 100
learning_rate = 0.01
for epoch in range(epochs):
    # Step 1: Train the VAE for anomaly detection
    output, real_output, fake_output, reconstruction, z_mean, z_log_var = unified_network.forward(network_traffic, phishing_emails)
    
    # VAE Loss: Reconstruction + KL divergence
    reconstruction_loss = np.mean((network_traffic - reconstruction) ** 2)
    kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss

    # Step 2: Train the GAN for phishing email generation
    gan.train(real_data=phishing_emails, epochs=1, learning_rate=learning_rate, noise_size=100)

    # Step 3: Train the classifier on both real and GAN-generated data
    classification_loss = binary_crossentropy(output, labels)  # Binary classification loss
    gan_loss_real = binary_crossentropy(real_output, np.ones_like(real_output))  # Real = 1
    gan_loss_fake = binary_crossentropy(fake_output, np.zeros_like(fake_output))  # Fake = 0
    gan_loss = gan_loss_real + gan_loss_fake

    # Total loss
    total_loss = classification_loss + gan_loss + vae_loss

    # Backpropagation
    unified_network.backward(total_loss, learning_rate)

    # Print losses for the current epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
