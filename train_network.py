# Import the necessary modules
from unified_network import UnifiedNetwork
from gan import GAN
import numpy as np
from data_loader import load_phishing_emails, load_network_traffic_from_txt
from loss_functions import binary_crossentropy, binary_crossentropy_prime

# Load phishing emails from CSV and labels
phishing_emails, labels = load_phishing_emails('datasets/Phishing_Legitimate_full.csv')  # Path to your phishing emails CSV file

# Load network traffic data from TXT
network_traffic = load_network_traffic_from_txt('datasets/Train.txt')  # Path to your network traffic data

# Check the size of network_traffic and labels
print(f"Size of network traffic: {network_traffic.shape[0]}")
print(f"Size of labels: {labels.shape[0]}")

# Trim network_traffic to match labels
min_samples = labels.shape[0]  # Use the number of labels as the limiting factor
network_traffic = network_traffic[:min_samples]

# Ensure the sizes match before proceeding to fit
if network_traffic.shape[0] != labels.shape[0]:
    raise ValueError(f"Mismatch in dataset sizes: network_traffic has {network_traffic.shape[0]} samples, "
                     f"but labels has {labels.shape[0]} samples.")

# Initialize the unified network
unified_network = UnifiedNetwork(input_shape=network_traffic.shape[1], phishing_email_shape=phishing_emails.shape[1])
# unified_network.network.use(binary_crossentropy, binary_crossentropy_prime)

# Training loop
epochs = 100
learning_rate = 0.01

# If network_traffic has more samples than labels, trim the data
min_samples = min(network_traffic.shape[0], labels.shape[0])
network_traffic = network_traffic[:min_samples]
labels = labels[:min_samples]

# Train the model (VAE, GAN, and Classification Network)
unified_network.fit(network_traffic, phishing_emails, labels, epochs, learning_rate)

# After training, you can use the model to predict new data
predictions = unified_network.predict(network_traffic)
print(predictions)