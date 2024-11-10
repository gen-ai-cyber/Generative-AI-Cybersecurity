# Generative AI and Cybersecurity Project

## Overview
This project explores the integration of Generative AI into cybersecurity applications. The main focus is on anomaly detection, deepfake phishing generation, Intrusion Detection Systems (IDS), malware analysis, and polymorphic malware detection. The project involves building a unified neural network that incorporates a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN), as well as a general-purpose neural network for classification tasks.

## Project Structure
The project is divided into the following main components:

### 1. Unified Network (`unified_network.py`)
This file defines the core architecture of the project, integrating:
- **VAE (Variational Autoencoder)**: Used for anomaly detection by reconstructing input data and calculating the reconstruction loss and KL divergence.
- **GAN (Generative Adversarial Network)**: Used for generating synthetic phishing emails, with a generator and a discriminator network.
- **Classification Network**: A fully connected network used for IDS, malware analysis, and polymorphic malware detection.

### 2. GAN Module (`gan.py`)
This module implements the structure and functions for training and generating synthetic data using a GAN.

### 3. VAE Module (`vae.py`)
This module defines the VAE architecture, including the encoder and decoder for anomaly detection purposes.

### 4. Neural Network Components
- **Network (`network.py`)**: The base class for creating and training neural networks.
- **Fully Connected Layer (`fc_layer.py`)**: Implements fully connected layers for the network.
- **Dropout Layer (`dropout_layer.py`)**: Adds dropout regularization to prevent overfitting.
- **Activation Layer (`activation_layer.py`)**: Implements activation functions and their derivatives.

### 5. Data Loader (`data_loader.py`)
Contains functions for loading and preprocessing phishing email datasets and network traffic data.

### 6. Loss Functions (`loss_functions.py`)
Implements the binary cross-entropy loss and its derivative, used for training the networks.

## Installation
To set up the project, follow these steps:
1. Clone this repository:
   git clone https://github.com/your-repo/generative-ai-cybersecurity.git
   cd generative-ai-cybersecurity
