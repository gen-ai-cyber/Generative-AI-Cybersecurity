import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load phishing email data (assuming CSV format)
def load_phishing_emails(file_path='datasets/Phishing_Legitimate_full.csv'):
    # Load the dataset into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Identify the non-numeric columns
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    
    # Convert non-numeric (categorical) columns to numeric using LabelEncoder
    label_encoders = {}
    for col in non_numeric_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Store the label encoder for later use if needed

    # Extract features (all columns except the last one, which is the label)
    email_features = data.iloc[:, :-1].values  # All columns except the last
    
    # Extract labels (last column)
    labels = data['CLASS_LABEL'].values
    
    # Normalize the features (optional, depending on your model)
    scaler = StandardScaler()
    email_features = scaler.fit_transform(email_features)
    
    return email_features, labels

# Load network traffic data (assuming CSV format)
def load_network_traffic(file_path='datasets/network_traffic.csv'):
    # Load data from CSV
    data = pd.read_csv(file_path)
    
    # Normalize the network traffic features (if needed)
    scaler = StandardScaler()
    network_traffic = scaler.fit_transform(data.values)  # Normalize the data
    return network_traffic


def load_phishing_emails_from_txt(file_path):
    with open(file_path, 'r') as file:
        emails = file.readlines()
    
    # Clean and remove any extra whitespace
    emails = [email.strip() for email in emails]
    
    # Vectorize the emails using TF-IDF (or other techniques)
    vectorizer = TfidfVectorizer(max_features=300)  # Adjust max_features as needed
    email_vectors = vectorizer.fit_transform(emails).toarray()
    
    return email_vectors

def load_labels_from_txt(file_path):
    with open(file_path, 'r') as file:
        labels = [int(label.strip()) for label in file.readlines()]
    
    return labels


def load_network_traffic_from_txt(file_path):
    # Load data using pandas to handle both numeric and non-numeric data
    data = pd.read_csv(file_path, delimiter=',', header=None)  # Assuming comma-separated values
    
    # Identify columns with non-numeric data
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    
    # Use LabelEncoder to convert categorical data to numeric data
    label_encoders = {}
    for col in non_numeric_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Store the label encoder for later use if needed
    
    # Normalize the numeric data (optional, depending on your model)
    scaler = StandardScaler()
    network_traffic = scaler.fit_transform(data)
    
    return network_traffic

