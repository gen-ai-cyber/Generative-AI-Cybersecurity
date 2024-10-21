import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

# Load phishing email data for BERT (assuming CSV format)
def load_phishing_emails_for_bert(file_path='../datasets/Phishing_Legitimate_full.csv'):
    # Load the dataset into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Use 'body' as the email content and 'label' as the classification labels
    email_text_column = 'body'  # Email text
    labels_column = 'label'  # Classification labels
    
    # Extract the email content and labels
    email_texts = data[email_text_column].values
    labels = data[labels_column].values
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(email_texts, labels, test_size=0.2, random_state=42)
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the email content
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)
    
    return train_encodings, test_encodings, y_train, y_test


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

