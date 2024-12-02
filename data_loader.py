# imports
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data, labels):
    # Step 1: Balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    data_resampled, labels_resampled = smote.fit_resample(data, labels)

    # Step 2: Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_resampled, labels_resampled, test_size=0.2, random_state=42)

    # Step 3: Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def load_network_traffic_data(file_path='datasets/Train.txt', label_column_index=-2):
    # Load the dataset into a DataFrame
    data = pd.read_csv(file_path, header=None)

    # Get labels and drop the label column from features
    raw_labels = data.iloc[:, label_column_index].values
    features = data.drop(data.columns[label_column_index], axis=1)
    
    # Encode categorical columns
    label_encoders = {}
    for column in features.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column])
        label_encoders[column] = le  # Store the encoder if you need to inverse-transform later

    # Normalize numerical data
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    labels = np.array([0 if label == 'normal' else 1 for label in raw_labels])

    return features, labels

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

def load_malware_data(file_path='./datasets/Malware dataset.csv'):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['classification', 'hash'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['classification'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
    

def load_intrusion_data(self):
    pass

