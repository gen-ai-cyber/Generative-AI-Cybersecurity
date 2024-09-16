import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load phishing email data (assuming CSV format)
def load_phishing_emails(file_path='datasets/phishing_emails.csv'):
    # Load data from CSV
    data = pd.read_csv(file_path)
    
    phishing_emails = data.values  # Convert dataframe to numpy array
    return phishing_emails

# Load network traffic data (assuming CSV format)
def load_network_traffic(file_path='datasets/network_traffic.csv'):
    # Load data from CSV
    data = pd.read_csv(file_path)
    
    # Normalize the network traffic features (if needed)
    scaler = StandardScaler()
    network_traffic = scaler.fit_transform(data.values)  # Normalize the data
    return network_traffic
