import pandas as pd

# Load the CSV file
file_path = '../datasets/Enron.csv'  # Replace with the correct path
data = pd.read_csv(file_path)

# Print column names to inspect
print(data.columns)
