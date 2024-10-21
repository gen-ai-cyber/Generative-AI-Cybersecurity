import numpy as np
from data_loader import load_phishing_emails, load_network_traffic_from_txt
from striprtf.striprtf import rtf_to_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

phishing_emails, labels = load_phishing_emails('./datasets/Phishing_Legitimate_full.csv')  # Path to your phishing emails CSV file
network_traffic = load_network_traffic_from_txt('datasets/Train.txt')  # Path to your network traffic data
min_samples = min(network_traffic.shape[0], labels.shape[0])
labels = labels[:min_samples]


# with open('./predict_output.rtf', 'r') as f:
#     predictions = f.readlines()

# # Extract numerical values from predictions (clean the raw file format)
# predictions = [list(map(float, line.strip().split(','))) for line in predictions]

# # Assuming that the labels are binary, you can compute accuracy
# predictions = np.array(predictions)
# predicted_classes = np.argmax(predictions, axis=1)

# # Compute accuracy
# accuracy = np.mean(predicted_classes == labels)
# Assuming `predictions` contains your model's output, and `true_labels` contains the actual labels
# You may need to threshold the predictions first
# threshold = 0.5
# binary_predictions = [1 if pred[0] > threshold else 0 for pred in predictions]
# binary_predictions = (predictions[:, 1] > 0.5).astype(int)
# Calculate performance metrics
# accuracy = accuracy_score(labels, binary_predictions)
# precision = precision_score(labels, binary_predictions)
# recall = recall_score(labels, binary_predictions)
# f1 = f1_score(labels, binary_predictions)

# Print metrics
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")


# Load and parse the .rtf file to extract raw text
# with open('./predict_output.rtf', 'r') as file:
#     rtf_content = file.read()

# # Convert the RTF content to plain text
# plain_text = rtf_to_text(rtf_content)

# # Print the plain text to see if it's correctly parsed
# print(plain_text)

# # Assuming predictions are in plain text now
# lines = plain_text.splitlines()

# Parse the lines and extract the numerical predictions
# predictions = []
# for line in plain_text.splitlines():
#     try:
#         # Convert each line to a float (assuming each line contains one prediction)
#         predictions.append(float(line.strip()))
#     except ValueError:
#         # If a line can't be converted to a float, skip it
#         continue

# # Convert to numpy array for further processing
# predictions = np.array(predictions)

predictions = [
    np.array([[0.45692521, 0.45769254]]), np.array([[0.4289505 , 0.44285925]]),
    np.array([[0.4168553 , 0.39744293]]), np.array([[0.45501281, 0.43337049]]),
    np.array([[0.45260073, 0.45975773]]), np.array([[0.43676078, 0.416115  ]]),
]


probabilities = np.array([pred[0][1] for pred in predictions])

# Now you can continue with your evaluation logic:
# For example, convert predictions to binary labels
# binary_predictions = (predictions[:, 1] > 0.5).astype(int)
# If predictions are 1D (a single probability value per sample)
binary_predictions = (probabilities > 0.5).astype(int)

# print(f"Predictions shape: {binary_predictions.shape}")
# print(f"Labels shape: {labels.shape}")
# print(f"Predictions: {binary_predictions}")
# print(f"Labels: {labels}")
print(labels[:6])
true_labels = np.array([1, 1, 1, 1, 1, 1])
# Assume you have the true labels already loaded in `true_labels`
# accuracy = np.mean(binary_predictions == labels)
if len(binary_predictions) != len(true_labels):
    raise ValueError("Mismatch between the number of predictions and true labels.")

# Calculate metrics
accuracy = accuracy_score(true_labels, binary_predictions)
precision = precision_score(true_labels, binary_predictions)
recall = recall_score(true_labels, binary_predictions)
f1 = f1_score(true_labels, binary_predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Accuracy: {accuracy}")


