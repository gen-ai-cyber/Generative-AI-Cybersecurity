from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pyod.models.iforest import IForest
import numpy as np
from data_loader_backup import load_network_traffic_from_txt, load_phishing_emails_for_bert

# Load network traffic data
network_traffic = load_network_traffic_from_txt('../datasets/Train.txt')

# Initialize Isolation Forest model
clf = IForest()

# phishing_emails, labels = load_phishing_emails('../datasets/Phishing_Legitimate_full.csv')
# # Load the dataset
# data = pd.read_csv('../datasets/Phishing_Legitimate_full.csv')
# X = data['Email_Content']  # Assuming 'Email_Content' column contains the email content
# y = data['Label']  # Assuming 'Label' contains 0 for legitimate, 1 for phishing

# Load phishing email data
train_encodings, test_encodings, y_train, y_test = load_phishing_emails_for_bert('../datasets/Enron.csv')

# Convert to torch datasets
class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PhishingDataset(train_encodings, y_train)
test_dataset = PhishingDataset(test_encodings, y_test)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    evaluation_strategy="steps",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the input text
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512)

# train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
# test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

# Convert to torch datasets
# class PhishingDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = PhishingDataset(train_encodings, y_train.values)
# test_dataset = PhishingDataset(test_encodings, y_test.values)

# # Training the model
# training_args = TrainingArguments(
#     output_dir='./results',
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     logging_dir='./logs',
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

# Fit the model to the network traffic data
clf.fit(network_traffic)
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./phishing-bert')
tokenizer.save_pretrained('./phishing-bert')

# Predict anomalies (1 for anomaly, 0 for normal)
anomaly_scores = clf.decision_function(network_traffic)
anomaly_labels = clf.predict(network_traffic)

# Print out the results
print("Anomaly Labels:", anomaly_labels)
