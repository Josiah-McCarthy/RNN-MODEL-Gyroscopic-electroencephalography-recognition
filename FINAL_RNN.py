import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load  training and testing data for GYRO
gyro_train_data = pd.read_csv('/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/RNN/Training_gyrofinal.csv')
gyro_test_data = pd.read_csv('/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/RNN/Testing_gyrofinal.csv')

# Load training and testing data for EEG
eeg_train_data = pd.read_csv('/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/RNN/Training_eegfinal.csv')
eeg_test_data = pd.read_csv('/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/RNN/Testing_eegfinal.csv')

# to run script type python lsl.py
# if you have missing libraries, do pip intsall 'missing_library' 
# define parametrs for model
lr_gyro = 0.0001 # learning rate has highest value to be 0.1, you can alwasy go dwon by 10
lr_eeg = 0.0001
epoch = 120 # from 10, 24, 32, 64, 120, 320, 500, 600 ....
b_s_eeg = 64 #  from 8, 16, 32, 64 ....
b_s_gyro = 64 #  from 8, 16, 32, 64 ....
hidden_layer = 164 # 24, 64, 200, 256 .....
dropout = 0.1 # from 0.1 to 0.9

class ImprovedGestureClassifier_gyro(nn.Module):
    def __init__(self, num_classes, input_channels_gyro):
        super(ImprovedGestureClassifier_gyro, self).__init__()
        self.input_channels = input_channels_gyro
        self.fc1 = nn.Linear(input_channels_gyro, hidden_layer)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_layer, num_classes)

    def forward(self, x):
        out_gyro = self.fc1(x)
        out_gyro = self.relu1(out_gyro)
        out_gyro = self.dropout1(out_gyro)
        out_gyro = self.fc2(out_gyro)
        return out_gyro

class ImprovedGestureClassifier_eeg(nn.Module):
    def __init__(self, num_classes, input_channels_eeg):
        super(ImprovedGestureClassifier_eeg, self).__init__()
        self.input_channels = input_channels_eeg
        self.fc1 = nn.Linear(input_channels_eeg, hidden_layer)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_layer, num_classes)

    def forward(self, y):
        out_eeg = self.fc1(y)
        out_eeg = self.relu1(out_eeg)
        out_eeg = self.dropout1(out_eeg)
        out_eeg = self.fc2(out_eeg)
        return out_eeg
    

# Extract input (X) and labels (Y) for GYRO
X_gyro_train = gyro_train_data[['MOT.Q0','MOT.Q1','MOT.Q3']].values
Y_gyro_train = gyro_train_data['Label'].values
X_gyro_test = gyro_test_data[['MOT.Q0','MOT.Q1','MOT.Q3']].values
Y_gyro_test = gyro_test_data['Label'].values

# Convert data to PyTorch tensors for GYRO
X_gyro_train_tensor = torch.tensor(X_gyro_train, dtype=torch.float32)
Y_gyro_train_tensor = torch.tensor(Y_gyro_train, dtype=torch.long)
X_gyro_test_tensor = torch.tensor(X_gyro_test, dtype=torch.float32)
Y_gyro_test_tensor = torch.tensor(Y_gyro_test, dtype=torch.long)

# Create DataLoader for GYRO training and testing data
gyro_train_data = TensorDataset(X_gyro_train_tensor, Y_gyro_train_tensor)
gyro_train_loader = DataLoader(gyro_train_data, batch_size=b_s_gyro, shuffle=True)
gyro_test_data = TensorDataset(X_gyro_test_tensor, Y_gyro_test_tensor)
gyro_test_loader = DataLoader(gyro_test_data, batch_size=b_s_gyro, shuffle=False)

# Initialize the GYRO model with dynamic num_classes
gyro_model = ImprovedGestureClassifier_gyro(num_classes=3, input_channels_gyro=3)


# Define loss and optimizer for GYRO
gyro_criterion = nn.CrossEntropyLoss()
gyro_optimizer = optim.Adam(gyro_model.parameters(), lr=lr_gyro)

# Define a start_time here before the GYRO training loop begins
start_time = time.time()  # Record start time

# Training loop for GYRO
num_epochs = epoch
for epoch in range(num_epochs):
    gyro_model.train()
    for inputs, targets in gyro_train_loader:
        gyro_optimizer.zero_grad()
        outputs = gyro_model(inputs)
        loss = gyro_criterion(outputs, targets)
        loss.backward()
        gyro_optimizer.step()
    print("GYRO")

# Evaluation loop for GYRO
gyro_model.eval()
gyro_total_correct = 0
gyro_total_samples = 0

with torch.no_grad():
    for inputs, targets in gyro_test_loader:
        outputs = gyro_model(inputs)
        print("Shape of outputs tensor:", outputs.shape)  # Add this line to check the shape
        predicted_classes_gyro = torch.argmax(outputs, dim=1)
        gyro_total_correct += (predicted_classes_gyro == targets.squeeze()).sum().item()
        gyro_total_samples += targets.size(0)


# Calculate accuracy for GYRO
gyro_accuracy = gyro_total_correct / gyro_total_samples
# print(f"GYRO Testing Accuracy: {gyro_accuracy * 100:.2f}%")



# Extract input (X) and labels (Y) for EEG
X_eeg_train = eeg_train_data[['EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']].values
Y_eeg_train = eeg_train_data['Label'].values
X_eeg_test = eeg_test_data[['EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']].values
Y_eeg_test = eeg_test_data['Label'].values

# Convert data to PyTorch tensors for EEG
X_eeg_train_tensor = torch.tensor(X_eeg_train, dtype=torch.float32)
Y_eeg_train_tensor = torch.tensor(Y_eeg_train, dtype=torch.long)
X_eeg_test_tensor = torch.tensor(X_eeg_test, dtype=torch.float32)
Y_eeg_test_tensor = torch.tensor(Y_eeg_test, dtype=torch.long)

# Create DataLoader for EEG training and testing data
eeg_train_data = TensorDataset(X_eeg_train_tensor, Y_eeg_train_tensor)
eeg_train_loader = DataLoader(eeg_train_data, batch_size=b_s_eeg, shuffle=True)
eeg_test_data = TensorDataset(X_eeg_test_tensor, Y_eeg_test_tensor)
eeg_test_loader = DataLoader(eeg_test_data, batch_size=b_s_eeg, shuffle=False)

# Initialize the EEG model with dynamic num_classes
eeg_model = ImprovedGestureClassifier_eeg(num_classes=3, input_channels_eeg=4)


# Define loss and optimizer for EEG
eeg_criterion = nn.CrossEntropyLoss()
eeg_optimizer = optim.Adam(eeg_model.parameters(), lr=lr_eeg)

# Training loop for EEG
num_epochs = epoch
for epoch in range(num_epochs):
    eeg_model.train()
    for inputs, targets in eeg_train_loader:
        eeg_optimizer.zero_grad()
        outputs = eeg_model(inputs)
        loss = eeg_criterion(outputs, targets)
        loss.backward()
        eeg_optimizer.step()
    print("EEG")

# Initialize eeg_total_correct
eeg_total_correct = 0

# Evaluation loop for EEG
eeg_model.eval()
eeg_total_correct = 0
eeg_total_samples = 0



with torch.no_grad():
    for inputs, targets in eeg_test_loader:
        outputs = eeg_model(inputs)
        print("Shape of outputs tensor:", outputs.shape)  # Add this line to check the shape
        predicted_classes_eeg = torch.argmax(outputs, dim=1)
        eeg_total_correct += (predicted_classes_eeg == targets.squeeze()).sum().item()
        eeg_total_samples += targets.size(0)

eeg_accuracy = eeg_total_correct / eeg_total_samples
end_time = time.time()  # Record end time
elapsed_time = end_time - start_time

# Define gesture labels for GYRO and EEG
gyro_labels = {i: f"Class_{i}" for i in range(len(set(Y_gyro_train)))}
eeg_labels = {i: f"Class_{i}" for i in range(len(set(Y_eeg_train)))}



# Print results
print(f"GYRO Testing Accuracy: {gyro_accuracy * 100:.2f}%")
print(f"EEG Testing Accuracy: {eeg_accuracy * 100:.2f}%")
print(f"Total Execution Time: {elapsed_time:.2f} seconds")


# save model
save_dir = '/Users/josiahmccarthy/Documents/SCHOOL/SNR CAP DESIGN/Database_Senior_Project/RNN/SAVED_MODELS'
gyro_model_path = os.path.join(save_dir, 'gyro_model.pth')
eeg_model_path = os.path.join(save_dir, 'eeg_model.pth')

torch.save(gyro_model.state_dict(), gyro_model_path)
torch.save(eeg_model.state_dict(), eeg_model_path)


# Calculate confusion matrix for GYRO
gyro_model.eval()
gyro_true_labels = []
gyro_predicted_labels = []

with torch.no_grad():
    for inputs, targets in gyro_test_loader:
        outputs = gyro_model(inputs)
        predicted_classes_gyro = torch.argmax(outputs, dim=1)
        gyro_true_labels.extend(targets.numpy())
        gyro_predicted_labels.extend(predicted_classes_gyro.numpy())

gyro_conf_matrix = confusion_matrix(gyro_true_labels, gyro_predicted_labels)
print("Confusion Matrix for GYRO:")
print(gyro_conf_matrix)

# Calculate confusion matrix for EEG
eeg_model.eval()
eeg_true_labels = []
eeg_predicted_labels = []

with torch.no_grad():
    for inputs, targets in eeg_test_loader:
        outputs = eeg_model(inputs)
        predicted_classes_eeg = torch.argmax(outputs, dim=1)
        eeg_true_labels.extend(targets.numpy())
        eeg_predicted_labels.extend(predicted_classes_eeg.numpy())

eeg_conf_matrix = confusion_matrix(eeg_true_labels, eeg_predicted_labels)
print("Confusion Matrix for EEG:")
print(eeg_conf_matrix)

# Print classification report for GYRO
gyro_report = classification_report(gyro_true_labels, gyro_predicted_labels)
print("Classification Report for GYRO:")
print(gyro_report)

# Print classification report for EEG
eeg_report = classification_report(eeg_true_labels, eeg_predicted_labels)
print("Classification Report for EEG:")
print(eeg_report)

# Plot confusion matrix for GYRO
plt.figure(figsize=(8, 6))
sns.heatmap(gyro_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=gyro_labels.values(), yticklabels=gyro_labels.values())
plt.title("Confusion Matrix for GYRO")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot confusion matrix for EEG
plt.figure(figsize=(8, 6))
sns.heatmap(eeg_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=eeg_labels.values(), yticklabels=eeg_labels.values())
plt.title("Confusion Matrix for EEG")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()