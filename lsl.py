"""Example program to show how to read a multi-channel time series from LSL."""
import time
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from pylsl import StreamInlet, resolve_stream
import numpy as np
import torch
import torch.nn as nn

import socket
import time 
from threading import Timer

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('',6000))
s.listen(5)
print('Server is now running.')

def background(message):
    if message == 0:
        print("blink")
        clientsocket.send(bytes("blink", "utf-8"))
    elif message == 1:
        print("raise_eyebrows")
        clientsocket.send(bytes("raise_eyebrows", "utf-8"))


gyro_train_data_path = 'C:/Users/nf01569-sw/Desktop/FINAL_DATA/gyro_model.pth'
eeg_train_data_path = 'C:/Users/nf01569-sw/Desktop/FINAL_DATA/eeg_model.pth'

cutoff_frequency = 0.8  # Set your desired cutoff frequency
sampling_rate_eeg = 256  # Set your sampling rate
sampling_rate_gyro = 256  # Set your sampling rate

class ImprovedGestureClassifier(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(ImprovedGestureClassifier, self).__init__()
        self.input_channels = input_channels
        self.fc1 = nn.Linear(input_channels, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        return out
    
def butter_highpass_filter(data, cutoff_freq, sampling_rate, order=7):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # Set padlen as a fraction of the length of the input data
    padlen_fraction = 0.1  # Adjust this fraction based on your data and filter order
    padlen = int(len(data) * padlen_fraction)

    filtered_data = filtfilt(b, a, data, padlen=padlen)
    return filtered_data

def apply_notch_filter(data, notch_freq, quality_factor, sampling_rate):
    nyquist = 0.5 * sampling_rate
    notch_freq_normalized = notch_freq / nyquist
    b, a = iirnotch(notch_freq_normalized, quality_factor)
    
    # Check if the length of the input data is sufficient
    if len(data) >= max(len(b), len(a)):
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    else:
        # If the input data is too short, you might choose to return it as is or handle it differently
        return data

# Function to apply filters to live stream data
def apply_filters_to_live_data(eeg_data, cutoff_freq, sampling_rate, notch_freq=60, quality_factor=30):
    # Apply notch filter
    notch_filtered_data = apply_notch_filter(eeg_data, notch_freq, quality_factor, sampling_rate)
    
    # Apply high-pass filter to the notch-filtered data
    filtered_data = butter_highpass_filter(notch_filtered_data, cutoff_freq, sampling_rate)
    
    return filtered_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2  # Replace with the actual number of classes in your classification task
input_channels = 2  # Replace with the actual number of input channels in your data

# Create instances of your model class
model_gyro = ImprovedGestureClassifier(num_classes, input_channels)
model_eeg = ImprovedGestureClassifier(num_classes, input_channels)

# Load the trained model weights
model_gyro.load_state_dict(torch.load(gyro_train_data_path, map_location=torch.device('cpu')))
model_gyro.eval()

model_eeg.load_state_dict(torch.load(eeg_train_data_path, map_location=torch.device('cpu')))
model_eeg.eval()

# model_gyro = torch.load(gyro_train_data_path)
# model_eeg = torch.load(eeg_train_data_path)
print("looking for a stream...")
# first resolve a Motion stream on the lab network
streams_eeg = resolve_stream('type', 'EEG')
streams_motion = resolve_stream('type', 'Motion')

# Combine the lists of EEG and Motion streams
streams = streams_eeg + streams_motion

# print(streams)

# Function to apply filters to live stream data
def apply_filters_to_live_data(eeg_data, cutoff_freq, sampling_rate, notch_freq=60, quality_factor=30):
    # Apply notch filter
    notch_filtered_data = apply_notch_filter(eeg_data, notch_freq, quality_factor, sampling_rate)
    
    # Apply high-pass filter to the notch-filtered data
    filtered_data = butter_highpass_filter(notch_filtered_data, cutoff_freq, sampling_rate)
    
    return filtered_data
 
# Your processing loop
while True:
    # Accept a connection from a client
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")

    # Inside the loop, read and process the LSL stream
    streams_eeg = resolve_stream('type', 'EEG')
    streams_motion = resolve_stream('type', 'Motion')
    # Combine the lists of EEG and Motion streams
    streams = streams_eeg + streams_motion
    print(streams)
    inlet = StreamInlet(streams[0])

    while True:
        sample, timestamp = inlet.pull_sample()
        if timestamp is not None:
            # print(sample)
            eeg_columns = [3, 16]
            gyro_columns = [17, 18, 19, ]
            eeg_data = np.array(sample)[eeg_columns]
            gyro_data = np.array(sample)[gyro_columns]

            # Apply filters to live stream data
            filtered_eeg_data = apply_filters_to_live_data(eeg_data, cutoff_frequency, sampling_rate_eeg)
            filtered_gyro_data = apply_filters_to_live_data(gyro_data, cutoff_frequency, sampling_rate_gyro)

            # Perform live classification
            with torch.no_grad():
                eeg_data_copy = np.copy(filtered_eeg_data)
                gyro_data_copy = np.copy(filtered_gyro_data)
                eeg_data_tensor = torch.from_numpy(eeg_data_copy).float().to(device)
                gyro_data_tensor = torch.from_numpy(gyro_data_copy).float().to(device)
                prediction_eeg = model_eeg(eeg_data_tensor)
                prediction_gyro = model_eeg(gyro_data_tensor)
                prediction_eeg = torch.argmax(prediction_eeg).item()
                prediction_gyro = torch.argmax(prediction_gyro).item()

            # Handle the prediction in the background
            background(prediction_eeg)
            background(prediction_gyro)

            # Sleep for a short duration to control the loop frequency
            time.sleep(0.2)