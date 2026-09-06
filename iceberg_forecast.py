#!/usr/bin/env python3
"""
Iceberg Drift Forecasting API
Based on the paper "IcebergDriftForecastingUsingMachineLearning.pdf"
Provides predictions for iceberg positions via HTTP API.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import random
import os
from flask import Flask, jsonify

# ======================
# CONFIGURATION
# ======================
SEQ_LENGTH = 10          # Number of past time steps to use for prediction
PRED_STEPS = 3           # Number of future steps to predict (for API response)
FEATURES = 2             # Latitude, Longitude
MODEL_PATH = 'iceberg_model.h5'
UPDATE_INTERVAL_SEC = 30 # Seconds between API calls (should match frontend setInterval)

# Simulation boundaries (Antarctic region focus)
LAT_MIN, LAT_MAX = -78.0, -40.0
LON_MIN, LON_MAX = -80.0, 20.0

# Drift simulation parameters
BASE_CURRENT_SPEED = 0.00005  # Degrees per update (~0.3 knots)
NOISE_LEVEL = 0.00002         # Random walk component
NUM_ICEBERGS = 20             # Number of icebergs to simulate

app = Flask(__name__)

# Global state to maintain iceberg histories
# Format: {iceberg_id: {'history': np.ndarray of shape (SEQ_LENGTH, 2), 'current_pos': [lat, lon]}}
iceberg_states = {}

class IcebergLSTM(nn.Module):
    """LSTM model for iceberg position prediction."""
    def __init__(self, input_size=FEATURES, hidden_size=64, num_layers=1, output_size=FEATURES):
        super(IcebergLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        # Take output from last time step
        out = out[:, -1, :]

        # Apply dropout and fully connected layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class IcebergDataset(Dataset):
    """Dataset for iceberg trajectory sequences."""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

def build_model():
    """Build and initialize the LSTM model for position prediction."""
    model = IcebergLSTM()
    return model

def generate_synthetic_trajectory(length=SEQ_LENGTH + PRED_STEPS + 5):
    """
    Generate a synthetic iceberg trajectory for training data.
    Returns: array of shape (length, 2) with [lat, lon] points.
    """
    # Start at random position within bounds
    lat = random.uniform(LAT_MIN, LAT_MAX)
    lon = random.uniform(LON_MIN, LON_MAX)

    trajectory = []

    for _ in range(length):
        trajectory.append([lat, lon])

        # Apply base current (simulating Antarctic Circumpolar Current)
        lat += BASE_CURRENT_SPEED * random.uniform(-0.5, 1.5)  # Bias towards east/north-east
        lon += BASE_CURRENT_SPEED * random.uniform(-0.5, 0.5)

        # Add random walk (turbulence, eddies)
        lat += random.uniform(-NOISE_LEVEL, NOISE_LEVEL)
        lon += random.uniform(-NOISE_LEVEL, NOISE_LEVEL)

        # Occasionally change direction significantly (simulate hitting ice edge or eddy)
        if random.random() < 0.02:  # 2% chance per step
            lat += random.uniform(-0.001, 0.001)
            lon += random.uniform(-0.001, 0.001)

        # Keep within simulation bounds
        lat = max(LAT_MIN, min(LAT_MAX, lat))
        lon = max(LON_MIN, min(LON_MAX, lon))

    return np.array(trajectory)

def create_sequences(trajectory):
    """
    Convert a trajectory into training sequences.
    Each sequence: (past SEQ_LENGTH points) -> (next 1 point)
    Returns: X (samples, SEQ_LENGTH, 2), y (samples, 2)
    """
    X, y = [], []
    for i in range(len(trajectory) - SEQ_LENGTH):
        X.append(trajectory[i:i+SEQ_LENGTH])
        y.append(trajectory[i+SEQ_LENGTH])
    return np.array(X), np.array(y)

def normalize_data(data, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX):
    """
    Normalize latitude and longitude to [0, 1] range.
    Handles both 2D (samples, 2) and 3D (samples, seq_length, 2) arrays.
    """
    data_norm = data.copy()
    if data_norm.ndim == 3:
        # 3D case: (samples, seq_length, 2)
        data_norm[:, :, 0] = (data_norm[:, :, 0] - lat_min) / (lat_max - lat_min)
        data_norm[:, :, 1] = (data_norm[:, :, 1] - lon_min) / (lon_max - lon_min)
    elif data_norm.ndim == 2:
        # 2D case: (samples, 2)
        data_norm[:, 0] = (data_norm[:, 0] - lat_min) / (lat_max - lat_min)
        data_norm[:, 1] = (data_norm[:, 1] - lon_min) / (lon_max - lon_min)
    else:
        raise ValueError(f"Unsupported data shape: {data_norm.shape}")
    return data_norm

def denormalize_data(data_norm, lat_min=LAT_MIN, lat_max=LAT_MAX, lon_min=LON_MIN, lon_max=LON_MAX):
    """
    Denormalize from [0, 1] back to actual lat/lon.
    Handles both 2D (samples, 2) and 3D (samples, seq_length, 2) arrays.
    """
    data = data_norm.copy()
    if data.ndim == 3:
        # 3D case: (samples, seq_length, 2)
        data[:, :, 0] = data_norm[:, :, 0] * (lat_max - lat_min) + lat_min
        data[:, :, 1] = data_norm[:, :, 1] * (lon_max - lon_min) + lon_min
    elif data.ndim == 2:
        # 2D case: (samples, 2)
        data[:, 0] = data_norm[:, 0] * (lat_max - lat_min) + lat_min
        data[:, 1] = data_norm[:, 1] * (lon_max - lon_min) + lon_min
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    return data

def load_or_train_model():
    """Load existing model or train a new one using synthetic data."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = IcebergLSTM()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        return model
    else:
        print("Generating training data and training new model...")

        # Generate synthetic trajectories
        all_trajectories = []
        for _ in range(100):  # Generate 100 different trajectories
            traj = generate_synthetic_trajectory(SEQ_LENGTH + PRED_STEPS + 50)
            all_trajectories.append(traj)

        # Create training sequences
        X_list, y_list = [], []
        for traj in all_trajectories:
            X_traj, y_traj = create_sequences(traj)
            X_list.append(X_traj)
            y_list.append(y_traj)

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        print(f"Generated {X.shape[0]} training samples")

        # Normalize data
        X_norm = normalize_data(X)
        y_norm = normalize_data(y)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_norm)
        y_tensor = torch.FloatTensor(y_norm)

        # Create dataset and dataloader
        dataset = IcebergDataset(X_norm, y_norm)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Build and train model
        model = build_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Model architecture:")
        print(model)

        # Train with validation split
        model.train()
        for epoch in range(15):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/15], Loss: {avg_loss:.6f}')

        # Save model
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        return model

def predict_next_position(model, history):
    """
    Predict the next position given a history of positions.

    Args:
        model: Trained PyTorch model
        history: numpy array of shape (SEQ_LENGTH, 2) with [lat, lon] points
                 (oldest to newest)

    Returns:
        (pred_lat, pred_lon): Predicted next position
    """
    # Normalize history
    history_norm = normalize_data(history.reshape(1, SEQ_LENGTH, FEATURES))[0]

    # Convert to tensor and add batch dimension
    history_tensor = torch.FloatTensor(history_norm).unsqueeze(0)

    # Predict
    model.eval()
    with torch.no_grad():
        pred_norm = model(history_tensor).squeeze().numpy()

    # Denormalize prediction
    pred_lat, pred_lon = denormalize_data(pred_norm.reshape(1, 2))[0]

    return float(pred_lat), float(pred_lon)

@app.route('/api/icebergs')
def get_icebergs():
    """
    API endpoint that returns current iceberg positions and predictions.
    Called by frontend every UPDATE_INTERVAL_SEC seconds.
    """
    global iceberg_states, model

    icebergs_data = []

    for iceberg_id in range(NUM_ICEBERGS):
        # Initialize iceberg if not present
        if iceberg_id not in iceberg_states:
            lat = random.uniform(LAT_MIN, LAT_MAX)
            lon = random.uniform(LON_MIN, LON_MAX)
            # Initialize history with current position repeated
            history = np.tile([[lat, lon]], (SEQ_LENGTH, 1))
            iceberg_states[iceberg_id] = {
                'history': history,
                'current_pos': [lat, lon]
            }

        state = iceberg_states[iceberg_id]
        history = state['history']
        current_lat, current_lon = state['current_pos']

        # Predict next position (step 1)
        pred1_lat, pred1_lon = predict_next_position(model, history)

        # For steps 2 and 3, we need to iteratively predict
        # Create temporary history for step 2 prediction
        history_step2 = np.vstack([history[1:], [[pred1_lat, pred1_lon]]])
        pred2_lat, pred2_lon = predict_next_position(model, history_step2)

        # Create temporary history for step 3 prediction
        history_step3 = np.vstack([history_step2[1:], [[pred2_lat, pred2_lon]]])
        pred3_lat, pred3_lon = predict_next_position(model, history_step3)

        # Update iceberg state for next call:
        # Shift history left and append current position
        new_history = np.vstack([history[1:], [[current_lat, current_lon]]])
        state['history'] = new_history
        # Set current position to the predicted position (step 1)
        state['current_pos'] = [pred1_lat, pred1_lon]

        # Format response for this iceberg
        icebergs_data.append({
            'iceberg_id': iceberg_id,
            'current_position': {
                'latitude': round(current_lat, 4),
                'longitude': round(current_lon, 4)
            },
            'predictions': [
                {'step': 1, 'latitude': round(pred1_lat, 4), 'longitude': round(pred1_lon, 4)},
                {'step': 2, 'latitude': round(pred2_lat, 4), 'longitude': round(pred2_lon, 4)},
                {'step': 3, 'latitude': round(pred3_lat, 4), 'longitude': round(pred3_lon, 4)}
            ]
        })

    return jsonify({'icebergs': icebergs_data})

if __name__ == '__main__':
    # Load or train the model
    model = load_or_train_model()

    # Start the Flask server
    print(f"Starting iceberg forecast API on port 5000...")
    print(f"Endpoint: http://localhost:5000/api/icebergs")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)