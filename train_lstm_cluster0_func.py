# -*- coding: utf-8 -*-
"""
LSTM Training for Time Series Prediction - Cluster 0 Data (PyTorch, Functions Only)

This script trains an LSTM to predict the next step in chemical processes
using the 54 files from cluster_0. Each file contains 13 process variables with 1000 time steps.
No custom classes are used for workflow, only functions (except for the LSTM nn.Module itself).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# LSTM Model Definition (still needs to inherit nn.Module)
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

def load_and_preprocess_data(data_dir, sequence_length):
    all_data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_dir, file_name)
            try:
                data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)
                if data.shape == (1000, 13):
                    all_data.append(data)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    if not all_data:
        raise ValueError("No valid data files found!")
    raw_data = np.array(all_data)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(raw_data.reshape(-1, 13)).reshape(raw_data.shape)
    return normalized_data

def create_sequences(data, sequence_length, target_steps=1):
    X, y = [], []
    for process in data:
        for i in range(sequence_length, len(process) - target_steps + 1):
            X.append(process[i-sequence_length:i])
            y.append(process[i:i+target_steps])
    return np.array(X), np.array(y)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze(1)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_y = batch_y.squeeze(1)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'lstm_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    model.load_state_dict(torch.load('lstm_best.pth'))
    return model, train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    all_predictions, all_actuals = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.squeeze(1)
            outputs = model(batch_X)
            all_predictions.append(outputs.cpu().numpy())
            all_actuals.append(batch_y.cpu().numpy())
    y_pred = np.vstack(all_predictions)
    y_test = np.vstack(all_actuals)
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    r2 = r2_score(y_test_flat, y_pred_flat)
    print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")
    return mse, mae, r2, y_test_flat, y_pred_flat

def predict_future_step(model, data, sequence_length):
    model.eval()
    last_sequence = data[-1, -sequence_length:, :]
    input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()
    print("Predicted values for step 501:")
    for i, val in enumerate(prediction[0]):
        print(f"Feature {i}: {val:.6f}")
    return prediction[0]

def plot_results(train_losses, val_losses, y_test, y_pred):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lstm_func_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    data_dir = 'cluster_0'
    sequence_length = 10
    data = load_and_preprocess_data(data_dir, sequence_length)
    X, y = create_sequences(data, sequence_length, target_steps=1)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)
    model = LSTMNet(input_size=13, hidden_size=128, num_layers=2, output_size=13, dropout_rate=0.3)
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=20)
    mse, mae, r2, y_test_flat, y_pred_flat = evaluate_model(model, test_loader)
    predict_future_step(model, data, sequence_length)
    plot_results(train_losses, val_losses, y_test_flat, y_pred_flat)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv('lstm_func_loss_history.csv', index=False)
    print('Results saved: lstm_func_training_results.png, lstm_func_loss_history.csv')

if __name__ == '__main__':
    main() 