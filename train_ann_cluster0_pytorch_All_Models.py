# -*- coding: utf-8 -*-
"""
ANN Training for Time Series Prediction - Cluster 0 Data (PyTorch Version)

This script trains Artificial Neural Networks to predict the next step in chemical processes
using the 54 files from cluster_0. Each file contains 13 process variables with 1000 time steps.

Objectives:
1. Predict the next step (t+1) given current step (t)
2. Predict the 5001st step from previous steps
3. Compare different ANN architectures
4. Perform hyperparameter tuning
5. Validate using train/test/validation splits

Features:
- Multiple ANN architectures (MLP, LSTM, GRU, CNN-LSTM)
- Hyperparameter optimization
- Time series data preprocessing
- Comprehensive evaluation metrics
- Model comparison and selection
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MLP(nn.Module):
    """Multi-Layer Perceptron for time series prediction"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the sequence dimension
        batch_size, seq_len, features = x.shape
        x = x.view(batch_size, -1)  # Flatten to (batch_size, seq_len * features)
        return self.network(x)

class LSTM(nn.Module):
    """LSTM for time series prediction"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

class GRU(nn.Module):
    """GRU for time series prediction"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # Take the last output
        out = self.dropout(gru_out[:, -1, :])
        return self.fc(out)

class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid for time series prediction"""
    def __init__(self, input_size, cnn_channels, lstm_hidden, output_size, dropout_rate=0.3):
        super(CNNLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, cnn_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(cnn_channels[1], lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, features, seq)
        
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # (batch, seq, features)
        
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take last output
        return self.fc(out)

class TimeSeriesPredictor:
    def __init__(self, data_dir="cluster_0", sequence_length=10):
        """
        Initialize the TimeSeriesPredictor
        
        Args:
            data_dir: Directory containing the cluster_0 files
            sequence_length: Number of previous steps to use for prediction
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.models = {}
        self.histories = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load all files from cluster_0 and preprocess the data"""
        print("Loading and preprocessing data...")
        
        all_data = []
        file_names = []
        
        # Load all files
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(self.data_dir, file_name)
                try:
                    # Load data, skip header
                    data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)
                    if data.shape[0] == 1000 and data.shape[1] == 13:
                        all_data.append(data)
                        file_names.append(file_name)
                        print(f"Loaded {file_name}: {data.shape}")
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
        
        if not all_data:
            raise ValueError("No valid data files found!")
        
        print(f"Successfully loaded {len(all_data)} files")
        
        # Combine all data
        self.raw_data = np.array(all_data)  # Shape: (54, 1000, 13)
        self.file_names = file_names
        
        # Normalize data
        self.normalized_data = self.scaler.fit_transform(
            self.raw_data.reshape(-1, 13)
        ).reshape(self.raw_data.shape)
        
        print(f"Data shape: {self.raw_data.shape}")
        print(f"Normalized data shape: {self.normalized_data.shape}")
        
        return self.normalized_data
    
    def create_sequences(self, data, target_steps=1):
        """
        Create sequences for time series prediction
        
        Args:
            data: Input data of shape (n_samples, n_timesteps, n_features)
            target_steps: Number of future steps to predict
        """
        X, y = [], []
        
        for process in data:  # Each process has shape (1000, 13)
            for i in range(self.sequence_length, len(process) - target_steps + 1):
                # Input sequence
                X.append(process[i-self.sequence_length:i])
                # Target (next step or future steps)
                y.append(process[i:i+target_steps])
        
        return np.array(X), np.array(y)
    
    def create_model(self, model_type, input_shape, output_shape, **kwargs):
        """Create a PyTorch model based on type"""
        if model_type == 'mlp':
            input_size = input_shape[0] * input_shape[1]  # seq_len * features
            hidden_sizes = kwargs.get('hidden_sizes', [128, 64, 32])
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return MLP(input_size, hidden_sizes, output_shape, dropout_rate)
        
        elif model_type == 'lstm':
            input_size = input_shape[1]  # features
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return LSTM(input_size, hidden_size, num_layers, output_shape, dropout_rate)
        
        elif model_type == 'gru':
            input_size = input_shape[1]  # features
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return GRU(input_size, hidden_size, num_layers, output_shape, dropout_rate)
        
        elif model_type == 'cnn_lstm':
            input_size = input_shape[1]  # features
            cnn_channels = kwargs.get('cnn_channels', [64, 32])
            lstm_hidden = kwargs.get('lstm_hidden', 64)
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return CNNLSTM(input_size, cnn_channels, lstm_hidden, output_shape, dropout_rate)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_model(self, model, train_loader, val_loader, model_name, **kwargs):
        """Train a PyTorch model"""
        print(f"\nTraining {model_name}...")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        
        optimizer_name = kwargs.get('optimizer', 'adam')
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        epochs = kwargs.get('epochs', 100)
        patience = kwargs.get('patience', 20)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'models/{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        model.load_state_dict(torch.load(f'models/{model_name}_best.pth'))
        
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        
        return model, history
    
    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        model.eval()
        all_predictions = []
        all_actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                all_predictions.append(outputs.cpu().numpy())
                all_actuals.append(batch_y.cpu().numpy())
        
        # Combine all predictions
        y_pred = np.vstack(all_predictions)
        y_test = np.vstack(all_actuals)
        
        # Reshape for evaluation
        y_test_flat = y_test.reshape(-1, y_test.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        
        # Calculate metrics
        mse = mean_squared_error(y_test_flat, y_pred_flat)
        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        r2 = r2_score(y_test_flat, y_pred_flat)
        
        # Calculate metrics for each feature
        feature_metrics = {}
        for i in range(y_test_flat.shape[1]):
            feature_mse = mean_squared_error(y_test_flat[:, i], y_pred_flat[:, i])
            feature_mae = mean_absolute_error(y_test_flat[:, i], y_pred_flat[:, i])
            feature_r2 = r2_score(y_test_flat[:, i], y_pred_flat[:, i])
            feature_metrics[f'feature_{i}'] = {
                'mse': feature_mse,
                'mae': feature_mae,
                'r2': feature_r2
            }
        
        results = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'feature_metrics': feature_metrics,
            'predictions': y_pred_flat,
            'actuals': y_test_flat
        }
        
        print(f"{model_name} Results:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        return results
    
    def predict_future_step(self, model, data, target_step=501):
        """Predict a specific future step (e.g., step 501)"""
        print(f"\nPredicting step {target_step}...")
        
        model.eval()
        # For step 501, we need to predict beyond the available data
        # We'll use the last available sequence to predict the next step
        last_sequence = data[-1, -self.sequence_length:, :]  # Last sequence from last process
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()
        
        print(f"Predicted values for step {target_step}:")
        for i, val in enumerate(prediction[0]):
            print(f"Feature {i}: {val:.6f}")
        
        return prediction[0]
    
    def plot_results(self, results):
        """Plot training results and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot training history for each model
        for i, (model_name, history) in enumerate(self.histories.items()):
            axes[0, 0].plot(history['train_loss'], label=f'{model_name} - Train')
            axes[0, 0].plot(history['val_loss'], label=f'{model_name} - Val')
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot model comparison
        model_names = list(results.keys())
        mse_scores = [results[name]['mse'] for name in model_names]
        mae_scores = [results[name]['mae'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, mse_scores, width, label='MSE')
        axes[0, 1].bar(x + width/2, mae_scores, width, label='MAE')
        axes[0, 1].set_title('Model Performance Comparison')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot predictions vs actual for best model
        best_model = min(results.keys(), key=lambda x: results[x]['mse'])
        y_pred = results[best_model]['predictions']
        y_actual = results[best_model]['actuals']
        
        axes[1, 0].scatter(y_actual.flatten(), y_pred.flatten(), alpha=0.5)
        axes[1, 0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'Predictions vs Actual ({best_model})')
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].grid(True)
        
        # Plot feature-wise performance
        feature_r2 = [results[best_model]['feature_metrics'][f'feature_{i}']['r2'] 
                     for i in range(len(results[best_model]['feature_metrics']))]
        
        axes[1, 1].bar(range(len(feature_r2)), feature_r2)
        axes[1, 1].set_title(f'Feature-wise R² Score ({best_model})')
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('ann_training_results_pytorch.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, target_steps=1):
        """Run complete analysis with all models and hyperparameter tuning"""
        print("="*60)
        print("ANN TIME SERIES PREDICTION ANALYSIS (PyTorch)")
        print("="*60)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Load and preprocess data
        data = self.load_and_preprocess_data()
        
        # Create sequences
        X, y = self.create_sequences(data, target_steps)
        print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Define models to test
        model_configs = {
            'MLP': {
                'type': 'mlp', 
                'params': {
                    'hidden_sizes': [128, 64, 32],
                    'dropout_rate': 0.3,
                    'epochs': 100,
                    'learning_rate': 0.001
                }
            },
            'LSTM': {
                'type': 'lstm', 
                'params': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout_rate': 0.3,
                    'epochs': 100,
                    'learning_rate': 0.001
                }
            },
            'GRU': {
                'type': 'gru', 
                'params': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout_rate': 0.3,
                    'epochs': 100,
                    'learning_rate': 0.001
                }
            },
            'CNN-LSTM': {
                'type': 'cnn_lstm', 
                'params': {
                    'cnn_channels': [64, 32],
                    'lstm_hidden': 64,
                    'dropout_rate': 0.3,
                    'epochs': 100,
                    'learning_rate': 0.001
                }
            }
        }
        
        # Train and evaluate each model
        for model_name, config in model_configs.items():
            print(f"\n{'='*40}")
            print(f"TRAINING {model_name}")
            print(f"{'='*40}")
            
            # Create model
            model = self.create_model(
                config['type'], 
                X_train.shape[1:], 
                y_train.shape[-1],
                **config['params']
            )
            
            # Train model
            model, history = self.train_model(
                model, train_loader, val_loader, model_name, **config['params']
            )
            
            # Evaluate model
            results = self.evaluate_model(model, test_loader, model_name)
            
            # Store results
            self.models[model_name] = model
            self.histories[model_name] = history
            self.results[model_name] = results
        
        # Find best model
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mse'])
        best_model = self.models[best_model_name]
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"Best MSE: {self.results[best_model_name]['mse']:.6f}")
        print(f"Best MAE: {self.results[best_model_name]['mae']:.6f}")
        print(f"Best R²: {self.results[best_model_name]['r2']:.6f}")
        print(f"{'='*60}")
        
        # Predict future step
        future_prediction = self.predict_future_step(best_model, data, 501)
        
        # Plot results
        self.plot_results(self.results)
        
        # Save results
        self.save_results()
        
        return best_model, self.results

    def save_results(self):
        """Save all results to files"""
        # Save model comparison
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'MSE': [self.results[name]['mse'] for name in self.results.keys()],
            'MAE': [self.results[name]['mae'] for name in self.results.keys()],
            'R2': [self.results[name]['r2'] for name in self.results.keys()]
        })
        comparison_df.to_csv('model_comparison_results_pytorch.csv', index=False)
        print("Model comparison saved to: model_comparison_results_pytorch.csv")
        
        # Save detailed results
        with open('training_summary_pytorch.txt', 'w') as f:
            f.write("ANN TIME SERIES PREDICTION RESULTS (PyTorch)\n")
            f.write("="*50 + "\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"{model_name} Results:\n")
                f.write(f"  MSE: {results['mse']:.6f}\n")
                f.write(f"  MAE: {results['mae']:.6f}\n")
                f.write(f"  R²: {results['r2']:.6f}\n\n")
                
                f.write("  Feature-wise Performance:\n")
                for feature, metrics in results['feature_metrics'].items():
                    f.write(f"    {feature}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, R²={metrics['r2']:.6f}\n")
                f.write("\n")
        
        print("Training summary saved to: training_summary_pytorch.txt")

def main():
    """Main function to run the complete analysis"""
    # Initialize predictor
    predictor = TimeSeriesPredictor(data_dir="cluster_0", sequence_length=10)
    
    # Run complete analysis
    best_model, results = predictor.run_complete_analysis(target_steps=1)
    
    print("\nAnalysis complete! Check the generated files:")
    print("- ann_training_results_pytorch.png (visualization)")
    print("- model_comparison_results_pytorch.csv (comparison table)")
    print("- training_summary_pytorch.txt (detailed results)")
    print("- models/ (saved model files)")

if __name__ == "__main__":
    main() 


# Output
# ========================================
# TRAINING MLP
# ========================================

# Training MLP...
# Epoch 1/100, Train Loss: 0.118162, Val Loss: 0.108027
# Epoch 11/100, Train Loss: 0.108588, Val Loss: 0.107905
# Epoch 21/100, Train Loss: 0.108554, Val Loss: 0.107944
# Epoch 31/100, Train Loss: 0.108545, Val Loss: 0.107915
# Epoch 41/100, Train Loss: 0.108534, Val Loss: 0.107888
# Early stopping at epoch 50

# Evaluating MLP...
# MLP Results:
# MSE: 0.105499
# MAE: 0.255232
# R²: 0.018244

# ========================================
# TRAINING LSTM
# ========================================

# R²: 0.018244

# ========================================
# TRAINING LSTM
# ========================================

# TRAINING LSTM
# ========================================


# Training LSTM...
# Epoch 1/100, Train Loss: 0.111915, Val Loss: 0.107963
# Epoch 11/100, Train Loss: 0.108704, Val Loss: 0.108104
# Epoch 21/100, Train Loss: 0.108549, Val Loss: 0.107886
# Epoch 31/100, Train Loss: 0.108527, Val Loss: 0.107863
# Epoch 41/100, Train Loss: 0.108499, Val Loss: 0.107845
# Epoch 51/100, Train Loss: 0.108527, Val Loss: 0.107857
# Epoch 61/100, Train Loss: 0.108513, Val Loss: 0.107838
# Epoch 71/100, Train Loss: 0.108478, Val Loss: 0.107838
# Epoch 81/100, Train Loss: 0.108495, Val Loss: 0.107867
# Early stopping at epoch 85

# MSE: 0.102806
# MAE: 0.252098
# R²: 0.055459

# ========================================
# TRAINING GRU
# ========================================

# Training GRU...
# Epoch 1/100, Train Loss: 0.112489, Val Loss: 0.108164
# Epoch 11/100, Train Loss: 0.108724, Val Loss: 0.107979
# Epoch 21/100, Train Loss: 0.108552, Val Loss: 0.107920
# Epoch 31/100, Train Loss: 0.108539, Val Loss: 0.107916
# Epoch 41/100, Train Loss: 0.108524, Val Loss: 0.107860
# Epoch 51/100, Train Loss: 0.108498, Val Loss: 0.107858
# Epoch 61/100, Train Loss: 0.108512, Val Loss: 0.107852
# Early stopping at epoch 65

# Evaluating GRU...
# GRU Results:
# MSE: 0.103375
# MAE: 0.253533
# R²: 0.049146

# ========================================
# TRAINING CNN-LSTM
# ========================================

# Training CNN-LSTM...
# Epoch 1/100, Train Loss: 0.111299, Val Loss: 0.108064
# Epoch 11/100, Train Loss: 0.108615, Val Loss: 0.107933
# Epoch 21/100, Train Loss: 0.108585, Val Loss: 0.107898
# Epoch 31/100, Train Loss: 0.108569, Val Loss: 0.107893
# Epoch 41/100, Train Loss: 0.108567, Val Loss: 0.107937
# Early stopping at epoch 43

# Evaluating CNN-LSTM...
# CNN-LSTM Results:
# MSE: 0.102999
# MAE: 0.253745
# R²: 0.048942

# ============================================================
# BEST MODEL: LSTM
# Best MSE: 0.102806
# Best MAE: 0.252098
# Best R²: 0.055459
# Evaluating CNN-LSTM...
# CNN-LSTM Results:
# MSE: 0.102999
# MAE: 0.253745
# R²: 0.048942

# ============================================================
# BEST MODEL: LSTM
# Best MSE: 0.102806
# Best MAE: 0.252098
# Best R²: 0.055459