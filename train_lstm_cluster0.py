# -*- coding: utf-8 -*-
"""
LSTM Training for Time Series Prediction - Cluster 0 Data (PyTorch)

This script trains LSTM (Long Short-Term Memory) networks to predict the next step in chemical processes
using the 54 files from cluster_0. Each file contains 13 process variables with 1000 time steps.

Objectives:
1. Predict the next step (t+1) given current step (t)
2. Predict the 501st step from previous steps
3. Compare different LSTM architectures
4. Perform hyperparameter tuning
5. Validate using train/test/validation splits

Features:
- Multiple LSTM architectures (Simple LSTM, Bidirectional LSTM, Stacked LSTM)
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
from sklearn.preprocessing import MinMaxScaler
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

class SimpleLSTM(nn.Module):
    """Simple LSTM for time series prediction"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(SimpleLSTM, self).__init__()
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

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for time series prediction"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
                           bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output from both directions
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

class StackedLSTM(nn.Module):
    """Stacked LSTM with additional dense layers"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(StackedLSTM, self).__init__()
        
        # Create stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(prev_size, hidden_size, batch_first=True, dropout=dropout_rate)
            )
            prev_size = hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # Pass through stacked LSTM layers
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
        
        # Take the last output
        out = self.dropout(x[:, -1, :])
        return self.dense_layers(out)

class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Calculate attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.dropout(attended_output)
        return self.fc(out)

class TimeSeriesLSTMPredictor:
    def __init__(self, data_dir="cluster_0", sequence_length=10):
        """
        Initialize the TimeSeriesLSTMPredictor
        
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
        """Create a PyTorch LSTM model based on type"""
        input_size = input_shape[1]  # features
        
        if model_type == 'simple_lstm':
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return SimpleLSTM(input_size, hidden_size, num_layers, output_shape, dropout_rate)
        
        elif model_type == 'bidirectional_lstm':
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return BidirectionalLSTM(input_size, hidden_size, num_layers, output_shape, dropout_rate)
        
        elif model_type == 'stacked_lstm':
            hidden_sizes = kwargs.get('hidden_sizes', [128, 64, 32])
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return StackedLSTM(input_size, hidden_sizes, output_shape, dropout_rate)
        
        elif model_type == 'attention_lstm':
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout_rate = kwargs.get('dropout_rate', 0.3)
            return AttentionLSTM(input_size, hidden_size, num_layers, output_shape, dropout_rate)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_model(self, model, train_loader, val_loader, model_name, **kwargs):
        """Train a PyTorch LSTM model"""
        print(f"\nTraining {model_name}...")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        
        optimizer_name = kwargs.get('optimizer', 'adam')
        learning_rate = kwargs.get('learning_rate', 0.001)
        weight_decay = kwargs.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        epochs = kwargs.get('epochs', 100)
        patience = kwargs.get('patience', 20)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
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
            
            # Update learning rate
            scheduler.step(val_loss)
            
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
        axes[0, 1].set_title('LSTM Model Performance Comparison')
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
        plt.savefig('lstm_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, target_steps=1):
        """Run complete analysis with all LSTM models and hyperparameter tuning"""
        print("="*60)
        print("LSTM TIME SERIES PREDICTION ANALYSIS")
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
        
        # Define LSTM models to test
        model_configs = {
            'Simple_LSTM': {
                'type': 'simple_lstm', 
                'params': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout_rate': 0.3,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'optimizer': 'adam'
                }
            },
            'Bidirectional_LSTM': {
                'type': 'bidirectional_lstm', 
                'params': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout_rate': 0.3,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'optimizer': 'adam'
                }
            },
            'Stacked_LSTM': {
                'type': 'stacked_lstm', 
                'params': {
                    'hidden_sizes': [128, 64, 32],
                    'dropout_rate': 0.3,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'optimizer': 'adam'
                }
            },
            'Attention_LSTM': {
                'type': 'attention_lstm', 
                'params': {
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout_rate': 0.3,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'optimizer': 'adam'
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
        print(f"BEST LSTM MODEL: {best_model_name}")
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
        comparison_df.to_csv('lstm_model_comparison_results.csv', index=False)
        print("LSTM model comparison saved to: lstm_model_comparison_results.csv")
        
        # Save detailed results
        with open('lstm_training_summary.txt', 'w') as f:
            f.write("LSTM TIME SERIES PREDICTION RESULTS\n")
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
        
        print("LSTM training summary saved to: lstm_training_summary.txt")

def main():
    """Main function to run the complete LSTM analysis"""
    # Initialize predictor
    predictor = TimeSeriesLSTMPredictor(data_dir="cluster_0", sequence_length=10)
    
    # Run complete analysis
    best_model, results = predictor.run_complete_analysis(target_steps=1)
    
    print("\nLSTM Analysis complete! Check the generated files:")
    print("- lstm_training_results.png (visualization)")
    print("- lstm_model_comparison_results.csv (comparison table)")
    print("- lstm_training_summary.txt (detailed results)")
    print("- models/ (saved model files)")

if __name__ == "__main__":
    main() 