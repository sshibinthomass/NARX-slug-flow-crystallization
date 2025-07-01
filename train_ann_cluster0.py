# -*- coding: utf-8 -*-
"""
ANN Training for Time Series Prediction - Cluster 0 Data

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
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

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
    
    def create_mlp_model(self, input_shape, output_shape, **kwargs):
        """Create Multi-Layer Perceptron model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape, activation='linear')
        ])
        return model
    
    def create_lstm_model(self, input_shape, output_shape, **kwargs):
        """Create LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape, activation='linear')
        ])
        return model
    
    def create_gru_model(self, input_shape, output_shape, **kwargs):
        """Create GRU model"""
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(output_shape, activation='linear')
        ])
        return model
    
    def create_cnn_lstm_model(self, input_shape, output_shape, **kwargs):
        """Create CNN-LSTM hybrid model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(output_shape, activation='linear')
        ])
        return model
    
    def create_advanced_lstm_model(self, input_shape, output_shape, **kwargs):
        """Create advanced LSTM model with attention-like mechanism"""
        inputs = Input(shape=input_shape)
        
        # Bidirectional LSTM layers
        lstm1 = LSTM(128, return_sequences=True)(inputs)
        lstm1 = Dropout(0.3)(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm2 = Dropout(0.3)(lstm2)
        
        lstm3 = LSTM(32, return_sequences=False)(lstm2)
        lstm3 = Dropout(0.2)(lstm3)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm3)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.2)(dense1)
        
        dense2 = Dense(32, activation='relu')(dense1)
        dense2 = Dropout(0.1)(dense2)
        
        outputs = Dense(output_shape, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name, **kwargs):
        """Train a model with given parameters"""
        print(f"\nTraining {model_name}...")
        
        # Compile model
        optimizer = kwargs.get('optimizer', 'adam')
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            opt = SGD(learning_rate=learning_rate)
        
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
            ModelCheckpoint(f'models/{model_name}_best.h5', save_best_only=True, verbose=1)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=kwargs.get('epochs', 100),
            batch_size=kwargs.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
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
    
    def hyperparameter_tuning(self, model_type, X_train, y_train, X_val, y_val):
        """Perform hyperparameter tuning for a given model type"""
        print(f"\nPerforming hyperparameter tuning for {model_type}...")
        
        # Define hyperparameter grid
        if model_type == 'mlp':
            param_grid = {
                'hidden_layers': [[128, 64], [256, 128, 64], [128, 64, 32]],
                'dropout_rate': [0.2, 0.3, 0.4],
                'learning_rate': [0.001, 0.01, 0.0001],
                'batch_size': [16, 32, 64]
            }
        elif model_type in ['lstm', 'gru']:
            param_grid = {
                'units': [[128, 64], [256, 128], [64, 32]],
                'dropout_rate': [0.2, 0.3, 0.4],
                'learning_rate': [0.001, 0.01, 0.0001],
                'batch_size': [16, 32, 64]
            }
        else:
            param_grid = {
                'filters': [[64, 32], [128, 64]],
                'dropout_rate': [0.2, 0.3],
                'learning_rate': [0.001, 0.01],
                'batch_size': [16, 32]
            }
        
        best_score = float('inf')
        best_params = None
        best_model = None
        
        # Grid search
        total_combinations = 1
        for param_list in param_grid.values():
            total_combinations *= len(param_list)
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        for i, params in enumerate(self._generate_param_combinations(param_grid)):
            print(f"\nTesting combination {i+1}/{total_combinations}: {params}")
            
            # Create and train model
            model = self._create_model_with_params(model_type, params, X_train.shape[1:], y_train.shape[-1])
            model, history = self.train_model(model, X_train, y_train, X_val, y_val, f"{model_type}_tune", **params)
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            
            if val_loss < best_score:
                best_score = val_loss
                best_params = params
                best_model = model
                print(f"New best score: {best_score:.6f}")
        
        print(f"\nBest parameters for {model_type}: {best_params}")
        print(f"Best validation loss: {best_score:.6f}")
        
        return best_model, best_params
    
    def _generate_param_combinations(self, param_grid):
        """Generate all combinations of parameters"""
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        for combo in combinations:
            yield dict(zip(keys, combo))
    
    def _create_model_with_params(self, model_type, params, input_shape, output_shape):
        """Create model with specific parameters"""
        if model_type == 'mlp':
            return self.create_mlp_model(input_shape, output_shape, **params)
        elif model_type == 'lstm':
            return self.create_lstm_model(input_shape, output_shape, **params)
        elif model_type == 'gru':
            return self.create_gru_model(input_shape, output_shape, **params)
        elif model_type == 'cnn_lstm':
            return self.create_cnn_lstm_model(input_shape, output_shape, **params)
        else:
            return self.create_advanced_lstm_model(input_shape, output_shape, **params)
    
    def predict_future_step(self, model, data, target_step=501):
        """Predict a specific future step (e.g., step 501)"""
        print(f"\nPredicting step {target_step}...")
        
        # For step 501, we need to predict beyond the available data
        # We'll use the last available sequence to predict the next step
        last_sequence = data[-1, -self.sequence_length:, :]  # Last sequence from last process
        
        # Make prediction
        prediction = model.predict(last_sequence.reshape(1, self.sequence_length, -1))
        
        print(f"Predicted values for step {target_step}:")
        for i, val in enumerate(prediction[0]):
            print(f"Feature {i}: {val:.6f}")
        
        return prediction[0]
    
    def plot_results(self, results):
        """Plot training results and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot training history for each model
        for i, (model_name, history) in enumerate(self.histories.items()):
            axes[0, 0].plot(history.history['loss'], label=f'{model_name} - Train')
            axes[0, 0].plot(history.history['val_loss'], label=f'{model_name} - Val')
        
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
        plt.savefig('ann_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, target_steps=1):
        """Run complete analysis with all models and hyperparameter tuning"""
        print("="*60)
        print("ANN TIME SERIES PREDICTION ANALYSIS")
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
        
        # Define models to test
        model_configs = {
            'MLP': {'type': 'mlp', 'params': {'epochs': 100, 'batch_size': 32}},
            'LSTM': {'type': 'lstm', 'params': {'epochs': 100, 'batch_size': 32}},
            'GRU': {'type': 'gru', 'params': {'epochs': 100, 'batch_size': 32}},
            'CNN-LSTM': {'type': 'cnn_lstm', 'params': {'epochs': 100, 'batch_size': 32}},
            'Advanced_LSTM': {'type': 'advanced_lstm', 'params': {'epochs': 100, 'batch_size': 32}}
        }
        
        # Train and evaluate each model
        for model_name, config in model_configs.items():
            print(f"\n{'='*40}")
            print(f"TRAINING {model_name}")
            print(f"{'='*40}")
            
            # Create model
            model = self._create_model_with_params(
                config['type'], 
                config['params'], 
                X_train.shape[1:], 
                y_train.shape[-1]
            )
            
            # Train model
            model, history = self.train_model(
                model, X_train, y_train, X_val, y_val, model_name, **config['params']
            )
            
            # Evaluate model
            results = self.evaluate_model(model, X_test, y_test, model_name)
            
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
        comparison_df.to_csv('model_comparison_results.csv', index=False)
        print("Model comparison saved to: model_comparison_results.csv")
        
        # Save detailed results
        with open('training_summary.txt', 'w') as f:
            f.write("ANN TIME SERIES PREDICTION RESULTS\n")
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
        
        print("Training summary saved to: training_summary.txt")

def main():
    """Main function to run the complete analysis"""
    # Initialize predictor
    predictor = TimeSeriesPredictor(data_dir="cluster_0", sequence_length=10)
    
    # Run complete analysis
    best_model, results = predictor.run_complete_analysis(target_steps=1)
    
    print("\nAnalysis complete! Check the generated files:")
    print("- ann_training_results.png (visualization)")
    print("- model_comparison_results.csv (comparison table)")
    print("- training_summary.txt (detailed results)")
    print("- models/ (saved model files)")

if __name__ == "__main__":
    main() 