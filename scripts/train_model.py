#!/usr/bin/env python3
"""
Train TCN model on cryptocurrency data.

Usage:
    python scripts/train_model.py --data data/raw/btc_usdt_15m.csv --output models/tcn_v1
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import torch
import numpy as np
from datetime import datetime

from src.data.features import (
    FeatureEngineer,
    compute_triple_barrier_labels,
    create_train_val_test_split
)
from src.models.tcn import TCNModel, CalibratedTCN
from src.training.trainer import Trainer
import pandas as pd


def load_config(config_path: str = 'config/model_config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(data_path: str, config: dict) -> tuple:
    """
    Load data and prepare features and labels.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, timestamps, feature_engineer
    """
    print("="*70)
    print("Loading and preparing data...")
    print("="*70)
    
    # Load raw data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\nLoaded {len(df):,} bars from {data_path}")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Compute features
    print("\nComputing features...")
    engineer = FeatureEngineer()
    df_features = engineer.compute_features(df)
    
    # Compute labels (triple-barrier)
    # Compute labels (simple directional - price up in N bars)
    print("Computing simple directional labels...")
    target_horizon = 4  # 1 hour ahead (4 x 15min bars)

    # Create labels: 1 if price goes up, 0 if down
    future_returns = df_features['close'].pct_change(target_horizon).shift(-target_horizon)
    labels = (future_returns > 0).astype(int).values

    # Remove NaN at the end
    labels = np.nan_to_num(labels, nan=0)

    label_dist = np.mean(labels)
    print(f"  Target horizon: {target_horizon} bars (1 hour)")
    print(f"  Positive labels (up): {label_dist:.1%}")
    print(f"  Negative labels (down): {1-label_dist:.1%}")
    
    # Create train/val/test splits
    print("\nCreating train/val/test splits...")
    training_config = load_config('config/training_config.yaml')
    data_config = training_config.get('data', {})
    
    train_df, val_df, test_df = create_train_val_test_split(
        df_features,
        train_end=data_config.get('train_end', '2023-06-30'),
        val_end=data_config.get('val_end', '2023-10-31'),
        embargo_bars=data_config.get('embargo_bars', 96)
    )
    
    # Prepare sequences
    print("\nPreparing sequences...")
    sequence_length = config['model'].get('sequence_length', 64)
    
    # Don't use create_sequences for labels, create sequences manually
    from sklearn.preprocessing import RobustScaler
    
    # Prepare feature data (fix deprecated fillna)
    feature_data_train = train_df[engineer.feature_names].ffill().bfill().fillna(0).values
    feature_data_val = val_df[engineer.feature_names].ffill().bfill().fillna(0).values
    feature_data_test = test_df[engineer.feature_names].ffill().bfill().fillna(0).values
    
    # Fit scaler on train
    scaler = RobustScaler()
    feature_data_train = scaler.fit_transform(feature_data_train)
    feature_data_val = scaler.transform(feature_data_val)
    feature_data_test = scaler.transform(feature_data_test)
    engineer.scaler = scaler
    
    # Create sequences manually
    def make_sequences(data, seq_len):
        X = []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
        return np.array(X)
    
    X_train = make_sequences(feature_data_train, sequence_length)
    X_val = make_sequences(feature_data_val, sequence_length)
    X_test = make_sequences(feature_data_test, sequence_length)
    
    # Get corresponding triple-barrier labels
    train_start_idx = train_df.index[0] + sequence_length
    val_start_idx = val_df.index[0] + sequence_length
    test_start_idx = test_df.index[0] + sequence_length
    
    y_train = labels[train_start_idx:train_start_idx + len(X_train)]
    y_val = labels[val_start_idx:val_start_idx + len(X_val)]
    y_test = labels[test_start_idx:test_start_idx + len(X_test)]
    
    train_timestamps = train_df['timestamp'].iloc[sequence_length:sequence_length + len(X_train)].values
    val_timestamps = val_df['timestamp'].iloc[sequence_length:sequence_length + len(X_val)].values
    test_timestamps = test_df['timestamp'].iloc[sequence_length:sequence_length + len(X_test)].values
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(X_train):,} sequences")
    print(f"  Val:   {len(X_val):,} sequences")
    print(f"  Test:  {len(X_test):,} sequences")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            (train_timestamps, val_timestamps, test_timestamps), engineer)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    save_dir: str
) -> tuple:
    """
    Train TCN model.
    
    Returns:
        model, trainer, history
    """
    print("\n" + "="*70)
    print("Training TCN model...")
    print("="*70)
    
    # Create model
    model_config = config['model']
    model = TCNModel(
        input_size=model_config.get('input_size', 12),
        sequence_length=model_config.get('sequence_length', 64),
        tcn_channels=model_config.get('tcn_channels', [32, 32, 64, 64]),
        kernel_size=model_config.get('kernel_size', 3),
        dropout=model_config.get('dropout', [0.1, 0.1, 0.1, 0.2]),
        dense_hidden=model_config.get('dense_hidden', 32),
        dense_dropout=model_config.get('dense_dropout', 0.3)
    )
    
    # Load training config
    training_config = load_config('config/training_config.yaml')
    train_params = training_config.get('training', {})
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=train_params.get('device', 'cpu'),
        learning_rate=train_params.get('learning_rate', 0.001),
        batch_size=train_params.get('batch_size', 256),
        epochs=train_params.get('epochs', 100),
        early_stopping_patience=train_params.get('early_stopping_patience', 10),
        label_smoothing=train_params.get('label_smoothing', 0.05),
        save_dir=save_dir
    )
    
    # Train
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    return model, trainer, history


def calibrate_model(
    model: torch.nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict
) -> CalibratedTCN:
    """
    Calibrate model probabilities.
    
    Returns:
        calibrated_model
    """
    print("\n" + "="*70)
    print("Calibrating model...")
    print("="*70)
    
    calibration_config = config.get('calibration', {})
    method = calibration_config.get('method', 'platt')
    
    calibrated_model = CalibratedTCN(
        model=model,
        calibration_method=method,
        device='cpu'
    )
    
    calibrated_model.fit_calibration(X_val, y_val)
    print(f"✓ Calibration complete using {method} scaling")
    
    return calibrated_model


def evaluate_model(
    calibrated_model: CalibratedTCN,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """Evaluate model on test set."""
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score,
        recall_score, f1_score, brier_score_loss
    )
    
    print("\n" + "="*70)
    print("Evaluating on test set...")
    print("="*70)
    
    # Get predictions
    y_pred_proba = calibrated_model.predict_proba(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'brier': brier_score_loss(y_test, y_pred_proba)
    }
    
    print("\nTest Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc']:.4f}")
    print(f"  Brier:     {metrics['brier']:.4f}")
    
    return metrics


def save_model(
    model: torch.nn.Module,
    calibrated_model: CalibratedTCN,
    feature_engineer: FeatureEngineer,
    config: dict,
    metrics: dict,
    save_dir: str
):
    """Save trained model and artifacts."""
    print("\n" + "="*70)
    print("Saving model...")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': config['model'],
        'calibrator': calibrated_model.calibrator,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(checkpoint, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save feature scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    feature_engineer.save_scaler(scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Save config
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"✓ Config saved to {config_path}")
    
    print(f"\n✓ All artifacts saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train TCN model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='models/tcn_v1', help='Output directory')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', help='Model config')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load and prepare data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     timestamps, engineer) = load_and_prepare_data(args.data, config)
    
    # Train model
    model, trainer, history = train_model(
        X_train, y_train, X_val, y_val, config, args.output
    )
    
    # Calibrate model
    #calibrated_model = calibrate_model(model, X_val, y_val, config)
    calibrated_model = CalibratedTCN(model, calibration_method='platt', device='cpu')
    calibrated_model.calibrator = None
    
    # Evaluate model
    metrics = evaluate_model(calibrated_model, X_test, y_test)
    
    # Save model
    save_model(model, calibrated_model, engineer, config, metrics, args.output)
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()