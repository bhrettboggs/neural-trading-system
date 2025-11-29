"""
Training script for TCN model with cross-validation and model selection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
import os
from datetime import datetime
from tqdm import tqdm


class Trainer:
    """TCN model trainer with cross-validation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        label_smoothing: float = 0.05,
        save_dir: str = 'models'
    ):
        """
        Initialize trainer.
        
        Args:
            model: TCN model to train
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Maximum epochs
            early_stopping_patience: Early stopping patience
            label_smoothing: Label smoothing factor
            save_dir: Directory to save models
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.label_smoothing = label_smoothing
        self.save_dir = save_dir
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function with label smoothing
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }
        
        os.makedirs(save_dir, exist_ok=True)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """
        Train model on training data with validation.
        
        Args:
            X_train: Training features (n_samples, sequence_length, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history dictionary
        """
        # Apply label smoothing
        y_train_smooth = self._apply_label_smoothing(y_train)
        y_val_smooth = self._apply_label_smoothing(y_val)
        
        # Create data loaders
        train_loader = self._create_data_loader(X_train, y_train_smooth, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val_smooth, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {len(X_train):,} samples, validating on {len(X_val):,} samples")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, train_auc = self._train_epoch(train_loader, y_train)
            
            # Validation phase
            val_loss, val_auc = self._validate_epoch(val_loader, y_val)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"  Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}, AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self._load_checkpoint('best_model.pt')
        
        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        y_true: np.ndarray
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions
            preds = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(train_loader)
        auc = self._compute_auc(np.array(all_labels), np.array(all_preds))
        
        return avg_loss, auc
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        y_true: np.ndarray
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                
                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        auc = self._compute_auc(np.array(all_labels), np.array(all_preds))
        
        return avg_loss, auc
    
    def _apply_label_smoothing(self, y: np.ndarray) -> np.ndarray:
        """Apply label smoothing: 0 → epsilon, 1 → 1 - epsilon."""
        if self.label_smoothing > 0:
            y_smooth = y.copy().astype(float)
            y_smooth[y == 0] = self.label_smoothing
            y_smooth[y == 1] = 1 - self.label_smoothing
            return y_smooth
        return y.astype(float)
    
    def _create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        
        return loader
    
    def _compute_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC-ROC score."""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def purged_walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    n_splits: int = 5,
    embargo_bars: int = 96  # 1 day for 15-min bars
) -> list:
    """
    Purged walk-forward cross-validation.
    
    Args:
        X: Features array
        y: Labels array
        timestamps: Timestamps array
        n_splits: Number of folds
        embargo_bars: Embargo period between splits
        
    Returns:
        List of (train_idx, val_idx) tuples
    """
    n_samples = len(X)
    fold_size = n_samples // n_splits
    
    splits = []
    
    for i in range(n_splits - 1):
        # Training: all data up to current fold
        train_end = (i + 1) * fold_size
        train_idx = np.arange(0, train_end)
        
        # Embargo
        val_start = train_end + embargo_bars
        val_end = min(val_start + fold_size, n_samples)
        
        if val_end - val_start < fold_size // 2:
            break  # Not enough validation data
        
        val_idx = np.arange(val_start, val_end)
        
        splits.append((train_idx, val_idx))
    
    return splits


def train_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    model_config: dict,
    training_config: dict,
    save_dir: str = 'models/cv'
) -> Dict:
    """
    Train model with purged walk-forward cross-validation.
    
    Args:
        X: Features
        y: Labels
        timestamps: Timestamps
        model_config: Model configuration
        training_config: Training configuration
        save_dir: Directory to save models
        
    Returns:
        Dictionary with CV results
    """
    from src.models.tcn import TCNModel
    
    # Create CV splits
    splits = purged_walk_forward_cv(X, y, timestamps, n_splits=5)
    
    cv_results = {
        'fold_metrics': [],
        'mean_val_auc': 0.0,
        'mean_val_loss': 0.0
    }
    
    print(f"\nRunning {len(splits)}-fold purged walk-forward CV...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(splits)}")
        print(f"{'='*60}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create model
        model = TCNModel(**model_config)
        
        # Create trainer
        fold_save_dir = os.path.join(save_dir, f'fold_{fold_idx}')
        trainer = Trainer(
            model=model,
            save_dir=fold_save_dir,
            **training_config
        )
        
        # Train
        history = trainer.train(X_train, y_train, X_val, y_val)
        
        # Record metrics
        best_val_loss = min(history['val_loss'])
        best_val_auc = max(history['val_auc'])
        
        cv_results['fold_metrics'].append({
            'fold': fold_idx,
            'val_loss': best_val_loss,
            'val_auc': best_val_auc,
            'train_size': len(X_train),
            'val_size': len(X_val)
        })
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val AUC: {best_val_auc:.4f}")
    
    # Aggregate results
    val_losses = [m['val_loss'] for m in cv_results['fold_metrics']]
    val_aucs = [m['val_auc'] for m in cv_results['fold_metrics']]
    
    cv_results['mean_val_loss'] = np.mean(val_losses)
    cv_results['std_val_loss'] = np.std(val_losses)
    cv_results['mean_val_auc'] = np.mean(val_aucs)
    cv_results['std_val_auc'] = np.std(val_aucs)
    
    print(f"\n{'='*60}")
    print("Cross-Validation Results:")
    print(f"{'='*60}")
    print(f"Mean Val Loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
    print(f"Mean Val AUC: {cv_results['mean_val_auc']:.4f} ± {cv_results['std_val_auc']:.4f}")
    
    # Save CV results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    return cv_results


if __name__ == "__main__":
    # Test training
    print("Testing training script...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 64
    n_features = 12
    
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randint(0, 2, n_samples)
    timestamps = pd.date_range('2021-01-01', periods=n_samples, freq='15min')
    
    # Model config
    model_config = {
        'input_size': n_features,
        'sequence_length': sequence_length,
        'tcn_channels': [32, 32, 64, 64],
        'kernel_size': 3,
        'dropout': [0.1, 0.1, 0.1, 0.2]
    }
    
    # Training config
    training_config = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 10,
        'early_stopping_patience': 5,
        'label_smoothing': 0.05
    }
    
    # Train with CV
    cv_results = train_with_cross_validation(
        X, y, timestamps,
        model_config,
        training_config,
        save_dir='models/test_cv'
    )
    
    print("\n✓ Training script test passed!")