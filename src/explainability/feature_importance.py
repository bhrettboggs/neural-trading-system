"""
Feature importance and explainability for TCN model.
"""

import numpy as np
import torch


def compute_feature_importance_permutation(
    model,
    X_test,
    y_test,
    n_repeats=10
):
    """
    Compute feature importance via permutation.
    
    Args:
        model: Trained model
        X_test: Test features (n_samples, seq_len, n_features)
        y_test: Test labels
        n_repeats: Number of permutations
        
    Returns:
        Feature importance scores
    """
    model.eval()
    
    # Baseline score
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = torch.sigmoid(model(X_tensor)).numpy().flatten()
    
    from sklearn.metrics import roc_auc_score
    baseline_score = roc_auc_score(y_test, y_pred)
    
    n_features = X_test.shape[2]
    importance = np.zeros(n_features)
    
    # Permute each feature
    for feat_idx in range(n_features):
        scores = []
        
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, :, feat_idx])
            
            with torch.no_grad():
                X_tensor = torch.tensor(X_permuted, dtype=torch.float32)
                y_pred = torch.sigmoid(model(X_tensor)).numpy().flatten()
            
            score = roc_auc_score(y_test, y_pred)
            scores.append(baseline_score - score)
        
        importance[feat_idx] = np.mean(scores)
    
    return importance


if __name__ == "__main__":
    print("Feature importance module")