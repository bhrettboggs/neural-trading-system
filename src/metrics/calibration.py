"""
Probability calibration metrics and visualization.
Evaluates how well predicted probabilities match empirical frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def compute_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Compute calibration metrics.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary with calibration metrics
    """
    metrics = {}
    
    # Brier score (MSE of probabilities)
    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
    
    # Log loss (cross-entropy)
    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
    
    # Expected Calibration Error (ECE)
    metrics['ece'] = compute_ece(y_true, y_pred_proba, n_bins=10)
    
    # Maximum Calibration Error (MCE)
    metrics['mce'] = compute_mce(y_true, y_pred_proba, n_bins=10)
    
    return metrics


def compute_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.
    
    ECE measures the average difference between predicted probabilities
    and empirical frequencies across bins.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        ECE score (lower is better, 0 is perfect)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges[1:-1])
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if bin_mask.sum() == 0:
            continue
        
        bin_prob = y_pred_proba[bin_mask].mean()
        bin_accuracy = y_true[bin_mask].mean()
        bin_size = bin_mask.sum()
        
        ece += (bin_size / total_samples) * abs(bin_prob - bin_accuracy)
    
    return ece


def compute_mce(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error.
    
    MCE is the maximum difference between predicted probabilities
    and empirical frequencies across all bins.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        MCE score (lower is better, 0 is perfect)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges[1:-1])
    
    max_error = 0.0
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if bin_mask.sum() == 0:
            continue
        
        bin_prob = y_pred_proba[bin_mask].mean()
        bin_accuracy = y_true[bin_mask].mean()
        
        error = abs(bin_prob - bin_accuracy)
        max_error = max(max_error, error)
    
    return max_error


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    save_path: str = None
):
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save plot (optional)
    """
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    ax1.set_xlabel('Predicted probability')
    ax1.set_ylabel('Empirical frequency')
    ax1.set_title('Calibration Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predicted probabilities
    ax2.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Predicted Probabilities')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Calibration plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_vs_accuracy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    save_path: str = None
):
    """
    Plot confidence vs accuracy.
    
    Shows how prediction accuracy varies with model confidence.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of confidence bins
        save_path: Path to save plot (optional)
    """
    # Compute confidence (distance from 0.5)
    confidence = np.abs(y_pred_proba - 0.5)
    
    # Bin by confidence
    bin_edges = np.linspace(0, 0.5, n_bins + 1)
    bin_indices = np.digitize(confidence, bin_edges[1:-1])
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if bin_mask.sum() == 0:
            continue
        
        bin_centers.append(bin_edges[i:i+2].mean())
        
        # Accuracy: fraction of correct predictions
        y_pred = (y_pred_proba[bin_mask] > 0.5).astype(int)
        accuracy = (y_pred == y_true[bin_mask]).mean()
        bin_accuracies.append(accuracy)
        bin_counts.append(bin_mask.sum())
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence vs accuracy
    ax1.plot(bin_centers, bin_accuracies, 'o-')
    ax1.axhline(0.5, color='r', linestyle='--', label='Random baseline')
    ax1.set_xlabel('Confidence (|p - 0.5|)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Confidence vs Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sample distribution
    ax2.bar(bin_centers, bin_counts, width=bin_edges[1]-bin_edges[0], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence (|p - 0.5|)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Confidence Levels')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confidence plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test calibration metrics
    print("Testing calibration metrics...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    # Poorly calibrated: probabilities don't match frequencies
    y_true = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.beta(2, 2, n_samples)  # Biased toward 0.5
    
    # Compute metrics
    metrics = compute_calibration_metrics(y_true, y_pred_proba)
    
    print("\nCalibration Metrics:")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  MCE: {metrics['mce']:.4f}")
    
    # Plot
    print("\nGenerating calibration plots...")
    plot_calibration_curve(y_true, y_pred_proba)
    plot_confidence_vs_accuracy(y_true, y_pred_proba)
    
    print("\n✓ Calibration metrics test passed!")