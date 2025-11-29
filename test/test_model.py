import pytest
import torch
from src.models.tcn import TCNModel


def test_tcn_forward():
    """Test TCN forward pass."""
    model = TCNModel(
        input_size=12,
        sequence_length=64,
        tcn_channels=[32, 32, 64, 64]
    )
    
    batch_size = 16
    X = torch.randn(batch_size, 64, 12)
    
    logits = model(X)
    
    assert logits.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__])