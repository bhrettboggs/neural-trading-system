import pytest
from src.trading.signals import TradingSignalGenerator, TradingConfig


def test_signal_generator():
    """Test signal generation."""
    config = TradingConfig()
    generator = TradingSignalGenerator(config)
    
    # Test high confidence long
    signal = generator.generate_signal(model_output=0.7, vol_forecast=0.02)
    assert signal > 0
    
    # Test high confidence short
    signal = generator.generate_signal(model_output=0.3, vol_forecast=0.02)
    assert signal < 0
    
    # Test low confidence (should be filtered by deadband)
    signal = generator.generate_signal(model_output=0.52, vol_forecast=0.02)
    assert signal == 0


if __name__ == "__main__":
    pytest.main([__file__])