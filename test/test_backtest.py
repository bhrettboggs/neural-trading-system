import pytest
import numpy as np
import pandas as pd
from src.backtest.engine import Backtester


def test_backtester():
    """Test backtester."""
    # Create sample data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='15min'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.rand(1000) * 1000
    })
    
    signals = np.random.uniform(-0.5, 0.5, len(df))
    
    backtester = Backtester(initial_capital=10000)
    results_df, metrics = backtester.run(df, signals)
    
    assert len(results_df) == len(df)
    assert metrics.total_trades > 0


if __name__ == "__main__":
    pytest.main([__file__])