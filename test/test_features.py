import pytest
import numpy as np
import pandas as pd
from src.data.features import FeatureEngineer, compute_triple_barrier_labels


def test_feature_engineer():
    """Test feature engineering."""
    # Create sample data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=200, freq='15min'),
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.rand(200) * 1000
    })
    
    engineer = FeatureEngineer()
    df_features = engineer.compute_features(df)
    
    # Check all features computed
    assert len(engineer.feature_names) == 12
    for feat in engineer.feature_names:
        assert feat in df_features.columns


def test_triple_barrier_labels():
    """Test triple-barrier labeling."""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='15min'),
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100, 110, 100),
        'volume': np.ones(100) * 1000
    })
    
    labels = compute_triple_barrier_labels(df)
    
    assert len(labels) == len(df)
    assert labels.min() >= 0
    assert labels.max() <= 1


if __name__ == "__main__":
    pytest.main([__file__])