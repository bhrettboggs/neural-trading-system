#!/usr/bin/env python3
"""
Run backtest on trained model.

Usage:
    python scripts/run_backtest.py \
        --model models/tcn_v1/final_model.pt \
        --data data/raw/btc_usdt_15m.csv \
        --output backtest_results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.data.features import FeatureEngineer
from src.models.tcn import TCNModel, CalibratedTCN
from src.models.volatility import VolatilityEstimator
from src.backtest.engine import Backtester, TransactionCostModel
from src.trading.signals import TradingSignalGenerator, TradingConfig


def load_model(model_path: str, scaler_path: str = None):
    """Load trained model and scaler."""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    # Create model
    model = TCNModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create calibrated model
    calibrator = checkpoint.get('calibrator', None)
    if calibrator:
        calibrated_model = CalibratedTCN(model, calibration_method='platt', device='cpu')
        calibrated_model.calibrator = calibrator
    else:
        calibrated_model = None
    
    # Load scaler
    engineer = FeatureEngineer()
    if scaler_path and os.path.exists(scaler_path):
        engineer.load_scaler(scaler_path)
    
    print("✓ Model loaded successfully")
    
    return model, calibrated_model, engineer


def prepare_backtest_data(data_path: str, engineer: FeatureEngineer, start_date: str = None, end_date: str = None):
    """Load and prepare data for backtesting."""
    print(f"\nLoading data from {data_path}...")
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter date range if specified
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
    
    print(f"  Loaded {len(df):,} bars")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Compute features
    print("  Computing features...")
    df_features = engineer.compute_features(df)
    
    # Prepare sequences
    X, timestamps, _ = engineer.prepare_sequences(
        df_features,
        sequence_length=64,
        normalize=True
    )
    
    print(f"  Prepared {len(X):,} sequences")
    
    return df, df_features, X, timestamps


def generate_predictions(model, calibrated_model, X: np.ndarray):
    """Generate model predictions."""
    print("\nGenerating predictions...")
    
    model.eval()
    
    if calibrated_model:
        predictions = calibrated_model.predict_proba(X)
    else:
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = model(X_tensor).numpy().flatten()
            predictions = 1 / (1 + np.exp(-logits))
    
    print(f"  Generated {len(predictions):,} predictions")
    print(f"  Mean probability: {predictions.mean():.3f}")
    print(f"  Std probability: {predictions.std():.3f}")
    
    return predictions


def estimate_volatility(df_features: pd.DataFrame, timestamps: np.ndarray):
    """Estimate volatility forecasts."""
    print("\nEstimating volatility...")
    
    # Compute returns
    returns = np.log(df_features['close'] / df_features['close'].shift(1)).values
    
    # Create volatility estimator
    vol_estimator = VolatilityEstimator(method='dual', fast_span=20, slow_span=96)
    
    # Estimate volatility
    vol_series = vol_estimator.estimate(returns)
    
    # Annualize
    bars_per_year = (365 * 24 * 60) / 15
    vol_annual = vol_series * np.sqrt(bars_per_year)
    
    # Align with timestamps
    vol_forecasts = []
    for ts in timestamps:
        idx = df_features[df_features['timestamp'] == ts].index
        if len(idx) > 0:
            vol_forecasts.append(vol_annual[idx[0]])
        else:
            vol_forecasts.append(0.02)  # Default 2%
    
    vol_forecasts = np.array(vol_forecasts)
    
    print(f"  Mean volatility: {vol_forecasts.mean():.2%}")
    print(f"  Volatility range: {vol_forecasts.min():.2%} - {vol_forecasts.max():.2%}")
    
    return vol_forecasts


def run_backtest(df: pd.DataFrame, predictions: np.ndarray, vol_forecasts: np.ndarray, config: dict):
    """Run backtest with predictions."""
    print("\n" + "="*70)
    print("Running backtest...")
    print("="*70)
    
    # Load backtest config
    with open('config/backtest_config.yaml', 'r') as f:
        backtest_config = yaml.safe_load(f)
    
    # Generate trading signals
    trading_config = TradingConfig(
        target_vol=backtest_config['trading']['target_volatility'],
        max_leverage=backtest_config['trading']['max_leverage'],
        deadband_threshold=backtest_config['trading']['deadband_threshold']
    )
    
    signal_generator = TradingSignalGenerator(trading_config)
    
    signals = np.array([
        signal_generator.generate_signal(p, v)
        for p, v in zip(predictions, vol_forecasts)
    ])
    
    print(f"\nSignal statistics:")
    print(f"  Mean signal: {signals.mean():.3f}")
    print(f"  Positive signals: {(signals > 0).sum()} ({(signals > 0).mean():.1%})")
    print(f"  Negative signals: {(signals < 0).sum()} ({(signals < 0).mean():.1%})")
    print(f"  Neutral (filtered): {(signals == 0).sum()} ({(signals == 0).mean():.1%})")
    
    # Determine order types (maker vs taker)
    confidence = np.abs(predictions - 0.5)
    is_maker = confidence > backtest_config['trading']['confidence_threshold']
    
    # Create cost model
    cost_model = TransactionCostModel(
        maker_fee=backtest_config['transaction_costs']['maker_fee'],
        taker_fee=backtest_config['transaction_costs']['taker_fee'],
        slippage_bps=backtest_config['transaction_costs']['slippage_bps']
    )
    
    # Run backtest
    backtester = Backtester(
        initial_capital=backtest_config['backtest']['initial_capital'],
        cost_model=cost_model
    )
    
    # Align df with signals
    df_backtest = df.iloc[-len(signals):].copy().reset_index(drop=True)
    
    results_df, metrics = backtester.run(df_backtest, signals, is_maker)
    
    return results_df, metrics, backtester


def print_results(metrics):
    """Print backtest results."""
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    
    print("\n📊 Returns:")
    print(f"  Total Return:     {metrics.total_return:>10.2%}")
    print(f"  CAGR:             {metrics.cagr:>10.2%}")
    print(f"  Annualized Vol:   {metrics.annualized_vol:>10.2%}")
    
    print("\n📈 Risk-Adjusted:")
    print(f"  Sharpe Ratio:     {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:    {metrics.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:     {metrics.calmar_ratio:>10.2f}")
    
    print("\n📉 Drawdown:")
    print(f"  Max Drawdown:     {metrics.max_drawdown:>10.2%}")
    print(f"  Avg Drawdown:     {metrics.avg_drawdown:>10.2%}")
    print(f"  Max DD Duration:  {metrics.max_drawdown_duration:>10.0f} bars")
    
    print("\n💼 Trading:")
    print(f"  Total Trades:     {metrics.total_trades:>10.0f}")
    print(f"  Win Rate:         {metrics.win_rate:>10.2%}")
    print(f"  Avg Win:          ${metrics.avg_win:>9.2f}")
    print(f"  Avg Loss:         ${metrics.avg_loss:>9.2f}")
    print(f"  Profit Factor:    {metrics.profit_factor:>10.2f}")
    
    print("\n💰 Costs:")
    print(f"  Total Fees:       ${metrics.total_fees:>9.2f}")
    print(f"  Turnover:         {metrics.turnover:>10.2f}x")
    
    print("\n⚠️  Risk:")
    print(f"  VaR (95%):        {metrics.var_95:>10.2%}")
    print(f"  CVaR (99%):       {metrics.cvar_99:>10.2%}")


def save_results(results_df, metrics, backtester, output_dir: str):
    """Save backtest results and plots."""
    print("\n" + "="*70)
    print("Saving results...")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results CSV
    results_path = os.path.join(output_dir, 'backtest_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to {results_path}")
    
    # Save metrics JSON
    import json
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")
    
    # Save trades CSV
    if backtester.trades:
        trades_df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'side': t.side,
            'size': t.size,
            'price': t.price,
            'notional': t.notional,
            'cost': t.cost
        } for t in backtester.trades])
        
        trades_path = os.path.join(output_dir, 'trades.csv')
        trades_df.to_csv(trades_path, index=False)
        print(f"✓ Trades saved to {trades_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Equity curve
    plt.figure(figsize=(14, 6))
    plt.plot(results_df['timestamp'], results_df['nav'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('NAV ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'), dpi=150)
    plt.close()
    print("  ✓ equity_curve.png")
    
    # 2. Drawdown
    plt.figure(figsize=(14, 6))
    plt.fill_between(results_df['timestamp'], 0, -results_df['drawdown'] * 100, alpha=0.3, color='red')
    plt.plot(results_df['timestamp'], -results_df['drawdown'] * 100, color='red')
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=150)
    plt.close()
    print("  ✓ drawdown.png")
    
    # 3. Returns distribution
    plt.figure(figsize=(10, 6))
    returns = results_df['returns'].dropna()
    plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    plt.title('Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'returns_distribution.png'), dpi=150)
    plt.close()
    print("  ✓ returns_distribution.png")
    
    # 4. Monthly returns heatmap
    results_df['year'] = pd.to_datetime(results_df['timestamp']).dt.year
    results_df['month'] = pd.to_datetime(results_df['timestamp']).dt.month
    
    monthly_returns = results_df.groupby(['year', 'month'])['returns'].sum().unstack()
    
    if len(monthly_returns) > 0:
        plt.figure(figsize=(12, 6))
        sns.heatmap(monthly_returns * 100, annot=True, fmt='.2f', cmap='RdYlGn', center=0, cbar_kws={'label': 'Return (%)'})
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_returns.png'), dpi=150)
        plt.close()
        print("  ✓ monthly_returns.png")
    
    print(f"\n✓ All results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--scaler', type=str, default=None, help='Path to scaler (optional)')
    parser.add_argument('--output', type=str, default='backtest_results', help='Output directory')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Auto-detect scaler path if not provided
    if args.scaler is None:
        model_dir = os.path.dirname(args.model)
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            args.scaler = scaler_path
    
    # Load model
    model, calibrated_model, engineer = load_model(args.model, args.scaler)
    
    # Prepare data
    df, df_features, X, timestamps = prepare_backtest_data(
        args.data, engineer, args.start, args.end
    )
    
    # Generate predictions
    predictions = generate_predictions(model, calibrated_model, X)
    
    # Estimate volatility
    vol_forecasts = estimate_volatility(df_features, timestamps)
    
    # Run backtest
    results_df, metrics, backtester = run_backtest(df, predictions, vol_forecasts, {})
    
    # Print results
    print_results(metrics)
    
    # Save results
    save_results(results_df, metrics, backtester, args.output)
    
    print("\n" + "="*70)
    print("✓ Backtest complete!")
    print("="*70)


if __name__ == "__main__":
    main()