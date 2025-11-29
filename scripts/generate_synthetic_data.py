#!/usr/bin/env python3
"""
Generate synthetic cryptocurrency data for testing.

Usage:
    python scripts/generate_synthetic_data.py --output data/synthetic/btc_usdt_15m.csv --bars 10000
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from src.data.synthetic_data import generate_synthetic_crypto_data, add_trend_regime, add_volatility_clusters


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic crypto data')
    parser.add_argument('--output', type=str, default='data/synthetic/btc_usdt_15m.csv', help='Output CSV path')
    parser.add_argument('--bars', type=int, default=10000, help='Number of bars to generate')
    parser.add_argument('--price', type=float, default=40000.0, help='Initial price')
    parser.add_argument('--drift', type=float, default=0.0001, help='Drift (annualized)')
    parser.add_argument('--vol', type=float, default=0.02, help='Volatility (annualized)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--enhanced', action='store_true', help='Add trend regimes and volatility clustering')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("="*70)
    print("Generating synthetic cryptocurrency data")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Bars: {args.bars:,}")
    print(f"  Initial price: ${args.price:,.2f}")
    print(f"  Drift: {args.drift:.4f}")
    print(f"  Volatility: {args.vol:.2%}")
    print(f"  Seed: {args.seed}")
    print(f"  Enhanced: {args.enhanced}")
    
    # Generate data
    df = generate_synthetic_crypto_data(
        output_path=None,  # Don't save yet
        n_bars=args.bars,
        initial_price=args.price,
        drift=args.drift,
        volatility=args.vol,
        seed=args.seed
    )
    
    # Add enhancements if requested
    if args.enhanced:
        print("\n" + "="*70)
        print("Adding market features...")
        print("="*70)
        
        df = add_trend_regime(df, regime_duration_bars=1000, trend_strength=0.0005)
        print("✓ Added trend regimes")
        
        df = add_volatility_clusters(df, cluster_probability=0.1, volatility_multiplier=3.0)
        print("✓ Added volatility clustering")
    
    # Save to file
    df.to_csv(args.output, index=False)
    print(f"\n✓ Saved to {args.output}")
    
    # Print summary
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"  Total bars: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    print(f"  Start price: ${df['close'].iloc[0]:,.2f}")
    print(f"  End price: ${df['close'].iloc[-1]:,.2f}")
    print(f"  Total return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.2%}")
    
    import numpy as np
    returns = df['close'].pct_change().dropna()
    print(f"  Return mean: {returns.mean():.6f}")
    print(f"  Return std: {returns.std():.6f}")
    print(f"  Annualized vol: {returns.std() * np.sqrt(365*24*4):.2%}")
    
    print("\n" + "="*70)
    print("✓ Data generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()