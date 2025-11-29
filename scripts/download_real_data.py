#!/usr/bin/env python3
"""
Download real cryptocurrency data from exchange.

Usage:
    python scripts/download_real_data.py --symbol BTC/USDT --start 2022-01-01 --end 2024-10-31
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import load_real_crypto_data
import argparse


def main():
    parser = argparse.ArgumentParser(description='Download crypto data')
    parser.add_argument('--exchange', type=str, default='binance')
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--output', type=str, default='data/raw/btc_usdt_15m.csv')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Download data
    df = load_real_crypto_data(
        exchange_name=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        save_path=args.output
    )
    
    print(f"\n✓ Downloaded {len(df):,} bars to {args.output}")


if __name__ == "__main__":
    main()