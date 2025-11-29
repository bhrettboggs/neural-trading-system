#!/usr/bin/env python3
"""
Start inference server for trading system.

Usage:
    # Paper trading mode
    python scripts/start_server.py --mode paper --model models/tcn_v1/final_model.pt
    
    # Live trading mode (requires consent)
    python scripts/start_server.py --mode live --model models/tcn_v1/final_model.pt
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import subprocess
from datetime import datetime


def check_consent_file():
    """Check if user has signed consent form for live trading."""
    consent_file = "LIVE_TRADING_CONSENT.txt"
    
    if not os.path.exists(consent_file):
        print("\n" + "="*70)
        print("⚠️  LIVE TRADING CONSENT REQUIRED")
        print("="*70)
        print("\nYou must create and sign a consent form before live trading.")
        print("\nCreate a file named 'LIVE_TRADING_CONSENT.txt' with:")
        print("""
I, [YOUR NAME], on [DATE], acknowledge that:

1. I have read and understood all risk warnings
2. I have consulted with a licensed financial professional (or chosen not to)
3. I am only trading with capital I can afford to lose completely
4. I understand this system can and will lose money
5. I am solely responsible for all trading decisions and losses
6. I have completed all regulatory and compliance requirements
7. I have tested the system in paper trading for minimum 30 days
8. I authorize this system to execute trades on my behalf within configured limits

Signature: ___________________
Date: ___________________
""")
        print("\n" + "="*70 + "\n")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Start trading server')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['paper', 'live'],
        required=True,
        help='Trading mode: paper or live'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/tcn_v1/final_model.pt',
        help='Path to trained model'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='models/tcn_v1/scaler.pkl',
        help='Path to feature scaler'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Server port'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Server host'
    )
    
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.scaler):
        print(f"⚠️  Warning: Scaler not found at {args.scaler}")
    
    # Live trading checks
    if args.mode == 'live':
        print("\n" + "="*70)
        print("⚠️  LIVE TRADING MODE")
        print("="*70)
        
        if not check_consent_file():
            print("❌ Cannot start live trading without signed consent")
            sys.exit(1)
        
        print("\n⚠️  FINAL WARNING:")
        print("You are about to start LIVE TRADING with REAL MONEY")
        print("Losses can and will occur. There are NO guarantees.")
        
        response = input("\nType 'I ACCEPT THE RISK' to continue: ")
        
        if response != "I ACCEPT THE RISK":
            print("Live trading cancelled")
            sys.exit(0)
    
    # Set environment variables
    os.environ['MODEL_PATH'] = args.model
    os.environ['SCALER_PATH'] = args.scaler
    os.environ['TRADING_MODE'] = args.mode
    
    # Print configuration
    print("\n" + "="*70)
    print("Trading Server Configuration")
    print("="*70)
    print(f"Mode:         {args.mode.upper()}")
    print(f"Model:        {args.model}")
    print(f"Scaler:       {args.scaler}")
    print(f"Host:         {args.host}")
    print(f"Port:         {args.port}")
    print(f"Start time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    if args.mode == 'paper':
        print("📄 Running in PAPER TRADING mode (no real money)")
    else:
        print("💰 Running in LIVE TRADING mode (REAL MONEY)")
    
    print("\nServer endpoints:")
    print(f"  Health:  http://localhost:{args.port}/health")
    print(f"  Predict: http://localhost:{args.port}/predict")
    print(f"  Metrics: http://localhost:{args.port}/metrics")
    print(f"  Docs:    http://localhost:{args.port}/docs")
    
    print("\nPress Ctrl+C to stop server\n")
    print("="*70 + "\n")
    
    # Start server
    try:
        subprocess.run([
            'python', '-m', 'uvicorn',
            'src.server.app:app',
            '--host', args.host,
            '--port', str(args.port),
            '--reload'
        ])
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        print("="*70)


if __name__ == "__main__":
    main()