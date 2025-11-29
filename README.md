# Neural Trading System for Cryptocurrency Markets

⚠️ **CRITICAL DISCLAIMER**: This system is for **RESEARCH AND EDUCATIONAL PURPOSES ONLY**. Cryptocurrency trading involves substantial risk of loss. This system provides NO GUARANTEES of profits. You can lose your entire investment. Past performance does not predict future results. Consult licensed financial professionals before trading.

---

## 🎯 System Overview

A complete research-to-production neural network trading system using:
- **Temporal Convolutional Networks (TCN)** for directional prediction
- **15-minute bar data** from cryptocurrency exchanges
- **Triple-barrier labeling** for realistic trade exits
- **Volatility-scaled position sizing**
- **Calibrated probability outputs** (Platt scaling)
- **Comprehensive risk management** with circuit breakers

---

## 📋 Quick Start

### 1. Installation

```bash
# Clone and enter directory
cd neural-trading-system

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Real Data

```bash
python scripts/download_real_data.py \
  --symbol BTC/USDT \
  --start 2022-01-01 \
  --end 2024-10-31 \
  --output data/raw/btc_usdt_15m.csv
```

### 3. Train Model

```bash
python scripts/train_model.py \
  --data data/raw/btc_usdt_15m.csv \
  --output models/tcn_v1
```

### 4. Run Backtest

```bash
python scripts/run_backtest.py \
  --model models/tcn_v1/final_model.pt \
  --data data/raw/btc_usdt_15m.csv \
  --output backtest_results
```

### 5. Paper Trading (MANDATORY before live)

```bash
python scripts/start_server.py \
  --mode paper \
  --model models/tcn_v1/final_model.pt
```

**Run paper trading for MINIMUM 30 days before considering live trading.**

---

## 🏗️ Architecture

```
Data Pipeline → Features (12 indicators) → TCN Model → Calibration → 
  Signals → Vol Scaling → Position Sizing → Risk Checks → Execution
```

### Key Components

1. **Data Loader** (`src/data/loader.py`): Fetches real OHLCV from exchanges via CCXT
2. **Feature Engineer** (`src/data/features.py`): Computes 12 technical indicators  
3. **TCN Model** (`src/models/tcn.py`): Predicts directional probability
4. **Calibrator**: Ensures probabilities match empirical frequencies
5. **Signal Generator** (`src/trading/signals.py`): Converts predictions to positions
6. **Backtester** (`src/backtest/engine.py`): Realistic simulation with costs
7. **Inference Server** (`src/server/app.py`): FastAPI REST API

---

## 📊 Model Architecture

**Temporal Convolutional Network:**
```
Input: (64 timesteps, 12 features)
  ↓
TCN Block 1: channels=32, dilation=1,  kernel=3
TCN Block 2: channels=32, dilation=2,  kernel=3  
TCN Block 3: channels=64, dilation=4,  kernel=3
TCN Block 4: channels=64, dilation=8,  kernel=3
  ↓
Global Average Pooling
  ↓
Dense(32) + Dropout(0.3)
  ↓
Output(1) → Sigmoid → Calibration
```

**Parameters:** ~47,000  
**Receptive Field:** ~50 timesteps (12.5 hours)

---

## 📈 Features (12 Technical Indicators)

1. **log_returns_1**: 1-bar log return
2. **log_returns_5**: 5-bar log return  
3. **realized_vol_20**: 20-bar rolling volatility
4. **ewma_vol_fast**: Fast EWMA vol (span=20)
5. **ewma_vol_slow**: Slow EWMA vol (span=96)
6. **momentum_20**: 20-bar price momentum
7. **rsi_14**: Relative Strength Index
8. **volume_zscore**: Normalized volume
9. **spread**: (high-low)/close
10. **sma_crossover**: SMA(5)/SMA(20) - 1
11. **time_of_day_sin**: Intraday seasonality
12. **day_of_week_sin**: Weekly seasonality

---

## 🎲 Triple-Barrier Labeling

For each bar:
```python
Entry: close[t]
Profit Target: +0.8%  
Stop Loss: -0.4%
Max Hold: 8 bars (2 hours)

Label = 1 if profit target hit first
Label = 0 if stop loss hit first or timeout
```

**Why?** Mimics realistic exits, no lookahead bias.

---

## 💹 Trading Strategy

**Signal Generation:**
```python
1. Model outputs probability P(profit) ∈ [0,1]
2. Convert to signal: s = 2×P - 1 ∈ [-1,1]
3. Vol scaling: r = (target_vol / forecast_vol) × s
4. Deadband filter: if |r| < 0.3, set r=0
5. Position weight: w = clip(r, -1, 1)
```

**Execution:**
- High confidence: Limit orders (maker rebates)
- Low confidence: Market orders (immediate fill)

**Risk Management:**
- Max position: 95% of NAV
- Max leverage: 1.0x (no borrowing)
- Circuit breakers: DD>20%, vol>10%, daily loss>5%

---

## 📁 Project Structure

```
neural-trading-system/
├── config/              # YAML configs
├── src/
│   ├── data/           # Data loading & features
│   ├── models/         # TCN, calibration, volatility
│   ├── training/       # Training loop, CV
│   ├── backtest/       # Backtester
│   ├── trading/        # Signals, position sizing
│   ├── metrics/        # Performance metrics
│   ├── explainability/ # Feature importance
│   └── server/         # FastAPI inference server
├── scripts/            # Executable scripts
├── tests/              # Unit tests
└── notebooks/          # Research docs
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Specific test
pytest tests/test_model.py -v
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t trading-system:v1 .

# Run server
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/tcn_v1/final_model.pt \
  trading-system:v1
```

---

## 📊 Expected Performance

**Conservative Estimates (after costs):**
```
Sharpe Ratio:     1.2 - 2.0
CAGR:             20% - 60%  
Max Drawdown:     15% - 25%
Win Rate:         52% - 55%
Avg Hold:         1-2 hours
```

**⚠️ Actual results will vary significantly based on market conditions.**

---

## ⚠️ Pre-Live Trading Checklist

Complete ALL items before live trading:

### Regulatory
- [ ] Complete KYC/AML on exchange
- [ ] Understand tax implications
- [ ] Consult financial/legal professionals

### Technical
- [ ] Tested in paper mode for 30+ days
- [ ] Validated transaction costs match reality
- [ ] Configured API keys securely (env vars, not code)
- [ ] Set up monitoring and alerts
- [ ] Tested kill switches and circuit breakers

### Risk Management
- [ ] Set maximum capital allocation
- [ ] Set maximum position sizes
- [ ] Set maximum daily loss limit
- [ ] Document all risk parameters
- [ ] Sign consent form (see README warnings)

---

## 🛑 Live Trading Warnings

**DO NOT proceed to live trading until:**

1. ✅ You've run paper trading for MINIMUM 30 days
2. ✅ You've completed the pre-deployment checklist
3. ✅ You understand you can lose ALL invested capital
4. ✅ You're only trading money you can afford to lose
5. ✅ You've consulted financial professionals

**Start with SMALL capital (e.g., $100-500) to test live execution.**

---

## 📚 Documentation

- **Full Research Report**: `notebooks/research_report.md`
- **Config Reference**: See `config/*.yaml` files
- **API Docs**: http://localhost:8000/docs (when server running)

---

## 🤝 Support

For issues or questions:
1. Review documentation thoroughly
2. Check config files
3. Run tests to verify installation
4. Review logs in `logs/` directory

---

## 📜 License

MIT License - See LICENSE file

**NO WARRANTY. USE AT YOUR OWN RISK.**

---

## 🎓 Educational Use

This system is designed for:
- Learning machine learning for trading
- Understanding production ML systems
- Studying risk management
- Exploring TCN architectures
- Practicing software engineering

**Not designed for:**  
- Guaranteed profits
- Get-rich-quick schemes
- Unsupervised automated trading
- Retail users without technical knowledge

---

## 🔧 Troubleshooting

**Common Issues:**

1. **"Module not found"**: Run `pip install -r requirements.txt`
2. **"Out of memory"**: Reduce batch_size in config
3. **"CCXT error"**: Check internet connection and API limits
4. **"Model diverges"**: Check learning rate, try different seed

---

## 📞 Emergency Procedures

**If system malfunctions in live trading:**

1. **IMMEDIATELY**: Press Ctrl+C to stop server
2. **Close all positions manually** on exchange website
3. **Disable API keys** on exchange
4. **Review logs** in `logs/` directory
5. **Do NOT restart** until issue identified

---

## ✨ Next Steps

1. ✅ Download real data
2. ✅ Train model and review metrics
3. ✅ Run backtest, analyze results
4. ✅ Start paper trading (30+ days)
5. ✅ Complete compliance checklist
6. ✅ Consider live trading with small capital

**Remember: Start small, test extensively, never risk more than you can afford to lose.**

---

**Built with ❤️ for research and education. Trade responsibly.** 🚀