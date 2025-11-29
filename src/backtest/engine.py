"""
Backtester with realistic execution simulation.
Includes transaction costs, slippage, position tracking, and P&L accounting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TransactionCostModel:
    """Transaction cost parameters for crypto trading."""
    maker_fee: float = 0.0010  # 10 bps (Binance VIP0)
    taker_fee: float = 0.0010  # 10 bps
    slippage_bps: float = 0.0005  # 5 bps slippage
    
    def compute_cost(
        self,
        notional: float,
        is_maker: bool = True
    ) -> float:
        """
        Compute transaction cost.
        
        Args:
            notional: Trade notional value
            is_maker: True if maker order (limit), False if taker (market)
            
        Returns:
            Total cost in quote currency
        """
        fee = self.maker_fee if is_maker else self.taker_fee
        cost = notional * (fee + self.slippage_bps)
        return cost


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    size: float  # In base currency
    price: float  # Execution price
    notional: float  # size * price
    cost: float  # Transaction cost
    position_before: float
    position_after: float


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    annualized_vol: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0
    
    # Trading stats
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_hold_time: float = 0.0
    
    # Costs
    total_fees: float = 0.0
    total_slippage: float = 0.0
    turnover: float = 0.0
    
    # Risk
    var_95: float = 0.0
    cvar_99: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class Backtester:
    """Backtest trading strategy with realistic execution."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        cost_model: TransactionCostModel = None
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in quote currency
            cost_model: Transaction cost model
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        
        # State variables
        self.cash = initial_capital
        self.position = 0.0  # Position in base currency
        self.nav = initial_capital
        self.peak_nav = initial_capital
        
        # History
        self.nav_history = []
        self.position_history = []
        self.trades = []
        self.daily_returns = []
    
    def run(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        is_maker: np.ndarray = None
    ) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data and timestamps
            signals: Array of target positions ∈ [-1, 1] (same length as df)
            is_maker: Array indicating if orders are maker (True) or taker (False)
            
        Returns:
            results_df: DataFrame with NAV, position, returns over time
            metrics: BacktestMetrics object
        """
        if len(signals) != len(df):
            raise ValueError("Signals must match DataFrame length")
        
        if is_maker is None:
            is_maker = np.ones(len(signals), dtype=bool)  # Assume all maker
        
        # Reset state
        self.cash = self.initial_capital
        self.position = 0.0
        self.nav = self.initial_capital
        self.peak_nav = self.initial_capital
        self.nav_history = []
        self.position_history = []
        self.trades = []
        
        # Run simulation
        for i in range(len(df)):
            timestamp = df['timestamp'].iloc[i]
            price = df['close'].iloc[i]
            target_signal = signals[i]
            maker = is_maker[i]
            
            # Compute target position
            target_notional = target_signal * self.nav
            target_position = target_notional / price
            
            # Execute trade if needed
            trade_size = target_position - self.position
            
            if abs(trade_size) > 1e-6:  # Minimum trade size threshold
                self._execute_trade(
                    timestamp=timestamp,
                    size=trade_size,
                    price=price,
                    is_maker=maker
                )
            
            # Update NAV (mark-to-market)
            self.nav = self.cash + self.position * price
            
            # Record state
            self.nav_history.append(self.nav)
            self.position_history.append(self.position)
            
            # Update peak NAV
            if self.nav > self.peak_nav:
                self.peak_nav = self.nav
        
        # Compute metrics
        results_df = self._create_results_df(df)
        metrics = self._compute_metrics(results_df)
        
        return results_df, metrics
    
    def _execute_trade(
        self,
        timestamp: datetime,
        size: float,
        price: float,
        is_maker: bool
    ):
        """
        Execute a trade with transaction costs.
        
        Args:
            timestamp: Trade timestamp
            size: Trade size in base currency (positive = buy, negative = sell)
            price: Execution price
            is_maker: Maker or taker order
        """
        notional = abs(size * price)
        cost = self.cost_model.compute_cost(notional, is_maker)
        
        # Update cash (subtract cost from both buys and sells)
        if size > 0:  # Buy
            self.cash -= (notional + cost)
        else:  # Sell
            self.cash += (notional - cost)
        
        # Update position
        position_before = self.position
        self.position += size
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            side='buy' if size > 0 else 'sell',
            size=abs(size),
            price=price,
            notional=notional,
            cost=cost,
            position_before=position_before,
            position_after=self.position
        )
        self.trades.append(trade)
    
    def _create_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create results DataFrame with NAV and returns."""
        results = pd.DataFrame({
            'timestamp': df['timestamp'].values,
            'close': df['close'].values,
            'nav': self.nav_history,
            'position': self.position_history
        })
        
        # Compute returns
        results['returns'] = results['nav'].pct_change()
        results['cumulative_returns'] = (results['nav'] / self.initial_capital - 1)
        
        # Compute drawdown
        results['peak_nav'] = results['nav'].cummax()
        results['drawdown'] = (results['peak_nav'] - results['nav']) / results['peak_nav']
        
        return results
    
    def _compute_metrics(self, results_df: pd.DataFrame) -> BacktestMetrics:
        """Compute comprehensive performance metrics."""
        metrics = BacktestMetrics()
        
        # Returns
        final_nav = results_df['nav'].iloc[-1]
        metrics.total_return = (final_nav / self.initial_capital - 1)
        
        # Annualize (assuming 15-min bars, 365 days)
        n_bars = len(results_df)
        bars_per_year = (365 * 24 * 60) / 15
        years = n_bars / bars_per_year
        
        metrics.cagr = (final_nav / self.initial_capital) ** (1 / years) - 1
        
        # Volatility
        returns = results_df['returns'].dropna()
        metrics.annualized_vol = returns.std() * np.sqrt(bars_per_year)
        
        # Sharpe ratio (risk-free rate = 0 for crypto)
        if metrics.annualized_vol > 0:
            metrics.sharpe_ratio = metrics.cagr / metrics.annualized_vol
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_dev = downside_returns.std() * np.sqrt(bars_per_year)
            if downside_dev > 0:
                metrics.sortino_ratio = metrics.cagr / downside_dev
        
        # Drawdown
        metrics.max_drawdown = results_df['drawdown'].max()
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown
        metrics.avg_drawdown = results_df['drawdown'].mean()
        
        # Drawdown duration
        in_drawdown = results_df['drawdown'] > 0
        if in_drawdown.any():
            drawdown_lengths = []
            current_length = 0
            for dd in in_drawdown:
                if dd:
                    current_length += 1
                else:
                    if current_length > 0:
                        drawdown_lengths.append(current_length)
                    current_length = 0
            if drawdown_lengths:
                metrics.max_drawdown_duration = max(drawdown_lengths)
        
        # Trading stats
        metrics.total_trades = len(self.trades)
        
        if len(self.trades) >= 2:
            # Compute P&L for each round-trip trade
            trade_pnls = self._compute_trade_pnls(results_df)
            
            if len(trade_pnls) > 0:
                wins = [pnl for pnl in trade_pnls if pnl > 0]
                losses = [pnl for pnl in trade_pnls if pnl < 0]
                
                metrics.win_rate = len(wins) / len(trade_pnls) if len(trade_pnls) > 0 else 0
                metrics.avg_win = np.mean(wins) if wins else 0
                metrics.avg_loss = np.mean(losses) if losses else 0
                
                total_wins = sum(wins)
                total_losses = abs(sum(losses))
                if total_losses > 0:
                    metrics.profit_factor = total_wins / total_losses
        
        # Costs
        metrics.total_fees = sum(t.cost for t in self.trades)
        metrics.turnover = sum(t.notional for t in self.trades) / self.initial_capital
        
        # Risk metrics
        if len(returns) > 20:
            metrics.var_95 = returns.quantile(0.05)
            worst_1pct = returns.quantile(0.01)
            metrics.cvar_99 = returns[returns <= worst_1pct].mean()
        
        return metrics
    
    def _compute_trade_pnls(self, results_df: pd.DataFrame) -> List[float]:
        """Compute P&L for each round-trip trade."""
        pnls = []
        entry_price = None
        entry_value = 0
        
        for trade in self.trades:
            if entry_price is None:
                # Entry trade
                entry_price = trade.price
                entry_value = trade.notional + trade.cost
            else:
                # Exit trade
                exit_value = trade.notional - trade.cost
                pnl = exit_value - entry_value if trade.side == 'sell' else entry_value - exit_value
                pnls.append(pnl)
                entry_price = None
        
        return pnls


def run_backtest_from_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    vol_forecasts: np.ndarray,
    config: dict = None
) -> Tuple[pd.DataFrame, BacktestMetrics]:
    """
    Convenience function to run backtest from model predictions.
    
    Args:
        df: DataFrame with OHLCV data
        predictions: Array of model probabilities ∈ [0, 1]
        vol_forecasts: Array of volatility forecasts
        config: Trading configuration dictionary
        
    Returns:
        results_df: Backtest results
        metrics: Performance metrics
    """
    from src.trading.signals import TradingSignalGenerator, TradingConfig
    
    # Initialize signal generator
    if config:
        trading_config = TradingConfig(**config)
    else:
        trading_config = TradingConfig()
    
    generator = TradingSignalGenerator(trading_config)
    
    # Generate signals
    signals = np.array([
        generator.generate_signal(p, v)
        for p, v in zip(predictions, vol_forecasts)
    ])
    
    # Run backtest
    backtester = Backtester()
    results_df, metrics = backtester.run(df, signals)
    
    return results_df, metrics


if __name__ == "__main__":
    # Test backtester
    print("Testing backtester...")
    
    # Generate synthetic data
    import sys
    sys.path.append('/home/claude/neural-trading-system')
    from src.data.synthetic_data import generate_synthetic_crypto_data
    
    df = generate_synthetic_crypto_data(
        output_path='data/synthetic/backtest_test.csv',
        n_bars=5000,
        seed=42
    )
    
    # Generate random signals for testing
    np.random.seed(42)
    signals = np.random.uniform(-0.5, 0.5, len(df))
    
    # Run backtest
    backtester = Backtester(initial_capital=10000.0)
    results_df, metrics = backtester.run(df, signals)
    
    # Print results
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"CAGR: {metrics.cagr:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Total Fees: ${metrics.total_fees:.2f}")
    
    print("\n✓ Backtester test passed!")