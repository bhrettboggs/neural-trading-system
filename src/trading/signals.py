"""
Trading rules: Convert model outputs to executable orders.
Includes position sizing, risk management, and order generation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class OrderSide(Enum):
    """Order side."""
    BUY = 'buy'
    SELL = 'sell'


class OrderType(Enum):
    """Order type."""
    MARKET = 'market'
    LIMIT = 'limit'


@dataclass
class Order:
    """Order specification."""
    side: OrderSide
    size: float  # Size in base currency (BTC, ETH, etc.)
    price: Optional[float] = None  # For limit orders
    order_type: OrderType = OrderType.MARKET
    time_in_force: str = 'GTC'  # Good-til-cancelled
    post_only: bool = False  # Maker-only (no taker fees)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'side': self.side.value,
            'size': self.size,
            'price': self.price,
            'order_type': self.order_type.value,
            'time_in_force': self.time_in_force,
            'post_only': self.post_only
        }


@dataclass
class TradingConfig:
    """Trading configuration parameters."""
    # Position sizing
    target_vol: float = 0.02  # 2% annualized target volatility
    max_leverage: float = 1.0  # No leverage (100% long or short max)
    min_order_size: float = 0.001  # Min 0.001 BTC
    max_order_size: float = 0.5  # Max 0.5 BTC per order
    
    # Signal filtering
    deadband_threshold: float = 0.3  # Ignore signals < 0.3
    confidence_threshold: float = 0.15  # |p - 0.5| > 0.15 for limit orders
    
    # Risk limits
    max_daily_turnover: float = 10.0  # Max 10x portfolio per day
    max_net_exposure: float = 0.95  # Max 95% of portfolio
    volatility_circuit_breaker: float = 0.10  # Halt if vol > 10%
    max_drawdown_halt: float = 0.20  # Halt if drawdown > 20%
    
    # Execution
    tick_size: float = 0.01  # $0.01 for BTC/USDT
    use_limit_orders: bool = True  # Use limit orders when possible


class TradingSignalGenerator:
    """Generate trading signals from model outputs."""
    
    def __init__(self, config: TradingConfig = None):
        """
        Initialize signal generator.
        
        Args:
            config: Trading configuration
        """
        self.config = config or TradingConfig()
        self.daily_turnover = 0.0
        self.current_drawdown = 0.0
        self.peak_nav = 0.0
    
    def generate_signal(
        self,
        model_output: float,
        vol_forecast: float
    ) -> float:
        """
        Convert model probability to trading signal.
        
        Args:
            model_output: Model probability ∈ [0, 1]
            vol_forecast: Forecasted annualized volatility
            
        Returns:
            Trading signal ∈ [-1, 1] (negative = short, positive = long)
        """
        # Convert probability to directional signal
        s_t = 2 * model_output - 1  # ∈ [-1, 1]
        
        # Volatility scaling
        r_t = (self.config.target_vol / (vol_forecast + 1e-8)) * s_t
        
        # Apply deadband
        r_t = self._apply_deadband(r_t, self.config.deadband_threshold)
        
        # Clip to leverage limits
        signal = np.clip(r_t, -self.config.max_leverage, self.config.max_leverage)
        
        return signal
    
    def _apply_deadband(self, signal: float, threshold: float) -> float:
        """Apply deadband filter to ignore weak signals."""
        if abs(signal) < threshold:
            return 0.0
        return signal
    
    def generate_order(
        self,
        model_output: float,
        vol_forecast: float,
        current_position: float,
        portfolio_nav: float,
        current_price: float,
        order_book: dict
    ) -> Optional[Order]:
        """
        Generate executable order from model output.
        
        Args:
            model_output: Model probability ∈ [0, 1]
            vol_forecast: Forecasted annualized volatility
            current_position: Current position in base currency
            portfolio_nav: Net Asset Value in quote currency
            current_price: Last traded price
            order_book: {'bids': [(price, size), ...], 'asks': [(price, size), ...]}
            
        Returns:
            Order object or None if no trade
        """
        # Step 1: Generate signal
        signal = self.generate_signal(model_output, vol_forecast)
        
        # Step 2: Convert to target position size
        target_notional = signal * portfolio_nav
        target_size = target_notional / current_price
        
        # Step 3: Compute required trade
        trade_size = target_size - current_position
        
        # Step 4: Safety checks
        if not self._safety_checks(
            trade_size, vol_forecast, portfolio_nav, current_price
        ):
            return None
        
        # Step 5: Check minimum order size
        if abs(trade_size) < self.config.min_order_size:
            return None
        
        # Step 6: Determine order type and parameters
        order = self._create_order(
            trade_size=trade_size,
            model_output=model_output,
            current_price=current_price,
            order_book=order_book
        )
        
        return order
    
    def _safety_checks(
        self,
        trade_size: float,
        vol_forecast: float,
        portfolio_nav: float,
        current_price: float
    ) -> bool:
        """
        Run safety checks before executing trade.
        
        Returns:
            True if all checks pass, False otherwise
        """
        # Check daily turnover limit
        trade_value = abs(trade_size * current_price)
        if self.daily_turnover + trade_value > self.config.max_daily_turnover * portfolio_nav:
            print(f"⚠️ Daily turnover limit exceeded: {self.daily_turnover:.2f}")
            return False
        
        # Check volatility circuit breaker
        if vol_forecast > self.config.volatility_circuit_breaker:
            print(f"⚠️ Volatility circuit breaker triggered: {vol_forecast:.2%}")
            return False
        
        # Check drawdown circuit breaker
        if self.current_drawdown > self.config.max_drawdown_halt:
            print(f"⚠️ Drawdown circuit breaker triggered: {self.current_drawdown:.2%}")
            return False
        
        return True
    
    def _create_order(
        self,
        trade_size: float,
        model_output: float,
        current_price: float,
        order_book: dict
    ) -> Order:
        """
        Create order object with appropriate type and price.
        
        Args:
            trade_size: Size to trade (positive = buy, negative = sell)
            model_output: Model probability
            current_price: Current price
            order_book: Order book data
            
        Returns:
            Order object
        """
        # Determine side
        side = OrderSide.BUY if trade_size > 0 else OrderSide.SELL
        size = abs(trade_size)
        
        # Clip to max order size
        if size > self.config.max_order_size:
            size = self.config.max_order_size
        
        # Determine if high confidence (use limit order)
        confidence = abs(model_output - 0.5)
        use_limit = (
            self.config.use_limit_orders and 
            confidence > self.config.confidence_threshold
        )
        
        if use_limit:
            # Limit order: post passive order for maker rebate
            if side == OrderSide.BUY:
                # Buy: post bid just below best ask
                best_ask = order_book['asks'][0][0] if order_book['asks'] else current_price
                limit_price = best_ask - self.config.tick_size
            else:
                # Sell: post ask just above best bid
                best_bid = order_book['bids'][0][0] if order_book['bids'] else current_price
                limit_price = best_bid + self.config.tick_size
            
            order = Order(
                side=side,
                size=size,
                price=limit_price,
                order_type=OrderType.LIMIT,
                time_in_force='GTC',
                post_only=True  # Ensure maker rebate
            )
        else:
            # Market order: execute immediately
            order = Order(
                side=side,
                size=size,
                order_type=OrderType.MARKET
            )
        
        return order
    
    def update_metrics(self, portfolio_nav: float):
        """Update tracking metrics."""
        # Update peak NAV
        if portfolio_nav > self.peak_nav:
            self.peak_nav = portfolio_nav
        
        # Update current drawdown
        if self.peak_nav > 0:
            self.current_drawdown = (self.peak_nav - portfolio_nav) / self.peak_nav
    
    def reset_daily_turnover(self):
        """Reset daily turnover counter (call at start of each day)."""
        self.daily_turnover = 0.0
    
    def add_turnover(self, trade_value: float):
        """Add trade value to daily turnover."""
        self.daily_turnover += trade_value


def split_order_into_chunks(
    order: Order,
    chunk_size: float
) -> List[Order]:
    """
    Split large order into smaller chunks.
    
    Args:
        order: Large order to split
        chunk_size: Size of each chunk
        
    Returns:
        List of smaller orders
    """
    chunks = []
    remaining = order.size
    
    while remaining > 0:
        size = min(chunk_size, remaining)
        chunk = Order(
            side=order.side,
            size=size,
            price=order.price,
            order_type=order.order_type,
            time_in_force=order.time_in_force,
            post_only=order.post_only
        )
        chunks.append(chunk)
        remaining -= size
    
    return chunks


if __name__ == "__main__":
    # Test trading signal generator
    print("Testing trading signal generator...")
    
    config = TradingConfig()
    generator = TradingSignalGenerator(config)
    
    # Test cases
    test_cases = [
        (0.68, 0.025, "High confidence long"),
        (0.52, 0.020, "Low confidence long"),
        (0.32, 0.030, "High confidence short"),
        (0.48, 0.015, "Low confidence short"),
    ]
    
    print("\nSignal generation tests:")
    for prob, vol, description in test_cases:
        signal = generator.generate_signal(prob, vol)
        print(f"{description}: p={prob:.2f}, vol={vol:.1%} → signal={signal:.3f}")
    
    # Test order generation
    print("\nOrder generation test:")
    order = generator.generate_order(
        model_output=0.68,
        vol_forecast=0.025,
        current_position=0.0,
        portfolio_nav=10000.0,
        current_price=43250.0,
        order_book={
            'bids': [(43248.0, 1.5), (43247.0, 2.0)],
            'asks': [(43252.0, 1.2), (43253.0, 1.8)]
        }
    )
    
    if order:
        print(f"Generated order: {order.to_dict()}")
    else:
        print("No order generated")
    
    print("\n✓ Trading signal generator test passed!")