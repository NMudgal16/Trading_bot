"""
Position Sizing - Calculate optimal position sizes based on risk
"""
import numpy as np
from typing import Optional
from strategies.base_strategy import TradingSignal
from utils.logger import get_logger

logger = get_logger()


class PositionSizer:
    """
    Calculate position sizes using various methods:
    1. Fixed Percentage
    2. Risk-Based (Kelly Criterion)
    3. Volatility-Adjusted
    4. Signal Strength Weighted
    """
    
    def __init__(self, max_position_size: float = 0.10, max_risk_per_trade: float = 0.02):
        """
        Initialize position sizer
        
        Args:
            max_position_size: Maximum position as % of portfolio (e.g., 0.10 = 10%)
            max_risk_per_trade: Maximum risk per trade as % of portfolio (e.g., 0.02 = 2%)
        """
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        
        logger.info(
            f"Position Sizer initialized | Max Position: {max_position_size:.1%} | "
            f"Max Risk: {max_risk_per_trade:.1%}"
        )
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float,
                               current_price: float, volatility: float = 0.02,
                               method: str = 'signal_weighted') -> int:
        """
        Calculate optimal position size
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current price of asset
            volatility: Asset volatility (annualized)
            method: 'fixed', 'risk_based', 'volatility_adjusted', 'signal_weighted'
            
        Returns:
            Number of shares to trade
        """
        if method == 'fixed':
            shares = self._fixed_percentage(portfolio_value, current_price)
        elif method == 'risk_based':
            shares = self._risk_based(portfolio_value, current_price, volatility)
        elif method == 'volatility_adjusted':
            shares = self._volatility_adjusted(portfolio_value, current_price, volatility)
        elif method == 'signal_weighted':
            shares = self._signal_weighted(signal, portfolio_value, current_price, volatility)
        else:
            logger.warning(f"Unknown method '{method}', using signal_weighted")
            shares = self._signal_weighted(signal, portfolio_value, current_price, volatility)
        
        # Ensure we don't exceed maximum position size
        max_shares = int((portfolio_value * self.max_position_size) / current_price)
        shares = min(shares, max_shares)
        
        # Must buy at least 1 share
        shares = max(shares, 1)
        
        logger.debug(
            f"Position size calculated for {signal.symbol}: {shares} shares "
            f"(${shares * current_price:,.2f})"
        )
        
        return shares
    
    def _fixed_percentage(self, portfolio_value: float, current_price: float) -> int:
        """
        Fixed percentage of portfolio
        Simple but doesn't account for risk
        """
        position_value = portfolio_value * self.max_position_size
        shares = int(position_value / current_price)
        return shares
    
    def _risk_based(self, portfolio_value: float, current_price: float,
                   volatility: float) -> int:
        """
        Risk-based sizing using volatility
        Invests more when volatility is low, less when high
        """
        # Calculate dollar risk
        max_dollar_risk = portfolio_value * self.max_risk_per_trade
        
        # Estimate potential loss per share (using 2 standard deviations)
        price_risk_per_share = current_price * volatility * 2
        
        # Calculate shares based on risk
        if price_risk_per_share > 0:
            shares = int(max_dollar_risk / price_risk_per_share)
        else:
            shares = int((portfolio_value * self.max_position_size) / current_price)
        
        return shares
    
    def _volatility_adjusted(self, portfolio_value: float, current_price: float,
                            volatility: float) -> int:
        """
        Adjust position size inversely with volatility
        Lower volatility = larger position, higher volatility = smaller position
        """
        # Base position
        base_shares = int((portfolio_value * self.max_position_size) / current_price)
        
        # Volatility adjustment (inverse relationship)
        # Normalize volatility (assume 2% is "normal")
        normal_vol = 0.02
        vol_adjustment = normal_vol / max(volatility, 0.005)  # Avoid division by zero
        
        # Clip adjustment to reasonable range [0.5, 1.5]
        vol_adjustment = np.clip(vol_adjustment, 0.5, 1.5)
        
        adjusted_shares = int(base_shares * vol_adjustment)
        
        return adjusted_shares
    
    def _signal_weighted(self, signal: TradingSignal, portfolio_value: float,
                        current_price: float, volatility: float) -> int:
        """
        Weight position size by signal strength and volatility
        Combines signal confidence with risk management
        """
        # Start with volatility-adjusted size
        base_shares = self._volatility_adjusted(portfolio_value, current_price, volatility)
        
        # Adjust by signal strength
        # Strong signals (0.8-1.0) get full position
        # Weak signals (0.5-0.6) get reduced position
        strength_multiplier = signal.strength
        
        # Adjust by confidence if available
        if signal.confidence > 0:
            confidence_multiplier = signal.confidence
        else:
            confidence_multiplier = 1.0
        
        # Combined adjustment
        adjustment = strength_multiplier * confidence_multiplier
        
        adjusted_shares = int(base_shares * adjustment)
        
        return adjusted_shares
    
    def calculate_stop_loss(self, entry_price: float, atr: float = None,
                           pct_stop: float = 0.05) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            atr: Average True Range (if available)
            pct_stop: Percentage stop loss (e.g., 0.05 = 5%)
            
        Returns:
            Stop loss price
        """
        if atr:
            # ATR-based stop (2x ATR below entry)
            stop_loss = entry_price - (2 * atr)
        else:
            # Percentage-based stop
            stop_loss = entry_price * (1 - pct_stop)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                             risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price based on risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_reward_ratio: Reward/Risk ratio (e.g., 2.0 = 2:1)
            
        Returns:
            Take profit price
        """
        risk_per_share = entry_price - stop_loss
        reward_per_share = risk_per_share * risk_reward_ratio
        take_profit = entry_price + reward_per_share
        
        return take_profit
    
    def validate_trade_risk(self, shares: int, entry_price: float,
                           stop_loss: float, portfolio_value: float) -> bool:
        """
        Validate that trade risk is within acceptable limits
        
        Returns:
            True if trade passes risk validation
        """
        # Calculate maximum loss
        max_loss_per_share = entry_price - stop_loss
        total_max_loss = shares * max_loss_per_share
        
        # Calculate risk as percentage of portfolio
        risk_pct = total_max_loss / portfolio_value
        
        if risk_pct > self.max_risk_per_trade:
            logger.warning(
                f"Trade risk ({risk_pct:.2%}) exceeds maximum ({self.max_risk_per_trade:.2%})"
            )
            return False
        
        logger.debug(f"Trade risk validated: {risk_pct:.2%} of portfolio")
        return True


# Example usage
if __name__ == '__main__':
    from strategies.base_strategy import TradingSignal, SignalType
    import pandas as pd
    
    # Create position sizer
    sizer = PositionSizer(max_position_size=0.10, max_risk_per_trade=0.02)
    
    # Create example signal
    signal = TradingSignal(
        symbol='AAPL',
        signal=SignalType.BUY,
        strength=0.75,
        price=150.0,
        timestamp=pd.Timestamp.now(),
        reason='Test signal',
        indicators={},
        confidence=0.8
    )
    
    # Calculate position size
    portfolio_value = 100000
    current_price = 150.0
    volatility = 0.02  # 2% volatility
    
    print("\n=== Position Sizing Examples ===\n")
    
    methods = ['fixed', 'risk_based', 'volatility_adjusted', 'signal_weighted']
    
    for method in methods:
        shares = sizer.calculate_position_size(
            signal, portfolio_value, current_price, volatility, method
        )
        position_value = shares * current_price
        position_pct = (position_value / portfolio_value) * 100
        
        print(f"{method:20} | {shares:4} shares | ${position_value:>10,.2f} ({position_pct:.2f}%)")
    
    # Calculate stop loss and take profit
    print("\n=== Risk Management ===\n")
    stop_loss = sizer.calculate_stop_loss(current_price, atr=3.0)
    take_profit = sizer.calculate_take_profit(current_price, stop_loss, risk_reward_ratio=2.0)
    
    print(f"Entry Price:   ${current_price:.2f}")
    print(f"Stop Loss:     ${stop_loss:.2f} ({((stop_loss/current_price - 1)*100):.2f}%)")
    print(f"Take Profit:   ${take_profit:.2f} ({((take_profit/current_price - 1)*100):.2f}%)")
    print(f"Risk/Reward:   1:2")