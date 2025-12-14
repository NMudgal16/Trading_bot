"""
Base Strategy Class - Abstract interface for all trading strategies
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from utils.logger import get_logger

logger = get_logger()


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    symbol: str
    signal: SignalType
    strength: float  # 0.0 to 1.0
    price: float
    timestamp: pd.Timestamp
    reason: str
    indicators: Dict[str, float]
    confidence: float = 0.0


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize strategy
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.signals_history = []
        
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals from market data
        
        Args:
            data: DataFrame with OHLCV and indicators
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, 
                               portfolio_value: float,
                               current_price: float) -> int:
        """
        Calculate position size for a signal
        
        Args:
            signal: Trading signal
            portfolio_value: Total portfolio value
            current_price: Current asset price
            
        Returns:
            Number of shares to trade
        """
        pass
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate if signal meets minimum requirements
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if signal is valid
        """
        # Minimum strength threshold
        if signal.strength < 0.5:
            return False
        
        # Must have a valid price
        if signal.price <= 0:
            return False
        
        return True
    
    def get_strategy_stats(self) -> Dict:
        """Get strategy performance statistics"""
        if not self.signals_history:
            return {}
        
        buy_signals = [s for s in self.signals_history if s.signal == SignalType.BUY]
        sell_signals = [s for s in self.signals_history if s.signal == SignalType.SELL]
        
        return {
            'total_signals': len(self.signals_history),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_buy_strength': sum(s.strength for s in buy_signals) / len(buy_signals) if buy_signals else 0,
            'avg_sell_strength': sum(s.strength for s in sell_signals) / len(sell_signals) if sell_signals else 0,
            'avg_confidence': sum(s.confidence for s in self.signals_history) / len(self.signals_history)
        }
    
    def reset(self):
        """Reset strategy state"""
        self.signals_history.clear()
        logger.info(f"Reset strategy: {self.name}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class StrategyEnsemble:
    """Combine multiple strategies with weighted voting"""
    
    def __init__(self, strategies: List[Tuple[BaseStrategy, float]]):
        """
        Initialize ensemble
        
        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies
        self.weights = [w for _, w in strategies]
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Created ensemble with {len(strategies)} strategies")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate ensemble signals by combining multiple strategies
        
        Returns:
            Combined trading signals
        """
        all_signals = {}
        
        # Collect signals from all strategies
        for (strategy, _), weight in zip(self.strategies, self.weights):
            signals = strategy.generate_signals(data)
            
            for signal in signals:
                if signal.symbol not in all_signals:
                    all_signals[signal.symbol] = {
                        'buy_votes': 0.0,
                        'sell_votes': 0.0,
                        'signals': []
                    }
                
                all_signals[signal.symbol]['signals'].append(signal)
                
                if signal.signal == SignalType.BUY:
                    all_signals[signal.symbol]['buy_votes'] += weight * signal.strength
                elif signal.signal == SignalType.SELL:
                    all_signals[signal.symbol]['sell_votes'] += weight * signal.strength
        
        # Create ensemble signals
        ensemble_signals = []
        
        for symbol, votes in all_signals.items():
            buy_score = votes['buy_votes']
            sell_score = votes['sell_votes']
            
            # Determine final signal
            if buy_score > sell_score and buy_score > 0.5:
                signal_type = SignalType.BUY
                strength = buy_score
            elif sell_score > buy_score and sell_score > 0.5:
                signal_type = SignalType.SELL
                strength = sell_score
            else:
                continue  # No clear signal
            
            # Get latest data point
            latest = data.iloc[-1] if isinstance(data, pd.DataFrame) else data[symbol].iloc[-1]
            
            # Create ensemble signal
            ensemble_signal = TradingSignal(
                symbol=symbol,
                signal=signal_type,
                strength=strength,
                price=latest['close'],
                timestamp=latest.name,
                reason=f"Ensemble ({len(votes['signals'])} strategies)",
                indicators={},
                confidence=strength
            )
            
            ensemble_signals.append(ensemble_signal)
        
        logger.info(f"Ensemble generated {len(ensemble_signals)} signals")
        return ensemble_signals
    
    def get_ensemble_stats(self) -> Dict:
        """Get statistics for all strategies in ensemble"""
        stats = {}
        for strategy, weight in self.strategies:
            stats[strategy.name] = {
                'weight': weight,
                'stats': strategy.get_strategy_stats()
            }
        return stats


# Example usage
if __name__ == '__main__':
    # This is an abstract class, so we can't instantiate it directly
    # We'll create concrete implementations in separate files
    print("Base Strategy Module Loaded Successfully")
    print(f"Available Signal Types: {[s.value for s in SignalType]}")