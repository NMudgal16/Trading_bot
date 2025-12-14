"""
Technical Analysis Trading Strategy
Combines multiple technical indicators with intelligent signal generation
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
from data.data_processor import TechnicalIndicators
from utils.logger import get_logger

logger = get_logger()


class TechnicalStrategy(BaseStrategy):
    """
    Multi-indicator technical strategy
    Combines RSI, MACD, Bollinger Bands, Moving Averages, and Volume
    """
    
    def __init__(self, params: Dict = None):
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'sma_short': 50,
            'sma_long': 200,
            'volume_threshold': 1.5,  # 1.5x average volume
            'min_signal_strength': 0.6
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name="TechnicalStrategy", params=default_params)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals based on technical indicators
        
        Strategy Logic:
        - BUY when: RSI oversold + MACD bullish + price near lower BB + trend up
        - SELL when: RSI overbought + MACD bearish + price near upper BB + trend down
        """
        signals = []
        
        # Ensure data has indicators
        if 'rsi' not in data.columns:
            data = TechnicalIndicators.add_all_indicators(data)
        
        # Get latest data point
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        
        symbol = latest.get('symbol', 'UNKNOWN')
        
        # Calculate individual indicator signals
        rsi_signal = self._analyze_rsi(latest)
        macd_signal = self._analyze_macd(latest, prev)
        bb_signal = self._analyze_bollinger_bands(latest)
        ma_signal = self._analyze_moving_averages(latest)
        volume_signal = self._analyze_volume(latest)
        trend_signal = self._analyze_trend(data)
        
        # Combine signals with weights
        buy_score = 0.0
        sell_score = 0.0
        
        weights = {
            'rsi': 0.25,
            'macd': 0.25,
            'bb': 0.20,
            'ma': 0.15,
            'volume': 0.10,
            'trend': 0.05
        }
        
        # Accumulate weighted scores
        if rsi_signal == 1:
            buy_score += weights['rsi']
        elif rsi_signal == -1:
            sell_score += weights['rsi']
        
        if macd_signal == 1:
            buy_score += weights['macd']
        elif macd_signal == -1:
            sell_score += weights['macd']
        
        if bb_signal == 1:
            buy_score += weights['bb']
        elif bb_signal == -1:
            sell_score += weights['bb']
        
        if ma_signal == 1:
            buy_score += weights['ma']
        elif ma_signal == -1:
            sell_score += weights['ma']
        
        if volume_signal > 0:
            buy_score += weights['volume'] * volume_signal
        
        if trend_signal > 0:
            buy_score += weights['trend'] * trend_signal
        elif trend_signal < 0:
            sell_score += weights['trend'] * abs(trend_signal)
        
        # Generate signal based on scores
        min_strength = self.params['min_signal_strength']
        
        if buy_score > sell_score and buy_score >= min_strength:
            signal = self._create_buy_signal(latest, buy_score, {
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_hist': latest['macd_hist'],
                'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']),
                'sma_50': latest['sma_50'],
                'volume_ratio': latest['volume_ratio']
            })
            signals.append(signal)
            
        elif sell_score > buy_score and sell_score >= min_strength:
            signal = self._create_sell_signal(latest, sell_score, {
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_hist': latest['macd_hist'],
                'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']),
                'sma_50': latest['sma_50'],
                'volume_ratio': latest['volume_ratio']
            })
            signals.append(signal)
        
        # Store in history
        self.signals_history.extend(signals)
        
        return signals
    
    def _analyze_rsi(self, data: pd.Series) -> int:
        """
        Analyze RSI indicator
        Returns: 1 (buy), -1 (sell), 0 (neutral)
        """
        rsi = data['rsi']
        
        if pd.isna(rsi):
            return 0
        
        if rsi < self.params['rsi_oversold']:
            return 1  # Oversold - buy signal
        elif rsi > self.params['rsi_overbought']:
            return -1  # Overbought - sell signal
        
        return 0
    
    def _analyze_macd(self, current: pd.Series, previous: pd.Series) -> int:
        """
        Analyze MACD crossovers
        Returns: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        curr_hist = current['macd_hist']
        prev_hist = previous['macd_hist']
        
        if pd.isna(curr_hist) or pd.isna(prev_hist):
            return 0
        
        # Bullish crossover (histogram crosses above zero)
        if prev_hist < 0 and curr_hist > 0:
            return 1
        
        # Bearish crossover (histogram crosses below zero)
        if prev_hist > 0 and curr_hist < 0:
            return -1
        
        # Check momentum
        if curr_hist > 0 and curr_hist > prev_hist:
            return 1  # Increasing bullish momentum
        elif curr_hist < 0 and curr_hist < prev_hist:
            return -1  # Increasing bearish momentum
        
        return 0
    
    def _analyze_bollinger_bands(self, data: pd.Series) -> int:
        """
        Analyze Bollinger Bands position
        Returns: 1 (near lower band), -1 (near upper band), 0 (neutral)
        """
        close = data['close']
        lower = data['bb_lower']
        upper = data['bb_upper']
        
        if pd.isna(lower) or pd.isna(upper):
            return 0
        
        # Calculate position within bands (0 to 1)
        position = (close - lower) / (upper - lower)
        
        if position < 0.2:  # Near lower band
            return 1
        elif position > 0.8:  # Near upper band
            return -1
        
        return 0
    
    def _analyze_moving_averages(self, data: pd.Series) -> int:
        """
        Analyze moving average crossovers
        Returns: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        close = data['close']
        sma_50 = data['sma_50']
        sma_200 = data['sma_200']
        
        if pd.isna(sma_50) or pd.isna(sma_200):
            return 0
        
        # Golden cross: SMA50 > SMA200 and price > SMA50
        if sma_50 > sma_200 and close > sma_50:
            return 1
        
        # Death cross: SMA50 < SMA200 and price < SMA50
        if sma_50 < sma_200 and close < sma_50:
            return -1
        
        return 0
    
    def _analyze_volume(self, data: pd.Series) -> float:
        """
        Analyze volume confirmation
        Returns: 0.0 to 1.0 (volume confirmation strength)
        """
        volume_ratio = data['volume_ratio']
        
        if pd.isna(volume_ratio):
            return 0.0
        
        # High volume confirms signal
        if volume_ratio > self.params['volume_threshold']:
            return min(volume_ratio / self.params['volume_threshold'] - 1, 1.0)
        
        return 0.0
    
    def _analyze_trend(self, data: pd.DataFrame, window: int = 10) -> float:
        """
        Analyze overall trend strength
        Returns: -1.0 to 1.0 (negative = downtrend, positive = uptrend)
        """
        recent = data.tail(window)
        
        if len(recent) < window:
            return 0.0
        
        # Calculate trend using linear regression slope
        x = np.arange(len(recent))
        y = recent['close'].values
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope
        avg_price = y.mean()
        normalized_slope = slope / avg_price * 100  # As percentage
        
        return np.clip(normalized_slope, -1.0, 1.0)
    
    def _create_buy_signal(self, data: pd.Series, strength: float, 
                          indicators: Dict) -> TradingSignal:
        """Create a BUY signal"""
        reasons = []
        
        if indicators['rsi'] < self.params['rsi_oversold']:
            reasons.append(f"RSI oversold ({indicators['rsi']:.1f})")
        
        if indicators['macd_hist'] > 0:
            reasons.append("MACD bullish")
        
        if indicators['bb_position'] < 0.2:
            reasons.append("Near lower BB")
        
        if data['close'] > indicators['sma_50']:
            reasons.append("Above SMA50")
        
        if indicators['volume_ratio'] > self.params['volume_threshold']:
            reasons.append(f"High volume ({indicators['volume_ratio']:.1f}x)")
        
        reason = " + ".join(reasons) if reasons else "Technical buy signal"
        
        return TradingSignal(
            symbol=data.get('symbol', 'UNKNOWN'),
            signal=SignalType.BUY,
            strength=strength,
            price=data['close'],
            timestamp=data.name,
            reason=reason,
            indicators=indicators,
            confidence=strength
        )
    
    def _create_sell_signal(self, data: pd.Series, strength: float,
                           indicators: Dict) -> TradingSignal:
        """Create a SELL signal"""
        reasons = []
        
        if indicators['rsi'] > self.params['rsi_overbought']:
            reasons.append(f"RSI overbought ({indicators['rsi']:.1f})")
        
        if indicators['macd_hist'] < 0:
            reasons.append("MACD bearish")
        
        if indicators['bb_position'] > 0.8:
            reasons.append("Near upper BB")
        
        if data['close'] < indicators['sma_50']:
            reasons.append("Below SMA50")
        
        reason = " + ".join(reasons) if reasons else "Technical sell signal"
        
        return TradingSignal(
            symbol=data.get('symbol', 'UNKNOWN'),
            signal=SignalType.SELL,
            strength=strength,
            price=data['close'],
            timestamp=data.name,
            reason=reason,
            indicators=indicators,
            confidence=strength
        )
    
    def calculate_position_size(self, signal: TradingSignal,
                               portfolio_value: float,
                               current_price: float) -> int:
        """
        Calculate position size based on signal strength and risk
        """
        # Base position size (as percentage of portfolio)
        base_allocation = 0.10  # 10% of portfolio
        
        # Adjust based on signal strength
        adjusted_allocation = base_allocation * signal.strength
        
        # Calculate dollar amount
        position_value = portfolio_value * adjusted_allocation
        
        # Calculate number of shares
        shares = int(position_value / current_price)
        
        return max(shares, 0)


# Example usage
if __name__ == '__main__':
    from data.data_fetcher import DataFetcher
    
    # Fetch data
    fetcher = DataFetcher()
    df = fetcher.get_historical_data('AAPL', period='3mo', interval='1d')
    
    # Create strategy
    strategy = TechnicalStrategy()
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    print(f"\n--- Generated {len(signals)} signals ---")
    for signal in signals:
        print(f"\n{signal.signal.value} {signal.symbol} @ ${signal.price:.2f}")
        print(f"Strength: {signal.strength:.2%}")
        print(f"Reason: {signal.reason}")
        print(f"Indicators: {signal.indicators}")