"""
Technical Indicators Calculator
Pure Python implementation - no C++ compilation required!
"""
import pandas as pd
import numpy as np
from typing import Tuple

# Try to import ta library, if not available use our own implementations
try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("⚠️ 'ta' library not found. Using built-in implementations.")
    print("Install with: pip install ta")

try:
    from utils.logger import get_logger
    logger = get_logger()
except:
    import logging
    logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for trading strategies"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Returns:
            macd_line, signal_line, histogram
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, 
                                 std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Returns:
            upper_band, middle_band, lower_band
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Average True Range (Volatility indicator)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 14, smooth_k: int = 3, 
                           smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        Returns:
            %K, %D
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k = k.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        
        return k, d
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Average Directional Index (Trend strength)"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = TechnicalIndicators.calculate_atr(high, low, close, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (np.abs(minus_dm).rolling(window=period).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20) -> dict:
        """
        Detect support and resistance levels
        
        Returns:
            Dict with support and resistance levels
        """
        close = data['close']
        
        # Find local minima (support)
        support = close.rolling(window=window, center=True).min()
        support_levels = close[close == support].unique()
        
        # Find local maxima (resistance)
        resistance = close.rolling(window=window, center=True).max()
        resistance_levels = close[close == resistance].unique()
        
        return {
            'support': sorted(support_levels[-3:]),  # Last 3 support levels
            'resistance': sorted(resistance_levels[-3:])  # Last 3 resistance levels
        }
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        logger.info(f"Calculating technical indicators for {len(df)} rows")
        
        # Moving Averages
        df['sma_20'] = TechnicalIndicators.calculate_sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.calculate_sma(df['close'], 50)
        df['sma_200'] = TechnicalIndicators.calculate_sma(df['close'], 200)
        df['ema_12'] = TechnicalIndicators.calculate_ema(df['close'], 12)
        df['ema_26'] = TechnicalIndicators.calculate_ema(df['close'], 26)
        
        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
        
        # MACD
        macd, signal, hist = TechnicalIndicators.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
        # ATR
        df['atr'] = TechnicalIndicators.calculate_atr(df['high'], df['low'], df['close'])
        
        # Stochastic
        k, d = TechnicalIndicators.calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k
        df['stoch_d'] = d
        
        # ADX
        df['adx'] = TechnicalIndicators.calculate_adx(df['high'], df['low'], df['close'])
        
        # OBV
        df['obv'] = TechnicalIndicators.calculate_obv(df['close'], df['volume'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['rate_of_change'] = df['close'].pct_change(periods=10)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        logger.info(f"Added {len(df.columns)} features including indicators")
        
        return df


# Example usage
if __name__ == '__main__':
    from data.data_fetcher import DataFetcher
    
    # Fetch data
    fetcher = DataFetcher()
    df = fetcher.get_historical_data('AAPL', period='6mo', interval='1d')
    
    # Add indicators
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Display results
    print("\nDataFrame with Technical Indicators:")
    print(df[['close', 'sma_20', 'sma_50', 'rsi', 'macd', 'bb_upper', 'bb_lower']].tail(10))
    
    # Check for signals
    latest = df.iloc[-1]
    print(f"\n--- Latest Signals for AAPL ---")
    print(f"RSI: {latest['rsi']:.2f} {'(Oversold)' if latest['rsi'] < 30 else '(Overbought)' if latest['rsi'] > 70 else ''}")
    print(f"MACD Histogram: {latest['macd_hist']:.4f}")
    print(f"Price vs SMA50: {((latest['close'] / latest['sma_50']) - 1) * 100:.2f}%")