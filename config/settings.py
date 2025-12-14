"""
Configuration Settings for Trading Bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    
    # Trading Parameters
    INITIAL_CAPITAL = 10000.0
    MAX_POSITION_SIZE = 0.1  # Max 10% per position
    MAX_TOTAL_RISK = 0.02    # Max 2% risk per trade
    STOP_LOSS_PCT = 0.05     # 5% stop loss
    TAKE_PROFIT_PCT = 0.15   # 15% take profit
    
    # Trading Mode
    PAPER_TRADING = True     # Set to False for live trading
    
    # Assets to Trade
    SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'JPM', 'V', 'WMT'
    ]
    
    # Timeframes
    TIMEFRAME = '1d'         # 1d, 1h, 15m, 5m
    LOOKBACK_DAYS = 365      # Historical data to fetch
    
    # Technical Indicators
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    BB_PERIOD = 20
    BB_STD = 2
    
    SMA_SHORT = 50
    SMA_LONG = 200
    
    # Machine Learning
    ML_ENABLED = True
    LSTM_LOOKBACK = 60       # 60 days for LSTM
    PREDICTION_HORIZON = 5   # Predict 5 days ahead
    MODEL_RETRAIN_DAYS = 30  # Retrain model every 30 days
    
    # Sentiment Analysis
    SENTIMENT_ENABLED = True
    SENTIMENT_WEIGHT = 0.3   # 30% weight to sentiment
    NEWS_SOURCES = [
        'https://newsapi.org/v2/everything',
        'https://finance.yahoo.com/rss/'
    ]
    
    # Risk Management
    MAX_DRAWDOWN = 0.20      # Stop trading if 20% drawdown
    DAILY_LOSS_LIMIT = 0.05  # Max 5% loss per day
    
    # Portfolio Optimization
    REBALANCE_FREQUENCY = 'weekly'  # daily, weekly, monthly
    OPTIMIZATION_METHOD = 'sharpe'  # sharpe, min_variance, max_return
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
    
    # API Keys (load from environment)
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'trading_bot.log'
    
    # Dashboard
    DASHBOARD_HOST = '0.0.0.0'
    DASHBOARD_PORT = 5000
    ENABLE_WEBSOCKET = True
    
    # Notifications
    ENABLE_EMAIL = False
    ENABLE_TELEGRAM = False
    EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Backtesting
    BACKTEST_START_DATE = '2020-01-01'
    BACKTEST_END_DATE = '2024-12-01'
    BACKTEST_COMMISSION = 0.001  # 0.1% commission
    
    # Performance Metrics
    BENCHMARK_SYMBOL = 'SPY'  # S&P 500 ETF for comparison
    RISK_FREE_RATE = 0.04     # 4% annual risk-free rate


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    PAPER_TRADING = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    PAPER_TRADING = False  # WARNING: Real money!
    LOG_LEVEL = 'INFO'


# Select configuration
config = DevelopmentConfig() if os.getenv('ENV') == 'dev' else ProductionConfig()