import os
from pathlib import Path

def create_file_contents():
    """Create basic content for all project files"""
    
    # 1. config/settings.py
    settings_content = '''import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Trading settings
DEFAULT_SYMBOL = "BTC-USDT"
TIME_FRAME = "1h"
INITIAL_CAPITAL = 10000.0

# API settings
YAHOO_API_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
BINANCE_API_URL = "https://api.binance.com/api/v3"

# Model settings
TRAIN_TEST_SPLIT = 0.8
SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 5

# Risk settings
MAX_POSITION_SIZE = 0.1  # 10% of portfolio
STOP_LOSS_PCT = 0.02     # 2%
TAKE_PROFIT_PCT = 0.05   # 5%
MAX_DRAWDOWN = 0.15      # 15%

# Dashboard settings
DASHBOARD_PORT = 5000
DASHBOARD_HOST = "0.0.0.0"
'''
    
    # 2. config/api_keys.py
    api_keys_content = '''"""
API Keys Configuration
IMPORTANT: Never commit this file to version control!
Add your actual keys in a .env file or environment variables.
"""

import os

# Load from environment variables (recommended)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# Alternative: Load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
except ImportError:
    pass

# News API
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Alert system
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trading_bot.db")
'''
    
    # 3. main.py
    main_content = '''#!/usr/bin/env python3
"""
Main entry point for the AI Trading Bot
"""

import logging
from utils.logger import setup_logging
from config.settings import BASE_DIR

def main():
    """Main application entry point"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting AI Trading Bot...")
    
    print("\\n" + "="*50)
    print("AI-Powered Trading Bot")
    print("="*50)
    print("\\nProject Structure Created Successfully!")
    print(f"Project Directory: {BASE_DIR}")
    print("\\nAvailable Modules:")
    print("1. Data Collection & Processing")
    print("2. Strategy Development")
    print("3. Machine Learning Models")
    print("4. Risk Management")
    print("5. Backtesting Engine")
    print("6. Trading Dashboard")
    print("\\nRun 'python -m pytest' to run tests")
    print("="*50)
    
    # TODO: Add your main trading logic here
    logger.info("Trading bot initialized successfully")
    
    return 0

if __name__ == "__main__":
    exit(main())
'''
    
    # 4. utils/logger.py
    logger_content = '''import logging
import sys
from pathlib import Path
from loguru import logger
from config.settings import LOG_DIR

def setup_logging():
    """Configure logging for the application"""
    
    # Remove default logger
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # File logging
    log_file = LOG_DIR / "trading_bot.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Error logging
    error_log = LOG_DIR / "errors.log"
    logger.add(
        error_log,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="90 days"
    )
    
    return logger
'''
    
    # 5. requirements.txt
    requirements_content = '''# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.13.0

# Data fetching
yfinance>=0.2.28
python-binance>=1.0.19
requests>=2.31.0
websocket-client>=1.6.0

# Technical analysis
ta-lib>=0.4.28
ta>=0.10.2

# Web framework
flask>=2.3.0
flask-cors>=4.0.0
flask-socketio>=5.3.0

# Data processing
beautifulsoup4>=4.12.0
textblob>=0.17.0
transformers>=4.30.0
nltk>=3.8.0

# Visualization
plotly>=5.15.0
dash>=2.11.0
matplotlib>=3.7.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
schedule>=1.2.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
'''
    
    # 6. data/data_fetcher.py
    data_fetcher_content = '''import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetch market data from various sources"""
    
    def __init__(self):
        self.cache = {}
        
    def fetch_yahoo_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            symbol: Stock/crypto symbol (e.g., "BTC-USD")
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol} (period: {period}, interval: {interval})")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        data = {}
        for symbol in symbols:
            df = self.fetch_yahoo_data(symbol, **kwargs)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.1)  # Rate limiting
        
        return data
    
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """Fetch real-time data (placeholder for real implementation)"""
        # This would connect to websocket for real-time data
        return {
            'symbol': symbol,
            'price': 0.0,
            'timestamp': datetime.now(),
            'volume': 0
        }
'''
    
    # File paths and contents mapping
    files_to_create = {
        'config/settings.py': settings_content,
        'config/api_keys.py': api_keys_content,
        'main.py': main_content,
        'utils/logger.py': logger_content,
        'requirements.txt': requirements_content,
        'data/data_fetcher.py': data_fetcher_content,
        
        # Empty template files
        'data/data_processor.py': '# Data processing and cleaning module\n',
        'data/sentiment_analyzer.py': '# Sentiment analysis module\n',
        'strategies/base_strategy.py': '# Base strategy abstract class\n',
        'strategies/technical_strategy.py': '# Technical analysis strategies\n',
        'strategies/ml_strategy.py': '# Machine learning strategies\n',
        'strategies/hybrid_strategy.py': '# Hybrid strategies\n',
        'models/lstm_model.py': '# LSTM model for price prediction\n',
        'models/ensemble_model.py': '# Ensemble learning models\n',
        'execution/broker.py': '# Broker interface module\n',
        'execution/order_manager.py': '# Order execution module\n',
        'execution/paper_trading.py': '# Paper trading simulation\n',
        'risk_management/position_sizer.py': '# Position sizing algorithms\n',
        'risk_management/portfolio_optimizer.py': '# Portfolio optimization\n',
        'risk_management/risk_calculator.py': '# Risk metrics calculation\n',
        'backtesting/backtest_engine.py': '# Backtesting framework\n',
        'backtesting/performance_metrics.py': '# Performance metrics\n',
        'dashboard/backend/app.py': '# Flask/FastAPI dashboard backend\n',
        'dashboard/backend/websocket.py': '# WebSocket for real-time updates\n',
        'utils/database.py': '# Database operations\n',
        'utils/notifications.py': '# Alert and notification system\n',
        'tests/test_strategies.py': '# Strategy tests\n',
        'tests/test_risk_management.py': '# Risk management tests\n',
        'tests/test_backtesting.py': '# Backtesting tests\n',
    }
    
    # Create all files with content
    print("Creating file contents...")
    for file_path, content in files_to_create.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Created: {file_path}")
        except Exception as e:
            print(f"✗ Error creating {file_path}: {e}")
    
    print("\\n✅ All files created successfully!")
    
    # Create .env.example
    env_example = '''# API Keys Configuration
# Copy this file to .env and fill in your actual keys

# Binance API
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# News API (optional)
NEWS_API_KEY=your_news_api_key_here

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Database
DATABASE_URL=sqlite:///trading_bot.db
'''
    
    with open('.env.example', 'w') as f:
        f.write(env_example)
    print("✓ Created: .env.example")
    
    # Create README.md
    readme_content = '''# AI-Powered Trading Bot

An automated trading system using machine learning and technical analysis.

## Quick Start

1. Create virtual environment:
```bash
python -m venv venv
venv\\Scripts\\activate