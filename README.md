# 🤖 AI-Powered Trading Bot
### Final Year Project - Advanced Algorithmic Trading System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

An intelligent, multi-strategy trading bot that combines technical analysis, machine learning, and advanced risk management to make automated trading decisions.

### 🌟 Unique Features

1. **Multi-Modal Analysis**
   - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
   - Machine Learning predictions (LSTM neural networks)
   - Sentiment analysis from news sources
   - Ensemble strategy combining multiple approaches

2. **Advanced Risk Management**
   - Dynamic position sizing based on volatility
   - Portfolio optimization using Modern Portfolio Theory
   - Real-time drawdown protection
   - Stop-loss and take-profit automation

3. **Professional Architecture**
   - Modular, extensible design
   - Comprehensive logging and monitoring
   - Paper trading mode for safe testing
   - Backtesting engine with detailed metrics

4. **Interactive Dashboard**
   - Real-time trade monitoring
   - Performance visualization
   - Risk metrics display
   - Strategy comparison tools

---

## 📊 Performance Metrics Tracked

- **Returns**: Total, Annual, Daily
- **Risk-Adjusted**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Risk Metrics**: Max Drawdown, VaR, CVaR, Beta, Alpha
- **Trading Metrics**: Win Rate, Profit Factor, Average Win/Loss

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
nano .env
```

**Required API Keys (All FREE):**
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
- **News API**: https://newsapi.org/register
- **Alpaca (Paper Trading)**: https://alpaca.markets/

### 3. Run the Bot

```bash
# Run in paper trading mode (safe!)
python main.py

# The bot will:
# 1. Fetch market data
# 2. Calculate technical indicators
# 3. Generate trading signals
# 4. Execute trades (simulated)
# 5. Display portfolio status
```

---

## 📁 Project Structure

```
trading_bot/
│
├── config/                  # Configuration files
│   ├── settings.py         # Main settings
│   └── api_keys.py         # API keys (gitignored)
│
├── data/                   # Data fetching and processing
│   ├── data_fetcher.py    # Market data retrieval
│   ├── data_processor.py  # Technical indicators
│   └── sentiment_analyzer.py
│
├── strategies/             # Trading strategies
│   ├── base_strategy.py   # Abstract base class
│   ├── technical_strategy.py  # Technical analysis
│   ├── ml_strategy.py     # Machine learning
│   └── hybrid_strategy.py # Combined approach
│
├── execution/              # Trade execution
│   ├── paper_trading.py   # Simulated trading
│   └── order_manager.py   # Order management
│
├── risk_management/        # Risk controls
│   ├── position_sizer.py  # Position sizing
│   └── risk_calculator.py # Risk metrics
│
├── backtesting/            # Historical testing
│   └── backtest_engine.py
│
├── dashboard/              # Web interface
│   ├── backend/           # API server
│   └── frontend/          # React UI
│
└── main.py                # Main entry point
```

---

## 🎓 For Your Presentation

### Key Points to Highlight

1. **Technical Complexity**
   - Multiple technical indicators integrated
   - Machine learning for price prediction
   - Ensemble learning with weighted voting

2. **Risk Management**
   - Dynamic position sizing
   - Volatility-adjusted allocations
   - Real-time portfolio monitoring

3. **Professional Standards**
   - Clean, modular architecture
   - Comprehensive error handling
   - Detailed logging and auditing
   - Unit tests and validation

4. **Practical Application**
   - Paper trading mode for safety
   - Real-world data from Yahoo Finance
   - Backtesting with historical data
   - Performance metrics comparable to industry standards

### Demo Suggestions

1. **Live Trading Simulation**
   - Show bot analyzing current market
   - Demonstrate signal generation
   - Execute simulated trades
   - Display portfolio updates

2. **Backtest Results**
   - Compare bot performance vs S&P 500
   - Show risk-adjusted returns (Sharpe ratio)
   - Demonstrate risk management (max drawdown)

3. **Dashboard Visualization**
   - Real-time trade monitoring
   - Interactive performance charts
   - Risk metrics display

---

## 📈 Example Usage

### Basic Trading

```python
from main import TradingBot

# Initialize bot
bot = TradingBot()

# Run single trading cycle
bot.run_once()

# Start continuous trading
bot.start()
```

### Custom Strategy

```python
from strategies.technical_strategy import TechnicalStrategy

# Create strategy with custom parameters
strategy = TechnicalStrategy(params={
    'rsi_oversold': 25,
    'rsi_overbought': 75,
    'min_signal_strength': 0.7
})

# Generate signals
signals = strategy.generate_signals(data)
```

### Backtesting

```python
from backtesting.backtest_engine import BacktestEngine

# Run backtest
engine = BacktestEngine(
    strategy=strategy,
    start_date='2023-01-01',
    end_date='2024-12-01'
)

results = engine.run()
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

---

## 🔧 Configuration Options

Edit `config/settings.py` to customize:

```python
# Trading Parameters
INITIAL_CAPITAL = 10000.0
MAX_POSITION_SIZE = 0.1      # Max 10% per position
STOP_LOSS_PCT = 0.05         # 5% stop loss

# Symbols to Trade
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Technical Indicators
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
```

---

## 📊 Performance Example

```
==========================================
📊 PORTFOLIO STATUS
==========================================
💵 Total Value: $105,250.00
💰 Cash: $45,250.00
📈 P&L: $5,250.00 (+5.25%)
📦 Open Positions: 4

Current Holdings:
  • AAPL: 100 shares @ $150.25 = $15,025.00
  • MSFT: 50 shares @ $380.50 = $19,025.00
  • GOOGL: 75 shares @ $140.00 = $10,500.00
  • TSLA: 80 shares @ $195.00 = $15,600.00
==========================================
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_strategies.py

# With coverage
pytest --cov=. tests/
```

---

## 📚 Learning Resources

- **Technical Analysis**: [Investopedia TA Guide](https://www.investopedia.com/technical-analysis-4689657)
- **Risk Management**: [Portfolio Theory](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)
- **Python Finance**: [QuantStart](https://www.quantstart.com/)
- **Backtesting**: [Backtrader Documentation](https://www.backtrader.com/)

---

##  Disclaimer

This is an educational project for academic purposes. **Do not use real money without thoroughly understanding the risks.** Always use paper trading mode for testing.

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Consult with a financial advisor before live trading



## 📝 License

MIT License - Feel free to use for educational purposes

---

## 👨‍💻 Author

NIHARIKA MUDGAL
- Department: COMPUTER SCIENCE
- Year: Final Year


---

##  Acknowledgments

- Yahoo Finance for market data
- TA-Lib for technical indicators
- scikit-learn and TensorFlow for ML capabilities
- Alpaca for paper trading API

---

** If you find this project useful for your studies, please star the repository!**
