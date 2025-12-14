"""
Complete Setup and Test Script
Verifies installation and runs a comprehensive test
"""
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("=" * 70)
    print(" Checking Dependencies...")
    print("=" * 70)
    
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'scikit-learn',
        'matplotlib', 'requests', 'flask'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f" {package:20} - OK")
        except ImportError:
            print(f" {package:20} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n All dependencies installed!\n")
    return True


def create_directory_structure():
    """Create necessary directories"""
    print("=" * 70)
    print(" Creating Directory Structure...")
    print("=" * 70)
    
    directories = [
        'config',
        'data',
        'strategies',
        'models',
        'execution',
        'risk_management',
        'backtesting',
        'dashboard/backend',
        'dashboard/frontend',
        'utils',
        'tests',
        'logs',
        'saved_models'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f" Created: {directory}")
        
        # Create __init__.py files
        if directory not in ['logs', 'saved_models', 'dashboard/frontend']:
            init_file = path / '__init__.py'
            init_file.touch(exist_ok=True)
    
    print("\n Directory structure created!\n")


def create_env_file():
    """Create .env file if it doesn't exist"""
    print("=" * 70)
    print(" Checking Environment Configuration...")
    print("=" * 70)
    
    env_file = Path('.env')
    
    if env_file.exists():
        print(" .env file already exists")
    else:
        print("  .env file not found. Creating template...")
        
        env_template = """# Environment Configuration
ENV=dev

# API Keys (Get free keys from these services)
ALPHA_VANTAGE_KEY=demo
NEWS_API_KEY=your_key_here
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_key_here

# Database
DATABASE_URL=sqlite:///trading_bot.db
"""
        
        with open('.env', 'w') as f:
            f.write(env_template)
        
        print(" Created .env template")
        print("  Please edit .env and add your API keys")
    
    print()


def run_comprehensive_test():
    """Run a comprehensive test of the system"""
    print("=" * 70)
    print(" Running Comprehensive Test...")
    print("=" * 70)
    
    try:
        # Test 1: Data Fetcher
        print("\n1️ Testing Data Fetcher...")
        from data.data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        df = fetcher.get_historical_data('AAPL', period='1mo', interval='1d')
        
        if df.empty:
            print(" Failed to fetch data")
            return False
        
        print(f" Fetched {len(df)} days of data for AAPL")
        print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
        
        # Test 2: Technical Indicators
        print("\n2️ Testing Technical Indicators...")
        from data.data_processor import TechnicalIndicators
        
        df = TechnicalIndicators.add_all_indicators(df)
        print(f" Added {len(df.columns)} features including indicators")
        print(f"   RSI: {df['rsi'].iloc[-1]:.2f}")
        print(f"   MACD: {df['macd'].iloc[-1]:.4f}")
        
        # Test 3: Trading Strategy
        print("\n3️ Testing Trading Strategy...")
        from strategies.technical_strategy import TechnicalStrategy
        
        strategy = TechnicalStrategy()
        signals = strategy.generate_signals(df)
        
        print(f" Strategy generated {len(signals)} signals")
        
        if signals:
            signal = signals[0]
            print(f"   Signal: {signal.signal.value}")
            print(f"   Strength: {signal.strength:.2%}")
            print(f"   Reason: {signal.reason}")
        
        # Test 4: Paper Trading Broker
        print("\n4️ Testing Paper Trading Broker...")
        from execution.paper_trading import PaperTradingBroker
        
        broker = PaperTradingBroker(initial_capital=10000)
        
        # Test buy order
        success = broker.place_order('AAPL', 10, 'BUY', 150.0)
        if success:
            print(" Successfully executed BUY order")
            print(f"   Cash remaining: ${broker.cash:,.2f}")
            print(f"   Position: {broker.get_position('AAPL')} shares")
        
        # Test sell order
        success = broker.place_order('AAPL', 5, 'SELL', 155.0)
        if success:
            print(" Successfully executed SELL order")
        
        # Test 5: Position Sizer
        print("\n5️ Testing Position Sizer...")
        from risk_management.position_sizer import PositionSizer
        from strategies.base_strategy import TradingSignal, SignalType
        import pandas as pd
        
        sizer = PositionSizer()
        
        test_signal = TradingSignal(
            symbol='AAPL',
            signal=SignalType.BUY,
            strength=0.75,
            price=150.0,
            timestamp=pd.Timestamp.now(),
            reason='Test signal',
            indicators={},
            confidence=0.8
        )
        
        shares = sizer.calculate_position_size(
            test_signal, 10000, 150.0, 0.02
        )
        
        print(f" Calculated position size: {shares} shares")
        
        # Test 6: Risk Calculator
        print("\n6️ Testing Risk Calculator...")
        from risk_management.risk_calculator import RiskCalculator
        import numpy as np
        
        risk_calc = RiskCalculator()
        
        # Create sample data
        sample_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        equity_curve = (1 + sample_returns).cumprod() * 10000
        
        sharpe = risk_calc.calculate_sharpe_ratio(sample_returns)
        dd = risk_calc.calculate_max_drawdown(equity_curve)
        
        print(f" Calculated risk metrics")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Max Drawdown: {dd['max_drawdown_pct']:.2f}%")
        
        # Test 7: Logger
        print("\n7️ Testing Logger...")
        from utils.logger import get_logger
        
        logger = get_logger('TestLogger')
        logger.info("Test log message")
        print(" Logger working correctly")
        
        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED!")
        print("=" * 70)
        print("\n Your trading bot is ready to use!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python main.py")
        print("3. Monitor the trading activity")
        
        return True
        
    except Exception as e:
        print(f"\n Test failed: {str(e)}")
        print("\nPlease check:")
        print("1. All dependencies are installed")
        print("2. All files are in the correct directories")
        print("3. .env file is configured")
        return False


def show_next_steps():
    """Show what to do next"""
    print("\n" + "=" * 70)
    print(" NEXT STEPS")
    print("=" * 70)
    print("""
1. Get API Keys (FREE):
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key
   - News API: https://newsapi.org/register
   - Alpaca Paper Trading: https://alpaca.markets/

2. Edit .env file:
   - Add your API keys
   - Configure trading parameters

3. Run the bot:
   python main.py

4. For presentation:
   - Run backtest: python backtesting/backtest_engine.py
   - Show live demo: python main.py
   - Display metrics and charts

5. Customize:
   - Modify strategies in strategies/
   - Adjust risk parameters in config/settings.py
   - Add new indicators in data/data_processor.py

📖 Read README.md for detailed documentation
🐛 Check logs/ directory for debugging

Good luck with your final year project! 🎓
""")


def main():
    """Main setup function"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + " TRADING BOT SETUP WIZARD " + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n Please install missing dependencies first!")
        print("Run: pip install -r requirements.txt\n")
        sys.exit(1)
    
    # Step 2: Create directory structure
    create_directory_structure()
    
    # Step 3: Create .env file
    create_env_file()
    
    # Step 4: Run comprehensive test
    if run_comprehensive_test():
        show_next_steps()
    else:
        print("\n Setup incomplete. Please fix errors and run again.\n")
        sys.exit(1)


if __name__ == '__main__':
    main()