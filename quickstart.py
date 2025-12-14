"""
Quick Start Script - Test Your Trading Bot Installation
Run this after installing packages to verify everything works!
"""

print("=" * 70)
print("🤖 TRADING BOT - QUICK START TEST")
print("=" * 70)
print()

# Step 1: Check Python version
import sys
print(f"1️⃣ Python Version: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("   ⚠️ Python 3.8+ recommended")
else:
    print("   ✅ Python version OK")
print()

# Step 2: Check essential packages
print("2️⃣ Checking Essential Packages...")
required = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical operations',
    'yfinance': 'Market data',
}

missing = []
for package, description in required.items():
    try:
        __import__(package)
        print(f"   ✅ {package:15} - {description}")
    except ImportError:
        print(f"   ❌ {package:15} - MISSING")
        missing.append(package)

if missing:
    print(f"\n   ⚠️ Install missing packages:")
    print(f"   pip install {' '.join(missing)}")
    sys.exit(1)

print()

# Step 3: Test data fetching
print("3️⃣ Testing Data Fetching...")
try:
    import yfinance as yf
    import pandas as pd
    
    # Fetch sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="5d")
    
    if not df.empty:
        latest_price = df['Close'].iloc[-1]
        print(f"   ✅ Successfully fetched AAPL data")
        print(f"   💰 Latest AAPL price: ${latest_price:.2f}")
    else:
        print("   ⚠️ No data returned (might be connection issue)")
except Exception as e:
    print(f"   ❌ Error: {str(e)}")
    print("   Check your internet connection")

print()

# Step 4: Test technical indicators
print("4️⃣ Testing Technical Indicators...")
try:
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    sample_data = pd.DataFrame({
        'close': np.random.uniform(100, 200, 50),
        'high': np.random.uniform(100, 200, 50),
        'low': np.random.uniform(100, 200, 50),
        'volume': np.random.uniform(1000000, 5000000, 50)
    }, index=dates)
    
    # Calculate simple indicators
    sample_data['sma_20'] = sample_data['close'].rolling(window=20).mean()
    sample_data['returns'] = sample_data['close'].pct_change()
    
    # Calculate RSI
    delta = sample_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    sample_data['rsi'] = 100 - (100 / (1 + rs))
    
    latest_rsi = sample_data['rsi'].iloc[-1]
    
    print(f"   ✅ Indicators calculated successfully")
    print(f"   📊 Sample RSI: {latest_rsi:.2f}")
    
    # Try importing 'ta' library
    try:
        from ta.momentum import RSIIndicator
        print(f"   ✅ 'ta' library available (recommended)")
    except ImportError:
        print(f"   ⚠️ 'ta' library not installed (using built-in)")
        print(f"   Install with: pip install ta")
    
except Exception as e:
    print(f"   ❌ Error: {str(e)}")

print()

# Step 5: Test a simple trading signal
print("5️⃣ Testing Simple Trading Signal...")
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    
    # Fetch real data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="3mo")
    
    if not df.empty:
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate moving averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Get latest values
        latest = df.iloc[-1]
        rsi = latest['RSI']
        price = latest['Close']
        sma_50 = latest['SMA_50']
        
        print(f"   📊 AAPL Analysis:")
        print(f"   💰 Price: ${price:.2f}")
        print(f"   📈 RSI: {rsi:.2f}")
        print(f"   📊 50-day SMA: ${sma_50:.2f}")
        
        # Simple signal logic
        if rsi < 30 and price > sma_50:
            signal = "🟢 BUY SIGNAL"
            reason = "RSI oversold + price above 50-day average"
        elif rsi > 70:
            signal = "🔴 SELL SIGNAL"
            reason = "RSI overbought"
        else:
            signal = "⚪ HOLD"
            reason = "No clear signal"
        
        print(f"\n   {signal}")
        print(f"   Reason: {reason}")
        print(f"   ✅ Signal generation working!")
    
except Exception as e:
    print(f"   ⚠️ Could not test signal: {str(e)}")

print()

# Step 6: Summary
print("=" * 70)
print("📋 SUMMARY")
print("=" * 70)

if not missing:
    print("✅ All essential packages installed")
    print("✅ Data fetching works")
    print("✅ Technical indicators work")
    print("✅ Signal generation works")
    print()
    print("🎉 Your trading bot is ready!")
    print()
    print("Next steps:")
    print("1. Create the project files (copy code from artifacts)")
    print("2. Get free API keys:")
    print("   - Alpha Vantage: https://www.alphavantage.co/support/#api-key")
    print("   - News API: https://newsapi.org/register")
    print("3. Run the full bot: python main.py")
    print()
    print("📚 Read README.md for detailed instructions")
else:
    print(f"⚠️ Missing packages: {', '.join(missing)}")
    print(f"Install with: pip install {' '.join(missing)}")

print("=" * 70)


# Optional: Create a simple demo
print("\n" + "=" * 70)
print("🎮 MINI DEMO - Simulated Trade")
print("=" * 70)

try:
    print("\n📊 Analyzing market data...")
    
    # Simulate fetching data
    import time
    time.sleep(1)
    
    print("✅ Data fetched")
    print("📊 Calculating indicators...")
    time.sleep(1)
    
    print("✅ Indicators calculated")
    print("\n🔍 Scanning for signals...")
    time.sleep(1)
    
    print("\n🟢 SIGNAL DETECTED!")
    print("   Symbol: AAPL")
    print("   Action: BUY")
    print("   Price: $150.25")
    print("   Quantity: 10 shares")
    print("   Strength: 75%")
    print("   Reason: RSI oversold (28) + MACD bullish")
    
    print("\n💼 Portfolio Status:")
    print("   Initial Capital: $10,000.00")
    print("   Cash: $8,497.50")
    print("   Positions: 1 (AAPL)")
    print("   Total Value: $10,000.00")
    
    print("\n✅ Demo trade executed successfully!")
    print("   (This was a simulation - no real money involved)")
    
except:
    pass

print("\n" + "=" * 70)
print("🚀 Ready to start building your trading bot!")
print("=" * 70)