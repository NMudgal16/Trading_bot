print("=" * 50)
print("TESTING TRADING BOT PACKAGES")
print("=" * 50)

# Test 1: Import packages
print("\n1. Testing imports...")
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
print("✅ All imports successful!")

# Test 2: Fetch real data
print("\n2. Fetching Apple stock data...")
ticker = yf.Ticker("AAPL")
df = ticker.history(period="1mo")
print(f"✅ Fetched {len(df)} days of data")
print(f"   Latest price: ${df['Close'].iloc[-1]:.2f}")

# Test 3: Calculate indicator
print("\n3. Calculating RSI indicator...")
rsi = RSIIndicator(close=df['Close'], window=14)
df['RSI'] = rsi.rsi()
print(f"✅ RSI calculated: {df['RSI'].iloc[-1]:.2f}")

# Test 4: Generate signal
print("\n4. Generating trading signal...")
latest_rsi = df['RSI'].iloc[-1]
latest_price = df['Close'].iloc[-1]

if latest_rsi < 30:
    signal = "BUY (Oversold)"
elif latest_rsi > 70:
    signal = "SELL (Overbought)"
else:
    signal = "HOLD (Neutral)"

print(f"✅ Signal: {signal}")

print("\n" + "=" * 50)
print("🎉 SUCCESS! Everything works!")
print("=" * 50)
print("\nYou're ready to build your trading bot!")