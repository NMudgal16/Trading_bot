import pandas as pd
import numpy as np
import yfinance as yf

# Fetch Apple stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="1mo")

# Calculate simple indicator (RSI)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rsi = 100 - (100 / (1 + gain/loss))

print(f"✅ Latest AAPL price: ${df['Close'].iloc[-1]:.2f}")
print(f"✅ RSI: {rsi.iloc[-1]:.2f}")