import yfinance as yf

print("Fetching Apple (AAPL) stock data...")
ticker = yf.Ticker("AAPL")
df = ticker.history(period="5d")

if not df.empty:
    print(f"✅ Success! Latest AAPL price: ${df['Close'].iloc[-1]:.2f}")
else:
    print("❌ Failed to fetch data")