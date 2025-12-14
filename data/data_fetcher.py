"""
Data Fetcher - Retrieve market data from multiple sources
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from alpha_vantage.timeseries import TimeSeries
from utils.logger import get_logger

logger = get_logger()


class DataFetcher:
    """Fetch and cache market data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)
        
    def get_historical_data(self, symbol: str, period: str = '1y', 
                           interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logger.debug(f"Using cached data for {symbol}")
                return cached_data
        
        try:
            logger.info(f"Fetching historical data for {symbol} ({period}, {interval})")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Cache the data
            self.cache[cache_key] = (df, datetime.now())
            
            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, symbols: List[str], period: str = '1y',
                            interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, period, interval)
            if not df.empty:
                data[symbol] = df
        
        logger.info(f"Fetched data for {len(data)}/{len(symbols)} symbols")
        return data
    
    def get_realtime_price(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time price data
        
        Returns:
            Dict with current price, volume, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            return None
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get company information and fundamentals"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName')),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'description': info.get('longBusinessSummary', '')[:500]
            }
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return None
    
    def get_financial_statements(self, symbol: str) -> Dict:
        """Get financial statements"""
        try:
            ticker = yf.Ticker(symbol)
            
            return {
                'income_statement': ticker.income_stmt,
                'balance_sheet': ticker.balance_sheet,
                'cash_flow': ticker.cashflow,
                'earnings': ticker.earnings
            }
            
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {str(e)}")
            return {}
    
    def download_batch(self, symbols: List[str], start_date: str, 
                      end_date: str, interval: str = '1d') -> pd.DataFrame:
        """
        Download data for multiple symbols efficiently
        
        Returns:
            Multi-index DataFrame with all symbols
        """
        try:
            logger.info(f"Batch downloading {len(symbols)} symbols from {start_date} to {end_date}")
            
            data = yf.download(
                tickers=' '.join(symbols),
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )
            
            logger.info(f"Successfully downloaded batch data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error in batch download: {str(e)}")
            return pd.DataFrame()
    
    def get_market_status(self) -> Dict:
        """Check if market is open"""
        try:
            # Use SPY as proxy for market
            ticker = yf.Ticker('SPY')
            info = ticker.info
            
            return {
                'is_open': info.get('marketState') == 'REGULAR',
                'market_state': info.get('marketState'),
                'timezone': info.get('timeZoneFullName'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return {'is_open': False, 'error': str(e)}
    
    def clear_cache(self):
        """Clear data cache"""
        self.cache.clear()
        logger.info("Data cache cleared")


# Example usage
if __name__ == '__main__':
    fetcher = DataFetcher()
    
    # Test single symbol
    df = fetcher.get_historical_data('AAPL', period='1mo', interval='1d')
    print(f"\nAAPL Data Shape: {df.shape}")
    print(df.head())
    
    # Test real-time price
    price_data = fetcher.get_realtime_price('AAPL')
    print(f"\nReal-time AAPL: ${price_data['price']:.2f}")
    
    # Test company info
    info = fetcher.get_company_info('AAPL')
    print(f"\nCompany: {info['name']}")
    print(f"Sector: {info['sector']}")
    print(f"Market Cap: ${info['market_cap']:,.0f}")
    
    # Test market status
    status = fetcher.get_market_status()
    print(f"\nMarket Status: {status}")