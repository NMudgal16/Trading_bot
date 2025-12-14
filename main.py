"""
Main Trading Bot Engine
Orchestrates data fetching, signal generation, and trade execution
"""
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict
import schedule

from config.settings import config
from data.data_fetcher import DataFetcher
from data.data_processor import TechnicalIndicators
from strategies.technical_strategy import TechnicalStrategy
from execution.paper_trading import PaperTradingBroker
from risk_management.position_sizer import PositionSizer
from risk_management.risk_calculator import RiskCalculator
from utils.logger import get_logger

logger = get_logger()


class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        logger.info("=" * 70)
        logger.info(" Initializing AI-Powered Trading Bot")
        logger.info("=" * 70)
        
        # Initialize components
        self.config = config
        self.data_fetcher = DataFetcher()
        self.strategy = TechnicalStrategy()
        self.broker = PaperTradingBroker(initial_capital=config.INITIAL_CAPITAL)
        self.position_sizer = PositionSizer(max_position_size=config.MAX_POSITION_SIZE)
        self.risk_calculator = RiskCalculator()
        
        # State
        self.is_running = False
        self.last_update = None
        self.symbols = config.SYMBOLS
        self.market_data = {}
        
        logger.info(f" Monitoring {len(self.symbols)} symbols: {', '.join(self.symbols)}")
        logger.info(f" Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")
        logger.info(f" Strategy: {self.strategy.name}")
        logger.info(f"  Mode: {'PAPER TRADING' if config.PAPER_TRADING else '🔴 LIVE TRADING'}")
    
    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch latest market data for all symbols"""
        logger.info(f" Fetching market data for {len(self.symbols)} symbols...")
        
        market_data = self.data_fetcher.get_multiple_symbols(
            self.symbols,
            period=f'{config.LOOKBACK_DAYS}d',
            interval=config.TIMEFRAME
        )
        
        # Add technical indicators
        for symbol, df in market_data.items():
            market_data[symbol] = TechnicalIndicators.add_all_indicators(df)
        
        logger.info(f" Successfully fetched data for {len(market_data)} symbols")
        return market_data
    
    def scan_for_signals(self) -> List:
        """Scan all symbols for trading signals"""
        all_signals = []
        
        logger.info(" Scanning for trading signals...")
        
        for symbol, data in self.market_data.items():
            try:
                signals = self.strategy.generate_signals(data)
                
                if signals:
                    for signal in signals:
                        logger.info(
                            f" SIGNAL DETECTED | {signal.signal.value} {symbol} | "
                            f"Strength: {signal.strength:.2%} | {signal.reason}"
                        )
                        all_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {str(e)}")
        
        logger.info(f"Found {len(all_signals)} total signals")
        return all_signals
    
    def execute_signals(self, signals: List):
        """Execute trading signals with risk management"""
        if not signals:
            logger.info("No signals to execute")
            return
        
        logger.info(f" Processing {len(signals)} signals for execution...")
        
        for signal in signals:
            try:
                # Check if we already have a position
                current_position = self.broker.get_position(signal.symbol)
                
                # Risk checks
                if not self._pass_risk_checks(signal):
                    logger.warning(f"  Signal for {signal.symbol} failed risk checks")
                    continue
                
                # Calculate position size
                portfolio_value = self.broker.get_portfolio_value()
                position_size = self.position_sizer.calculate_position_size(
                    signal=signal,
                    portfolio_value=portfolio_value,
                    current_price=signal.price,
                    volatility=self._get_volatility(signal.symbol)
                )
                
                # Execute trade
                if signal.signal.name == 'BUY':
                    if current_position > 0:
                        logger.info(f"Already holding {signal.symbol}, skipping")
                        continue
                    
                    self.broker.place_order(
                        symbol=signal.symbol,
                        quantity=position_size,
                        order_type='BUY',
                        price=signal.price
                    )
                    
                    logger.info(
                        f" EXECUTED BUY | {position_size} shares of {signal.symbol} @ ${signal.price:.2f}"
                    )
                
                elif signal.signal.name == 'SELL':
                    if current_position <= 0:
                        logger.info(f"No position in {signal.symbol}, skipping")
                        continue
                    
                    self.broker.place_order(
                        symbol=signal.symbol,
                        quantity=current_position,
                        order_type='SELL',
                        price=signal.price
                    )
                    
                    logger.info(
                        f" EXECUTED SELL | {current_position} shares of {signal.symbol} @ ${signal.price:.2f}"
                    )
                
            except Exception as e:
                logger.error(f"Error executing signal for {signal.symbol}: {str(e)}")
    
    def _pass_risk_checks(self, signal) -> bool:
        """Perform risk management checks"""
        portfolio_value = self.broker.get_portfolio_value()
        
        # Check maximum drawdown
        initial_value = config.INITIAL_CAPITAL
        current_drawdown = (initial_value - portfolio_value) / initial_value
        
        if current_drawdown > config.MAX_DRAWDOWN:
            logger.warning(f"  Maximum drawdown exceeded: {current_drawdown:.2%}")
            return False
        
        # Check daily loss limit
        daily_return = self.broker.get_daily_return()
        if daily_return < -config.DAILY_LOSS_LIMIT:
            logger.warning(f"  Daily loss limit exceeded: {daily_return:.2%}")
            return False
        
        # Check signal strength
        if signal.strength < config.MIN_SIGNAL_STRENGTH:
            return False
        
        return True
    
    def _get_volatility(self, symbol: str) -> float:
        """Calculate recent volatility for a symbol"""
        if symbol not in self.market_data:
            return 0.02  # Default 2%
        
        data = self.market_data[symbol]
        if 'volatility' in data.columns:
            return data['volatility'].iloc[-1]
        
        # Calculate if not available
        returns = data['close'].pct_change()
        return returns.std()
    
    def update_portfolio_status(self):
        """Log current portfolio status"""
        portfolio_value = self.broker.get_portfolio_value()
        cash = self.broker.cash
        positions = self.broker.positions
        
        total_pnl = portfolio_value - config.INITIAL_CAPITAL
        total_return = total_pnl / config.INITIAL_CAPITAL
        
        logger.info("=" * 70)
        logger.info(" PORTFOLIO STATUS")
        logger.info("=" * 70)
        logger.info(f" Total Value: ${portfolio_value:,.2f}")
        logger.info(f" Cash: ${cash:,.2f}")
        logger.info(f" P&L: ${total_pnl:,.2f} ({total_return:+.2%})")
        logger.info(f" Open Positions: {len(positions)}")
        
        if positions:
            logger.info("\nCurrent Holdings:")
            for symbol, quantity in positions.items():
                current_price = self._get_current_price(symbol)
                position_value = quantity * current_price
                logger.info(f"  • {symbol}: {quantity} shares @ ${current_price:.2f} = ${position_value:,.2f}")
        
        logger.info("=" * 70)
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if symbol in self.market_data:
            return self.market_data[symbol]['close'].iloc[-1]
        return 0.0
    
    def run_once(self):
        """Execute one complete trading cycle"""
        logger.info("\n Starting trading cycle...")
        
        try:
            # 1. Fetch market data
            self.market_data = self.fetch_market_data()
            
            # 2. Scan for signals
            signals = self.scan_for_signals()
            
            # 3. Execute signals
            self.execute_signals(signals)
            
            # 4. Update portfolio
            self.update_portfolio_status()
            
            self.last_update = datetime.now()
            logger.info(f" Trading cycle completed at {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f" Error in trading cycle: {str(e)}")
    
    def start(self):
        """Start the trading bot"""
        self.is_running = True
        logger.info("\n Trading Bot Started!")
        logger.info("Press Ctrl+C to stop\n")
        
        # Run immediately
        self.run_once()
        
        # Schedule periodic updates (every day at market open)
        if config.TIMEFRAME == '1d':
            schedule.every().day.at("09:35").do(self.run_once)
            logger.info(" Scheduled to run daily at 09:35 AM")
        
        # Keep running
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("\n Stopping Trading Bot...")
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        
        # Final portfolio status
        self.update_portfolio_status()
        
        # Save trade history
        self.broker.save_trade_history('trade_history.csv')
        
        logger.info(" Trading Bot Stopped Successfully")
        logger.info("Trade history saved to trade_history.csv")


def main():
    """Main entry point"""
    # Create and start bot
    bot = TradingBot()
    
    # For development: run once and exit
    if config.DEBUG:
        logger.info(" Running in DEBUG mode (single cycle)")
        bot.run_once()
    else:
        # Production: run continuously
        bot.start()


if __name__ == '__main__':
    main()