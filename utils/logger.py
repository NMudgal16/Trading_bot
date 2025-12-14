"""
Enhanced Logging System for Trading Bot
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import colorlog

class TradingLogger:
    """Custom logger with colored output and file logging"""
    
    def __init__(self, name='TradingBot', log_file='trading_bot.log', level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # File handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_dir / f'{log_file.replace(".log", "")}_{timestamp}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger
    
    # Trading-specific log methods
    def log_trade(self, action, symbol, quantity, price, reason=''):
        """Log trade execution"""
        self.logger.info(
            f"🔔 TRADE | {action.upper()} {quantity} {symbol} @ ${price:.2f} | {reason}"
        )
    
    def log_signal(self, signal_type, symbol, strength, indicators):
        """Log trading signals"""
        self.logger.info(
            f"📊 SIGNAL | {signal_type} for {symbol} | Strength: {strength:.2%} | {indicators}"
        )
    
    def log_portfolio(self, total_value, cash, positions_count, pnl, pnl_pct):
        """Log portfolio status"""
        pnl_symbol = "📈" if pnl >= 0 else "📉"
        self.logger.info(
            f"{pnl_symbol} PORTFOLIO | Value: ${total_value:,.2f} | Cash: ${cash:,.2f} | "
            f"Positions: {positions_count} | P&L: ${pnl:,.2f} ({pnl_pct:+.2%})"
        )
    
    def log_risk_alert(self, alert_type, message):
        """Log risk management alerts"""
        self.logger.warning(f"⚠️  RISK ALERT | {alert_type}: {message}")
    
    def log_ml_prediction(self, symbol, prediction, confidence):
        """Log ML predictions"""
        self.logger.info(
            f"🤖 ML PREDICTION | {symbol}: {prediction} | Confidence: {confidence:.2%}"
        )
    
    def log_backtest_result(self, strategy, total_return, sharpe, max_dd, win_rate):
        """Log backtest results"""
        self.logger.info(
            f"📋 BACKTEST | {strategy} | Return: {total_return:.2%} | "
            f"Sharpe: {sharpe:.2f} | Max DD: {max_dd:.2%} | Win Rate: {win_rate:.2%}"
        )


# Singleton instance
_logger_instance = None

def get_logger(name='TradingBot', log_file='trading_bot.log', level=logging.INFO):
    """Get or create logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingLogger(name, log_file, level)
    return _logger_instance.get_logger()


# Example usage
if __name__ == '__main__':
    logger = get_logger()
    logger.info("Trading bot initialized")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")
    
    # Trading-specific logs
    trading_logger = TradingLogger()
    trading_logger.log_trade('BUY', 'AAPL', 10, 150.25, 'RSI oversold + bullish MACD')
    trading_logger.log_signal('BUY', 'TSLA', 0.75, 'RSI: 28, MACD: bullish cross')
    trading_logger.log_portfolio(105000, 5000, 8, 5000, 0.05)