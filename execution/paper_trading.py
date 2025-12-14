"""
Paper Trading Broker - Simulated trading without real money
"""
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from utils.logger import get_logger

logger = get_logger()


@dataclass
class Trade:
    """Trade record"""
    timestamp: datetime
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    price: float
    commission: float
    total_cost: float
    portfolio_value_after: float
    
    def to_dict(self):
        return asdict(self)


class PaperTradingBroker:
    """
    Simulated broker for paper trading
    Tracks positions, cash, and P&L without real money
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission_rate: float = 0.001):
        """
        Initialize paper trading broker
        
        Args:
            initial_capital: Starting cash amount
            commission_rate: Commission as percentage (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        
        self.positions: Dict[str, int] = {}  # symbol -> quantity
        self.avg_costs: Dict[str, float] = {}  # symbol -> average cost
        self.trades: List[Trade] = []
        
        self.daily_start_value = initial_capital
        self.daily_trades_count = 0
        
        logger.info(f"💼 Paper Trading Broker initialized with ${initial_capital:,.2f}")
    
    def place_order(self, symbol: str, quantity: int, order_type: str, 
                   price: float) -> bool:
        """
        Place a trading order
        
        Args:
            symbol: Stock ticker
            quantity: Number of shares
            order_type: 'BUY' or 'SELL'
            price: Execution price
            
        Returns:
            True if order executed successfully
        """
        order_type = order_type.upper()
        
        if order_type == 'BUY':
            return self._execute_buy(symbol, quantity, price)
        elif order_type == 'SELL':
            return self._execute_sell(symbol, quantity, price)
        else:
            logger.error(f"Invalid order type: {order_type}")
            return False
    
    def _execute_buy(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute a buy order"""
        if quantity <= 0:
            logger.warning(f"Invalid quantity for BUY: {quantity}")
            return False
        
        # Calculate costs
        gross_cost = quantity * price
        commission = gross_cost * self.commission_rate
        total_cost = gross_cost + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(
                f"Insufficient funds to BUY {quantity} {symbol}. "
                f"Required: ${total_cost:,.2f}, Available: ${self.cash:,.2f}"
            )
            return False
        
        # Execute trade
        self.cash -= total_cost
        
        # Update position
        current_position = self.positions.get(symbol, 0)
        current_avg_cost = self.avg_costs.get(symbol, 0.0)
        
        # Calculate new average cost
        total_shares = current_position + quantity
        total_cost_basis = (current_position * current_avg_cost) + (quantity * price)
        new_avg_cost = total_cost_basis / total_shares
        
        self.positions[symbol] = total_shares
        self.avg_costs[symbol] = new_avg_cost
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            action='BUY',
            quantity=quantity,
            price=price,
            commission=commission,
            total_cost=total_cost,
            portfolio_value_after=self.get_portfolio_value()
        )
        self.trades.append(trade)
        self.daily_trades_count += 1
        
        logger.info(
            f"✅ BUY ORDER FILLED | {quantity} {symbol} @ ${price:.2f} | "
            f"Commission: ${commission:.2f} | Total: ${total_cost:,.2f}"
        )
        
        return True
    
    def _execute_sell(self, symbol: str, quantity: int, price: float) -> bool:
        """Execute a sell order"""
        if quantity <= 0:
            logger.warning(f"Invalid quantity for SELL: {quantity}")
            return False
        
        # Check if we have the position
        current_position = self.positions.get(symbol, 0)
        
        if current_position < quantity:
            logger.warning(
                f"Insufficient shares to SELL {quantity} {symbol}. "
                f"Current position: {current_position}"
            )
            return False
        
        # Calculate proceeds
        gross_proceeds = quantity * price
        commission = gross_proceeds * self.commission_rate
        net_proceeds = gross_proceeds - commission
        
        # Calculate P&L
        avg_cost = self.avg_costs.get(symbol, 0.0)
        cost_basis = quantity * avg_cost
        pnl = net_proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        # Execute trade
        self.cash += net_proceeds
        
        # Update position
        new_position = current_position - quantity
        
        if new_position == 0:
            # Closed entire position
            del self.positions[symbol]
            del self.avg_costs[symbol]
        else:
            # Partial close
            self.positions[symbol] = new_position
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            action='SELL',
            quantity=quantity,
            price=price,
            commission=commission,
            total_cost=net_proceeds,
            portfolio_value_after=self.get_portfolio_value()
        )
        self.trades.append(trade)
        self.daily_trades_count += 1
        
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        logger.info(
            f"✅ SELL ORDER FILLED | {quantity} {symbol} @ ${price:.2f} | "
            f"Commission: ${commission:.2f} | {pnl_emoji} P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)"
        )
        
        return True
    
    def get_position(self, symbol: str) -> int:
        """Get current position size for a symbol"""
        return self.positions.get(symbol, 0)
    
    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total portfolio value
        
        Args:
            current_prices: Dict of symbol -> current price
                          If None, uses average cost as approximation
        """
        positions_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if current_prices and symbol in current_prices:
                price = current_prices[symbol]
            else:
                # Use average cost as fallback
                price = self.avg_costs.get(symbol, 0.0)
            
            positions_value += quantity * price
        
        return self.cash + positions_value
    
    def get_daily_return(self) -> float:
        """Calculate return for current day"""
        current_value = self.get_portfolio_value()
        daily_return = (current_value - self.daily_start_value) / self.daily_start_value
        return daily_return
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at end of day)"""
        self.daily_start_value = self.get_portfolio_value()
        self.daily_trades_count = 0
    
    def get_position_pnl(self, symbol: str, current_price: float) -> Dict:
        """
        Calculate P&L for a specific position
        
        Returns:
            Dict with P&L metrics
        """
        quantity = self.positions.get(symbol, 0)
        
        if quantity == 0:
            return {
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'quantity': 0,
                'avg_cost': 0.0,
                'current_value': 0.0
            }
        
        avg_cost = self.avg_costs[symbol]
        cost_basis = quantity * avg_cost
        current_value = quantity * current_price
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis) * 100
        
        return {
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'quantity': quantity,
            'avg_cost': avg_cost,
            'current_value': current_value,
            'cost_basis': cost_basis
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get complete trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([trade.to_dict() for trade in self.trades])
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        current_value = self.get_portfolio_value()
        total_pnl = current_value - self.initial_capital
        total_return = (total_pnl / self.initial_capital) * 100
        
        trades_df = self.get_trade_history()
        
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            total_commissions = trades_df['commission'].sum()
            
            # Calculate win rate (simplified)
            winning_trades = 0
            losing_trades = 0
            
            for trade in self.trades:
                if trade.action == 'SELL':
                    # Compare with average cost
                    if trade.symbol in self.avg_costs:
                        if trade.price > self.avg_costs[trade.symbol]:
                            winning_trades += 1
                        else:
                            losing_trades += 1
            
            total_closed_trades = winning_trades + losing_trades
            win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
        else:
            total_commissions = 0
            win_rate = 0
            buy_trades = pd.DataFrame()
            sell_trades = pd.DataFrame()
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'cash': self.cash,
            'positions_count': len(self.positions),
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_commissions': total_commissions,
            'win_rate': win_rate
        }
    
    def save_trade_history(self, filename: str = 'trade_history.csv'):
        """Save trade history to CSV"""
        df = self.get_trade_history()
        if not df.empty:
            df.to_csv(filename, index=False)
            logger.info(f"💾 Trade history saved to {filename}")
        else:
            logger.info("No trades to save")
    
    def print_summary(self):
        """Print performance summary"""
        summary = self.get_performance_summary()
        
        print("\n" + "=" * 70)
        print("📊 TRADING PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Initial Capital:     ${summary['initial_capital']:>15,.2f}")
        print(f"Current Value:       ${summary['current_value']:>15,.2f}")
        print(f"Cash:                ${summary['cash']:>15,.2f}")
        print(f"Open Positions:      {summary['positions_count']:>15}")
        print("-" * 70)
        pnl_symbol = "📈" if summary['total_pnl'] >= 0 else "📉"
        print(f"{pnl_symbol} Total P&L:          ${summary['total_pnl']:>15,.2f} ({summary['total_return_pct']:+.2f}%)")
        print("-" * 70)
        print(f"Total Trades:        {summary['total_trades']:>15}")
        print(f"Buy Orders:          {summary['buy_trades']:>15}")
        print(f"Sell Orders:         {summary['sell_trades']:>15}")
        print(f"Win Rate:            {summary['win_rate']:>14.2f}%")
        print(f"Total Commissions:   ${summary['total_commissions']:>15,.2f}")
        print("=" * 70 + "\n")


# Example usage
if __name__ == '__main__':
    # Create broker
    broker = PaperTradingBroker(initial_capital=100000)
    
    # Simulate some trades
    broker.place_order('AAPL', 10, 'BUY', 150.00)
    broker.place_order('GOOGL', 5, 'BUY', 140.00)
    broker.place_order('AAPL', 5, 'SELL', 155.00)
    
    # Print summary
    broker.print_summary()
    
    # Get trade history
    print("\nTrade History:")
    print(broker.get_trade_history())