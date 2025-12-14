"""
Risk Calculator - Calculate portfolio risk metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger()


class RiskCalculator:
    """Calculate various risk and performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize risk calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.04 = 4%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"Risk Calculator initialized | Risk-free rate: {risk_free_rate:.2%}")
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year (252 for daily, 12 for monthly)
            
        Returns:
            Sharpe ratio (annualized)
        """
        if len(returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        
        # Annualize
        mean_excess = excess_returns.mean() * periods_per_year
        std_excess = excess_returns.std() * np.sqrt(periods_per_year)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = mean_excess / std_excess
        return sharpe
    
    def calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (like Sharpe but only considers downside volatility)
        
        Returns:
            Sortino ratio (annualized)
        """
        if len(returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        
        # Annualize mean
        mean_excess = excess_returns.mean() * periods_per_year
        
        # Calculate downside deviation (only negative returns)
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = negative_returns.std() * np.sqrt(periods_per_year)
        
        if downside_std == 0:
            return 0.0
        
        sortino = mean_excess / downside_std
        return sortino
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Dict:
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            Dict with max drawdown metrics
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'drawdown_duration': 0,
                'current_drawdown': 0.0
            }
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = equity_curve - running_max
        drawdown_pct = (drawdown / running_max) * 100
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()
        
        # Calculate drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = is_drawdown.astype(int).groupby((~is_drawdown).cumsum()).cumsum()
        max_duration = drawdown_periods.max()
        
        # Current drawdown
        current_dd = drawdown.iloc[-1]
        current_dd_pct = drawdown_pct.iloc[-1]
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'drawdown_duration': max_duration,
            'current_drawdown': current_dd,
            'current_drawdown_pct': current_dd_pct
        }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 = 95%)
            
        Returns:
            VaR as a percentage
        """
        if len(returns) < 2:
            return 0.0
        
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        Average loss in worst (1-confidence_level) cases
        
        Returns:
            CVaR as a percentage
        """
        if len(returns) < 2:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Beta (portfolio volatility relative to market)
        
        Args:
            returns: Portfolio returns
            market_returns: Market (benchmark) returns
            
        Returns:
            Beta coefficient
        """
        if len(returns) < 2 or len(market_returns) < 2:
            return 1.0
        
        # Align series
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return 1.0
        
        # Calculate covariance and variance
        covariance = aligned_data['portfolio'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_alpha(self, returns: pd.Series, market_returns: pd.Series,
                       beta: float = None) -> float:
        """
        Calculate Alpha (excess return vs market after adjusting for risk)
        
        Returns:
            Alpha as percentage
        """
        if beta is None:
            beta = self.calculate_beta(returns, market_returns)
        
        # Calculate average returns
        portfolio_return = returns.mean()
        market_return = market_returns.mean()
        
        # Alpha = Portfolio Return - (Risk-free Rate + Beta * (Market Return - Risk-free Rate))
        alpha = portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
        
        return alpha
    
    def calculate_calmar_ratio(self, returns: pd.Series, equity_curve: pd.Series) -> float:
        """
        Calculate Calmar Ratio (annual return / max drawdown)
        
        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0
        
        # Annual return
        annual_return = returns.mean() * 252  # Assuming daily returns
        
        # Max drawdown
        dd_metrics = self.calculate_max_drawdown(equity_curve)
        max_dd_pct = abs(dd_metrics['max_drawdown_pct'])
        
        if max_dd_pct == 0:
            return float('inf')
        
        calmar = (annual_return * 100) / max_dd_pct
        return calmar
    
    def calculate_win_rate(self, trades: List[Dict]) -> Dict:
        """
        Calculate win rate and average win/loss
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Dict with win rate metrics
        """
        if not trades:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
        
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        
        if not pnls:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = (len(wins) / len(pnls)) * 100 if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        # Profit factor = gross profit / gross loss
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(pnls),
            'winning_trades': len(wins),
            'losing_trades': len(losses)
        }
    
    def calculate_comprehensive_metrics(self, equity_curve: pd.Series,
                                       returns: pd.Series,
                                       market_returns: pd.Series = None,
                                       trades: List[Dict] = None) -> Dict:
        """
        Calculate all risk metrics in one call
        
        Returns:
            Dict with all metrics
        """
        metrics = {}
        
        # Return metrics
        metrics['total_return'] = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
        metrics['annual_return'] = returns.mean() * 252 * 100  # Annualized
        metrics['volatility'] = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        
        # Drawdown
        dd_metrics = self.calculate_max_drawdown(equity_curve)
        metrics.update(dd_metrics)
        
        # Risk measures
        metrics['var_95'] = self.calculate_var(returns, 0.95) * 100
        metrics['cvar_95'] = self.calculate_cvar(returns, 0.95) * 100
        
        # Calmar ratio
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns, equity_curve)
        
        # Market comparison
        if market_returns is not None:
            metrics['beta'] = self.calculate_beta(returns, market_returns)
            metrics['alpha'] = self.calculate_alpha(returns, market_returns) * 252 * 100  # Annualized
        
        # Trade metrics
        if trades:
            win_metrics = self.calculate_win_rate(trades)
            metrics.update(win_metrics)
        
        return metrics


# Example usage
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Simulate returns (slightly positive drift)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    
    # Calculate equity curve
    equity_curve = (1 + returns).cumprod() * 100000
    
    # Market returns (for comparison)
    market_returns = pd.Series(np.random.normal(0.0008, 0.015, 252), index=dates)
    
    # Create risk calculator
    risk_calc = RiskCalculator()
    
    # Calculate metrics
    metrics = risk_calc.calculate_comprehensive_metrics(
        equity_curve, returns, market_returns
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RISK & PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Total Return:        {metrics['total_return']:>10.2f}%")
    print(f"Annual Return:       {metrics['annual_return']:>10.2f}%")
    print(f"Volatility:          {metrics['volatility']:>10.2f}%")
    print("-" * 70)
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
    print("-" * 70)
    print(f"Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%")
    print(f"Current Drawdown:    {metrics['current_drawdown_pct']:>10.2f}%")
    print("-" * 70)
    print(f"VaR (95%):           {metrics['var_95']:>10.2f}%")
    print(f"CVaR (95%):          {metrics['cvar_95']:>10.2f}%")
    print("-" * 70)
    print(f"Beta:                {metrics['beta']:>10.2f}")
    print(f"Alpha:               {metrics['alpha']:>10.2f}%")
    print("=" * 70)