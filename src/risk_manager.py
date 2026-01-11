"""
Risk Manager Module
===================
Institutional-grade portfolio risk management.
Prevents catastrophic losses through multiple safety layers.

Key Features:
- Portfolio heat calculation
- Position correlation analysis
- Maximum drawdown protection
- Position sizing with Kelly Criterion
- Exposure limits
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from config import (
        MAX_PORTFOLIO_RISK, MAX_POSITIONS, MAX_CORRELATED_EXPOSURE,
        MAX_DRAWDOWN_STOP, MIN_POSITION_SIZE, MAX_POSITION_SIZE
    )
except ImportError:
    # Institutional defaults
    MAX_PORTFOLIO_RISK = 0.02  # 2% max portfolio risk
    MAX_POSITIONS = 10
    MAX_CORRELATED_EXPOSURE = 0.05  # 5% max in correlated assets
    MAX_DRAWDOWN_STOP = 0.15  # Stop at 15% drawdown
    MIN_POSITION_SIZE = 0.001  # 0.1% min
    MAX_POSITION_SIZE = 0.02  # 2% max


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    portfolio_heat: float  # Current total risk exposure
    num_positions: int
    correlated_exposure: float
    current_drawdown: float
    available_risk: float
    is_safe_to_trade: bool
    rejection_reason: Optional[str] = None


class RiskManager:
    """
    Institutional-grade risk management system.
    
    Prevents:
    - Over-leverage
    - Correlated position accumulation
    - Excessive drawdowns
    - Undersized/oversized positions
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_portfolio_risk: float = MAX_PORTFOLIO_RISK,
        max_positions: int = MAX_POSITIONS,
        max_correlated_exposure: float = MAX_CORRELATED_EXPOSURE,
        max_drawdown: float = MAX_DRAWDOWN_STOP
    ):
        """
        Initialize risk manager.
        
        Args:
            initial_capital: Starting capital in USD
            max_portfolio_risk: Maximum total portfolio risk (0.02 = 2%)
            max_positions: Maximum number of concurrent positions
            max_correlated_exposure: Max exposure to correlated assets
            max_drawdown: Maximum drawdown before circuit breaker
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_positions = max_positions
        self.max_correlated_exposure = max_correlated_exposure
        self.max_drawdown = max_drawdown
        
        # Correlation matrix cache
        self.correlation_cache: Dict[str, Dict[str, float]] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_duration = timedelta(hours=1)
        
        logger.info(f"RiskManager initialized: Capital=${initial_capital:,.2f}, "
                   f"MaxRisk={max_portfolio_risk*100}%, MaxPositions={max_positions}")
    
    def update_capital(self, current_capital: float):
        """Update current capital and track peak."""
        self.current_capital = current_capital
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
    
    def get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_capital == 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital
    
    def calculate_portfolio_heat(self, positions: List) -> float:
        """
        Calculate total portfolio risk (heat).
        
        Portfolio heat = sum of all position risks as % of capital
        
        Args:
            positions: List of Position objects
            
        Returns:
            Total risk as decimal (0.05 = 5% portfolio risk)
        """
        total_risk = 0.0
        
        for pos in positions:
            # Risk per position = (entry - SL) / entry * position_size
            if pos.direction == "long":
                risk_per_unit = (pos.entry_price - pos.stop_loss) / pos.entry_price
            else:
                risk_per_unit = (pos.stop_loss - pos.entry_price) / pos.entry_price
            
            # Assume equal position sizing for now (will be dynamic later)
            position_size = 0.01  # 1% of capital per position
            position_risk = risk_per_unit * position_size
            total_risk += position_risk
        
        return total_risk
    
    def calculate_correlation(
        self, 
        symbol1: str, 
        symbol2: str,
        fetcher=None
    ) -> float:
        """
        Calculate price correlation between two symbols.
        
        Uses 30-day rolling correlation of returns.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            fetcher: DataFetcher instance
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Check cache
        cache_key = f"{symbol1}_{symbol2}"
        if (self.cache_timestamp and 
            datetime.now() - self.cache_timestamp < self.cache_duration and
            cache_key in self.correlation_cache):
            return self.correlation_cache[cache_key]
        
        # Calculate correlation
        try:
            if fetcher is None:
                from data_fetcher import get_fetcher
                fetcher = get_fetcher()
            
            # Fetch 30 days of data
            df1 = fetcher.fetch_ohlcv(symbol1, '1d', limit=30)
            df2 = fetcher.fetch_ohlcv(symbol2, '1d', limit=30)
            
            if len(df1) < 20 or len(df2) < 20:
                logger.warning(f"Insufficient data for correlation: {symbol1}, {symbol2}")
                return 0.5  # Conservative assumption
            
            # Calculate returns
            returns1 = df1['close'].pct_change().dropna()
            returns2 = df2['close'].pct_change().dropna()
            
            # Align timestamps
            min_len = min(len(returns1), len(returns2))
            returns1 = returns1.iloc[-min_len:]
            returns2 = returns2.iloc[-min_len:]
            
            # Calculate correlation
            correlation = returns1.corr(returns2)
            
            # Cache result
            self.correlation_cache[cache_key] = correlation
            self.correlation_cache[f"{symbol2}_{symbol1}"] = correlation
            self.cache_timestamp = datetime.now()
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.5  # Conservative assumption
    
    def calculate_correlated_exposure(
        self, 
        positions: List,
        new_symbol: str,
        fetcher=None
    ) -> float:
        """
        Calculate exposure to assets correlated with new symbol.
        
        Args:
            positions: Current positions
            new_symbol: Symbol being considered
            fetcher: DataFetcher instance
            
        Returns:
            Total correlated exposure as decimal
        """
        correlated_exposure = 0.0
        
        for pos in positions:
            correlation = self.calculate_correlation(
                pos.symbol, 
                new_symbol,
                fetcher
            )
            
            # If correlation > 0.7, consider it correlated
            if abs(correlation) > 0.7:
                # Add this position's size to correlated exposure
                position_size = 0.01  # Will be dynamic later
                correlated_exposure += position_size
        
        return correlated_exposure
    
    def calculate_position_size(
        self,
        signal,
        positions: List,
        win_rate: float = 0.4,
        avg_win: float = 0.015,
        avg_loss: float = 0.005
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
        Use half-Kelly for safety.
        
        Args:
            signal: Signal object
            positions: Current positions
            win_rate: Historical win rate
            avg_win: Average win size
            avg_loss: Average loss size
            
        Returns:
            Position size as decimal (0.02 = 2% of capital)
        """
        # Kelly Criterion
        if avg_win == 0:
            return MIN_POSITION_SIZE
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Half-Kelly for safety
        half_kelly = kelly * 0.5
        
        # Adjust for confidence
        confidence_multiplier = signal.confidence_score / 100.0
        adjusted_size = half_kelly * confidence_multiplier
        
        # Clamp to limits
        position_size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, adjusted_size))
        
        return position_size
    
    def validate_new_position(
        self,
        signal,
        positions: List,
        fetcher=None
    ) -> Tuple[bool, RiskMetrics]:
        """
        Validate if new position is safe to open.
        
        Checks:
        1. Maximum positions limit
        2. Portfolio heat limit
        3. Correlated exposure limit
        4. Drawdown circuit breaker
        
        Args:
            signal: Signal object
            positions: Current active positions
            fetcher: DataFetcher instance
            
        Returns:
            (is_valid, risk_metrics)
        """
        # Calculate current metrics
        current_heat = self.calculate_portfolio_heat(positions)
        current_drawdown = self.get_current_drawdown()
        num_positions = len(positions)
        
        # Check 1: Drawdown circuit breaker
        if current_drawdown >= self.max_drawdown:
            return False, RiskMetrics(
                portfolio_heat=current_heat,
                num_positions=num_positions,
                correlated_exposure=0.0,
                current_drawdown=current_drawdown,
                available_risk=0.0,
                is_safe_to_trade=False,
                rejection_reason=f"CIRCUIT BREAKER: Drawdown {current_drawdown*100:.1f}% >= {self.max_drawdown*100:.1f}%"
            )
        
        # Check 2: Maximum positions
        if num_positions >= self.max_positions:
            return False, RiskMetrics(
                portfolio_heat=current_heat,
                num_positions=num_positions,
                correlated_exposure=0.0,
                current_drawdown=current_drawdown,
                available_risk=0.0,
                is_safe_to_trade=False,
                rejection_reason=f"Max positions reached: {num_positions}/{self.max_positions}"
            )
        
        # Check 3: Portfolio heat
        # Estimate new position risk
        if signal.direction.value == "long":
            new_risk = (signal.entry_price - signal.stop_loss) / signal.entry_price
        else:
            new_risk = (signal.stop_loss - signal.entry_price) / signal.entry_price
        
        position_size = 0.01  # Will use Kelly later
        new_position_risk = new_risk * position_size
        projected_heat = current_heat + new_position_risk
        
        if projected_heat > self.max_portfolio_risk:
            return False, RiskMetrics(
                portfolio_heat=current_heat,
                num_positions=num_positions,
                correlated_exposure=0.0,
                current_drawdown=current_drawdown,
                available_risk=self.max_portfolio_risk - current_heat,
                is_safe_to_trade=False,
                rejection_reason=f"Portfolio heat {projected_heat*100:.2f}% > {self.max_portfolio_risk*100:.1f}%"
            )
        
        # Check 4: Correlated exposure
        correlated_exp = self.calculate_correlated_exposure(
            positions,
            signal.symbol,
            fetcher
        )
        
        if correlated_exp + position_size > self.max_correlated_exposure:
            return False, RiskMetrics(
                portfolio_heat=current_heat,
                num_positions=num_positions,
                correlated_exposure=correlated_exp,
                current_drawdown=current_drawdown,
                available_risk=self.max_portfolio_risk - current_heat,
                is_safe_to_trade=False,
                rejection_reason=f"Correlated exposure {(correlated_exp+position_size)*100:.1f}% > {self.max_correlated_exposure*100:.1f}%"
            )
        
        # All checks passed
        return True, RiskMetrics(
            portfolio_heat=current_heat,
            num_positions=num_positions,
            correlated_exposure=correlated_exp,
            current_drawdown=current_drawdown,
            available_risk=self.max_portfolio_risk - current_heat,
            is_safe_to_trade=True
        )
    
    def get_correlation_matrix(
        self,
        positions: List,
        fetcher=None
    ) -> pd.DataFrame:
        """
        Build correlation matrix for all active positions.
        
        Args:
            positions: List of Position objects
            fetcher: DataFetcher instance
            
        Returns:
            DataFrame with correlation coefficients
        """
        if not positions:
            return pd.DataFrame()
        
        symbols = [pos.symbol for pos in positions]
        n = len(symbols)
        
        # Initialize matrix
        corr_matrix = np.ones((n, n))
        
        # Calculate pairwise correlations
        for i in range(n):
            for j in range(i+1, n):
                corr = self.calculate_correlation(
                    symbols[i],
                    symbols[j],
                    fetcher
                )
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        # Create DataFrame
        df = pd.DataFrame(
            corr_matrix,
            index=symbols,
            columns=symbols
        )
        
        return df
    
    def calculate_portfolio_beta(
        self,
        positions: List,
        market_symbol: str = "BTC/USDT:USDT",
        fetcher=None
    ) -> float:
        """
        Calculate portfolio beta to market (BTC).
        
        Beta measures portfolio volatility relative to market:
        - Beta > 1: More volatile than market
        - Beta = 1: Same as market
        - Beta < 1: Less volatile than market
        
        Args:
            positions: List of Position objects
            market_symbol: Market benchmark symbol
            fetcher: DataFetcher instance
            
        Returns:
            Portfolio beta
        """
        if not positions:
            return 0.0
        
        try:
            if fetcher is None:
                from data_fetcher import get_fetcher
                fetcher = get_fetcher()
            
            # Fetch market data
            market_df = fetcher.fetch_ohlcv(market_symbol, '1d', limit=30)
            market_returns = market_df['close'].pct_change().dropna()
            
            # Calculate weighted beta
            total_beta = 0.0
            total_weight = 0.0
            
            for pos in positions:
                # Fetch position data
                pos_df = fetcher.fetch_ohlcv(pos.symbol, '1d', limit=30)
                
                if len(pos_df) < 20:
                    continue
                
                pos_returns = pos_df['close'].pct_change().dropna()
                
                # Align returns
                min_len = min(len(market_returns), len(pos_returns))
                market_ret = market_returns.iloc[-min_len:]
                pos_ret = pos_returns.iloc[-min_len:]
                
                # Calculate beta = cov(pos, market) / var(market)
                covariance = np.cov(pos_ret, market_ret)[0, 1]
                market_variance = np.var(market_ret)
                
                if market_variance > 0:
                    beta = covariance / market_variance
                else:
                    beta = 1.0
                
                # Weight by position size (assume equal for now)
                weight = 1.0 / len(positions)
                total_beta += beta * weight
                total_weight += weight
            
            portfolio_beta = total_beta / total_weight if total_weight > 0 else 1.0
            
            return portfolio_beta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0  # Neutral assumption
    
    def calculate_diversification_score(
        self,
        positions: List,
        fetcher=None
    ) -> float:
        """
        Calculate portfolio diversification score (0-100).
        
        Score based on:
        - Number of positions
        - Average correlation between positions
        - Sector/category diversity
        
        100 = perfectly diversified
        0 = all positions identical
        
        Args:
            positions: List of Position objects
            fetcher: DataFetcher instance
            
        Returns:
            Diversification score (0-100)
        """
        if not positions:
            return 100.0  # No positions = fully diversified
        
        if len(positions) == 1:
            return 50.0  # Single position = moderate
        
        # Get correlation matrix
        corr_matrix = self.get_correlation_matrix(positions, fetcher)
        
        if corr_matrix.empty:
            return 50.0
        
        # Calculate average correlation (excluding diagonal)
        n = len(corr_matrix)
        total_corr = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                total_corr += abs(corr_matrix.iloc[i, j])
                count += 1
        
        avg_correlation = total_corr / count if count > 0 else 0.5
        
        # Score components
        # 1. Number of positions (more = better, up to 10)
        position_score = min(len(positions) / 10.0, 1.0) * 50
        
        # 2. Low correlation (lower = better)
        correlation_score = (1 - avg_correlation) * 50
        
        # Total score
        diversification_score = position_score + correlation_score
        
        return max(0, min(100, diversification_score))
    
    def get_advanced_risk_report(
        self,
        positions: List,
        fetcher=None
    ) -> str:
        """
        Generate advanced risk report with correlation matrix.
        
        Args:
            positions: List of Position objects
            fetcher: DataFetcher instance
            
        Returns:
            Formatted report string
        """
        heat = self.calculate_portfolio_heat(positions)
        drawdown = self.get_current_drawdown()
        
        # Calculate advanced metrics
        portfolio_beta = self.calculate_portfolio_beta(positions, fetcher=fetcher)
        diversification = self.calculate_diversification_score(positions, fetcher=fetcher)
        
        report = f"""
📊 Advanced Risk Report
━━━━━━━━━━━━━━━━━━━━

💰 Capital:
  Current: ${self.current_capital:,.2f}
  Peak: ${self.peak_capital:,.2f}
  Drawdown: {drawdown*100:.1f}%

🔥 Portfolio Heat:
  Current: {heat*100:.2f}%
  Max: {self.max_portfolio_risk*100:.1f}%
  Available: {(self.max_portfolio_risk - heat)*100:.2f}%

📊 Positions: {len(positions)}/{self.max_positions}

📈 Portfolio Metrics:
  Beta: {portfolio_beta:.2f}
  Diversification: {diversification:.0f}/100

"""
        
        # Add correlation matrix if positions exist
        if len(positions) > 1:
            corr_matrix = self.get_correlation_matrix(positions, fetcher)
            
            if not corr_matrix.empty:
                report += "🔗 Correlation Matrix:\n"
                
                # Format matrix for display
                for i, symbol1 in enumerate(corr_matrix.index):
                    symbol1_short = symbol1.split('/')[0][:4]
                    row_str = f"  {symbol1_short}: "
                    
                    for j, symbol2 in enumerate(corr_matrix.columns):
                        if i != j:
                            corr = corr_matrix.iloc[i, j]
                            # Color code: high correlation = warning
                            if abs(corr) > 0.7:
                                row_str += f"🔴{corr:.2f} "
                            elif abs(corr) > 0.5:
                                row_str += f"🟡{corr:.2f} "
                            else:
                                row_str += f"🟢{corr:.2f} "
                    
                    report += row_str + "\n"
                
                report += "\n"
        
        # Status
        if drawdown >= self.max_drawdown:
            status = "🔴 CIRCUIT BREAKER"
        elif heat >= self.max_portfolio_risk * 0.8:
            status = "🟡 HIGH RISK"
        else:
            status = "🟢 SAFE"
        
        report += f"Status: {status}\n"
        
        return report
    
    def get_risk_report(self, positions: List) -> str:
        """Generate human-readable risk report."""
        heat = self.calculate_portfolio_heat(positions)
        drawdown = self.get_current_drawdown()
        
        report = f"""
📊 Risk Report
━━━━━━━━━━━━━━
Capital: ${self.current_capital:,.2f}
Peak: ${self.peak_capital:,.2f}
Drawdown: {drawdown*100:.1f}%

Portfolio Heat: {heat*100:.2f}%
Max Allowed: {self.max_portfolio_risk*100:.1f}%
Available: {(self.max_portfolio_risk - heat)*100:.2f}%

Positions: {len(positions)}/{self.max_positions}

Status: {'🟢 SAFE' if drawdown < self.max_drawdown else '🔴 CIRCUIT BREAKER'}
"""
        return report


# Singleton instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager(initial_capital: float = 10000.0) -> RiskManager:
    """Get or create the global risk manager."""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager(initial_capital=initial_capital)
    return _risk_manager


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    rm = RiskManager(initial_capital=10000.0)
    print("✅ RiskManager initialized")
    print(rm.get_risk_report([]))
