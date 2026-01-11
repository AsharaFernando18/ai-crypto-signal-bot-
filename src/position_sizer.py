"""
Dynamic Position Sizer - Phase 3.1
===================================
Kelly Criterion-based position sizing for optimal capital growth.

Features:
- Kelly Criterion calculation
- Half-Kelly for safety
- Confidence-based adjustments
- Regime-aware sizing
- Performance tracking
- Risk limit enforcement
"""
import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from config import MIN_POSITION_SIZE, MAX_POSITION_SIZE
except ImportError:
    MIN_POSITION_SIZE = 0.005  # 0.5%
    MAX_POSITION_SIZE = 0.03   # 3%


@dataclass
class PerformanceStats:
    """Historical performance statistics."""
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    total_trades: int
    recent_trades: int
    sharpe_ratio: float
    
    def __str__(self):
        return f"WR: {self.win_rate:.1%}, Avg Win: {self.avg_win_pct:.2f}%, Avg Loss: {self.avg_loss_pct:.2f}%"


class DynamicPositionSizer:
    """
    Dynamic position sizing using Kelly Criterion.
    
    Kelly Formula:
    K = (p × b - q) / b
    
    Where:
    - p = win rate
    - q = loss rate (1 - p)
    - b = avg win / avg loss ratio
    
    We use Half-Kelly for safety and adjust for:
    - Signal confidence
    - Market regime
    - Recent performance
    """
    
    def __init__(
        self,
        min_trades_for_kelly: int = 20,
        kelly_fraction: float = 0.5,  # Half-Kelly
        lookback_trades: int = 50
    ):
        """
        Initialize position sizer.
        
        Args:
            min_trades_for_kelly: Minimum trades before using Kelly
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
            lookback_trades: Number of recent trades to analyze
        """
        self.min_trades_for_kelly = min_trades_for_kelly
        self.kelly_fraction = kelly_fraction
        self.lookback_trades = lookback_trades
        
        logger.info(f"DynamicPositionSizer initialized: {kelly_fraction*100}% Kelly, {lookback_trades} trade lookback")
    
    def calculate_performance_stats(
        self,
        trades: List,
        lookback: Optional[int] = None
    ) -> PerformanceStats:
        """
        Calculate performance statistics from trade history.
        
        Args:
            trades: List of completed trades
            lookback: Number of recent trades to analyze
            
        Returns:
            PerformanceStats object
        """
        if not trades:
            # Default conservative stats
            return PerformanceStats(
                win_rate=0.4,
                avg_win_pct=1.5,
                avg_loss_pct=0.5,
                total_trades=0,
                recent_trades=0,
                sharpe_ratio=0.0
            )
        
        # Use recent trades if specified
        if lookback:
            recent_trades = trades[-lookback:]
        else:
            recent_trades = trades
        
        # Separate wins and losses
        wins = [t for t in recent_trades if t.pnl_percent > 0]
        losses = [t for t in recent_trades if t.pnl_percent <= 0]
        
        # Calculate metrics
        win_rate = len(wins) / len(recent_trades) if recent_trades else 0.4
        avg_win = np.mean([t.pnl_percent for t in wins]) if wins else 1.5
        avg_loss = abs(np.mean([t.pnl_percent for t in losses])) if losses else 0.5
        
        # Calculate Sharpe ratio
        if len(recent_trades) > 1:
            returns = [t.pnl_percent for t in recent_trades]
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe = 0.0
        
        return PerformanceStats(
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            total_trades=len(trades),
            recent_trades=len(recent_trades),
            sharpe_ratio=sharpe
        )
    
    def calculate_kelly_size(
        self,
        performance: PerformanceStats
    ) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Args:
            performance: Performance statistics
            
        Returns:
            Kelly position size (0-1)
        """
        p = performance.win_rate
        q = 1 - p
        
        # Avoid division by zero
        if performance.avg_loss_pct == 0:
            return MIN_POSITION_SIZE
        
        # b = avg win / avg loss
        b = performance.avg_win_pct / performance.avg_loss_pct
        
        # Kelly = (p × b - q) / b
        kelly = (p * b - q) / b
        
        # Apply Kelly fraction (half-Kelly for safety)
        fractional_kelly = kelly * self.kelly_fraction
        
        # Ensure positive
        return max(0, fractional_kelly)
    
    def calculate_position_size(
        self,
        signal,
        trades: List,
        regime=None,
        signal_strength: str = "moderate"
    ) -> float:
        """
        Calculate optimal position size for signal.
        
        Args:
            signal: Signal object
            trades: Historical trades
            regime: Market regime (optional)
            signal_strength: Signal strength classification
            
        Returns:
            Position size as decimal (0.01 = 1%)
        """
        # Get performance stats
        performance = self.calculate_performance_stats(
            trades,
            lookback=self.lookback_trades
        )
        
        logger.debug(f"Performance: {performance}")
        
        # Base size calculation
        if performance.total_trades < self.min_trades_for_kelly:
            # Not enough data - use conservative fixed size
            base_size = 0.01  # 1%
            logger.debug(f"Using fixed size (only {performance.total_trades} trades)")
        else:
            # Use Kelly Criterion
            kelly_size = self.calculate_kelly_size(performance)
            base_size = kelly_size
            logger.debug(f"Kelly size: {kelly_size*100:.2f}%")
        
        # Adjustment 1: Signal Confidence
        confidence_score = getattr(signal, 'confidence_score', 75)
        confidence_multiplier = confidence_score / 100.0
        
        # Adjustment 2: Signal Strength
        strength_multipliers = {
            'weak': 0.5,
            'moderate': 1.0,
            'strong': 1.3,
            'very_strong': 1.5
        }
        strength_multiplier = strength_multipliers.get(signal_strength, 1.0)
        
        # Adjustment 3: Market Regime
        regime_multiplier = 1.0
        if regime:
            if regime.trend == "ranging":
                regime_multiplier = 1.2  # Boost in ideal conditions
            elif "trending" in regime.trend:
                regime_multiplier = 0.7  # Reduce in poor conditions
            
            if regime.volatility == "high_volatility":
                regime_multiplier *= 0.8  # Reduce in high vol
            elif regime.volatility == "low_volatility":
                regime_multiplier *= 1.1  # Increase in low vol
        
        # Adjustment 4: Recent Performance
        if performance.recent_trades >= 10:
            # Reduce size if recent performance poor
            if performance.sharpe_ratio < 0:
                performance_multiplier = 0.7
            elif performance.sharpe_ratio < 0.5:
                performance_multiplier = 0.85
            else:
                performance_multiplier = 1.0
        else:
            performance_multiplier = 1.0
        
        # Calculate final size
        final_size = (
            base_size *
            confidence_multiplier *
            strength_multiplier *
            regime_multiplier *
            performance_multiplier
        )
        
        # Clamp to limits
        final_size = max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, final_size))
        
        logger.info(
            f"Position size: {final_size*100:.2f}% "
            f"(base: {base_size*100:.1f}%, conf: {confidence_multiplier:.2f}, "
            f"strength: {strength_multiplier:.1f}, regime: {regime_multiplier:.1f})"
        )
        
        return final_size
    
    def get_sizing_report(
        self,
        trades: List,
        signal,
        regime=None
    ) -> str:
        """
        Generate detailed sizing report.
        
        Args:
            trades: Historical trades
            signal: Current signal
            regime: Market regime
            
        Returns:
            Formatted report string
        """
        performance = self.calculate_performance_stats(trades)
        size = self.calculate_position_size(signal, trades, regime)
        
        report = f"""
📊 Position Sizing Report
━━━━━━━━━━━━━━━━━━━━━━

Performance Stats:
  Win Rate: {performance.win_rate:.1%}
  Avg Win: {performance.avg_win_pct:.2f}%
  Avg Loss: {performance.avg_loss_pct:.2f}%
  Sharpe: {performance.sharpe_ratio:.2f}
  Trades: {performance.recent_trades}/{performance.total_trades}

Kelly Calculation:
  Full Kelly: {self.calculate_kelly_size(performance)*100:.2f}%
  Half Kelly: {self.calculate_kelly_size(performance)*self.kelly_fraction*100:.2f}%

Adjustments:
  Confidence: {signal.confidence_score}/100
  Regime: {regime.trend if regime else 'N/A'}
  
Recommended Size: {size*100:.2f}%
"""
        return report


# Singleton instance
_position_sizer: Optional[DynamicPositionSizer] = None


def get_position_sizer() -> DynamicPositionSizer:
    """Get or create the global position sizer."""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = DynamicPositionSizer()
    return _position_sizer


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Dynamic Position Sizer Test")
    print("="*60)
    
    # Mock trades
    from dataclasses import dataclass as dc
    
    @dc
    class MockTrade:
        pnl_percent: float
    
    # Simulate 30 trades: 60% win rate
    trades = []
    for i in range(30):
        if i % 5 < 3:  # 60% wins
            trades.append(MockTrade(pnl_percent=np.random.uniform(1.0, 3.0)))
        else:
            trades.append(MockTrade(pnl_percent=np.random.uniform(-1.5, -0.5)))
    
    # Mock signal
    @dc
    class MockSignal:
        confidence_score: float = 85
    
    signal = MockSignal()
    
    # Test sizer
    sizer = DynamicPositionSizer()
    
    # Calculate size
    size = sizer.calculate_position_size(signal, trades, signal_strength="strong")
    
    print(f"\n✅ Calculated Position Size: {size*100:.2f}%")
    print(f"   (Range: {MIN_POSITION_SIZE*100:.1f}% - {MAX_POSITION_SIZE*100:.1f}%)")
    
    # Show report
    print(sizer.get_sizing_report(trades, signal))
    
    print("\n✅ Dynamic Position Sizer working!")
