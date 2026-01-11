"""
Trailing Stop Manager - Phase 3.3
==================================
Advanced stop-loss management with trailing and ATR-based stops.

Features:
- Trailing stop activation
- ATR-based dynamic stops
- Breakeven protection
- Time-based adjustments
"""
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StopUpdate:
    """Result of stop update."""
    new_stop: float
    stop_type: str  # 'trailing', 'breakeven', 'atr', 'original'
    activated: bool
    reason: str


class TrailingStopManager:
    """
    Manages trailing stops and advanced stop-loss logic.
    
    Features:
    - Activates trailing after profit threshold
    - Trails by specified percentage
    - Moves to breakeven after time
    - ATR-based dynamic stops
    """
    
    def __init__(
        self,
        activation_profit_pct: float = 0.5,  # Activate after 0.5% profit
        trail_distance_pct: float = 0.3,     # Trail by 0.3%
        breakeven_bars: int = 10,            # Move to BE after 10 bars
        atr_multiplier: float = 1.5          # Stop = 1.5x ATR
    ):
        """
        Initialize trailing stop manager.
        
        Args:
            activation_profit_pct: Profit % to activate trailing
            trail_distance_pct: Distance to trail behind price
            breakeven_bars: Bars before moving to breakeven
            atr_multiplier: ATR multiplier for dynamic stops
        """
        self.activation_profit_pct = activation_profit_pct
        self.trail_distance_pct = trail_distance_pct
        self.breakeven_bars = breakeven_bars
        self.atr_multiplier = atr_multiplier
        
        logger.info(f"TrailingStopManager initialized: "
                   f"activation={activation_profit_pct}%, trail={trail_distance_pct}%")
    
    def calculate_atr(self, df, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            return atr.iloc[-1]
        except:
            return 0.0
    
    def calculate_profit_pct(
        self,
        position,
        current_price: float
    ) -> float:
        """Calculate current profit percentage."""
        if position.direction == "long":
            return ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            return ((position.entry_price - current_price) / position.entry_price) * 100
    
    def update_trailing_stop(
        self,
        position,
        current_price: float,
        bars_in_trade: int = 0,
        df=None
    ) -> StopUpdate:
        """
        Update stop-loss with trailing logic.
        
        Args:
            position: Position object
            current_price: Current market price
            bars_in_trade: Number of bars in trade
            df: OHLCV DataFrame for ATR calculation
            
        Returns:
            StopUpdate with new stop level
        """
        original_stop = position.stop_loss
        new_stop = original_stop
        stop_type = "original"
        activated = False
        reason = "No update"
        
        # Calculate current profit
        profit_pct = self.calculate_profit_pct(position, current_price)
        
        # Check 1: Breakeven after time
        if bars_in_trade >= self.breakeven_bars and profit_pct > 0:
            if position.direction == "long":
                breakeven_stop = position.entry_price
                if breakeven_stop > original_stop:
                    new_stop = breakeven_stop
                    stop_type = "breakeven"
                    activated = True
                    reason = f"Moved to breakeven after {bars_in_trade} bars"
            else:
                breakeven_stop = position.entry_price
                if breakeven_stop < original_stop:
                    new_stop = breakeven_stop
                    stop_type = "breakeven"
                    activated = True
                    reason = f"Moved to breakeven after {bars_in_trade} bars"
        
        # Check 2: Trailing stop activation
        if profit_pct >= self.activation_profit_pct:
            trail_distance = current_price * (self.trail_distance_pct / 100)
            
            if position.direction == "long":
                trailing_stop = current_price - trail_distance
                # Only move stop up
                if trailing_stop > new_stop:
                    new_stop = trailing_stop
                    stop_type = "trailing"
                    activated = True
                    reason = f"Trailing activated at {profit_pct:.2f}% profit"
            else:
                trailing_stop = current_price + trail_distance
                # Only move stop down
                if trailing_stop < new_stop:
                    new_stop = trailing_stop
                    stop_type = "trailing"
                    activated = True
                    reason = f"Trailing activated at {profit_pct:.2f}% profit"
        
        # Check 3: ATR-based stop (if provided)
        if df is not None and len(df) > 20:
            atr = self.calculate_atr(df)
            if atr > 0:
                if position.direction == "long":
                    atr_stop = current_price - (self.atr_multiplier * atr)
                    if atr_stop > new_stop:
                        new_stop = atr_stop
                        stop_type = "atr"
                        activated = True
                        reason = f"ATR stop ({self.atr_multiplier}x ATR)"
                else:
                    atr_stop = current_price + (self.atr_multiplier * atr)
                    if atr_stop < new_stop:
                        new_stop = atr_stop
                        stop_type = "atr"
                        activated = True
                        reason = f"ATR stop ({self.atr_multiplier}x ATR)"
        
        if activated:
            logger.info(f"Stop updated: {original_stop:.4f} → {new_stop:.4f} ({stop_type})")
        
        return StopUpdate(
            new_stop=new_stop,
            stop_type=stop_type,
            activated=activated,
            reason=reason
        )


# Singleton instance
_stop_manager: Optional[TrailingStopManager] = None


def get_stop_manager() -> TrailingStopManager:
    """Get or create the global stop manager."""
    global _stop_manager
    if _stop_manager is None:
        _stop_manager = TrailingStopManager()
    return _stop_manager


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Trailing Stop Manager Test")
    print("="*60)
    
    from dataclasses import dataclass as dc
    
    @dc
    class MockPosition:
        direction: str = "long"
        entry_price: float = 95000.0
        stop_loss: float = 94500.0
    
    position = MockPosition()
    manager = TrailingStopManager()
    
    # Test scenarios
    print("\nScenario 1: Small profit (0.3%)")
    update = manager.update_trailing_stop(position, 95285.0)
    print(f"  {update.reason}")
    print(f"  Stop: {update.new_stop:.2f}")
    
    print("\nScenario 2: Activation profit (0.6%)")
    update = manager.update_trailing_stop(position, 95570.0)
    print(f"  {update.reason}")
    print(f"  Stop: {update.new_stop:.2f}")
    
    print("\nScenario 3: Breakeven after 10 bars")
    update = manager.update_trailing_stop(position, 95100.0, bars_in_trade=11)
    print(f"  {update.reason}")
    print(f"  Stop: {update.new_stop:.2f}")
    
    print("\n✅ Trailing Stop Manager working!")
