"""
Partial Profit Manager - Phase 3.4
===================================
Staged profit-taking to lock in gains while letting winners run.

Strategy:
- Take 25% at 0.5% profit
- Take 25% at 1.0% profit  
- Take 25% at 1.5% profit
- Let 25% run to full TP

Benefits:
- Locks in early profits
- Improves win rate
- Reduces risk
- Lets winners run
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PartialExit:
    """Record of a partial exit."""
    level_pct: float
    exit_pct: float
    price: float
    profit_pct: float
    timestamp: str


@dataclass
class PartialProfitConfig:
    """Configuration for partial profit levels."""
    levels: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 0.25),  # Take 25% at 0.5% profit
        (1.0, 0.25),  # Take 25% at 1.0% profit
        (1.5, 0.25),  # Take 25% at 1.5% profit
        # Remaining 25% runs to full TP
    ])


class PartialProfitManager:
    """
    Manages partial profit taking at staged levels.
    
    Improves win rate by locking in profits early while
    still letting winners run to full target.
    """
    
    def __init__(self, config: Optional[PartialProfitConfig] = None):
        """
        Initialize partial profit manager.
        
        Args:
            config: Partial profit configuration
        """
        self.config = config or PartialProfitConfig()
        self.partial_exits: Dict[str, List[PartialExit]] = {}
        
        logger.info(f"PartialProfitManager initialized with {len(self.config.levels)} levels")
    
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
    
    def get_position_key(self, position) -> str:
        """Get unique key for position."""
        return f"{position.symbol}_{position.entry_time}"
    
    def check_partial_exits(
        self,
        position,
        current_price: float
    ) -> List[PartialExit]:
        """
        Check if any partial exit levels have been hit.
        
        Args:
            position: Position object
            current_price: Current market price
            
        Returns:
            List of PartialExit objects for levels hit
        """
        position_key = self.get_position_key(position)
        
        # Initialize tracking for this position if needed
        if position_key not in self.partial_exits:
            self.partial_exits[position_key] = []
        
        # Get already taken exits
        taken_levels = {exit.level_pct for exit in self.partial_exits[position_key]}
        
        # Calculate current profit
        profit_pct = self.calculate_profit_pct(position, current_price)
        
        # Check each level
        new_exits = []
        
        for level_pct, exit_pct in self.config.levels:
            # Skip if already taken
            if level_pct in taken_levels:
                continue
            
            # Check if level hit
            if profit_pct >= level_pct:
                exit = PartialExit(
                    level_pct=level_pct,
                    exit_pct=exit_pct,
                    price=current_price,
                    profit_pct=profit_pct,
                    timestamp=str(pd.Timestamp.now())
                )
                
                new_exits.append(exit)
                self.partial_exits[position_key].append(exit)
                
                logger.info(
                    f"💰 Partial exit: {exit_pct*100:.0f}% at {level_pct}% profit "
                    f"(price: {current_price:.2f}, profit: {profit_pct:.2f}%)"
                )
        
        return new_exits
    
    def get_remaining_size(self, position) -> float:
        """
        Get remaining position size after partial exits.
        
        Args:
            position: Position object
            
        Returns:
            Remaining size as decimal (1.0 = 100%)
        """
        position_key = self.get_position_key(position)
        
        if position_key not in self.partial_exits:
            return 1.0  # Full size
        
        # Calculate total exited
        total_exited = sum(exit.exit_pct for exit in self.partial_exits[position_key])
        
        return max(0.0, 1.0 - total_exited)
    
    def calculate_realized_profit(
        self,
        position,
        position_size_usd: float
    ) -> float:
        """
        Calculate total realized profit from partial exits.
        
        Args:
            position: Position object
            position_size_usd: Position size in USD
            
        Returns:
            Realized profit in USD
        """
        position_key = self.get_position_key(position)
        
        if position_key not in self.partial_exits:
            return 0.0
        
        total_profit = 0.0
        
        for exit in self.partial_exits[position_key]:
            # Profit = size × exit_pct × profit_pct
            exit_profit = position_size_usd * exit.exit_pct * (exit.profit_pct / 100)
            total_profit += exit_profit
        
        return total_profit
    
    def get_exit_report(self, position) -> str:
        """
        Generate report of partial exits for position.
        
        Args:
            position: Position object
            
        Returns:
            Formatted report string
        """
        position_key = self.get_position_key(position)
        
        if position_key not in self.partial_exits or not self.partial_exits[position_key]:
            return "No partial exits taken"
        
        report = "Partial Exits:\n"
        
        for exit in self.partial_exits[position_key]:
            report += f"  • {exit.exit_pct*100:.0f}% at {exit.level_pct}% profit (${exit.price:.2f})\n"
        
        remaining = self.get_remaining_size(position)
        report += f"\nRemaining: {remaining*100:.0f}%"
        
        return report
    
    def cleanup_position(self, position):
        """Remove tracking for closed position."""
        position_key = self.get_position_key(position)
        if position_key in self.partial_exits:
            del self.partial_exits[position_key]


# Singleton instance
_profit_manager: Optional[PartialProfitManager] = None


def get_profit_manager() -> PartialProfitManager:
    """Get or create the global profit manager."""
    global _profit_manager
    if _profit_manager is None:
        _profit_manager = PartialProfitManager()
    return _profit_manager


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Partial Profit Manager Test")
    print("="*60)
    
    import pandas as pd
    from dataclasses import dataclass as dc
    from datetime import datetime
    
    @dc
    class MockPosition:
        symbol: str = "BTC/USDT:USDT"
        direction: str = "long"
        entry_price: float = 95000.0
        entry_time: datetime = datetime.now()
    
    position = MockPosition()
    manager = PartialProfitManager()
    
    # Test scenarios
    print("\nScenario 1: 0.3% profit (no exit)")
    exits = manager.check_partial_exits(position, 95285.0)
    print(f"  Exits: {len(exits)}")
    print(f"  Remaining: {manager.get_remaining_size(position)*100:.0f}%")
    
    print("\nScenario 2: 0.6% profit (first exit)")
    exits = manager.check_partial_exits(position, 95570.0)
    print(f"  Exits: {len(exits)}")
    print(f"  Remaining: {manager.get_remaining_size(position)*100:.0f}%")
    
    print("\nScenario 3: 1.2% profit (second exit)")
    exits = manager.check_partial_exits(position, 96140.0)
    print(f"  Exits: {len(exits)}")
    print(f"  Remaining: {manager.get_remaining_size(position)*100:.0f}%")
    
    print("\nScenario 4: 1.8% profit (third exit)")
    exits = manager.check_partial_exits(position, 96710.0)
    print(f"  Exits: {len(exits)}")
    print(f"  Remaining: {manager.get_remaining_size(position)*100:.0f}%")
    
    print("\n" + manager.get_exit_report(position))
    
    print("\n✅ Partial Profit Manager working!")
