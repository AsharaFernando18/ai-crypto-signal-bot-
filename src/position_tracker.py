"""
Position Tracker Module
========================
Tracks active trading positions and monitors for TP/SL hits.
Records trade outcomes and calculates performance statistics.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_generator import Signal, SignalDirection

logger = logging.getLogger(__name__)


class PositionOutcome(Enum):
    """Outcome of a closed position."""
    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"
    MANUAL_CLOSE = "manual_close"
    TIMEOUT = "timeout"


@dataclass
class Position:
    """Represents an active or closed trading position."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    take_profit: float
    stop_loss: float
    timeframe: str
    confidence_score: float
    entry_time: str
    
    # DCA Support (NEW)
    initial_entry: Optional[float] = None  # First entry price
    entries: Optional[List[Dict]] = None  # List of all entries [{price, size, time}]
    average_entry: Optional[float] = None  # Weighted average entry
    dca_count: int = 0  # Number of DCA entries made
    
    # Outcome tracking (filled when closed)
    outcome: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl_percent: Optional[float] = None
    duration_minutes: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return asdict(self)
    
    @classmethod
    def from_signal(cls, signal: Signal) -> 'Position':
        """Create Position from Signal object."""
        entry_price = signal.entry_price
        return cls(
            symbol=signal.symbol,
            direction=signal.direction.value,
            entry_price=entry_price,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            timeframe=signal.timeframe,
            confidence_score=signal.confidence_score or 0,
            entry_time=datetime.now().isoformat(),
            # DCA initialization
            initial_entry=entry_price,
            entries=[{
                'price': entry_price,
                'size': 1.0,
                'time': datetime.now().isoformat()
            }],
            average_entry=entry_price,
            dca_count=0
        )


class PositionTracker:
    """
    Tracks active positions and monitors for TP/SL hits.
    """
    
    def __init__(self, data_file: str = "positions.json"):
        """Initialize tracker with data file path."""
        self.data_file = Path(data_file)
        self.active_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # Statistics
        self.total_wins = 0
        self.total_losses = 0
        
        # Load existing data
        self._load_data()
        logger.info(f"Position Tracker initialized: {len(self.active_positions)} active positions")
    
    def _load_data(self):
        """Load positions from JSON file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load active positions
                for pos_data in data.get('active', []):
                    pos = Position(**pos_data)
                    self.active_positions.append(pos)
                
                # Load closed positions
                for pos_data in data.get('closed', []):
                    pos = Position(**pos_data)
                    self.closed_positions.append(pos)
                    # Update stats
                    if pos.outcome == PositionOutcome.TP_HIT.value:
                        self.total_wins += 1
                    elif pos.outcome == PositionOutcome.SL_HIT.value:
                        self.total_losses += 1
                        
            except Exception as e:
                logger.error(f"Error loading positions: {e}")
    
    def _save_data(self):
        """Save positions to JSON file."""
        try:
            data = {
                'active': [pos.to_dict() for pos in self.active_positions],
                'closed': [pos.to_dict() for pos in self.closed_positions[-100:]]  # Keep last 100
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def add_position(self, signal: Signal) -> Position:
        """
        Add a new position from a signal.
        
        Args:
            signal: Signal object
        
        Returns:
            Created Position object
        """
        position = Position.from_signal(signal)
        self.active_positions.append(position)
        self._save_data()
        
        logger.info(
            f"📈 Position opened: {position.symbol} {position.direction.upper()} "
            f"@ {position.entry_price:.2f} | TP: {position.take_profit:.2f} | SL: {position.stop_loss:.2f}"
        )
        
        return position
    
    def add_dca_entry(self, symbol: str, dca_price: float, size: float = 1.0) -> Optional[Position]:
        """
        Add a DCA entry to an existing position.
        
        Args:
            symbol: Symbol of the position
            dca_price: DCA entry price
            size: Size of DCA entry (default 1.0)
        
        Returns:
            Updated Position object or None if not found
        """
        # Find the position
        position = None
        for pos in self.active_positions:
            if pos.symbol == symbol:
                position = pos
                break
        
        if not position:
            logger.warning(f"Position not found for DCA: {symbol}")
            return None
        
        # Add new entry
        new_entry = {
            'price': dca_price,
            'size': size,
            'time': datetime.now().isoformat()
        }
        
        if position.entries is None:
            position.entries = []
        position.entries.append(new_entry)
        
        # Recalculate average entry
        total_value = sum(e['price'] * e['size'] for e in position.entries)
        total_size = sum(e['size'] for e in position.entries)
        position.average_entry = total_value / total_size
        
        # Update DCA count
        position.dca_count += 1
        
        # Update entry_price to average (for PnL calculations)
        position.entry_price = position.average_entry
        
        # Save
        self._save_data()
        
        logger.info(
            f"🔄 DCA Entry added: {symbol} @ ${dca_price:.2f} | "
            f"New Average: ${position.average_entry:.2f} | DCA Count: {position.dca_count}"
        )
        
        return position
    
    def check_positions(self, current_prices: Dict[str, float]) -> List[tuple]:
        """
        Check all active positions against current prices.
        
        Args:
            current_prices: Dict mapping symbol -> current price
        
        Returns:
            List of (position, outcome) tuples for closed positions
        """
        closed = []
        
        for position in self.active_positions[:]:  # Copy to allow modification
            symbol = position.symbol
            
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Check TP/SL based on direction
            if position.direction == 'long':
                if current_price >= position.take_profit:
                    outcome = PositionOutcome.TP_HIT
                    closed.append((position, outcome, current_price))
                elif current_price <= position.stop_loss:
                    outcome = PositionOutcome.SL_HIT
                    closed.append((position, outcome, current_price))
            else:  # short
                if current_price <= position.take_profit:
                    outcome = PositionOutcome.TP_HIT
                    closed.append((position, outcome, current_price))
                elif current_price >= position.stop_loss:
                    outcome = PositionOutcome.SL_HIT
                    closed.append((position, outcome, current_price))
        
        # Process closures
        for position, outcome, exit_price in closed:
            self._close_position(position, outcome, exit_price)
        
        return closed
    
    def _close_position(self, position: Position, outcome: PositionOutcome, exit_price: float):
        """Close a position and record outcome."""
        # Calculate PnL
        if position.direction == 'long':
            pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100
        
        # Calculate duration
        entry_time = datetime.fromisoformat(position.entry_time)
        duration = datetime.now() - entry_time
        duration_minutes = int(duration.total_seconds() / 60)
        
        # Update position
        position.outcome = outcome.value
        position.exit_price = exit_price
        position.exit_time = datetime.now().isoformat()
        position.pnl_percent = round(pnl_pct, 2)
        position.duration_minutes = duration_minutes
        
        # Move to closed
        self.active_positions.remove(position)
        self.closed_positions.append(position)
        
        # Update stats
        if outcome == PositionOutcome.TP_HIT:
            self.total_wins += 1
            emoji = "🎯"
            logger.info(f"🎯 TP HIT! {position.symbol} | PnL: +{pnl_pct:.2f}%")
        else:
            self.total_losses += 1
            emoji = "❌"
            logger.info(f"❌ SL HIT! {position.symbol} | PnL: {pnl_pct:.2f}%")
        
        self._save_data()
        
        return position
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        total_trades = self.total_wins + self.total_losses
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate average PnL
        pnls = [p.pnl_percent for p in self.closed_positions if p.pnl_percent is not None]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        
        return {
            'total_trades': total_trades,
            'wins': self.total_wins,
            'losses': self.total_losses,
            'win_rate': round(win_rate, 1),
            'avg_pnl': round(avg_pnl, 2),
            'active_positions': len(self.active_positions)
        }
    
    def format_stats_message(self) -> str:
        """Format stats as Telegram message."""
        stats = self.get_stats()
        
        return f"""
📊 **Performance Stats**

✅ Wins: {stats['wins']}
❌ Losses: {stats['losses']}
📈 Win Rate: {stats['win_rate']}%
💰 Avg PnL: {stats['avg_pnl']:+.2f}%
🔄 Active: {stats['active_positions']}
"""
    
    def format_outcome_message(self, position: Position) -> str:
        """Format position outcome as Telegram message."""
        if position.outcome == PositionOutcome.TP_HIT.value:
            header_emoji = "🎯"
            status = "TARGET HIT"
            result_emoji = "💰"
            pnl_style = "✅"
        else:
            header_emoji = "❌"
            status = "STOPPED OUT"
            result_emoji = "💸"
            pnl_style = "🔻"
        
        # Duration formatting
        mins = position.duration_minutes or 0
        if mins < 60:
            duration_str = f"{mins}m"
        else:
            duration_str = f"{mins // 60}h {mins % 60}m"
        
        # Direction emoji
        dir_emoji = "📈" if position.direction == "long" else "📉"
        
        return f"""
╔═══════════════════╗
    {header_emoji} <b>{status}</b> {header_emoji}
╚═══════════════════╝

🪙 <b>{position.symbol.split('/')[0]}</b>
{dir_emoji} {position.direction.upper()}

┏━━━ RESULT ━━━━┓

   💵 Entry: <code>${position.entry_price:.4f}</code>
   🏁 Exit:   <code>${position.exit_price:.4f}</code>
   {result_emoji} PnL:   <code>{position.pnl_percent:+.2f}%</code>

┗━━━━━━━━━━━━━━━┛

⏱ <b>Duration:</b> {duration_str}
"""


# Singleton instance
_tracker: Optional[PositionTracker] = None

def get_tracker() -> PositionTracker:
    """Get or create the global position tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PositionTracker()
    return _tracker


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    tracker = PositionTracker("test_positions.json")
    
    print("=" * 50)
    print("Position Tracker Test")
    print("=" * 50)
    
    print(f"\nStats: {tracker.get_stats()}")
    print("\n✅ Position Tracker module is working!")
