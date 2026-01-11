"""
Paper Trading Tracker
=====================
Tracks paper trading performance without executing real trades.

Features:
- Virtual position tracking
- Simulated P&L calculation
- Performance metrics
- Trade history
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Paper trade record."""
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    position_size: float
    confidence_score: int
    
    # Exit info (filled when closed)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl_percent: Optional[float] = None
    pnl_usd: Optional[float] = None
    
    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'confidence_score': self.confidence_score,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'pnl_percent': self.pnl_percent,
            'pnl_usd': self.pnl_usd
        }


class PaperTradingTracker:
    """
    Tracks paper trading performance.
    
    Simulates trade execution and tracks P&L without real capital.
    """
    
    def __init__(self, initial_capital: float = 10000):
        """Initialize paper trading tracker."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.active_trades: List[PaperTrade] = []
        self.closed_trades: List[PaperTrade] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Create results directory
        self.results_dir = Path("paper_trading_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Paper Trading Tracker initialized with ${initial_capital:,.2f}")
    
    def open_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence_score: int,
        position_size_pct: float = 0.01
    ) -> PaperTrade:
        """
        Open a paper trade.
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence_score: Signal confidence
            position_size_pct: Position size as % of capital
            
        Returns:
            PaperTrade object
        """
        # Calculate position size in USD
        position_size_usd = self.current_capital * position_size_pct
        
        trade = PaperTrade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size_usd,
            confidence_score=confidence_score
        )
        
        self.active_trades.append(trade)
        self.total_trades += 1
        
        logger.info(f"📝 Paper trade opened: {direction.upper()} {symbol} @ ${entry_price:,.2f} | Size: ${position_size_usd:,.2f}")
        
        return trade
    
    def check_exits(self, current_prices: Dict[str, float]) -> List[PaperTrade]:
        """
        Check if any active trades hit SL/TP.
        
        Args:
            current_prices: Dict of symbol -> current price
            
        Returns:
            List of closed trades
        """
        closed = []
        
        for trade in self.active_trades[:]:  # Copy list to modify during iteration
            if trade.symbol not in current_prices:
                continue
            
            current_price = current_prices[trade.symbol]
            
            # Check stop loss
            if trade.direction == "long":
                if current_price <= trade.stop_loss:
                    self._close_trade(trade, current_price, "stop_loss")
                    closed.append(trade)
                elif current_price >= trade.take_profit:
                    self._close_trade(trade, current_price, "take_profit")
                    closed.append(trade)
            else:  # short
                if current_price >= trade.stop_loss:
                    self._close_trade(trade, current_price, "stop_loss")
                    closed.append(trade)
                elif current_price <= trade.take_profit:
                    self._close_trade(trade, current_price, "take_profit")
                    closed.append(trade)
        
        return closed
    
    def _close_trade(self, trade: PaperTrade, exit_price: float, reason: str):
        """Close a paper trade and calculate P&L."""
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_reason = reason
        
        # Calculate P&L
        if trade.direction == "long":
            pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100
        
        pnl_usd = trade.position_size * (pnl_pct / 100)
        
        trade.pnl_percent = pnl_pct
        trade.pnl_usd = pnl_usd
        
        # Update capital
        self.current_capital += pnl_usd
        
        # Update stats
        if pnl_pct > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Move to closed trades
        self.active_trades.remove(trade)
        self.closed_trades.append(trade)
        
        logger.info(f"✅ Paper trade closed: {trade.symbol} | P&L: {pnl_pct:+.2f}% (${pnl_usd:+,.2f}) | Reason: {reason}")
        
        # Save results
        self._save_results()
    
    def get_performance_report(self) -> str:
        """Generate performance report."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        total_pnl = self.current_capital - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        report = f"""
📊 Paper Trading Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Capital:
  Initial: ${self.initial_capital:,.2f}
  Current: ${self.current_capital:,.2f}
  P&L: ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)

📈 Trades:
  Total: {self.total_trades}
  Active: {len(self.active_trades)}
  Closed: {len(self.closed_trades)}
  
🎯 Performance:
  Win Rate: {win_rate:.1f}%
  Winners: {self.winning_trades}
  Losers: {self.losing_trades}

━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report
    
    def _save_results(self):
        """Save results to JSON file."""
        try:
            results = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'active_trades': [t.to_dict() for t in self.active_trades],
                'closed_trades': [t.to_dict() for t in self.closed_trades],
                'last_updated': datetime.now().isoformat()
            }
            
            filename = self.results_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving paper trading results: {e}")


# Singleton
_paper_tracker: Optional[PaperTradingTracker] = None


def get_paper_tracker(initial_capital: float = 10000) -> PaperTradingTracker:
    """Get or create paper trading tracker."""
    global _paper_tracker
    if _paper_tracker is None:
        _paper_tracker = PaperTradingTracker(initial_capital)
    return _paper_tracker


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    tracker = PaperTradingTracker(10000)
    
    # Open trade
    trade = tracker.open_trade(
        symbol="BTC/USDT",
        direction="long",
        entry_price=95000,
        stop_loss=94500,
        take_profit=97000,
        confidence_score=85,
        position_size_pct=0.02
    )
    
    # Simulate price movement
    tracker.check_exits({"BTC/USDT": 97100})  # Hit TP
    
    # Show report
    print(tracker.get_performance_report())
    
    print("\n✅ Paper Trading Tracker working!")
