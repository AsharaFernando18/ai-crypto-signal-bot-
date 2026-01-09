"""
Backtesting Module
===================
Test signal performance on historical data before going live.

Features:
- Simulates signals using full strategy logic
- Tracks trade outcomes (TP hit, SL hit, timeout)
- Calculates key performance metrics
- Generates detailed reports
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MIN_RR_RATIO, MIN_CONFIDENCE_SCORE
from data_fetcher import DataFetcher
from channel_builder import detect_channel, Channel
from signal_generator import check_signal, SignalDirection, Signal
from signal_filters import apply_signal_filters

logger = logging.getLogger(__name__)


class TradeOutcome(Enum):
    """Possible trade outcomes."""
    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"
    TIMEOUT = "timeout"
    OPEN = "open"


@dataclass
class BacktestTrade:
    """Represents a single backtested trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: SignalDirection
    symbol: str
    timeframe: str
    
    entry_price: float
    take_profit: float
    stop_loss: float
    exit_price: Optional[float]
    
    outcome: TradeOutcome
    pnl_percent: float
    rr_achieved: float
    
    confidence_score: float
    channel_type: str
    volatility: str = 'normal'
    
    # Time in trade (bars)
    bars_held: int = 0


@dataclass
class BacktestResult:
    """Complete backtesting results."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    timeout_trades: int
    
    win_rate: float
    avg_win_percent: float
    avg_loss_percent: float
    profit_factor: float
    total_return_percent: float
    
    max_drawdown_percent: float
    avg_rr_achieved: float
    avg_bars_held: float
    
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'period': f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate:.1%}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'total_return': f"{self.total_return_percent:.2f}%",
            'max_drawdown': f"{self.max_drawdown_percent:.2f}%",
            'avg_rr_achieved': f"{self.avg_rr_achieved:.2f}",
            'avg_bars_held': f"{self.avg_bars_held:.1f}"
        }


class Backtester:
    """
    Backtesting engine for signal strategy.
    
    Simulates trading based on historical data and measures performance.
    """
    
    def __init__(
        self,
        symbol: str = 'BTC/USDT:USDT',
        timeframe: str = '15m',
        lookback_days: int = 30,
        max_bars_in_trade: int = 20  # Max bars before timeout
    ):
        """
        Initialize backtester.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to test
            lookback_days: Days of historical data to test
            max_bars_in_trade: Maximum bars before force-closing trade
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.max_bars_in_trade = max_bars_in_trade
        
        self.fetcher = DataFetcher()
        self.trades: List[BacktestTrade] = []
        
    def _get_candles_needed(self) -> int:
        """Calculate how many candles we need for the lookback period."""
        tf_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '1d': 1440
        }
        minutes_per_candle = tf_minutes.get(self.timeframe, 15)
        minutes_in_period = self.lookback_days * 24 * 60
        return min(minutes_in_period // minutes_per_candle, 1000)  # CCXT limit
    
    def _simulate_trade(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        signal: Signal
    ) -> BacktestTrade:
        """
        Simulate a trade from entry to exit.
        
        Args:
            df: Full OHLCV DataFrame
            entry_idx: Index where trade was entered
            signal: Signal that triggered the trade
        
        Returns:
            BacktestTrade with outcome
        """
        entry_time = df.index[entry_idx]
        entry_price = signal.entry_price
        tp = signal.take_profit
        sl = signal.stop_loss
        
        outcome = TradeOutcome.OPEN
        exit_price = None
        exit_time = None
        bars_held = 0
        
        # Walk forward through candles to find exit
        for i in range(entry_idx + 1, min(entry_idx + self.max_bars_in_trade, len(df))):
            candle = df.iloc[i]
            bars_held += 1
            
            if signal.direction == SignalDirection.LONG:
                # Check SL first (hit if low <= sl)
                if candle['low'] <= sl:
                    outcome = TradeOutcome.SL_HIT
                    exit_price = sl
                    exit_time = df.index[i]
                    break
                # Check TP (hit if high >= tp)
                if candle['high'] >= tp:
                    outcome = TradeOutcome.TP_HIT
                    exit_price = tp
                    exit_time = df.index[i]
                    break
            else:  # SHORT
                # Check SL first (hit if high >= sl)
                if candle['high'] >= sl:
                    outcome = TradeOutcome.SL_HIT
                    exit_price = sl
                    exit_time = df.index[i]
                    break
                # Check TP (hit if low <= tp)
                if candle['low'] <= tp:
                    outcome = TradeOutcome.TP_HIT
                    exit_price = tp
                    exit_time = df.index[i]
                    break
        
        # Timeout if neither TP nor SL hit
        if outcome == TradeOutcome.OPEN:
            outcome = TradeOutcome.TIMEOUT
            exit_idx = min(entry_idx + self.max_bars_in_trade, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            exit_time = df.index[exit_idx]
        
        # Calculate PnL
        if signal.direction == SignalDirection.LONG:
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            risk = entry_price - sl
            reward = exit_price - entry_price
        else:
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100
            risk = sl - entry_price
            reward = entry_price - exit_price
        
        rr_achieved = reward / risk if risk > 0 else 0
        
        return BacktestTrade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=signal.direction,
            symbol=self.symbol,
            timeframe=self.timeframe,
            entry_price=entry_price,
            take_profit=tp,
            stop_loss=sl,
            exit_price=exit_price,
            outcome=outcome,
            pnl_percent=pnl_percent,
            rr_achieved=rr_achieved,
            confidence_score=signal.confidence_score,
            channel_type=signal.channel_type.value,
            bars_held=bars_held
        )
    
    def run(self, apply_filters: bool = True) -> BacktestResult:
        """
        Run the backtest.
        
        Args:
            apply_filters: Whether to apply signal filters
        
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Starting backtest for {self.symbol} {self.timeframe} ({self.lookback_days} days)")
        
        # Fetch historical data
        candles_needed = self._get_candles_needed()
        df = self.fetcher.fetch_ohlcv(self.symbol, self.timeframe, limit=candles_needed)
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data: {len(df)} candles")
        
        logger.info(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        
        self.trades = []
        equity = [100.0]  # Start with $100
        current_equity = 100.0
        
        # Minimum window needed for channel detection
        min_window = 100
        
        # Walk forward through the data
        in_trade = False
        trade_cooldown = 0
        
        for i in range(min_window, len(df) - self.max_bars_in_trade):
            # Skip if in cooldown
            if trade_cooldown > 0:
                trade_cooldown -= 1
                equity.append(current_equity)
                continue
            
            # Get window of data up to current bar
            window_df = df.iloc[i - min_window:i + 1].copy()
            
            # Detect channel
            channel = detect_channel(window_df)
            
            if not channel.is_valid:
                equity.append(current_equity)
                continue
            
            # Check for signal
            signal = check_signal(window_df, channel, self.symbol, self.timeframe, lookback_bars=1)
            
            if signal is None:
                equity.append(current_equity)
                continue
            
            # Apply filters if enabled
            if apply_filters:
                filter_result = apply_signal_filters(
                    window_df,
                    signal.direction,
                    self.timeframe,
                    channel=channel,
                    symbol=self.symbol
                )
                
                if not filter_result.passed or filter_result.confidence_score < MIN_CONFIDENCE_SCORE:
                    equity.append(current_equity)
                    continue
                
                signal.confidence_score = filter_result.confidence_score
            
            # Simulate the trade
            trade = self._simulate_trade(df, i, signal)
            self.trades.append(trade)
            
            # Update equity
            position_size = 0.02  # 2% risk per trade
            pnl_dollar = current_equity * position_size * (trade.pnl_percent / 100 * trade.rr_achieved)
            current_equity += pnl_dollar
            equity.append(current_equity)
            
            # Set cooldown (don't enter new trade while in one)
            trade_cooldown = trade.bars_held
        
        # Fill remaining equity curve
        while len(equity) < len(df) - min_window:
            equity.append(current_equity)
        
        # Calculate results
        result = self._calculate_results(df, equity)
        
        logger.info(f"Backtest complete: {result.total_trades} trades, {result.win_rate:.1%} win rate")
        
        return result
    
    def _calculate_results(self, df: pd.DataFrame, equity: List[float]) -> BacktestResult:
        """Calculate performance metrics from trades."""
        if not self.trades:
            return BacktestResult(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=df.index[0],
                end_date=df.index[-1],
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                timeout_trades=0,
                win_rate=0,
                avg_win_percent=0,
                avg_loss_percent=0,
                profit_factor=0,
                total_return_percent=0,
                max_drawdown_percent=0,
                avg_rr_achieved=0,
                avg_bars_held=0,
                trades=[],
                equity_curve=equity
            )
        
        # Categorize trades
        wins = [t for t in self.trades if t.outcome == TradeOutcome.TP_HIT]
        losses = [t for t in self.trades if t.outcome == TradeOutcome.SL_HIT]
        timeouts = [t for t in self.trades if t.outcome == TradeOutcome.TIMEOUT]
        
        # Win rate
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        # Average win/loss
        avg_win = np.mean([t.pnl_percent for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl_percent for t in losses])) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.pnl_percent for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_percent for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Total return
        total_return = equity[-1] - 100 if equity else 0
        
        # Max drawdown
        peak = 100
        max_dd = 0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Averages
        avg_rr = np.mean([t.rr_achieved for t in self.trades])
        avg_bars = np.mean([t.bars_held for t in self.trades])
        
        return BacktestResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=df.index[0],
            end_date=df.index[-1],
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            timeout_trades=len(timeouts),
            win_rate=win_rate,
            avg_win_percent=avg_win,
            avg_loss_percent=avg_loss,
            profit_factor=profit_factor,
            total_return_percent=total_return,
            max_drawdown_percent=max_dd,
            avg_rr_achieved=avg_rr,
            avg_bars_held=avg_bars,
            trades=self.trades,
            equity_curve=equity
        )
    
    def print_report(self, result: BacktestResult):
        """Print a formatted backtest report."""
        print("\n" + "=" * 60)
        print("📊 BACKTEST REPORT")
        print("=" * 60)
        
        print(f"\n🎯 Symbol: {result.symbol}")
        print(f"⏱️  Timeframe: {result.timeframe}")
        print(f"📅 Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        
        print("\n" + "-" * 40)
        print("📈 PERFORMANCE SUMMARY")
        print("-" * 40)
        
        print(f"Total Trades: {result.total_trades}")
        print(f"   ✅ Winners: {result.winning_trades}")
        print(f"   ❌ Losers: {result.losing_trades}")
        print(f"   ⏱️  Timeouts: {result.timeout_trades}")
        
        print(f"\n🎯 Win Rate: {result.win_rate:.1%}")
        print(f"📊 Profit Factor: {result.profit_factor:.2f}")
        print(f"💰 Total Return: {result.total_return_percent:+.2f}%")
        print(f"📉 Max Drawdown: {result.max_drawdown_percent:.2f}%")
        
        print("\n" + "-" * 40)
        print("📐 TRADE STATISTICS")
        print("-" * 40)
        
        print(f"Avg Win: +{result.avg_win_percent:.2f}%")
        print(f"Avg Loss: -{result.avg_loss_percent:.2f}%")
        print(f"Avg R/R Achieved: {result.avg_rr_achieved:.2f}")
        print(f"Avg Bars Held: {result.avg_bars_held:.1f}")
        
        # Trade breakdown by direction
        longs = [t for t in result.trades if t.direction == SignalDirection.LONG]
        shorts = [t for t in result.trades if t.direction == SignalDirection.SHORT]
        
        print("\n" + "-" * 40)
        print("🔀 DIRECTION BREAKDOWN")
        print("-" * 40)
        
        long_wins = len([t for t in longs if t.outcome == TradeOutcome.TP_HIT])
        short_wins = len([t for t in shorts if t.outcome == TradeOutcome.TP_HIT])
        
        print(f"🟢 Long trades: {len(longs)} ({long_wins} wins, {len(longs) - long_wins} losses)")
        print(f"🔴 Short trades: {len(shorts)} ({short_wins} wins, {len(shorts) - short_wins} losses)")
        
        print("\n" + "=" * 60)
        print("✅ Backtest Complete")
        print("=" * 60)


def run_multi_symbol_backtest(
    symbols: List[str] = None,
    timeframe: str = '15m',
    lookback_days: int = 14
) -> Dict[str, BacktestResult]:
    """
    Run backtest across multiple symbols.
    
    Args:
        symbols: List of symbols to test
        timeframe: Timeframe to use
        lookback_days: Days of data
    
    Returns:
        Dict mapping symbol to BacktestResult
    """
    if symbols is None:
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    results = {}
    
    for symbol in symbols:
        try:
            print(f"\n🔍 Backtesting {symbol}...")
            bt = Backtester(symbol, timeframe, lookback_days)
            result = bt.run()
            results[symbol] = result
            bt.print_report(result)
        except Exception as e:
            print(f"❌ Error backtesting {symbol}: {e}")
            logger.error(f"Backtest error for {symbol}: {e}", exc_info=True)
    
    return results


# Run backtest when executed directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           📊 Crypto Signal Bot - Backtesting                  ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run single symbol backtest
    symbol = 'BTC/USDT:USDT'
    timeframe = '15m'
    days = 7  # 7 days of data
    
    print(f"Testing {symbol} on {timeframe} for {days} days...")
    
    backtester = Backtester(symbol, timeframe, days)
    result = backtester.run(apply_filters=True)
    backtester.print_report(result)
    
    # Show last 5 trades
    if result.trades:
        print("\n📋 Recent Trades (last 5):")
        print("-" * 80)
        for trade in result.trades[-5:]:
            emoji = "✅" if trade.outcome == TradeOutcome.TP_HIT else "❌" if trade.outcome == TradeOutcome.SL_HIT else "⏱️"
            direction = "LONG" if trade.direction == SignalDirection.LONG else "SHORT"
            print(f"  {emoji} {direction} @ ${trade.entry_price:,.2f} → ${trade.exit_price:,.2f} | {trade.pnl_percent:+.2f}% | {trade.bars_held} bars")
