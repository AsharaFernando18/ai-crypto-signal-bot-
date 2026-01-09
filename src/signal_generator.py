"""
Signal Generator Module
========================
Detects trading signals when price touches channel boundaries.
Calculates smart TP/SL levels based on channel geometry.
"""
import pandas as pd
import numpy as np
from typing import Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        TOUCH_TOLERANCE_ATR, SL_BUFFER_ATR, MIN_RR_RATIO
    )
except ImportError:
    TOUCH_TOLERANCE_ATR = 0.1
    SL_BUFFER_ATR = 0.5
    MIN_RR_RATIO = 1.5

from channel_builder import Channel, ChannelType, find_sr_levels
from swing_detector import calculate_atr

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Signal direction enum."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Signal:
    """Represents a trading signal with entry, TP, SL levels."""
    direction: SignalDirection
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Price levels
    entry_price: float
    take_profit: float
    stop_loss: float
    
    # Channel context
    channel_type: ChannelType
    upper_line_price: float
    lower_line_price: float
    
    # Quality metrics
    touch_type: str        # 'wick_touch', 'body_touch', 'rejection'
    rr_ratio: float        # Risk/Reward ratio
    channel_width_pct: float  # Channel width as % of price
    
    # Filter results (NEW)
    confidence_score: float = 50.0  # 0-100 confidence score
    filter_reasons: tuple = ()  # Tuple of filter reason strings
    
    @property
    def risk(self) -> float:
        """Calculate risk in price terms."""
        if self.direction == SignalDirection.LONG:
            return self.entry_price - self.stop_loss
        else:
            return self.stop_loss - self.entry_price
    
    @property
    def reward(self) -> float:
        """Calculate reward in price terms."""
        if self.direction == SignalDirection.LONG:
            return self.take_profit - self.entry_price
        else:
            return self.entry_price - self.take_profit
    
    def format_message(self) -> str:
        """Format signal as Telegram message."""
        emoji = "🟢" if self.direction == SignalDirection.LONG else "🔴"
        direction = "LONG" if self.direction == SignalDirection.LONG else "SHORT"
        
        # Format filter reasons
        filter_text = "\n".join([f"   {r}" for r in self.filter_reasons]) if self.filter_reasons else "   ℹ️ No filters applied"
        
        return f"""
{emoji} {direction} SIGNAL - {self.symbol} ({self.timeframe})
━━━━━━━━━━━━━━━━━━━━
📊 Channel: {self.channel_type.value.capitalize()}
🎯 Touch: {self.touch_type.replace('_', ' ').title()}
━━━━━━━━━━━━━━━━━━━━
💰 Entry: ${self.entry_price:,.2f}
🎯 TP: ${self.take_profit:,.2f}
🛑 SL: ${self.stop_loss:,.2f}
━━━━━━━━━━━━━━━━━━━━
📈 R/R: 1:{self.rr_ratio:.2f}
🎯 Confidence: {self.confidence_score:.0f}/100
━━━━━━━━━━━━━━━━━━━━
📋 Filters:
{filter_text}
━━━━━━━━━━━━━━━━━━━━
⏰ {self.timestamp.strftime('%Y-%m-%d %H:%M')}
"""


def calculate_touch_tolerance(df: pd.DataFrame, atr_multiple: float = TOUCH_TOLERANCE_ATR) -> float:
    """
    Calculate the price tolerance for touch detection.
    
    Args:
        df: DataFrame with OHLCV data
        atr_multiple: Multiple of ATR to use as tolerance
    
    Returns:
        Tolerance in price terms
    """
    atr_values = calculate_atr(df)
    current_atr = atr_values.iloc[-1]
    
    # Smart Dynamic Tolerance 🧠
    # If market is volatile, we widen tolerance to catch fast wicks.
    # If market is calm, we keep it strict to avoid noise.
    multiplier, vol_level = calculate_volatility_multiplier(df, current_atr)
    
    if vol_level == 'high':
        # High volatility: Widen significantly (catch wild wicks)
        final_multiplier = 2.5
    elif vol_level == 'low':
        # Low volatility: Keep Strict (noise reduction)
        final_multiplier = 1.0
    else:
        # Normal: Slightly wider than base
        final_multiplier = 1.5
        
    return current_atr * atr_multiple * final_multiplier


def detect_touch_type(
    candle_high: float,
    candle_low: float,
    candle_open: float,
    candle_close: float,
    line_price: float,
    tolerance: float,
    is_upper_line: bool
) -> Optional[str]:
    """
    Detect the type of touch on a channel line.
    
    Args:
        candle_high: Candle high price
        candle_low: Candle low price
        candle_open: Candle open price
        candle_close: Candle close price
        line_price: Channel line price at this bar
        tolerance: Touch tolerance in price terms
        is_upper_line: True if checking upper line, False for lower
    
    Returns:
        Touch type string or None if no touch
    """
    body_high = max(candle_open, candle_close)
    body_low = min(candle_open, candle_close)
    
    if is_upper_line:
        # Upper line touch: High should reach or exceed line
        if candle_high >= line_price - tolerance:
            # Check if it's a rejection (closed below the line)
            if candle_close < line_price:
                if body_high < line_price:
                    return 'wick_rejection'  # Only wick touched
                else:
                    return 'body_rejection'  # Body touched but closed below
            else:
                return 'breakout_attempt'  # Closed above - might be breakout
    else:
        # Lower line touch: Low should reach or go below line
        if candle_low <= line_price + tolerance:
            # Check if it's a rejection (closed above the line)
            if candle_close > line_price:
                if body_low > line_price:
                    return 'wick_rejection'  # Only wick touched
                else:
                    return 'body_rejection'  # Body touched but closed above
            else:
                return 'breakout_attempt'  # Closed below - might be breakdown
    
    return None


def calculate_volatility_multiplier(df: pd.DataFrame, atr: float) -> tuple:
    """
    Calculate dynamic multiplier based on current volatility.
    
    Logic:
    - Compare current ATR to historical ATR
    - High volatility (ATR > 75th percentile) → Wider stops (1.5x)
    - Normal volatility → Standard stops (1.0x)
    - Low volatility (ATR < 25th percentile) → Tighter stops (0.7x)
    
    Args:
        df: DataFrame with OHLCV data
        atr: Current ATR value
    
    Returns:
        Tuple of (multiplier, volatility_level)
    """
    # Calculate historical ATR for comparison
    atr_series = calculate_atr(df)
    
    if len(atr_series) < 20:
        return 1.0, 'normal'
    
    # Get percentiles
    atr_25 = atr_series.quantile(0.25)
    atr_75 = atr_series.quantile(0.75)
    
    if atr > atr_75:
        # High volatility - widen stops to avoid noise
        multiplier = 1.5
        volatility = 'high'
    elif atr < atr_25:
        # Low volatility - tighten stops for better R/R
        multiplier = 0.7
        volatility = 'low'
    else:
        # Normal volatility
        multiplier = 1.0
        volatility = 'normal'
    
    logger.debug(f"Volatility: {volatility} (ATR={atr:.2f}, 25th={atr_25:.2f}, 75th={atr_75:.2f}) → {multiplier}x multiplier")
    return multiplier, volatility


def calculate_tp_sl(
    direction: SignalDirection,
    entry_price: float,
    upper_line: float,
    lower_line: float,
    atr: float,
    df: pd.DataFrame = None,
    sl_buffer_atr: float = SL_BUFFER_ATR
) -> tuple:
    """
    Calculate Take Profit and Stop Loss levels with DYNAMIC volatility adjustment.
    
    TP Strategy:
    - LONG: Target opposite channel line (upper)
    - SHORT: Target opposite channel line (lower)
    
    SL Strategy (DYNAMIC):
    - Base: Place SL just outside the touched line by ATR buffer
    - High ATR: 1.5x wider stops (avoid getting stopped by noise)
    - Low ATR: 0.7x tighter stops (better risk/reward)
    
    Args:
        direction: Signal direction
        entry_price: Entry price
        upper_line: Upper channel line price
        lower_line: Lower channel line price
        atr: Average True Range value
        df: DataFrame for volatility calculation (optional)
        sl_buffer_atr: Base ATR multiple for SL buffer
    
    Returns:
        Tuple of (take_profit, stop_loss, volatility_info)
    """
    # Calculate dynamic volatility multiplier
    if df is not None:
        vol_multiplier, volatility_level = calculate_volatility_multiplier(df, atr)
    else:
        vol_multiplier, volatility_level = 1.0, 'normal'
    
    # Apply dynamic multiplier to SL buffer
    dynamic_sl_buffer = sl_buffer_atr * vol_multiplier
    sl_buffer = atr * dynamic_sl_buffer
    
    # Calculate channel width for safety margin
    # Note: For parallel channels, width is constant. For others, use current lines.
    width = abs(upper_line - lower_line)
    
    # Import config safely
    try:
        from config import TP_SAFETY_MARGIN
    except ImportError:
        TP_SAFETY_MARGIN = 0.90
        
    if direction.value == "long":
        # LONG: Target upper line, but pull back by (1 - margin)
        # Entry is near lower_line. Target is upper_line.
        # Conservative Target = Entry + (Width * Margin)
        full_target_dist = upper_line - entry_price
        # Initial Target
        take_profit = entry_price + (full_target_dist * TP_SAFETY_MARGIN)
        
        # Obstacle Check
        if df is not None:
            sr_levels = find_sr_levels(df)
            # Find closest S/R level above entry but below target
            obstacles = [lvl for lvl in sr_levels if lvl > (entry_price * 1.002) and lvl < take_profit]
            if obstacles:
                nearest_obstacle = min(obstacles)
                logger.debug(f"Adapting TP for Obstacle: {take_profit:.2f} -> {nearest_obstacle:.2f}")
                take_profit = nearest_obstacle * 0.999 # Just below resistance
                
        stop_loss = lower_line - sl_buffer
        
    else:  # SHORT
        # SHORT: Target lower line, but pull back by (1 - margin)
        # Entry is near upper_line. Target is lower_line.
        # Conservative Target = Entry - (Width * Margin)
        full_target_dist = entry_price - lower_line
        # Initial Target
        take_profit = entry_price - (full_target_dist * TP_SAFETY_MARGIN)
        
        # Obstacle Check
        if df is not None:
            sr_levels = find_sr_levels(df)
            # Find closest S/R level below entry but above target
            obstacles = [lvl for lvl in sr_levels if lvl < (entry_price * 0.998) and lvl > take_profit]
            if obstacles:
                nearest_obstacle = max(obstacles)
                logger.debug(f"Adapting TP for Obstacle: {take_profit:.2f} -> {nearest_obstacle:.2f}")
                take_profit = nearest_obstacle * 1.001 # Just above support
                
        stop_loss = upper_line + sl_buffer
    
    return take_profit, stop_loss, volatility_level


def check_signal(
    df: pd.DataFrame,
    channel: Channel,
    symbol: str = "UNKNOWN",
    timeframe: str = "15m",
    lookback_bars: int = 1  # How many recent bars to check
) -> Optional[Signal]:
    """
    Check if the current candle(s) generate a trading signal.
    
    Args:
        df: DataFrame with OHLCV data
        channel: Channel object with trendlines
        symbol: Trading symbol for the signal
        timeframe: Timeframe string
        lookback_bars: Number of recent bars to check for signals
    
    Returns:
        Signal object if signal detected, None otherwise
    """
    if not channel.is_valid:
        logger.debug("Channel is not valid, skipping signal check")
        return None
    
    # Get ATR for calculations
    atr_series = calculate_atr(df)
    current_atr = atr_series.iloc[-1]
    tolerance = current_atr * TOUCH_TOLERANCE_ATR
    
    # Get current channel line prices
    # IMPORTANT: Apply offset correction - channel was built on sliced data
    channel_offset = len(df) - channel.df_length
    current_idx = channel.df_length - 1  # Last index in channel coordinate system
    current_upper = channel.upper_line.get_price_at(current_idx)
    current_lower = channel.lower_line.get_price_at(current_idx)
    channel_width = current_upper - current_lower
    avg_price = df['close'].iloc[-1]
    channel_width_pct = (channel_width / avg_price) * 100
    
    # Check recent candles for signals
    for i in range(1, lookback_bars + 1):
        idx = -i  # Check from most recent
        
        candle = df.iloc[idx]
        candle_high = candle['high']
        candle_low = candle['low']
        candle_open = candle['open']
        candle_close = candle['close']
        candle_time = df.index[idx]
        
        # Calculate line prices at this bar's position
        # CRITICAL: Convert df index to channel index using offset
        df_bar_idx = len(df) - i
        channel_bar_idx = df_bar_idx - channel_offset
        
        # Skip if outside channel bounds
        if channel_bar_idx < 0 or channel_bar_idx >= channel.df_length:
            continue
            
        upper_at_bar = channel.upper_line.get_price_at(channel_bar_idx)
        lower_at_bar = channel.lower_line.get_price_at(channel_bar_idx)
        
        # Check for SHORT signal (upper line touch/rejection)
        upper_touch = detect_touch_type(
            candle_high, candle_low, candle_open, candle_close,
            upper_at_bar, tolerance, is_upper_line=True
        )
        
        if upper_touch and 'rejection' in upper_touch:
            # Entry should be at the CHANNEL LINE (resistance), not current close
            # This is where the rejection happened - optimal short entry
            entry = upper_at_bar
            tp, sl, volatility = calculate_tp_sl(
                SignalDirection.SHORT, entry,
                upper_at_bar, lower_at_bar, current_atr, df
            )
            
            # Calculate R/R ratio
            risk = sl - entry
            reward = entry - tp
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Filter by minimum R/R (Smart Logic)
            # If Confidence Score is HIGH (>80), accept lower R/R (scalp mode)
            # Confidence score isn't calculated yet (it's in main filters), 
            # so we use a structural proxy: "Is this a Perfect Touch?"
            
            # Base requirement
            required_rr = MIN_RR_RATIO
            
            # If high volatility (likely big move coming), we can be stricter or looser?
            # Actually, let's keep it simple: strict R/R unless we implement the Score check here.
            # Since Score comes LATER, we stick to base for now, but relax slightly for High Vol
            if volatility == 'high':
                required_rr = 1.2 # Allow 1.2 in volatile markets (price moves fast)
            
            if rr_ratio >= required_rr:
                logger.info(f"SHORT signal detected at {upper_at_bar:.2f} ({upper_touch})")
                return Signal(
                    direction=SignalDirection.SHORT,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=candle_time.to_pydatetime() if hasattr(candle_time, 'to_pydatetime') else candle_time,
                    entry_price=entry,
                    take_profit=tp,
                    stop_loss=sl,
                    channel_type=channel.channel_type,
                    upper_line_price=upper_at_bar,
                    lower_line_price=lower_at_bar,
                    touch_type=upper_touch,
                    rr_ratio=rr_ratio,
                    channel_width_pct=channel_width_pct
                )
        
        # Check for LONG signal (lower line touch/rejection)
        lower_touch = detect_touch_type(
            candle_high, candle_low, candle_open, candle_close,
            lower_at_bar, tolerance, is_upper_line=False
        )
        
        if lower_touch and 'rejection' in lower_touch:
            # Entry should be at the CHANNEL LINE (support), not current close
            # This is where the bounce happened - optimal long entry
            entry = lower_at_bar
            tp, sl, volatility = calculate_tp_sl(
                SignalDirection.LONG, entry,
                upper_at_bar, lower_at_bar, current_atr, df
            )
            
            # Calculate R/R ratio
            risk = entry - sl
            reward = tp - entry
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Filter by minimum R/R
            if rr_ratio >= MIN_RR_RATIO:
                logger.info(f"LONG signal detected at {lower_at_bar:.2f} ({lower_touch})")
                return Signal(
                    direction=SignalDirection.LONG,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=candle_time.to_pydatetime() if hasattr(candle_time, 'to_pydatetime') else candle_time,
                    entry_price=entry,
                    take_profit=tp,
                    stop_loss=sl,
                    channel_type=channel.channel_type,
                    upper_line_price=upper_at_bar,
                    lower_line_price=lower_at_bar,
                    touch_type=lower_touch,
                    rr_ratio=rr_ratio,
                    channel_width_pct=channel_width_pct
                )
    
    return None


# Test the module when run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_fetcher import fetch_ohlcv
    from channel_builder import detect_channel
    
    print("=" * 60)
    print("Signal Generator Module Test")
    print("=" * 60)
    
    # Test with multiple symbols
    test_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    for symbol in test_symbols:
        print(f"\n📈 Testing {symbol} (1h)...")
        
        try:
            df = fetch_ohlcv(symbol, '1h', 150)
            channel = detect_channel(df)
            
            if not channel.is_valid:
                print(f"   ⚠️ No valid channel found")
                continue
            
            print(f"   📊 Channel: {channel.channel_type.value}")
            print(f"   📏 Upper: ${channel.get_current_upper():,.2f} | Lower: ${channel.get_current_lower():,.2f}")
            
            # Check for signals (look back 3 bars for testing)
            signal = check_signal(df, channel, symbol, '1h', lookback_bars=3)
            
            if signal:
                print(f"\n   🚨 SIGNAL FOUND!")
                print(signal.format_message())
            else:
                print(f"   ℹ️ No signal at current price")
                
                # Show how close we are to channel boundaries
                current_price = df['close'].iloc[-1]
                dist_upper = channel.get_current_upper() - current_price
                dist_lower = current_price - channel.get_current_lower()
                print(f"   💰 Current: ${current_price:,.2f}")
                print(f"   ↑ Distance to upper: ${dist_upper:,.2f}")
                print(f"   ↓ Distance to lower: ${dist_lower:,.2f}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Signal generator test complete!")
    print("=" * 60)
