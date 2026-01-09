"""
Channel Builder Module
=======================
Constructs price channels by fitting trendlines through swing points.
Uses linear regression for robust line fitting.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        MIN_TOUCHES_PER_LINE, MIN_R_SQUARED, 
        SLOPE_THRESHOLD, CHANNEL_LOOKBACK_BARS
    )
except ImportError:
    MIN_TOUCHES_PER_LINE = 2
    MIN_R_SQUARED = 0.5
    SLOPE_THRESHOLD = 0.0001
    CHANNEL_LOOKBACK_BARS = 100

from swing_detector import SwingPoint, detect_all_swings, calculate_atr

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Channel classification based on slope."""
    ASCENDING = "ascending"      # Both lines sloping up
    DESCENDING = "descending"    # Both lines sloping down
    HORIZONTAL = "horizontal"    # Lines roughly flat
    CONVERGING = "converging"    # Lines coming together (wedge)
    DIVERGING = "diverging"      # Lines spreading apart
    INVALID = "invalid"          # Cannot determine valid channel


@dataclass
class TrendLine:
    """Represents a fitted trendline."""
    slope: float              # Price change per bar
    intercept: float          # Y-intercept (price at bar 0)
    r_squared: float          # Quality of fit (0-1)
    touch_points: List[SwingPoint]  # Swing points used to fit
    
    def get_price_at(self, bar_index: int) -> float:
        """Get the projected price at a specific bar index."""
        return self.slope * bar_index + self.intercept
    
    def get_prices(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Get array of projected prices for a range of bars."""
        indices = np.arange(start_idx, end_idx + 1)
        return self.slope * indices + self.intercept


@dataclass 
class Channel:
    """Represents a complete price channel."""
    upper_line: TrendLine
    lower_line: TrendLine
    channel_type: ChannelType
    is_valid: bool
    df_length: int            # Length of DataFrame used
    
    # Convenience properties
    @property
    def upper_slope(self) -> float:
        return self.upper_line.slope
    
    @property
    def lower_slope(self) -> float:
        return self.lower_line.slope
    
    @property
    def avg_slope(self) -> float:
        return (self.upper_slope + self.lower_slope) / 2
    
    def get_current_upper(self) -> float:
        """Get upper line price at most recent bar."""
        return self.upper_line.get_price_at(self.df_length - 1)
    
    def get_current_lower(self) -> float:
        """Get lower line price at most recent bar."""
        return self.lower_line.get_price_at(self.df_length - 1)
    
    def get_channel_width(self) -> float:
        """Get current channel width in price."""
        return self.get_current_upper() - self.get_current_lower()
    
    def get_upper_prices(self) -> np.ndarray:
        """Get upper line prices for all bars."""
        return self.upper_line.get_prices(0, self.df_length - 1)
    
    def get_lower_prices(self) -> np.ndarray:
        """Get lower line prices for all bars."""
        return self.lower_line.get_prices(0, self.df_length - 1)


def fit_trendline(swing_points: List[SwingPoint]) -> Optional[TrendLine]:
    """
    Fit a trendline through swing points using linear regression.
    
    Args:
        swing_points: List of SwingPoint objects to fit
    
    Returns:
        TrendLine object or None if fit fails
    """
    if len(swing_points) < 2:
        return None
    
    # Extract x (bar index) and y (price) values
    x = np.array([sp.index for sp in swing_points])
    y = np.array([sp.price for sp in swing_points])
    
    try:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        return TrendLine(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            touch_points=swing_points
        )
    except Exception as e:
        logger.warning(f"Failed to fit trendline: {e}")
        return None


def classify_channel(upper_line: TrendLine, lower_line: TrendLine, 
                     price_scale: float) -> ChannelType:
    """
    Classify the channel type based on slopes.
    
    Args:
        upper_line: Upper trendline
        lower_line: Lower trendline
        price_scale: Average price for normalizing slope threshold
    
    Returns:
        ChannelType enum value
    """
    # Smart Slope Classification 🧠
    # Standard threshold (0.0001) is too sensitive. Trivial moves become "Trends".
    # We widen the "Horizontal Zone" to treating weak trends as Ranges.
    # This allows Long+Short scalping in slow drifts.
    
    # Base threshold: 0.01% per bar
    base_threshold = price_scale * SLOPE_THRESHOLD
    
    # Smart threshold: 0.025% per bar (2.5x base)
    # If slope is between -0.025% and +0.025%, it's Horizontal.
    relative_threshold = base_threshold * 2.5
    
    upper_slope = upper_line.slope
    lower_slope = lower_line.slope
    
    upper_up = upper_slope > relative_threshold
    upper_down = upper_slope < -relative_threshold
    upper_flat = not upper_up and not upper_down
    
    lower_up = lower_slope > relative_threshold
    lower_down = lower_slope < -relative_threshold
    lower_flat = not lower_up and not lower_down
    
    # Classification logic
    if upper_up and lower_up:
        return ChannelType.ASCENDING
    elif upper_down and lower_down:
        return ChannelType.DESCENDING
    elif upper_flat and lower_flat:
        return ChannelType.HORIZONTAL
    elif upper_down and lower_up:
        return ChannelType.CONVERGING
    elif upper_up and lower_down:
        return ChannelType.DIVERGING
    else:
        # Mixed case - classify by average slope direction
        avg_slope = (upper_slope + lower_slope) / 2
        if avg_slope > relative_threshold:
            return ChannelType.ASCENDING
        elif avg_slope < -relative_threshold:
            return ChannelType.DESCENDING
        else:
            return ChannelType.HORIZONTAL


def validate_channel(channel: Channel, df: pd.DataFrame) -> bool:
    """
    Validate if a channel meets quality criteria.
    
    Args:
        channel: Channel object to validate
        df: Original DataFrame for context
    
    Returns:
        True if channel is valid
    """
    # Check minimum touch points
    if len(channel.upper_line.touch_points) < MIN_TOUCHES_PER_LINE:
        logger.debug(f"Upper line has only {len(channel.upper_line.touch_points)} touch points")
        return False
    
    if len(channel.lower_line.touch_points) < MIN_TOUCHES_PER_LINE:
        logger.debug(f"Lower line has only {len(channel.lower_line.touch_points)} touch points")
        return False
    
    # For horizontal channels, R² can be very low (flat line has no correlation)
    # So we relax R² requirement based on slope magnitude
    avg_price = df['close'].mean()
    price_scale_threshold = avg_price * SLOPE_THRESHOLD * 10  # Relaxed threshold
    
    upper_is_flat = abs(channel.upper_slope) < price_scale_threshold
    lower_is_flat = abs(channel.lower_slope) < price_scale_threshold
    
    # Only check R² for non-flat trendlines
    if not upper_is_flat and channel.upper_line.r_squared < MIN_R_SQUARED:
        logger.debug(f"Upper line R² too low: {channel.upper_line.r_squared:.3f}")
        return False
    
    if not lower_is_flat and channel.lower_line.r_squared < MIN_R_SQUARED:
        logger.debug(f"Lower line R² too low: {channel.lower_line.r_squared:.3f}")
        return False
    
    # Check that upper line is actually above lower line throughout
    current_upper = channel.get_current_upper()
    current_lower = channel.get_current_lower()
    
    if current_upper <= current_lower:
        logger.debug("Upper line is not above lower line")
        return False
    
    # Check channel width is reasonable (0.5% to 25% of price)
    channel_width = channel.get_channel_width()
    width_pct = channel_width / avg_price * 100
    
    if width_pct < 0.3:
        logger.debug(f"Channel too narrow: {width_pct:.2f}%")
        return False
    
    if width_pct > 25:
        logger.debug(f"Channel too wide: {width_pct:.2f}%")
        return False
    
    return True


def build_channel(
    df: pd.DataFrame,
    swing_highs: Optional[List[SwingPoint]] = None,
    swing_lows: Optional[List[SwingPoint]] = None,
    lookback: int = CHANNEL_LOOKBACK_BARS
) -> Channel:
    """
    Build a PARALLEL OUTER Channel from DataFrame.
    
    Logic:
    1. Detect swing points
    2. Calculate "Center Trend" slope using all swing points
    3. Project Upper Line with same slope, anchored to the HIGHEST High
    4. Project Lower Line with same slope, anchored to the LOWEST Low
    
    This creates a channel that contains 100% of the swing points.
    
    Args:
        df: DataFrame with OHLCV data
        swing_highs: Pre-calculated swing highs (optional)
        swing_lows: Pre-calculated swing lows (optional)
        lookback: Number of bars to analyze (uses last N bars)
    
    Returns:
        Channel object
    """
    # Use last N bars if df is longer
    if len(df) > lookback:
        df = df.iloc[-lookback:].copy()
        # Reset index positions for regression
        df_reset = df.reset_index(drop=True)
    else:
        df_reset = df.reset_index(drop=True)
    
    # Detect swings if not provided
    if swing_highs is None or swing_lows is None:
        swing_highs, swing_lows = detect_all_swings(df_reset, lookback=5, min_significance=0)
    
    # Need at least 2 points for each line
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        logger.debug(f"Insufficient swings: {len(swing_highs)} highs, {len(swing_lows)} lows")
        return Channel(
            upper_line=TrendLine(0, 0, 0, []),
            lower_line=TrendLine(0, 0, 0, []),
            channel_type=ChannelType.INVALID,
            is_valid=False,
            df_length=len(df_reset)
        )
    
    # --- PARALLEL OUTER CHANNEL LOGIC ---
    
    # 1. Calculate Center Trend Slope
    # Combine all points to find the general market direction
    all_points = swing_highs + swing_lows
    x = np.array([sp.index for sp in all_points])
    y = np.array([sp.price for sp in all_points])
    
    try:
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        r_squared = r_value ** 2
    except Exception:
        slope, r_squared = 0, 0
        
    # 2. Find Upper Intercept (Highest deviation above slope)
    # y = mx + b  =>  b = y - mx
    # IMPORTANT: Use MAX to ensure line touches the highest high (full containment)
    upper_intercepts = [sp.price - (slope * sp.index) for sp in swing_highs]
    max_intercept = max(upper_intercepts)
    
    # 3. Find Lower Intercept (Lowest deviation below slope)
    # IMPORTANT: Use MIN to ensure line touches the lowest low (full containment)
    lower_intercepts = [sp.price - (slope * sp.index) for sp in swing_lows]
    min_intercept = min(lower_intercepts)
    
    # Construct TrendLines
    upper_line = TrendLine(
        slope=slope,
        intercept=max_intercept,
        r_squared=r_squared, # Shared R^2 of the trend
        touch_points=swing_highs
    )
    
    lower_line = TrendLine(
        slope=slope,
        intercept=min_intercept,
        r_squared=r_squared,
        touch_points=swing_lows
    )
    
    # Classify channel
    avg_price = df_reset['close'].mean()
    channel_type = classify_channel(upper_line, lower_line, avg_price)
    
    # Create channel object
    channel = Channel(
        upper_line=upper_line,
        lower_line=lower_line,
        channel_type=channel_type,
        is_valid=True,
        df_length=len(df_reset)
    )
    
    # Validate
    # Note: Validate checks R^2. For parallel channels, the "fit" R^2 applies to the center line.
    # The outer lines technically "fit" perfectly to at least 1 point, so we skip R^2 check for them individually.
    # We reuse validate_channel but might need to relax checks if needed.
    channel.is_valid = validate_channel(channel, df_reset)
    
    if channel.is_valid:
        logger.info(
            f"Built {channel_type.value} PARALLEL channel | "
            f"Slope: {slope:.4f} | Width: {channel.get_channel_width():.2f}"
        )
    
    return channel


def find_sr_levels(df: pd.DataFrame, lookback: int = 100) -> List[float]:
    """
    Find key Support/Resistance levels from price history.
    
    Uses swing highs/lows and price clusters.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of periods to analyze
    
    Returns:
        List of S/R price levels
    """
    if len(df) < lookback:
        lookback = len(df)
    
    recent_df = df.iloc[-lookback:]
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    
    sr_levels = []
    
    # Find swing highs (local maxima)
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            sr_levels.append(highs[i])
    
    # Find swing lows (local minima)
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            sr_levels.append(lows[i])
    
    # Cluster nearby levels (within 0.5% of each other)
    if sr_levels:
        sr_levels = sorted(sr_levels)
        clustered = []
        current_cluster = [sr_levels[0]]
        
        for level in sr_levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < 0.005:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        clustered.append(np.mean(current_cluster))
        
        return clustered
    
    try:
        from config import MIN_SWING_SIGNIFICANCE
    except ImportError:
        pass

    return []


def detect_channel(df: pd.DataFrame) -> Channel:
    """
    Main entry point: Detect channel from DataFrame.
    
    This is a convenience wrapper around build_channel().
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Channel object with lines and classification
    """
    return build_channel(df)


# Test the module when run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_fetcher import fetch_ohlcv
    
    print("=" * 60)
    print("Channel Builder Module Test")
    print("=" * 60)
    
    # Fetch test data
    print("\n📈 Fetching BTC/USDT 1h data (200 candles)...")
    df = fetch_ohlcv('BTC/USDT:USDT', '1h', 200)
    print(f"✅ Fetched {len(df)} candles")
    
    # Build channel
    print("\n🔧 Building channel...")
    channel = detect_channel(df)
    
    print(f"\n📊 Channel Analysis:")
    print(f"   Type: {channel.channel_type.value.upper()}")
    print(f"   Valid: {'✅ Yes' if channel.is_valid else '❌ No'}")
    
    if channel.is_valid:
        print(f"\n📈 Upper Trendline:")
        print(f"   Slope: {channel.upper_slope:.4f} per bar")
        print(f"   R²: {channel.upper_line.r_squared:.4f}")
        print(f"   Touch points: {len(channel.upper_line.touch_points)}")
        print(f"   Current price level: ${channel.get_current_upper():,.2f}")
        
        print(f"\n📉 Lower Trendline:")
        print(f"   Slope: {channel.lower_slope:.4f} per bar")
        print(f"   R²: {channel.lower_line.r_squared:.4f}")
        print(f"   Touch points: {len(channel.lower_line.touch_points)}")
        print(f"   Current price level: ${channel.get_current_lower():,.2f}")
        
        print(f"\n📏 Channel Width: ${channel.get_channel_width():,.2f}")
        
        current_close = df['close'].iloc[-1]
        print(f"\n💰 Current Price: ${current_close:,.2f}")
        print(f"   Distance to upper: ${channel.get_current_upper() - current_close:,.2f}")
        print(f"   Distance to lower: ${current_close - channel.get_current_lower():,.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Channel builder test complete!")
    print("=" * 60)
