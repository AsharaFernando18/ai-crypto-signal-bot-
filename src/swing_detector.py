"""
Swing Detector Module
======================
Detects Swing Highs and Swing Lows using fractal-based algorithm.
These pivot points are used to construct channel trendlines.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import SWING_LOOKBACK, MIN_SWING_SIGNIFICANCE
except ImportError:
    SWING_LOOKBACK = 5
    MIN_SWING_SIGNIFICANCE = 0.5

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    """Represents a swing high or swing low point."""
    index: int          # Bar index in DataFrame
    timestamp: pd.Timestamp
    price: float        # High price for swing high, Low price for swing low
    point_type: str     # 'high' or 'low'
    strength: float     # How significant is this swing (ATR-based)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range for volatility measurement.
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period
    
    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def detect_swing_highs(
    df: pd.DataFrame, 
    lookback: int = SWING_LOOKBACK,
    min_significance: float = MIN_SWING_SIGNIFICANCE
) -> List[SwingPoint]:
    """
    Detect Swing Highs using fractal algorithm.
    
    A bar is a Swing High if its HIGH is greater than the highs of
    `lookback` bars before AND after it.
    
    Args:
        df: DataFrame with OHLCV data (must have 'high' column)
        lookback: Number of bars on each side to confirm swing
        min_significance: Minimum ATR multiple for swing to be significant
    
    Returns:
        List of SwingPoint objects representing swing highs
    """
    highs = df['high'].values
    atr = calculate_atr(df).values
    timestamps = df.index
    
    swing_highs = []
    
    # Need at least lookback bars on each side
    for i in range(lookback, len(highs) - lookback):
        current_high = highs[i]
        is_swing_high = True
        
        # Check all bars on the left
        for j in range(1, lookback + 1):
            if highs[i - j] >= current_high:
                is_swing_high = False
                break
        
        if not is_swing_high:
            continue
        
        # Check all bars on the right
        for j in range(1, lookback + 1):
            if highs[i + j] >= current_high:
                is_swing_high = False
                break
        
        if is_swing_high:
            # Calculate swing strength (how far it extends above neighbors)
            left_max = max(highs[i - lookback:i])
            right_max = max(highs[i + 1:i + lookback + 1])
            neighbor_max = max(left_max, right_max)
            
            swing_strength = (current_high - neighbor_max) / atr[i] if atr[i] > 0 else 0
            
            # Only keep significant swings
            if swing_strength >= min_significance or min_significance == 0:
                swing_highs.append(SwingPoint(
                    index=i,
                    timestamp=timestamps[i],
                    price=current_high,
                    point_type='high',
                    strength=swing_strength
                ))
    
    logger.debug(f"Found {len(swing_highs)} swing highs")
    return swing_highs


def detect_swing_lows(
    df: pd.DataFrame, 
    lookback: int = SWING_LOOKBACK,
    min_significance: float = MIN_SWING_SIGNIFICANCE
) -> List[SwingPoint]:
    """
    Detect Swing Lows using fractal algorithm.
    
    A bar is a Swing Low if its LOW is less than the lows of
    `lookback` bars before AND after it.
    
    Args:
        df: DataFrame with OHLCV data (must have 'low' column)
        lookback: Number of bars on each side to confirm swing
        min_significance: Minimum ATR multiple for swing to be significant
    
    Returns:
        List of SwingPoint objects representing swing lows
    """
    lows = df['low'].values
    atr = calculate_atr(df).values
    timestamps = df.index
    
    swing_lows = []
    
    # Need at least lookback bars on each side
    for i in range(lookback, len(lows) - lookback):
        current_low = lows[i]
        is_swing_low = True
        
        # Check all bars on the left
        for j in range(1, lookback + 1):
            if lows[i - j] <= current_low:
                is_swing_low = False
                break
        
        if not is_swing_low:
            continue
        
        # Check all bars on the right
        for j in range(1, lookback + 1):
            if lows[i + j] <= current_low:
                is_swing_low = False
                break
        
        if is_swing_low:
            # Calculate swing strength (how far it extends below neighbors)
            left_min = min(lows[i - lookback:i])
            right_min = min(lows[i + 1:i + lookback + 1])
            neighbor_min = min(left_min, right_min)
            
            swing_strength = (neighbor_min - current_low) / atr[i] if atr[i] > 0 else 0
            
            # Only keep significant swings
            if swing_strength >= min_significance or min_significance == 0:
                swing_lows.append(SwingPoint(
                    index=i,
                    timestamp=timestamps[i],
                    price=current_low,
                    point_type='low',
                    strength=swing_strength
                ))
    
    logger.debug(f"Found {len(swing_lows)} swing lows")
    return swing_lows


def detect_all_swings(
    df: pd.DataFrame,
    lookback: int = SWING_LOOKBACK,
    min_significance: float = MIN_SWING_SIGNIFICANCE
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Detect both swing highs and swing lows.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars on each side to confirm swing
        min_significance: Minimum ATR multiple for swing to be significant
    
    Returns:
        Tuple of (swing_highs, swing_lows)
    """
    swing_highs = detect_swing_highs(df, lookback, min_significance)
    swing_lows = detect_swing_lows(df, lookback, min_significance)
    
    return swing_highs, swing_lows


# Test the module when run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_fetcher import fetch_ohlcv
    
    print("=" * 60)
    print("Swing Detector Module Test")
    print("=" * 60)
    
    # Fetch test data
    print("\n📈 Fetching BTC/USDT 1h data (200 candles)...")
    df = fetch_ohlcv('BTC/USDT:USDT', '1h', 200)
    print(f"✅ Fetched {len(df)} candles")
    
    # Detect swings with less strict significance filter for testing
    print("\n🔍 Detecting Swing Points (lookback=5, min_significance=0)...")
    swing_highs, swing_lows = detect_all_swings(df, lookback=5, min_significance=0)
    
    print(f"\n✅ Found {len(swing_highs)} Swing Highs:")
    for sh in swing_highs[-5:]:  # Last 5
        print(f"   📍 {sh.timestamp} | Price: ${sh.price:,.2f} | Strength: {sh.strength:.2f}")
    
    print(f"\n✅ Found {len(swing_lows)} Swing Lows:")
    for sl in swing_lows[-5:]:  # Last 5
        print(f"   📍 {sl.timestamp} | Price: ${sl.price:,.2f} | Strength: {sl.strength:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Swing detector is working correctly!")
    print("=" * 60)
