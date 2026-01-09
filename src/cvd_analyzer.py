"""
CVD Analyzer Module
====================
Calculates Cumulative Volume Delta (CVD) and detects
divergences between price action and order flow.

CVD = Running sum of (Buy Volume - Sell Volume)
Divergence = Price and CVD disagree → Smart Money signal
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_generator import SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class CVDResult:
    """Result from CVD analysis."""
    cvd_trend: str  # 'bullish', 'bearish', 'neutral'
    has_divergence: bool
    divergence_type: Optional[str]  # 'bullish_divergence', 'bearish_divergence', None
    cvd_value: float  # Current CVD value
    cvd_change_pct: float  # Recent CVD change percentage
    confirms_signal: bool  # Does CVD confirm the signal direction?


def calculate_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Cumulative Volume Delta from OHLCV data.
    
    Uses "Close Location Value" method to estimate buy/sell pressure:
    - If candle closes near high → More buying pressure
    - If candle closes near low → More selling pressure
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Series with CVD values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Calculate Close Location Value (0 to 1)
    # 1 = Close at High (bullish), 0 = Close at Low (bearish)
    range_hl = high - low
    
    # Avoid division by zero for doji candles
    clv = np.where(
        range_hl > 0,
        (close - low) / range_hl,
        0.5  # Doji = neutral
    )
    
    # Calculate Delta (positive = buying, negative = selling)
    # Delta = Volume * (CLV * 2 - 1) to normalize to [-1, 1]
    delta = volume * (clv * 2 - 1)
    
    # Cumulative sum
    cvd = delta.cumsum()
    
    return cvd


def get_cvd_trend(df: pd.DataFrame, lookback: int = 20) -> Tuple[str, float]:
    """
    Determine the current CVD trend direction.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to analyze
    
    Returns:
        Tuple of (trend_direction, change_percentage)
    """
    cvd = calculate_cvd(df)
    
    if len(cvd) < lookback:
        return 'neutral', 0.0
    
    recent_cvd = cvd.iloc[-lookback:]
    start_cvd = recent_cvd.iloc[0]
    end_cvd = recent_cvd.iloc[-1]
    
    # Calculate percentage change
    if abs(start_cvd) > 0:
        change_pct = (end_cvd - start_cvd) / abs(start_cvd) * 100
    else:
        change_pct = 100 if end_cvd > 0 else -100 if end_cvd < 0 else 0
    
    # Determine trend based on CVD direction
    if change_pct > 10:
        trend = 'bullish'
    elif change_pct < -10:
        trend = 'bearish'
    else:
        trend = 'neutral'
    
    return trend, change_pct


def detect_cvd_divergence(
    df: pd.DataFrame,
    direction: SignalDirection,
    lookback: int = 20
) -> Tuple[bool, Optional[str]]:
    """
    Detect divergence between price and CVD.
    
    Bullish Divergence: Price Lower Low + CVD Higher Low = Hidden Accumulation
    Bearish Divergence: Price Higher High + CVD Lower High = Hidden Distribution
    
    Args:
        df: DataFrame with OHLCV data
        direction: Expected signal direction
        lookback: Number of bars to analyze
    
    Returns:
        Tuple of (has_divergence, divergence_type)
    """
    if len(df) < lookback * 2:
        return False, None
    
    cvd = calculate_cvd(df)
    
    # Get recent and previous sections
    mid_point = len(df) - lookback
    
    recent_price_low = df['low'].iloc[-lookback:].min()
    prev_price_low = df['low'].iloc[mid_point - lookback:mid_point].min()
    
    recent_price_high = df['high'].iloc[-lookback:].max()
    prev_price_high = df['high'].iloc[mid_point - lookback:mid_point].max()
    
    recent_cvd_min = cvd.iloc[-lookback:].min()
    prev_cvd_min = cvd.iloc[mid_point - lookback:mid_point].min()
    
    recent_cvd_max = cvd.iloc[-lookback:].max()
    prev_cvd_max = cvd.iloc[mid_point - lookback:mid_point].max()
    
    # Check for Bullish Divergence (for Long signals)
    # Price makes Lower Low, but CVD makes Higher Low
    if direction == SignalDirection.LONG:
        if recent_price_low < prev_price_low and recent_cvd_min > prev_cvd_min:
            logger.info("🔍 BULLISH CVD DIVERGENCE detected! Hidden accumulation.")
            return True, 'bullish_divergence'
    
    # Check for Bearish Divergence (for Short signals)
    # Price makes Higher High, but CVD makes Lower High
    if direction == SignalDirection.SHORT:
        if recent_price_high > prev_price_high and recent_cvd_max < prev_cvd_max:
            logger.info("🔍 BEARISH CVD DIVERGENCE detected! Hidden distribution.")
            return True, 'bearish_divergence'
    
    return False, None


def check_cvd_filter(
    df: pd.DataFrame,
    direction: SignalDirection,
    lookback: int = 20
) -> CVDResult:
    """
    Main CVD filter function for signal validation.
    
    Checks:
    1. CVD Trend alignment with signal direction
    2. CVD Divergence for extra confirmation
    
    Args:
        df: DataFrame with OHLCV data
        direction: Signal direction to validate
        lookback: Number of bars to analyze
    
    Returns:
        CVDResult with all analysis data
    """
    cvd = calculate_cvd(df)
    current_cvd = cvd.iloc[-1]
    
    # Get CVD trend
    cvd_trend, cvd_change = get_cvd_trend(df, lookback)
    
    # Check for divergence
    has_divergence, divergence_type = detect_cvd_divergence(df, direction, lookback)
    
    # Determine if CVD confirms the signal
    confirms = False
    
    if direction == SignalDirection.LONG:
        # Long confirmed if CVD is bullish or bullish divergence
        if cvd_trend == 'bullish' or divergence_type == 'bullish_divergence':
            confirms = True
    else:  # SHORT
        # Short confirmed if CVD is bearish or bearish divergence
        if cvd_trend == 'bearish' or divergence_type == 'bearish_divergence':
            confirms = True
    
    logger.debug(
        f"CVD Analysis: Trend={cvd_trend}, Change={cvd_change:.1f}%, "
        f"Divergence={divergence_type}, Confirms={confirms}"
    )
    
    return CVDResult(
        cvd_trend=cvd_trend,
        has_divergence=has_divergence,
        divergence_type=divergence_type,
        cvd_value=current_cvd,
        cvd_change_pct=cvd_change,
        confirms_signal=confirms
    )


# Test the module when run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_fetcher import fetch_ohlcv
    
    print("=" * 60)
    print("CVD Analyzer Module Test")
    print("=" * 60)
    
    symbol = 'BTC/USDT:USDT'
    
    print(f"\n📈 Fetching {symbol} 15m data...")
    df = fetch_ohlcv(symbol, '15m', 150)
    
    print(f"\n🔍 Calculating CVD...")
    cvd = calculate_cvd(df)
    print(f"   Current CVD: {cvd.iloc[-1]:,.0f}")
    print(f"   CVD 20 bars ago: {cvd.iloc[-20]:,.0f}")
    
    print(f"\n📊 Checking CVD Trend...")
    trend, change = get_cvd_trend(df)
    print(f"   Trend: {trend}")
    print(f"   Change: {change:.1f}%")
    
    print(f"\n🔎 Testing CVD Filter for LONG signal...")
    result = check_cvd_filter(df, SignalDirection.LONG)
    print(f"   Trend: {result.cvd_trend}")
    print(f"   Divergence: {result.divergence_type}")
    print(f"   Confirms LONG: {'✅' if result.confirms_signal else '❌'}")
    
    print("\n" + "=" * 60)
    print("✅ CVD Analyzer test complete!")
    print("=" * 60)
