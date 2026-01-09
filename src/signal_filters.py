"""
Signal Filters Module
======================
Advanced filters to improve signal accuracy:
- Volume confirmation
- RSI filter
- Multi-timeframe confluence
- Candle pattern detection
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        VOLUME_MULTIPLIER, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD
    )
except ImportError:
    VOLUME_MULTIPLIER = 1.5
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

from channel_builder import Channel, ChannelType
from signal_generator import SignalDirection

logger = logging.getLogger(__name__)


class CandlePattern(Enum):
    """Detected candle patterns."""
    NONE = "none"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    PIN_BAR_BULLISH = "pin_bar_bullish"
    PIN_BAR_BEARISH = "pin_bar_bearish"


@dataclass
class FilterResult:
    """Result from applying signal filters."""
    passed: bool
    volume_ok: bool
    rsi_ok: bool
    mtf_confluence: bool
    candle_pattern: CandlePattern
    confidence_score: float  # 0-100
    reasons: List[str]
    
    # Advanced filters (NEW)
    adx_value: float = 0.0
    adx_trend_type: str = 'neutral'
    sr_confluence: bool = False
    funding_rate: Optional[float] = None
    funding_suitable: bool = True
    
    # CVD Order Flow (NEW)
    cvd_confirmation: bool = False
    cvd_divergence: Optional[str] = None


def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    close = df['close']
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Use EMA for smoother RSI
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def check_volume_filter(
    df: pd.DataFrame, 
    lookback: int = 20,
    multiplier: float = VOLUME_MULTIPLIER
) -> Tuple[bool, float]:
    """
    Check if current volume is above average.
    
    Args:
        df: DataFrame with 'volume' column
        lookback: Periods for average calculation
        multiplier: Required multiple of average volume
    
    Returns:
        Tuple of (passed, volume_ratio)
    """
    if len(df) < lookback:
        return True, 1.0  # Not enough data, pass by default
    
    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].iloc[-lookback-1:-1].mean()
    
    if avg_volume == 0:
        return True, 1.0
    
    volume_ratio = current_volume / avg_volume
    passed = volume_ratio >= multiplier
    
    logger.debug(f"Volume filter: {volume_ratio:.2f}x avg (need {multiplier}x) - {'PASS' if passed else 'FAIL'}")
    return passed, volume_ratio


def check_rsi_filter(
    df: pd.DataFrame,
    direction: SignalDirection,
    overbought: float = RSI_OVERBOUGHT,
    oversold: float = RSI_OVERSOLD
) -> Tuple[bool, float]:
    """
    Check if RSI allows the signal direction.
    
    Args:
        df: DataFrame with 'close' column
        direction: Signal direction
        overbought: RSI overbought threshold
        oversold: RSI oversold threshold
    
    Returns:
        Tuple of (passed, rsi_value)
    """
    rsi = calculate_rsi(df)
    current_rsi = rsi.iloc[-1]
    
    if pd.isna(current_rsi):
        return True, 50.0  # Not enough data, pass by default
    
    if direction == SignalDirection.LONG:
        # Long signals: RSI should not be overbought
        passed = current_rsi < overbought
        logger.debug(f"RSI filter (LONG): {current_rsi:.1f} < {overbought} - {'PASS' if passed else 'FAIL'}")
    else:
        # Short signals: RSI should not be oversold
        passed = current_rsi > oversold
        logger.debug(f"RSI filter (SHORT): {current_rsi:.1f} > {oversold} - {'PASS' if passed else 'FAIL'}")
    
    return passed, current_rsi


def check_mtf_confluence(
    signal_direction: SignalDirection,
    signal_timeframe: str,
    higher_tf_channel: Optional[Channel]
) -> Tuple[bool, str]:
    """
    Check multi-timeframe confluence.
    
    Higher probability when:
    - LONG signal + higher TF ascending channel
    - SHORT signal + higher TF descending channel
    
    Args:
        signal_direction: Direction of the signal
        signal_timeframe: Timeframe of the signal
        higher_tf_channel: Channel from higher timeframe (if available)
    
    Returns:
        Tuple of (has_confluence, reason)
    """
    if higher_tf_channel is None or not higher_tf_channel.is_valid:
        return False, "No higher TF data"
    
    htf_type = higher_tf_channel.channel_type
    
    # Check confluence
    if signal_direction == SignalDirection.LONG:
        if htf_type in [ChannelType.ASCENDING, ChannelType.HORIZONTAL]:
            return True, f"LONG aligned with {htf_type.value} HTF"
        else:
            return False, f"LONG against {htf_type.value} HTF"
    else:  # SHORT
        if htf_type in [ChannelType.DESCENDING, ChannelType.HORIZONTAL]:
            return True, f"SHORT aligned with {htf_type.value} HTF"
        else:
            return False, f"SHORT against {htf_type.value} HTF"


def detect_candle_pattern(df: pd.DataFrame) -> CandlePattern:
    """
    Detect reversal candle patterns on the last candle.
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Detected CandlePattern
    """
    if len(df) < 2:
        return CandlePattern.NONE
    
    # Current candle
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    curr_open = curr['open']
    curr_high = curr['high']
    curr_low = curr['low']
    curr_close = curr['close']
    
    prev_open = prev['open']
    prev_close = prev['close']
    
    # Calculate body and wick sizes
    body_size = abs(curr_close - curr_open)
    upper_wick = curr_high - max(curr_open, curr_close)
    lower_wick = min(curr_open, curr_close) - curr_low
    total_range = curr_high - curr_low
    
    if total_range == 0:
        return CandlePattern.NONE
    
    body_ratio = body_size / total_range
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    
    prev_body_size = abs(prev_close - prev_open)
    
    # 1. DOJI - Very small body
    if body_ratio < 0.1:
        logger.debug("Detected: DOJI")
        return CandlePattern.DOJI
    
    # 2. BULLISH ENGULFING - Current green candle engulfs previous red
    if (curr_close > curr_open and  # Current is green
        prev_close < prev_open and  # Previous is red
        curr_close > prev_open and  # Current close above prev open
        curr_open < prev_close and  # Current open below prev close
        body_size > prev_body_size * 1.1):  # Current body larger
        logger.debug("Detected: BULLISH_ENGULFING")
        return CandlePattern.BULLISH_ENGULFING
    
    # 3. BEARISH ENGULFING - Current red candle engulfs previous green
    if (curr_close < curr_open and  # Current is red
        prev_close > prev_open and  # Previous is green
        curr_open > prev_close and  # Current open above prev close
        curr_close < prev_open and  # Current close below prev open
        body_size > prev_body_size * 1.1):  # Current body larger
        logger.debug("Detected: BEARISH_ENGULFING")
        return CandlePattern.BEARISH_ENGULFING
    
    # 4. HAMMER - Small body at top, long lower wick
    if (body_ratio < 0.35 and 
        lower_wick_ratio > 0.5 and 
        upper_wick_ratio < 0.15):
        logger.debug("Detected: HAMMER")
        return CandlePattern.HAMMER
    
    # 5. SHOOTING STAR - Small body at bottom, long upper wick
    if (body_ratio < 0.35 and 
        upper_wick_ratio > 0.5 and 
        lower_wick_ratio < 0.15):
        logger.debug("Detected: SHOOTING_STAR")
        return CandlePattern.SHOOTING_STAR
    
    # 6. PIN BAR BULLISH - Long lower wick, body in upper third
    if (lower_wick_ratio > 0.6 and 
        body_ratio < 0.3 and
        (min(curr_open, curr_close) - curr_low) > 2 * body_size):
        logger.debug("Detected: PIN_BAR_BULLISH")
        return CandlePattern.PIN_BAR_BULLISH
    
    # 7. PIN BAR BEARISH - Long upper wick, body in lower third
    if (upper_wick_ratio > 0.6 and 
        body_ratio < 0.3 and
        (curr_high - max(curr_open, curr_close)) > 2 * body_size):
        logger.debug("Detected: PIN_BAR_BEARISH")
        return CandlePattern.PIN_BAR_BEARISH
    
    return CandlePattern.NONE


def is_pattern_valid_for_direction(
    pattern: CandlePattern, 
    direction: SignalDirection
) -> bool:
    """
    Check if detected pattern supports the signal direction.
    
    Args:
        pattern: Detected candle pattern
        direction: Signal direction
    
    Returns:
        True if pattern supports direction
    """
    bullish_patterns = [
        CandlePattern.BULLISH_ENGULFING,
        CandlePattern.HAMMER,
        CandlePattern.PIN_BAR_BULLISH,
        CandlePattern.DOJI  # Neutral, can go either way
    ]
    
    bearish_patterns = [
        CandlePattern.BEARISH_ENGULFING,
        CandlePattern.SHOOTING_STAR,
        CandlePattern.PIN_BAR_BEARISH,
        CandlePattern.DOJI  # Neutral, can go either way
    ]
    
    if direction == SignalDirection.LONG:
        return pattern in bullish_patterns
    else:
        return pattern in bearish_patterns


def apply_signal_filters(
    df: pd.DataFrame,
    direction: SignalDirection,
    timeframe: str,
    higher_tf_channel: Optional[Channel] = None,
    symbol: str = None,
    channel: Optional[Channel] = None,
    funding_rate: Optional[float] = None
) -> FilterResult:
    """
    Apply all signal filters (basic + advanced) and return comprehensive result.
    
    Args:
        df: DataFrame with OHLCV data
        direction: Signal direction
        timeframe: Signal timeframe
        higher_tf_channel: Channel from higher timeframe (optional)
        symbol: Trading symbol for funding rate lookup
        channel: Current channel for S/R confluence check
    
    Returns:
        FilterResult with all filter outcomes
    """
    reasons = []
    score = 50  # Base score
    
    # === BASIC FILTERS ===
    
    # 1. Volume Filter
    volume_ok, volume_ratio = check_volume_filter(df)
    if volume_ok:
        score += 15
        reasons.append(f"✅ Volume: {volume_ratio:.1f}x avg")
    else:
        score -= 10
        reasons.append(f"⚠️ Low volume: {volume_ratio:.1f}x avg")
    
    # 2. RSI Filter
    rsi_ok, rsi_value = check_rsi_filter(df, direction)
    if rsi_ok:
        score += 15
        reasons.append(f"✅ RSI: {rsi_value:.1f}")
    else:
        score -= 15
        reasons.append(f"❌ RSI extreme: {rsi_value:.1f}")
    
    # 3. Multi-Timeframe Confluence
    mtf_ok, mtf_reason = check_mtf_confluence(direction, timeframe, higher_tf_channel)
    if mtf_ok:
        score += 20
        reasons.append(f"✅ MTF: {mtf_reason}")
    else:
        reasons.append(f"⚠️ MTF: {mtf_reason}")
    
    # 4. Candle Pattern
    pattern = detect_candle_pattern(df)
    pattern_valid = is_pattern_valid_for_direction(pattern, direction)
    
    if pattern != CandlePattern.NONE:
        if pattern_valid:
            score += 20
            reasons.append(f"✅ Pattern: {pattern.value}")
        else:
            score -= 10
            reasons.append(f"⚠️ Wrong pattern: {pattern.value}")
    else:
        reasons.append("ℹ️ No clear pattern")
    
    # === ADVANCED FILTERS ===
    adx_value = 0.0
    adx_trend_type = 'neutral'
    sr_confluence = False
    funding_rate = None
    funding_suitable = True
    cvd_confirmation = False
    cvd_divergence = None
    
    try:
        from advanced_filters import (
            check_adx_filter, check_sr_confluence, check_funding_rate_filter
        )
        
        # 5. ADX Trend Strength
        if channel:
            adx_suitable, adx_value, adx_trend_type = check_adx_filter(df, direction, channel.channel_type)
            if adx_suitable:
                score += 5
                if adx_trend_type == 'ranging':
                    score += 5  # Bonus for ranging markets (ideal for channel trading)
                    reasons.append(f"✅ ADX: {adx_value:.1f} (ranging - ideal)")
                else:
                    reasons.append(f"✅ ADX: {adx_value:.1f} ({adx_trend_type})")
            else:
                score -= 5
                # Extra penalty if ADX is too high (too trending)
                try:
                    from config import ADX_MAX_THRESHOLD
                    if adx_value > ADX_MAX_THRESHOLD:
                        score -= 10
                        reasons.append(f"⚠️ ADX: {adx_value:.1f} (too trending)")
                    else:
                        reasons.append(f"⚠️ ADX: {adx_value:.1f} ({adx_trend_type})")
                except:
                    reasons.append(f"⚠️ ADX: {adx_value:.1f} ({adx_trend_type})")
        
        # 6. S/R Confluence (REQUIRED for high scores)
        if channel:
            sr_confluence, sr_levels = check_sr_confluence(df, channel)
            if sr_confluence:
                score += 10
                reasons.append(f"✅ S/R confluence: {len(sr_levels)} levels")
            else:
                # Penalty for no S/R confluence
                score -= 15
                reasons.append(f"⚠️ No S/R confluence")
        
        # 7. Funding Rate
        if symbol:
            funding_suitable, funding_rate, funding_bias = check_funding_rate_filter(
                symbol, direction, pre_fetched_rate=funding_rate
            )
            if funding_rate is not None:
                if funding_suitable:
                    score += 5
                    reasons.append(f"✅ Funding: {funding_rate:.4%} ({funding_bias})")
                else:
                    score -= 10
                    reasons.append(f"⚠️ Funding: {funding_rate:.4%} ({funding_bias})")
        
        # 8. CVD Order Flow Analysis 🧠
        try:
            from cvd_analyzer import check_cvd_filter
            
            cvd_result = check_cvd_filter(df, direction)
            cvd_confirmation = cvd_result.confirms_signal
            cvd_divergence = cvd_result.divergence_type
            
            if cvd_result.has_divergence:
                # Divergence is a STRONG signal
                score += 15
                reasons.append(f"🔥 CVD: {cvd_result.divergence_type.replace('_', ' ').title()}!")
            elif cvd_confirmation:
                # Trend alignment is good
                score += 10
                reasons.append(f"✅ CVD: {cvd_result.cvd_trend} ({cvd_result.cvd_change_pct:+.1f}%)")
            else:
                # CVD conflicts with signal - STRICTER PENALTY
                score -= 20
                reasons.append(f"⚠️ CVD: {cvd_result.cvd_trend} conflicts with signal")
        except Exception as e:
            logger.debug(f"CVD analysis error: {e}")
            cvd_confirmation = False
            cvd_divergence = None
    
    except ImportError:
        logger.debug("Advanced filters not available, skipping")
    except Exception as e:
        logger.debug(f"Error applying advanced filters: {e}")
    
    # === FINAL DETERMINATION ===
    # Must have: RSI OK + at least one of (volume OK or pattern valid)
    passed = rsi_ok and (volume_ok or pattern_valid)
    
    # Cap score
    score = max(0, min(100, score))
    
    logger.info(f"Filter result: {'PASS' if passed else 'FAIL'} | Score: {score} | {', '.join(reasons)}")
    
    return FilterResult(
        passed=passed,
        volume_ok=volume_ok,
        rsi_ok=rsi_ok,
        mtf_confluence=mtf_ok,
        candle_pattern=pattern,
        confidence_score=score,
        reasons=reasons,
        adx_value=adx_value,
        adx_trend_type=adx_trend_type,
        sr_confluence=sr_confluence,
        funding_rate=funding_rate,
        funding_suitable=funding_suitable,
        cvd_confirmation=cvd_confirmation,
        cvd_divergence=cvd_divergence
    )


# Test the module when run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_fetcher import fetch_ohlcv
    from channel_builder import detect_channel
    
    print("=" * 60)
    print("Signal Filters Module Test")
    print("=" * 60)
    
    symbol = 'BTC/USDT:USDT'
    
    # Fetch data for multiple timeframes
    print(f"\n📈 Fetching {symbol} data...")
    df_15m = fetch_ohlcv(symbol, '15m', 100)
    df_1h = fetch_ohlcv(symbol, '1h', 100)
    
    # Build 1h channel for MTF
    channel_1h = detect_channel(df_1h)
    
    print(f"\n🔍 Testing filters for a simulated LONG signal...")
    result = apply_signal_filters(
        df_15m,
        SignalDirection.LONG,
        '15m',
        channel_1h
    )
    
    print(f"\n📊 Filter Results:")
    print(f"   Passed: {'✅ YES' if result.passed else '❌ NO'}")
    print(f"   Volume OK: {result.volume_ok}")
    print(f"   RSI OK: {result.rsi_ok}")
    print(f"   MTF Confluence: {result.mtf_confluence}")
    print(f"   Candle Pattern: {result.candle_pattern.value}")
    print(f"   Confidence Score: {result.confidence_score}/100")
    print(f"\n   Reasons:")
    for reason in result.reasons:
        print(f"      {reason}")
    
    print("\n" + "=" * 60)
    print("✅ Signal filters test complete!")
    print("=" * 60)
