"""
Advanced Signal Filters Module
===============================
Medium-impact filters for enhanced signal accuracy:
- ADX Trend Strength
- Support/Resistance Zones
- Funding Rate Filter
- Liquidation Heatmap Integration
"""
import pandas as pd
import numpy as np
import requests
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        ADX_PERIOD, ADX_TRENDING_THRESHOLD, ADX_RANGING_THRESHOLD,
        SR_LOOKBACK_PERIODS, SR_TOUCH_TOLERANCE,
        FUNDING_RATE_THRESHOLD
    )
except ImportError:
    ADX_PERIOD = 14
    ADX_TRENDING_THRESHOLD = 25
    ADX_RANGING_THRESHOLD = 20
    SR_LOOKBACK_PERIODS = 50
    SR_TOUCH_TOLERANCE = 0.002  # 0.2% tolerance
    FUNDING_RATE_THRESHOLD = 0.0005  # 0.05% considered high

from signal_generator import SignalDirection
from channel_builder import Channel, find_sr_levels, ChannelType

logger = logging.getLogger(__name__)


@dataclass
class AdvancedFilterResult:
    """Result from advanced filters."""
    adx_value: float
    adx_trend_type: str  # 'trending', 'ranging', 'neutral'
    adx_suitable: bool
    
    sr_confluence: bool
    sr_levels_nearby: List[float]
    
    funding_rate: Optional[float]
    funding_suitable: bool
    funding_bias: str  # 'bullish', 'bearish', 'neutral'
    
    liquidation_data: Optional[dict]
    liquidation_suitable: bool
    
    overall_score_modifier: int  # Points to add/subtract


def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index) and +DI/-DI.
    
    ADX measures trend strength:
    - ADX > 25: Strong trend
    - ADX < 20: Ranging/weak trend
    
    Args:
        df: DataFrame with OHLC data
        period: ADX calculation period
    
    Returns:
        Tuple of (ADX, +DI, -DI) Series
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    alpha = 1 / period
    
    atr = true_range.ewm(alpha=alpha, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, min_periods=period).mean()
    
    return adx, plus_di, minus_di


def check_adx_filter(
    df: pd.DataFrame,
    direction: SignalDirection,
    channel_type: ChannelType
) -> Tuple[bool, float, str]:
    """
    Check if ADX supports the signal type.
    
    Logic:
    - ADX > 25 (trending): Favor breakout signals
    - ADX < 20 (ranging): Favor channel bounce signals (our main strategy)
    
    Args:
        df: DataFrame with OHLC data
        direction: Signal direction
        channel_type: Type of channel detected
    
    Returns:
        Tuple of (suitable, adx_value, trend_type)
    """
    adx, plus_di, minus_di = calculate_adx(df)
    current_adx = adx.iloc[-1]
    
    if pd.isna(current_adx):
        return True, 0, 'neutral'
    
    # Determine trend type
    if current_adx > ADX_TRENDING_THRESHOLD:
        trend_type = 'trending'
        # For channel bounce signals, ranging is better
        # But we still allow trending if it's a clear channel
        suitable = channel_type in [ChannelType.ASCENDING, ChannelType.DESCENDING]
    elif current_adx < ADX_RANGING_THRESHOLD:
        trend_type = 'ranging'
        # Ranging markets are IDEAL for channel bounce strategy
        suitable = True
    else:
        trend_type = 'neutral'
        suitable = True
    
    logger.debug(f"ADX: {current_adx:.1f} ({trend_type}) - {'PASS' if suitable else 'FAIL'}")
    
    # Dead Zone Filter 💀
    # If ADX < 15, the market is "dead" (no momentum).
    # We avoid trading unless there is a Volume Spike to wake it up.
    if current_adx < 15:
        # Check Volume
        vol = df['volume'].iloc[-1]
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        
        if vol < (vol_sma * 1.5): # Use 1.5x volume spike as exception
            logger.info(f"Filtered out by DEAD ZONE (ADX {current_adx:.1f} < 15 and no volume spike)")
            return False, current_adx, 'dead'
            
    return suitable, current_adx, trend_type





def check_sr_confluence(
    df: pd.DataFrame,
    channel: Channel,
    tolerance: float = SR_TOUCH_TOLERANCE
) -> Tuple[bool, List[float]]:
    """
    Check if channel lines align with key S/R levels.
    
    Args:
        df: DataFrame with OHLC data
        channel: Channel object
        tolerance: Percentage tolerance for level matching
    
    Returns:
        Tuple of (has_confluence, nearby_levels)
    """
    sr_levels = find_sr_levels(df)
    
    if not sr_levels:
        return False, []
    
    upper_line = channel.get_current_upper()
    lower_line = channel.get_current_lower()
    
    nearby_levels = []
    
    for level in sr_levels:
        # Check if level is near upper or lower channel line
        upper_diff = abs(level - upper_line) / upper_line
        lower_diff = abs(level - lower_line) / lower_line
        
        if upper_diff <= tolerance or lower_diff <= tolerance:
            nearby_levels.append(level)
    
    has_confluence = len(nearby_levels) >= 1
    
    logger.debug(f"S/R confluence: {len(nearby_levels)} levels near channel lines")
    return has_confluence, nearby_levels


def fetch_funding_rate(symbol: str) -> Optional[float]:
    """
    Fetch current funding rate from Binance Futures.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
    
    Returns:
        Funding rate as decimal (e.g., 0.0001 = 0.01%)
    """
    try:
        # Convert symbol format
        binance_symbol = symbol.split('/')[0] + 'USDT'
        
        url = f"https://fapi.binance.com/fapi/v1/fundingRate"
        params = {'symbol': binance_symbol, 'limit': 1}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            funding_rate = float(data[0]['fundingRate'])
            logger.debug(f"Funding rate for {symbol}: {funding_rate:.4%}")
            return funding_rate
        
    except Exception as e:
        logger.debug(f"Could not fetch funding rate for {symbol}: {e}")
    
    return None


def check_funding_rate_filter(
    symbol: str,
    direction: SignalDirection,
    threshold: float = FUNDING_RATE_THRESHOLD,
    pre_fetched_rate: Optional[float] = None
) -> Tuple[bool, Optional[float], str]:
    """
    Check if funding rate supports the signal direction.
    
    Logic:
    - High positive funding → Market is too long → Avoid longs
    - High negative funding → Market is too short → Avoid shorts
    
    Args:
        symbol: Trading symbol
        direction: Signal direction
        threshold: Funding rate threshold (e.g., 0.0005 = 0.05%)
        pre_fetched_rate: Optional pre-fetched rate to avoid API call
    
    Returns:
        Tuple of (suitable, funding_rate, bias)
    """
    if pre_fetched_rate is not None:
        funding_rate = pre_fetched_rate
    else:
        funding_rate = fetch_funding_rate(symbol)
    
    if funding_rate is None:
        return True, None, 'neutral'
    
    # Determine market bias
    if funding_rate > threshold:
        bias = 'bearish'  # Too many longs, expect short squeeze
        suitable = direction.value != "long"
    elif funding_rate < -threshold:
        bias = 'bullish'  # Too many shorts, expect long squeeze
        suitable = direction.value != "short"
    else:
        bias = 'neutral'
        suitable = True
    
    logger.debug(f"Funding rate: {funding_rate:.4%} ({bias}) - {'PASS' if suitable else 'FAIL'}")
    return suitable, funding_rate, bias


def fetch_long_short_ratio(symbol: str) -> Optional[dict]:
    """
    Fetch Long/Short Ratio from Binance Futures (FREE API).
    
    This is MORE useful than liquidation heatmap because it shows
    current trader positioning in real-time.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT:USDT')
    
    Returns:
        Dict with long/short ratio data
    """
    try:
        binance_symbol = symbol.split('/')[0] + 'USDT'
        
        # Top Trader Long/Short Ratio (Accounts)
        url = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"
        params = {'symbol': binance_symbol, 'period': '15m', 'limit': 1}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            latest = data[0]
            long_ratio = float(latest['longAccount'])
            short_ratio = float(latest['shortAccount'])
            ls_ratio = float(latest['longShortRatio'])
            
            logger.debug(f"Long/Short Ratio for {symbol}: {ls_ratio:.2f} (Long: {long_ratio:.1%}, Short: {short_ratio:.1%})")
            
            return {
                'long_account': long_ratio,
                'short_account': short_ratio,
                'long_short_ratio': ls_ratio,
                # Tuned Thresholds: >2.5 is crowded long, <0.5 is crowded short
                'bias': 'crowded_long' if ls_ratio > 2.5 else 
                        'crowded_short' if ls_ratio < 0.50 else 'balanced'
            }
    
    except Exception as e:
        logger.debug(f"Could not fetch Long/Short ratio for {symbol}: {e}")
    
    return None


def fetch_open_interest_change(symbol: str) -> Optional[dict]:
    """
    Fetch Open Interest change from Binance Futures (FREE API).
    
    Rising OI + Rising Price = Strong uptrend
    Rising OI + Falling Price = Strong downtrend
    Falling OI = Trend exhaustion
    
    Args:
        symbol: Trading symbol
    
    Returns:
        Dict with open interest data
    """
    try:
        binance_symbol = symbol.split('/')[0] + 'USDT'
        
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {'symbol': binance_symbol, 'period': '15m', 'limit': 5}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if len(data) >= 2:
            current_oi = float(data[-1]['sumOpenInterest'])
            prev_oi = float(data[0]['sumOpenInterest'])
            
            oi_change = (current_oi - prev_oi) / prev_oi if prev_oi > 0 else 0
            
            logger.debug(f"Open Interest change for {symbol}: {oi_change:.2%}")
            
            return {
                'current_oi': current_oi,
                'prev_oi': prev_oi,
                'oi_change_pct': oi_change,
                'trend': 'increasing' if oi_change > 0.02 else 
                         'decreasing' if oi_change < -0.02 else 'stable'
            }
    
    except Exception as e:
        logger.debug(f"Could not fetch Open Interest for {symbol}: {e}")
    
    return None


def check_market_sentiment_filter(
    symbol: str,
    direction: SignalDirection
) -> Tuple[bool, Optional[dict]]:
    """
    Check market sentiment based on Long/Short Ratio + Open Interest.
    
    BETTER than liquidation heatmap because:
    1. It's real-time current positioning, not past liquidations
    2. Combined with OI, shows strength of current trend
    
    Logic:
    - Crowded longs (ratio > 1.5) + LONG signal = ⚠️ (contrarian risk)
    - Crowded shorts (ratio < 0.67) + SHORT signal = ⚠️ (contrarian risk)
    - Balanced ratio = ✅ (safe entry)
    - Rising OI = ✅ Fresh positions (stronger moves)
    - Falling OI = ⚠️ Closing positions (trend exhaustion)
    
    Args:
        symbol: Trading symbol
        direction: Signal direction
    
    Returns:
        Tuple of (suitable, sentiment_data)
    """
    ls_data = fetch_long_short_ratio(symbol)
    oi_data = fetch_open_interest_change(symbol)
    
    if ls_data is None:
        return True, None
    
    sentiment = {
        'long_short_ratio': ls_data['long_short_ratio'],
        'ls_bias': ls_data['bias'],
        'oi_trend': oi_data['trend'] if oi_data else 'unknown',
        'oi_change': oi_data['oi_change_pct'] if oi_data else 0
    }
    
    # Check if signal goes against crowded positioning (contrarian risk)
    suitable = True
    
    if ls_data['bias'] == 'crowded_long' and direction.value == "long":
        # Too many longs already, risky to add more
        suitable = False
        sentiment['warning'] = 'Crowded long - contrarian risk'
    elif ls_data['bias'] == 'crowded_short' and direction.value == "short":
        # Too many shorts already, risky to add more
        suitable = False
        sentiment['warning'] = 'Crowded short - contrarian risk'
    
    # Bonus check: OI trend
    if oi_data and oi_data['trend'] == 'increasing':
        sentiment['oi_confirmation'] = True
    else:
        sentiment['oi_confirmation'] = False
    
    logger.debug(f"Market sentiment: L/S={ls_data['long_short_ratio']:.2f} ({ls_data['bias']}), OI={oi_data['trend'] if oi_data else 'N/A'} - {'PASS' if suitable else 'FAIL'}")
    return suitable, sentiment


def apply_advanced_filters(
    df: pd.DataFrame,
    channel: Channel,
    direction: SignalDirection,
    symbol: str
) -> AdvancedFilterResult:
    """
    Apply all advanced filters and return comprehensive result.
    
    Args:
        df: DataFrame with OHLCV data
        channel: Channel object
        direction: Signal direction
        symbol: Trading symbol
    
    Returns:
        AdvancedFilterResult with all filter outcomes
    """
    score_modifier = 0
    
    # 1. ADX Filter
    adx_suitable, adx_value, adx_trend = check_adx_filter(df, direction, channel.channel_type)
    if adx_suitable:
        score_modifier += 5
        if adx_trend == 'ranging':
            score_modifier += 5  # Extra bonus for ranging markets
    
    # 2. S/R Confluence
    sr_confluence, sr_levels = check_sr_confluence(df, channel)
    if sr_confluence:
        score_modifier += 10
    
    # 3. Funding Rate
    funding_suitable, funding_rate, funding_bias = check_funding_rate_filter(symbol, direction)
    if funding_suitable:
        score_modifier += 5
    else:
        score_modifier -= 10  # Penalty for going against funding
    
    # 4. Market Sentiment (Long/Short Ratio + Open Interest)
    sentiment_suitable, sentiment_data = check_market_sentiment_filter(symbol, direction)
    if sentiment_suitable:
        score_modifier += 5
        # Bonus for OI confirmation
        if sentiment_data and sentiment_data.get('oi_confirmation'):
            score_modifier += 5
    else:
        score_modifier -= 10  # Penalty for crowded positioning
    
    return AdvancedFilterResult(
        adx_value=adx_value,
        adx_trend_type=adx_trend,
        adx_suitable=adx_suitable,
        sr_confluence=sr_confluence,
        sr_levels_nearby=sr_levels,
        funding_rate=funding_rate,
        funding_suitable=funding_suitable,
        funding_bias=funding_bias,
        liquidation_data=sentiment_data,  # Now contains L/S ratio + OI data
        liquidation_suitable=sentiment_suitable,
        overall_score_modifier=score_modifier
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
    print("Advanced Signal Filters Module Test")
    print("=" * 60)
    
    symbol = 'BTC/USDT:USDT'
    
    print(f"\n📈 Fetching {symbol} data...")
    df = fetch_ohlcv(symbol, '15m', 150)
    channel = detect_channel(df)
    
    if channel.is_valid:
        print(f"✅ Valid {channel.channel_type.value} channel")
        
        print("\n🔍 Testing advanced filters for LONG signal...")
        result = apply_advanced_filters(df, channel, SignalDirection.LONG, symbol)
        
        print(f"\n📊 Advanced Filter Results:")
        print(f"   ADX: {result.adx_value:.1f} ({result.adx_trend_type}) - {'✅' if result.adx_suitable else '⚠️'}")
        print(f"   S/R Confluence: {'✅ Yes' if result.sr_confluence else '❌ No'} ({len(result.sr_levels_nearby)} levels)")
        print(f"   Funding Rate: {result.funding_rate:.4%} ({result.funding_bias}) - {'✅' if result.funding_suitable else '⚠️'}" if result.funding_rate else "   Funding Rate: N/A")
        
        # Show L/S Ratio and OI data
        if result.liquidation_data:
            ls_ratio = result.liquidation_data.get('long_short_ratio', 0)
            ls_bias = result.liquidation_data.get('ls_bias', 'N/A')
            oi_trend = result.liquidation_data.get('oi_trend', 'N/A')
            oi_change = result.liquidation_data.get('oi_change', 0)
            print(f"   L/S Ratio: {ls_ratio:.2f} ({ls_bias}) - {'✅' if result.liquidation_suitable else '⚠️'}")
            print(f"   Open Interest: {oi_trend} ({oi_change:+.2%})")
        else:
            print(f"   Market Sentiment: N/A")
        
        print(f"   Score Modifier: {result.overall_score_modifier:+d} points")
    else:
        print("❌ No valid channel")
    
    print("\n" + "=" * 60)
    print("✅ Advanced filters test complete!")
    print("=" * 60)

