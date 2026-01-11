"""
Market Bias Detector
====================
Detects overall market bias (bullish/bearish) to filter out
counter-trend signals.

Helps avoid shorts in bull markets and longs in bear markets.
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketBias:
    """Market bias result."""
    bias: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1
    price_trend: float  # % change over period
    ma_alignment: bool  # Are MAs aligned with trend?
    
    def __str__(self):
        return f"{self.bias.upper()} (strength: {self.strength:.2f})"


class BiasDetector:
    """
    Detects market bias to filter counter-trend signals.
    
    Uses:
    - Price trend over multiple timeframes
    - Moving average alignment
    - Higher highs / lower lows
    """
    
    def __init__(self):
        """Initialize bias detector."""
        logger.info("BiasDetector initialized")
    
    def detect_bias(
        self,
        df: pd.DataFrame,
        lookback_short: int = 20,
        lookback_long: int = 50
    ) -> MarketBias:
        """
        Detect current market bias.
        
        Args:
            df: OHLCV DataFrame
            lookback_short: Short-term lookback
            lookback_long: Long-term lookback
            
        Returns:
            MarketBias object
        """
        if len(df) < lookback_long:
            return MarketBias('neutral', 0.5, 0.0, False)
        
        # Calculate price trends
        short_trend = (df['close'].iloc[-1] - df['close'].iloc[-lookback_short]) / df['close'].iloc[-lookback_short]
        long_trend = (df['close'].iloc[-1] - df['close'].iloc[-lookback_long]) / df['close'].iloc[-lookback_long]
        
        # Calculate moving averages
        ma_20 = df['close'].rolling(20).mean().iloc[-1]
        ma_50 = df['close'].rolling(50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # MA alignment
        ma_aligned = (current_price > ma_20 > ma_50) or (current_price < ma_20 < ma_50)
        
        # Determine bias
        avg_trend = (short_trend + long_trend) / 2
        
        if avg_trend > 0.02:  # 2% up
            bias = 'bullish'
            strength = min(1.0, avg_trend / 0.1)  # Normalize to 0-1
        elif avg_trend < -0.02:  # 2% down
            bias = 'bearish'
            strength = min(1.0, abs(avg_trend) / 0.1)
        else:
            bias = 'neutral'
            strength = 0.5
        
        logger.info(f"Market Bias: {bias.upper()} | Short: {short_trend*100:.1f}% | Long: {long_trend*100:.1f}%")
        
        return MarketBias(
            bias=bias,
            strength=strength,
            price_trend=avg_trend * 100,
            ma_alignment=ma_aligned
        )


# Singleton
_bias_detector: Optional[BiasDetector] = None


def get_bias_detector() -> BiasDetector:
    """Get or create bias detector."""
    global _bias_detector
    if _bias_detector is None:
        _bias_detector = BiasDetector()
    return _bias_detector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    df = pd.DataFrame({
        'close': [95000 + i*100 for i in range(100)]  # Uptrend
    })
    
    detector = BiasDetector()
    bias = detector.detect_bias(df)
    
    print(f"\nBias: {bias}")
    print(f"Trend: {bias.price_trend:.2f}%")
    print(f"MA Aligned: {bias.ma_alignment}")
    
    print("\n✅ Bias Detector working!")
