"""
Market Regime Detector
======================
Detects current market regime to adapt trading strategy.

Regimes Detected:
- Trend: Trending (bull/bear) vs Ranging
- Volatility: High / Normal / Low
- Liquidity: High / Normal / Low

Strategy adapts based on regime for optimal performance.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)

try:
    from config import (
        ADX_TRENDING_THRESHOLD, ADX_RANGING_THRESHOLD,
        ADX_PERIOD
    )
except ImportError:
    ADX_TRENDING_THRESHOLD = 25
    ADX_RANGING_THRESHOLD = 20
    ADX_PERIOD = 14


class MarketRegime:
    """Market regime classifications."""
    # Trend regimes
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    
    # Volatility regimes
    HIGH_VOLATILITY = "high_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    LOW_VOLATILITY = "low_volatility"
    
    # Liquidity regimes
    HIGH_LIQUIDITY = "high_liquidity"
    NORMAL_LIQUIDITY = "normal_liquidity"
    LOW_LIQUIDITY = "low_liquidity"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    trend: str
    volatility: str
    liquidity: str
    
    # Metrics
    adx_value: float
    atr_percentile: float
    volume_percentile: float
    price_change_20: float
    
    # Confidence
    confidence: float
    
    # Strategy adjustments
    confidence_adjustment: int
    sl_multiplier: float
    position_size_multiplier: float
    should_trade: bool
    
    def __str__(self):
        return (f"Regime: {self.trend.upper()} | "
                f"Vol: {self.volatility} | "
                f"Liq: {self.liquidity} | "
                f"Confidence: {self.confidence:.0f}%")


class RegimeDetector:
    """
    Detects market regime and provides strategy adjustments.
    
    Uses:
    - ADX for trend strength
    - ATR percentile for volatility
    - Volume percentile for liquidity
    - Price action for trend direction
    """
    
    def __init__(self):
        """Initialize regime detector."""
        self.min_data_points = 100
        logger.info("RegimeDetector initialized")
    
    def calculate_adx(self, df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures trend strength (0-100):
        - 0-20: Weak/no trend (ranging)
        - 20-25: Emerging trend
        - 25-50: Strong trend
        - 50+: Very strong trend
        
        Args:
            df: DataFrame with high, low, close
            period: ADX period (default 14)
            
        Returns:
            Series with ADX values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with Wilder's smoothing
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame with high, low, close
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    def detect_trend_regime(
        self, 
        df: pd.DataFrame,
        adx: pd.Series
    ) -> Tuple[str, float]:
        """
        Detect trend regime using ADX and price action.
        
        Args:
            df: DataFrame with OHLCV
            adx: ADX series
            
        Returns:
            (regime, confidence)
        """
        current_adx = adx.iloc[-1]
        
        # Price change over 20 periods
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        
        # Determine regime
        if current_adx > ADX_TRENDING_THRESHOLD:
            # Strong trend
            if price_change > 0.05:  # 5% up
                regime = MarketRegime.TRENDING_BULL
                confidence = min(100, (current_adx / 50) * 100)
            elif price_change < -0.05:  # 5% down
                regime = MarketRegime.TRENDING_BEAR
                confidence = min(100, (current_adx / 50) * 100)
            else:
                # ADX high but price flat - transition period
                regime = MarketRegime.RANGING
                confidence = 50
        else:
            # Weak trend = ranging
            regime = MarketRegime.RANGING
            confidence = min(100, ((ADX_TRENDING_THRESHOLD - current_adx) / ADX_TRENDING_THRESHOLD) * 100)
        
        return regime, confidence
    
    def detect_volatility_regime(
        self,
        df: pd.DataFrame,
        atr: pd.Series
    ) -> Tuple[str, float]:
        """
        Detect volatility regime using ATR percentile.
        
        Args:
            df: DataFrame with OHLCV
            atr: ATR series
            
        Returns:
            (regime, percentile)
        """
        current_atr = atr.iloc[-1]
        
        # Calculate percentile (where current ATR ranks)
        atr_percentile = percentileofscore(atr.dropna(), current_atr)
        
        if atr_percentile > 75:
            regime = MarketRegime.HIGH_VOLATILITY
        elif atr_percentile < 25:
            regime = MarketRegime.LOW_VOLATILITY
        else:
            regime = MarketRegime.NORMAL_VOLATILITY
        
        return regime, atr_percentile
    
    def detect_liquidity_regime(
        self,
        df: pd.DataFrame
    ) -> Tuple[str, float]:
        """
        Detect liquidity regime using volume percentile.
        
        Args:
            df: DataFrame with volume
            
        Returns:
            (regime, percentile)
        """
        current_volume = df['volume'].iloc[-1]
        
        # Calculate percentile
        volume_percentile = percentileofscore(df['volume'].dropna(), current_volume)
        
        if volume_percentile > 75:
            regime = MarketRegime.HIGH_LIQUIDITY
        elif volume_percentile < 25:
            regime = MarketRegime.LOW_LIQUIDITY
        else:
            regime = MarketRegime.NORMAL_LIQUIDITY
        
        return regime, volume_percentile
    
    def get_strategy_adjustments(
        self,
        trend: str,
        volatility: str,
        liquidity: str
    ) -> Dict:
        """
        Get strategy adjustments based on regime.
        
        Args:
            trend: Trend regime
            volatility: Volatility regime
            liquidity: Liquidity regime
            
        Returns:
            Dict with adjustments
        """
        adjustments = {
            'confidence_adjustment': 0,
            'sl_multiplier': 1.0,
            'position_size_multiplier': 1.0,
            'should_trade': True
        }
        
        # Trend adjustments
        if trend == MarketRegime.TRENDING_BULL or trend == MarketRegime.TRENDING_BEAR:
            # Channels work poorly in strong trends
            adjustments['confidence_adjustment'] -= 15
            logger.debug("Trending market: -15 confidence (channels prefer ranging)")
            
        elif trend == MarketRegime.RANGING:
            # Channels work best in ranging markets
            adjustments['confidence_adjustment'] += 10
            logger.debug("Ranging market: +10 confidence (ideal for channels)")
        
        # Volatility adjustments
        if volatility == MarketRegime.HIGH_VOLATILITY:
            # Wider stops needed
            adjustments['sl_multiplier'] = 1.5
            adjustments['confidence_adjustment'] -= 5
            logger.debug("High volatility: 1.5x stops, -5 confidence")
            
        elif volatility == MarketRegime.LOW_VOLATILITY:
            # Tighter stops acceptable
            adjustments['sl_multiplier'] = 0.8
            adjustments['confidence_adjustment'] += 5
            logger.debug("Low volatility: 0.8x stops, +5 confidence")
        
        # Liquidity adjustments
        if liquidity == MarketRegime.LOW_LIQUIDITY:
            # Reduce size or skip
            adjustments['position_size_multiplier'] = 0.5
            adjustments['confidence_adjustment'] -= 10
            logger.debug("Low liquidity: 0.5x size, -10 confidence")
            
            # If very low confidence, don't trade
            if adjustments['confidence_adjustment'] < -20:
                adjustments['should_trade'] = False
                logger.warning("Regime unsuitable for trading")
                
        elif liquidity == MarketRegime.HIGH_LIQUIDITY:
            # Can trade with confidence
            adjustments['confidence_adjustment'] += 5
            logger.debug("High liquidity: +5 confidence")
        
        return adjustments
    
    def detect_regime(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> Optional[RegimeAnalysis]:
        """
        Detect complete market regime.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for logging
            
        Returns:
            RegimeAnalysis object or None if insufficient data
        """
        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data for regime detection: {len(df)} < {self.min_data_points}")
            return None
        
        try:
            # Calculate indicators
            adx = self.calculate_adx(df)
            atr = self.calculate_atr(df)
            
            # Detect regimes
            trend, trend_conf = self.detect_trend_regime(df, adx)
            volatility, atr_pct = self.detect_volatility_regime(df, atr)
            liquidity, vol_pct = self.detect_liquidity_regime(df)
            
            # Get strategy adjustments
            adjustments = self.get_strategy_adjustments(trend, volatility, liquidity)
            
            # Calculate overall confidence
            overall_confidence = trend_conf
            
            # Price change for reference
            price_change_20 = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            
            analysis = RegimeAnalysis(
                trend=trend,
                volatility=volatility,
                liquidity=liquidity,
                adx_value=adx.iloc[-1],
                atr_percentile=atr_pct,
                volume_percentile=vol_pct,
                price_change_20=price_change_20,
                confidence=overall_confidence,
                confidence_adjustment=adjustments['confidence_adjustment'],
                sl_multiplier=adjustments['sl_multiplier'],
                position_size_multiplier=adjustments['position_size_multiplier'],
                should_trade=adjustments['should_trade']
            )
            
            logger.info(f"{symbol} Regime: {analysis}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}")
            return None


# Singleton instance
_regime_detector: Optional[RegimeDetector] = None


def get_regime_detector() -> RegimeDetector:
    """Get or create the global regime detector."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = RegimeDetector()
    return _regime_detector


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'high': np.random.uniform(95000, 96000, 150),
        'low': np.random.uniform(94000, 95000, 150),
        'close': np.random.uniform(94500, 95500, 150),
        'volume': np.random.uniform(10000, 100000, 150)
    })
    
    print("="*60)
    print("Regime Detector Test")
    print("="*60)
    
    detector = RegimeDetector()
    regime = detector.detect_regime(test_df, "BTC/USDT")
    
    if regime:
        print(f"\n{regime}")
        print(f"\nAdjustments:")
        print(f"  Confidence: {regime.confidence_adjustment:+d}")
        print(f"  SL Multiplier: {regime.sl_multiplier:.1f}x")
        print(f"  Size Multiplier: {regime.position_size_multiplier:.1f}x")
        print(f"  Should Trade: {regime.should_trade}")
    
    print("\n✅ Regime Detector working!")
