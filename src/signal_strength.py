"""
Signal Strength Classifier - Phase 3.2
=======================================
Multi-confirmation signal strength classification.

Analyzes multiple factors to rank signal quality:
- Channel quality
- Volume confirmation
- RSI divergence
- CVD alignment
- S/R confluence
- MTF alignment

Classifies signals as: WEAK, MODERATE, STRONG, VERY_STRONG
"""
import logging
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength classifications."""
    WEAK = 1          # 1-2 confirmations
    MODERATE = 2      # 3 confirmations
    STRONG = 3        # 4 confirmations
    VERY_STRONG = 4   # 5+ confirmations


@dataclass
class ConfirmationResult:
    """Result of signal confirmation analysis."""
    strength: SignalStrength
    confirmations: List[str]
    confirmation_count: int
    confidence_boost: int
    
    def __str__(self):
        return f"{self.strength.name}: {self.confirmation_count} confirmations ({', '.join(self.confirmations)})"


class SignalStrengthClassifier:
    """
    Classifies signal strength based on multiple confirmations.
    
    Confirmations checked:
    1. Channel quality (R² > 0.7)
    2. Volume spike (> 1.5x average)
    3. RSI divergence
    4. CVD alignment
    5. S/R confluence
    6. MTF alignment
    7. Momentum confirmation
    """
    
    def __init__(self):
        """Initialize classifier."""
        self.min_confirmations = {
            SignalStrength.WEAK: 1,
            SignalStrength.MODERATE: 3,
            SignalStrength.STRONG: 4,
            SignalStrength.VERY_STRONG: 5
        }
        
        logger.info("SignalStrengthClassifier initialized")
    
    def check_channel_quality(self, channel) -> bool:
        """Check if channel has high quality (R² > 0.7)."""
        try:
            return channel.r_squared > 0.7
        except:
            return False
    
    def check_volume_confirmation(self, df, volume_multiplier: float = 1.5) -> bool:
        """Check if current volume is above average."""
        try:
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            return current_volume > avg_volume * volume_multiplier
        except:
            return False
    
    def check_rsi_divergence(self, df, signal_direction: str) -> bool:
        """Check for RSI divergence (simplified)."""
        try:
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Check for divergence pattern
            price_trend = df['close'].iloc[-5:].is_monotonic_increasing
            rsi_trend = rsi.iloc[-5:].is_monotonic_increasing
            
            if signal_direction == "long":
                # Bullish divergence: price down, RSI up
                return not price_trend and rsi_trend
            else:
                # Bearish divergence: price up, RSI down
                return price_trend and not rsi_trend
        except:
            return False
    
    def check_cvd_alignment(self, signal) -> bool:
        """Check if CVD aligns with signal direction."""
        try:
            # Check if signal has CVD data
            if hasattr(signal, 'cvd_trend'):
                if signal.direction.value == "long":
                    return signal.cvd_trend == "bullish"
                else:
                    return signal.cvd_trend == "bearish"
            return False
        except:
            return False
    
    def check_sr_confluence(self, signal) -> bool:
        """Check if signal is near support/resistance."""
        try:
            # Check if signal has S/R data
            return hasattr(signal, 'near_sr') and signal.near_sr
        except:
            return False
    
    def check_mtf_alignment(self, signal, higher_tf_channel) -> bool:
        """Check if higher timeframe aligns."""
        try:
            if not higher_tf_channel or not higher_tf_channel.is_valid:
                return False
            
            # Check alignment
            if signal.direction.value == "long":
                return higher_tf_channel.channel_type.value == "ascending"
            else:
                return higher_tf_channel.channel_type.value == "descending"
        except:
            return False
    
    def check_momentum_confirmation(self, df, signal_direction: str) -> bool:
        """Check if momentum confirms direction."""
        try:
            # Simple momentum: 5-period price change
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            if signal_direction == "long":
                return momentum > 0.001  # Positive momentum
            else:
                return momentum < -0.001  # Negative momentum
        except:
            return False
    
    def classify_signal(
        self,
        signal,
        df,
        channel,
        higher_tf_channel=None
    ) -> ConfirmationResult:
        """
        Classify signal strength based on confirmations.
        
        Args:
            signal: Signal object
            df: OHLCV DataFrame
            channel: Channel object
            higher_tf_channel: Higher timeframe channel (optional)
            
        Returns:
            ConfirmationResult with strength classification
        """
        confirmations = []
        
        # Check each confirmation
        if self.check_channel_quality(channel):
            confirmations.append("high_quality_channel")
        
        if self.check_volume_confirmation(df):
            confirmations.append("volume_spike")
        
        if self.check_rsi_divergence(df, signal.direction.value):
            confirmations.append("rsi_divergence")
        
        if self.check_cvd_alignment(signal):
            confirmations.append("cvd_aligned")
        
        if self.check_sr_confluence(signal):
            confirmations.append("sr_confluence")
        
        if self.check_mtf_alignment(signal, higher_tf_channel):
            confirmations.append("mtf_aligned")
        
        if self.check_momentum_confirmation(df, signal.direction.value):
            confirmations.append("momentum_confirmed")
        
        # Count confirmations
        count = len(confirmations)
        
        # Classify strength
        if count >= 5:
            strength = SignalStrength.VERY_STRONG
            confidence_boost = +15
        elif count >= 4:
            strength = SignalStrength.STRONG
            confidence_boost = +10
        elif count >= 3:
            strength = SignalStrength.MODERATE
            confidence_boost = 0
        else:
            strength = SignalStrength.WEAK
            confidence_boost = -10
        
        result = ConfirmationResult(
            strength=strength,
            confirmations=confirmations,
            confirmation_count=count,
            confidence_boost=confidence_boost
        )
        
        logger.info(f"Signal classified: {result}")
        
        return result


# Singleton instance
_classifier: Optional[SignalStrengthClassifier] = None


def get_signal_classifier() -> SignalStrengthClassifier:
    """Get or create the global signal classifier."""
    global _classifier
    if _classifier is None:
        _classifier = SignalStrengthClassifier()
    return _classifier


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Signal Strength Classifier Test")
    print("="*60)
    
    classifier = SignalStrengthClassifier()
    
    # Mock data
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'close': np.random.uniform(95000, 96000, 100),
        'volume': np.random.uniform(10000, 100000, 100)
    })
    
    from dataclasses import dataclass as dc
    
    @dc
    class MockChannel:
        r_squared: float = 0.75
        is_valid: bool = True
    
    @dc  
    class MockSignal:
        direction: object
        
        class Direction:
            value = "long"
        
        direction = Direction()
    
    channel = MockChannel()
    signal = MockSignal()
    
    # Classify
    result = classifier.classify_signal(signal, df, channel)
    
    print(f"\n{result}")
    print(f"Confidence Boost: {result.confidence_boost:+d}")
    
    print("\n✅ Signal Strength Classifier working!")
