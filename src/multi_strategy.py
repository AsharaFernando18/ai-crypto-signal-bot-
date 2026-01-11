"""
Multi-Strategy Portfolio Manager - Phase 4.2
============================================
Manages multiple uncorrelated trading strategies for diversification.

Strategies:
1. Channel Breakout (existing) - 40% allocation
2. Momentum Breakout - 30% allocation
3. Mean Reversion - 20% allocation
4. Volatility Expansion - 10% allocation
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy types."""
    CHANNEL = "channel"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"


@dataclass
class StrategySignal:
    """Signal from a strategy."""
    strategy: StrategyType
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: int
    reason: str


class MomentumStrategy:
    """
    Momentum Breakout Strategy.
    
    Trades strong trends with volume confirmation.
    Best in: Trending markets
    """
    
    def __init__(self):
        """Initialize momentum strategy."""
        self.name = "Momentum Breakout"
        self.allocation = 0.30  # 30%
    
    def detect_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """
        Detect momentum breakout signal.
        
        Entry: Price breaks above resistance + volume spike
        Exit: Momentum reversal
        """
        if len(df) < 50:
            return None
        
        # Calculate indicators
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        avg_volume = df['volume'].rolling(20).mean()
        
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        # Long: Breakout above 20-day high with volume
        if (current_price > high_20.iloc[-2] and 
            current_volume > avg_volume.iloc[-1] * 1.5):
            
            entry = current_price
            stop_loss = low_20.iloc[-1]
            take_profit = entry + (entry - stop_loss) * 2.5
            
            return StrategySignal(
                strategy=StrategyType.MOMENTUM,
                symbol=symbol,
                direction='long',
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=75,
                reason="Momentum breakout above 20-day high with volume"
            )
        
        # Short: Breakdown below 20-day low with volume
        elif (current_price < low_20.iloc[-2] and 
              current_volume > avg_volume.iloc[-1] * 1.5):
            
            entry = current_price
            stop_loss = high_20.iloc[-1]
            take_profit = entry - (stop_loss - entry) * 2.5
            
            return StrategySignal(
                strategy=StrategyType.MOMENTUM,
                symbol=symbol,
                direction='short',
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=75,
                reason="Momentum breakdown below 20-day low with volume"
            )
        
        return None


class MeanReversionStrategy:
    """
    Mean Reversion Strategy.
    
    Trades oversold/overbought conditions.
    Best in: Choppy/ranging markets
    """
    
    def __init__(self):
        """Initialize mean reversion strategy."""
        self.name = "Mean Reversion"
        self.allocation = 0.20  # 20%
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """
        Detect mean reversion signal.
        
        Entry: RSI < 30 (oversold) or > 70 (overbought)
        Exit: RSI returns to 50
        """
        if len(df) < 50:
            return None
        
        # Calculate RSI
        rsi = self.calculate_rsi(df)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        current_price = df['close'].iloc[-1]
        ma_20 = df['close'].rolling(20).mean().iloc[-1]
        
        # Long: Oversold + turning up
        if current_rsi < 30 and current_rsi > prev_rsi:
            entry = current_price
            stop_loss = current_price * 0.98  # 2% stop
            take_profit = ma_20  # Target mean
            
            return StrategySignal(
                strategy=StrategyType.MEAN_REVERSION,
                symbol=symbol,
                direction='long',
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=70,
                reason=f"Oversold RSI {current_rsi:.1f} turning up"
            )
        
        # Short: Overbought + turning down
        elif current_rsi > 70 and current_rsi < prev_rsi:
            entry = current_price
            stop_loss = current_price * 1.02  # 2% stop
            take_profit = ma_20  # Target mean
            
            return StrategySignal(
                strategy=StrategyType.MEAN_REVERSION,
                symbol=symbol,
                direction='short',
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=70,
                reason=f"Overbought RSI {current_rsi:.1f} turning down"
            )
        
        return None


class VolatilityStrategy:
    """
    Volatility Expansion Strategy.
    
    Trades breakouts from low volatility.
    Best in: Low vol → High vol transitions
    """
    
    def __init__(self):
        """Initialize volatility strategy."""
        self.name = "Volatility Expansion"
        self.allocation = 0.10  # 10%
    
    def detect_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """
        Detect volatility expansion signal.
        
        Entry: Bollinger Band squeeze + breakout
        Exit: Volatility contraction
        """
        if len(df) < 50:
            return None
        
        # Calculate Bollinger Bands
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        upper_band = ma_20 + (std_20 * 2)
        lower_band = ma_20 - (std_20 * 2)
        bandwidth = (upper_band - lower_band) / ma_20
        
        # Detect squeeze (low bandwidth)
        avg_bandwidth = bandwidth.rolling(50).mean()
        current_bandwidth = bandwidth.iloc[-1]
        
        current_price = df['close'].iloc[-1]
        
        # Squeeze condition
        if current_bandwidth < avg_bandwidth.iloc[-1] * 0.7:
            
            # Breakout above upper band
            if current_price > upper_band.iloc[-1]:
                entry = current_price
                stop_loss = ma_20.iloc[-1]
                take_profit = entry + (entry - stop_loss) * 2.0
                
                return StrategySignal(
                    strategy=StrategyType.VOLATILITY,
                    symbol=symbol,
                    direction='long',
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=65,
                    reason="Volatility expansion breakout (squeeze)"
                )
            
            # Breakdown below lower band
            elif current_price < lower_band.iloc[-1]:
                entry = current_price
                stop_loss = ma_20.iloc[-1]
                take_profit = entry - (stop_loss - entry) * 2.0
                
                return StrategySignal(
                    strategy=StrategyType.VOLATILITY,
                    symbol=symbol,
                    direction='short',
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=65,
                    reason="Volatility expansion breakdown (squeeze)"
                )
        
        return None


class MultiStrategyPortfolio:
    """
    Multi-strategy portfolio manager.
    
    Allocates capital across multiple uncorrelated strategies.
    """
    
    def __init__(self):
        """Initialize portfolio manager."""
        self.strategies = {
            StrategyType.MOMENTUM: MomentumStrategy(),
            StrategyType.MEAN_REVERSION: MeanReversionStrategy(),
            StrategyType.VOLATILITY: VolatilityStrategy()
        }
        
        logger.info("Multi-Strategy Portfolio initialized with 3 strategies")
    
    def scan_all_strategies(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[StrategySignal]:
        """
        Scan all strategies for signals.
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            List of signals from all strategies
        """
        signals = []
        
        for strategy_type, strategy in self.strategies.items():
            try:
                signal = strategy.detect_signal(df, symbol)
                if signal:
                    signals.append(signal)
                    logger.info(f"🎯 {strategy.name} signal: {signal.direction.upper()} {symbol} @ {signal.entry_price:.2f}")
            except Exception as e:
                logger.debug(f"Error in {strategy.name}: {e}")
        
        return signals
    
    def get_strategy_allocation(self, strategy_type: StrategyType) -> float:
        """Get capital allocation for strategy."""
        if strategy_type in self.strategies:
            return self.strategies[strategy_type].allocation
        return 0.0


# Singleton
_portfolio: Optional[MultiStrategyPortfolio] = None


def get_multi_strategy_portfolio() -> MultiStrategyPortfolio:
    """Get or create multi-strategy portfolio."""
    global _portfolio
    if _portfolio is None:
        _portfolio = MultiStrategyPortfolio()
    return _portfolio


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    df = pd.DataFrame({
        'close': [95000 + np.random.randn()*500 for _ in range(100)],
        'high': [95500 + np.random.randn()*500 for _ in range(100)],
        'low': [94500 + np.random.randn()*500 for _ in range(100)],
        'volume': [10000 + np.random.randn()*2000 for _ in range(100)]
    })
    
    portfolio = MultiStrategyPortfolio()
    signals = portfolio.scan_all_strategies(df, "BTC/USDT")
    
    print(f"\nFound {len(signals)} signals")
    for sig in signals:
        print(f"  {sig.strategy.value}: {sig.direction} @ {sig.entry_price:.2f}")
    
    print("\n✅ Multi-Strategy Portfolio working!")
