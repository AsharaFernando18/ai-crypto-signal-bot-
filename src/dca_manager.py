"""
DCA Manager Module
==================
Intelligent DCA (Dollar Cost Averaging) opportunity detection.
Detects when new channels form in same direction as existing positions.
"""
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DCAOpportunity:
    """Represents a DCA opportunity for an existing position."""
    
    def __init__(
        self,
        symbol: str,
        direction: str,
        original_entry: float,
        dca_entry: float,
        current_price: float,
        new_average: float,
        take_profit: float,
        confidence_score: float,
        channel_type: str,
        timeframe: str,
        unrealized_pct: float,
        entries_count: int
    ):
        self.symbol = symbol
        self.direction = direction
        self.original_entry = original_entry
        self.dca_entry = dca_entry
        self.current_price = current_price
        self.new_average = new_average
        self.take_profit = take_profit
        self.confidence_score = confidence_score
        self.channel_type = channel_type
        self.timeframe = timeframe
        self.unrealized_pct = unrealized_pct
        self.entries_count = entries_count
        self.timestamp = datetime.now()


class DCAManager:
    """Manages DCA opportunity detection and position averaging."""
    
    def __init__(
        self,
        enable_dca_alerts: bool = True,
        min_dca_distance_pct: float = 1.0,
        max_dca_count: int = 3,
        dca_zone_spacing_pct: float = 1.5,
        min_dca_confidence: float = 60
    ):
        """
        Initialize DCA Manager.
        
        Args:
            enable_dca_alerts: Enable DCA opportunity alerts
            min_dca_distance_pct: Minimum % worse than entry for DCA
            max_dca_count: Maximum number of DCA entries
            dca_zone_spacing_pct: % spacing between DCA zones
            min_dca_confidence: Minimum confidence for DCA signal
        """
        self.enable_dca_alerts = enable_dca_alerts
        self.min_dca_distance_pct = min_dca_distance_pct
        self.max_dca_count = max_dca_count
        self.dca_zone_spacing_pct = dca_zone_spacing_pct
        self.min_dca_confidence = min_dca_confidence
        
        logger.info(f"DCA Manager initialized: max_dca={max_dca_count}, min_distance={min_dca_distance_pct}%")
    
    def detect_dca_opportunity(
        self,
        position: Dict,
        new_signal: 'Signal'
    ) -> Optional[DCAOpportunity]:
        """
        Detect if new signal is valid DCA opportunity for existing position.
        
        Args:
            position: Existing position dict
            new_signal: New signal object
        
        Returns:
            DCAOpportunity if valid, None otherwise
        """
        if not self.enable_dca_alerts:
            return None
        
        # 1. Check if same symbol
        if position['symbol'] != new_signal.symbol:
            return None
        
        # 2. Check if same direction
        if position['direction'] != new_signal.direction.value:
            logger.debug(f"DCA rejected: different direction ({position['direction']} vs {new_signal.direction.value})")
            return None
        
        # 3. Check if already at max DCA count
        dca_count = position.get('dca_count', 0)
        if dca_count >= self.max_dca_count:
            logger.debug(f"DCA rejected: max DCA count reached ({dca_count}/{self.max_dca_count})")
            return None
        
        # 4. Check if price is worse than average entry
        avg_entry = position.get('average_entry', position['entry_price'])
        current_price = new_signal.entry_price
        
        if position['direction'] == 'long':
            # For LONG: DCA price should be LOWER than avg entry
            price_diff_pct = ((avg_entry - current_price) / avg_entry) * 100
            is_worse = current_price < avg_entry
        else:
            # For SHORT: DCA price should be HIGHER than avg entry
            price_diff_pct = ((current_price - avg_entry) / avg_entry) * 100
            is_worse = current_price > avg_entry
        
        if not is_worse or price_diff_pct < self.min_dca_distance_pct:
            logger.debug(f"DCA rejected: price not worse enough ({price_diff_pct:.2f}% < {self.min_dca_distance_pct}%)")
            return None
        
        # 5. Check confidence score
        signal_confidence = getattr(new_signal, 'confidence_score', 0)
        if signal_confidence < self.min_dca_confidence:
            logger.debug(f"DCA rejected: low confidence ({signal_confidence} < {self.min_dca_confidence})")
            return None
        
        # 6. Calculate new average entry
        entries = position.get('entries', [{'price': position['entry_price'], 'size': 1.0}])
        new_average = self.calculate_average_entry(
            entries + [{'price': current_price, 'size': 1.0}]
        )
        
        # 7. Calculate unrealized %
        if position['direction'] == 'long':
            unrealized_pct = ((current_price - avg_entry) / avg_entry) * 100
        else:
            unrealized_pct = ((avg_entry - current_price) / avg_entry) * 100
        
        # Create DCA opportunity
        opportunity = DCAOpportunity(
            symbol=position['symbol'],
            direction=position['direction'],
            original_entry=position.get('initial_entry', position['entry_price']),
            dca_entry=current_price,
            current_price=current_price,
            new_average=new_average,
            take_profit=position['take_profit'],
            confidence_score=signal_confidence,
            channel_type=new_signal.channel_type.value,
            timeframe=new_signal.timeframe,
            unrealized_pct=unrealized_pct,
            entries_count=len(entries) + 1
        )
        
        logger.info(f"✅ DCA Opportunity detected: {position['symbol']} {position['direction'].upper()} @ ${current_price:,.2f}")
        return opportunity
    
    def calculate_average_entry(self, entries: List[Dict]) -> float:
        """
        Calculate weighted average entry price.
        
        Args:
            entries: List of entry dicts with 'price' and 'size'
        
        Returns:
            Weighted average price
        """
        if not entries:
            return 0.0
        
        total_value = sum(e['price'] * e['size'] for e in entries)
        total_size = sum(e['size'] for e in entries)
        
        return total_value / total_size if total_size > 0 else 0.0
    
    def suggest_dca_zones(
        self,
        entry_price: float,
        direction: str,
        num_zones: int = 2
    ) -> List[float]:
        """
        Suggest potential DCA price levels.
        
        Args:
            entry_price: Initial entry price
            direction: 'long' or 'short'
            num_zones: Number of DCA zones to suggest
        
        Returns:
            List of suggested DCA prices
        """
        zones = []
        
        for i in range(1, num_zones + 1):
            distance_pct = self.dca_zone_spacing_pct * i
            
            if direction == 'long':
                # For LONG: zones below entry
                zone_price = entry_price * (1 - distance_pct / 100)
            else:
                # For SHORT: zones above entry
                zone_price = entry_price * (1 + distance_pct / 100)
            
            zones.append(zone_price)
        
        return zones
    
    def calculate_potential_profit(
        self,
        average_entry: float,
        take_profit: float,
        direction: str
    ) -> float:
        """
        Calculate potential profit % from average entry to TP.
        
        Args:
            average_entry: Average entry price
            take_profit: Take profit price
            direction: 'long' or 'short'
        
        Returns:
            Potential profit percentage
        """
        if direction == 'long':
            return ((take_profit - average_entry) / average_entry) * 100
        else:
            return ((average_entry - take_profit) / average_entry) * 100


# Global instance
_dca_manager: Optional[DCAManager] = None


def get_dca_manager() -> DCAManager:
    """Get or create the global DCA manager instance."""
    global _dca_manager
    if _dca_manager is None:
        try:
            from config import (
                ENABLE_DCA_ALERTS,
                MIN_DCA_DISTANCE_PERCENT,
                MAX_DCA_COUNT,
                DCA_ZONE_SPACING_PERCENT,
                MIN_DCA_CONFIDENCE
            )
            _dca_manager = DCAManager(
                enable_dca_alerts=ENABLE_DCA_ALERTS,
                min_dca_distance_pct=MIN_DCA_DISTANCE_PERCENT,
                max_dca_count=MAX_DCA_COUNT,
                dca_zone_spacing_pct=DCA_ZONE_SPACING_PERCENT,
                min_dca_confidence=MIN_DCA_CONFIDENCE
            )
        except ImportError:
            # Use defaults if config not available
            _dca_manager = DCAManager()
    
    return _dca_manager
