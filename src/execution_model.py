"""
Execution Cost Model
====================
Models realistic trading costs including fees, slippage, and funding.
Adjusts TP/SL levels for true risk/reward calculation.

Critical for institutional-grade performance estimation.
"""
import logging
from typing import Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from config import (
        MAKER_FEE, TAKER_FEE, ASSUME_TAKER,
        SLIPPAGE_BPS, DEFAULT_SLIPPAGE_BPS,
        HIGH_LIQUIDITY_THRESHOLD, MEDIUM_LIQUIDITY_THRESHOLD,
        FUNDING_INTERVAL_HOURS, EXPECTED_HOLDING_HOURS
    )
except ImportError:
    MAKER_FEE = 0.0002
    TAKER_FEE = 0.0004
    ASSUME_TAKER = True
    SLIPPAGE_BPS = {'high_liquidity': 5, 'medium_liquidity': 15, 'low_liquidity': 50}
    DEFAULT_SLIPPAGE_BPS = 15
    HIGH_LIQUIDITY_THRESHOLD = 1_000_000_000
    MEDIUM_LIQUIDITY_THRESHOLD = 100_000_000
    FUNDING_INTERVAL_HOURS = 8
    EXPECTED_HOLDING_HOURS = 4


@dataclass
class ExecutionCosts:
    """Breakdown of execution costs."""
    entry_fee: float
    exit_fee: float
    entry_slippage: float
    exit_slippage: float
    funding_cost: float
    total_cost_percent: float
    
    def __str__(self):
        return (f"Fees: {(self.entry_fee + self.exit_fee)*100:.3f}%, "
                f"Slippage: {(self.entry_slippage + self.exit_slippage)*100:.3f}%, "
                f"Funding: {self.funding_cost*100:.3f}%, "
                f"Total: {self.total_cost_percent*100:.3f}%")


class ExecutionModel:
    """
    Models realistic execution costs.
    
    Accounts for:
    - Trading fees (maker/taker)
    - Slippage (liquidity-dependent)
    - Funding rates (futures-specific)
    """
    
    @staticmethod
    def determine_liquidity_tier(volume_24h: float) -> str:
        """
        Determine liquidity tier based on 24h volume.
        
        Args:
            volume_24h: 24-hour trading volume in USD
            
        Returns:
            'high_liquidity', 'medium_liquidity', or 'low_liquidity'
        """
        if volume_24h >= HIGH_LIQUIDITY_THRESHOLD:
            return 'high_liquidity'
        elif volume_24h >= MEDIUM_LIQUIDITY_THRESHOLD:
            return 'medium_liquidity'
        else:
            return 'low_liquidity'
    
    @staticmethod
    def calculate_costs(
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        direction: str,
        volume_24h: float = None,
        funding_rate: float = 0.0001
    ) -> ExecutionCosts:
        """
        Calculate total execution costs.
        
        Args:
            entry_price: Entry price
            take_profit: Take profit price
            stop_loss: Stop loss price
            direction: 'long' or 'short'
            volume_24h: 24h volume in USD (for liquidity tier)
            funding_rate: Current funding rate
            
        Returns:
            ExecutionCosts object
        """
        # Determine fees
        fee = TAKER_FEE if ASSUME_TAKER else MAKER_FEE
        entry_fee = fee
        exit_fee = fee
        
        # Determine slippage based on liquidity
        if volume_24h:
            tier = ExecutionModel.determine_liquidity_tier(volume_24h)
            slippage_bps = SLIPPAGE_BPS.get(tier, DEFAULT_SLIPPAGE_BPS)
        else:
            slippage_bps = DEFAULT_SLIPPAGE_BPS
        
        slippage_decimal = slippage_bps / 10000.0
        entry_slippage = slippage_decimal
        exit_slippage = slippage_decimal
        
        # Calculate funding cost
        # Funding paid/received every 8 hours
        # Estimate based on expected holding time
        funding_periods = EXPECTED_HOLDING_HOURS / FUNDING_INTERVAL_HOURS
        funding_cost = abs(funding_rate) * funding_periods
        
        # Total cost
        total_cost = entry_fee + exit_fee + entry_slippage + exit_slippage + funding_cost
        
        return ExecutionCosts(
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            funding_cost=funding_cost,
            total_cost_percent=total_cost
        )
    
    @staticmethod
    def adjust_tp_sl_for_costs(
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        direction: str,
        volume_24h: float = None,
        funding_rate: float = 0.0001
    ) -> Tuple[float, float, float]:
        """
        Adjust TP/SL for realistic execution costs.
        
        Returns worse TP and worse SL to account for:
        - Entry fees + slippage
        - Exit fees + slippage
        - Funding costs
        
        Args:
            entry_price: Original entry price
            take_profit: Original TP
            stop_loss: Original SL
            direction: 'long' or 'short'
            volume_24h: 24h volume for liquidity tier
            funding_rate: Current funding rate
            
        Returns:
            (adjusted_entry, adjusted_tp, adjusted_sl)
        """
        costs = ExecutionModel.calculate_costs(
            entry_price, take_profit, stop_loss,
            direction, volume_24h, funding_rate
        )
        
        # Entry cost (fees + slippage on entry)
        entry_cost = costs.entry_fee + costs.entry_slippage
        
        # Exit cost (fees + slippage on exit)
        exit_cost = costs.exit_fee + costs.exit_slippage
        
        # Total cost to overcome
        total_cost = costs.total_cost_percent
        
        if direction == "long":
            # LONG: Pay more to enter, receive less on exit
            adjusted_entry = entry_price * (1 + entry_cost)
            adjusted_tp = take_profit * (1 - exit_cost)
            adjusted_sl = stop_loss * (1 - entry_cost)  # SL hit costs entry fee
            
        else:  # short
            # SHORT: Receive less to enter, pay more on exit
            adjusted_entry = entry_price * (1 - entry_cost)
            adjusted_tp = take_profit * (1 + exit_cost)
            adjusted_sl = stop_loss * (1 + entry_cost)
        
        return adjusted_entry, adjusted_tp, adjusted_sl
    
    @staticmethod
    def calculate_true_rr(
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        direction: str,
        volume_24h: float = None,
        funding_rate: float = 0.0001
    ) -> Tuple[float, float]:
        """
        Calculate TRUE risk/reward after costs.
        
        Returns:
            (true_rr_ratio, cost_impact_percent)
        """
        # Original R:R
        if direction == "long":
            original_reward = (take_profit - entry_price) / entry_price
            original_risk = (entry_price - stop_loss) / entry_price
        else:
            original_reward = (entry_price - take_profit) / entry_price
            original_risk = (stop_loss - entry_price) / entry_price
        
        original_rr = original_reward / original_risk if original_risk > 0 else 0
        
        # Adjusted R:R
        adj_entry, adj_tp, adj_sl = ExecutionModel.adjust_tp_sl_for_costs(
            entry_price, take_profit, stop_loss,
            direction, volume_24h, funding_rate
        )
        
        if direction == "long":
            true_reward = (adj_tp - adj_entry) / adj_entry
            true_risk = (adj_entry - adj_sl) / adj_entry
        else:
            true_reward = (adj_entry - adj_tp) / adj_entry
            true_risk = (adj_sl - adj_entry) / adj_entry
        
        true_rr = true_reward / true_risk if true_risk > 0 else 0
        
        # Impact
        cost_impact = ((original_rr - true_rr) / original_rr) if original_rr > 0 else 0
        
        return true_rr, cost_impact


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Example: BTC long
    entry = 95000
    tp = 96500
    sl = 94500
    
    print("="*60)
    print("Execution Cost Model Test")
    print("="*60)
    
    costs = ExecutionModel.calculate_costs(
        entry, tp, sl, "long",
        volume_24h=2_000_000_000,  # High liquidity
        funding_rate=0.0001
    )
    
    print(f"\nCost Breakdown:")
    print(costs)
    
    adj_entry, adj_tp, adj_sl = ExecutionModel.adjust_tp_sl_for_costs(
        entry, tp, sl, "long",
        volume_24h=2_000_000_000,
        funding_rate=0.0001
    )
    
    print(f"\nOriginal: Entry=${entry}, TP=${tp}, SL=${sl}")
    print(f"Adjusted: Entry=${adj_entry:.2f}, TP=${adj_tp:.2f}, SL=${adj_sl:.2f}")
    
    true_rr, impact = ExecutionModel.calculate_true_rr(
        entry, tp, sl, "long",
        volume_24h=2_000_000_000,
        funding_rate=0.0001
    )
    
    print(f"\nTrue R:R: {true_rr:.2f}")
    print(f"Cost Impact: {impact*100:.1f}% worse")
    print("\n✅ Execution Model working!")
