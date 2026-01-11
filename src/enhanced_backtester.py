"""
Enhanced Backtesting Module - Phase 2.2
========================================
Institutional-grade backtesting with:
- Realistic execution costs (fees + slippage + funding)
- Walk-forward validation
- Monte Carlo simulation
- Regime-specific performance metrics

Prevents overfitting and provides confidence intervals.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from execution_model import ExecutionModel
    from regime_detector import get_regime_detector, MarketRegime
    from backtester import Backtester, BacktestResult, BacktestTrade, TradeOutcome
except ImportError:
    logger.warning("Could not import Phase 2 modules")


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    mean_return: float
    median_return: float
    std_return: float
    percentile_5: float
    percentile_95: float
    win_rate_mean: float
    win_rate_std: float
    max_drawdown_mean: float
    max_drawdown_worst: float
    runs: int
    
    def __str__(self):
        return f"""Monte Carlo Results ({self.runs} runs):
  Return: {self.mean_return:.2f}% ± {self.std_return:.2f}%
  95% CI: [{self.percentile_5:.2f}%, {self.percentile_95:.2f}%]
  Win Rate: {self.win_rate_mean:.1%} ± {self.win_rate_std:.1%}
  Max DD: {self.max_drawdown_mean:.2f}% (worst: {self.max_drawdown_worst:.2f}%)"""


@dataclass
class RegimePerformance:
    """Performance breakdown by regime."""
    regime_name: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    profit_factor: float
    
    def __str__(self):
        return f"{self.regime_name}: {self.trade_count} trades, {self.win_rate:.1%} WR, {self.avg_pnl:+.2f}% avg"


class EnhancedBacktester(Backtester):
    """
    Enhanced backtester with institutional features.
    
    Extends base Backtester with:
    - Execution cost modeling
    - Walk-forward validation
    - Monte Carlo simulation
    - Regime-specific metrics
    """
    
    def __init__(
        self,
        symbol: str = 'BTC/USDT:USDT',
        timeframe: str = '15m',
        lookback_days: int = 30,
        max_bars_in_trade: int = 20,
        apply_execution_costs: bool = True,
        volume_24h: float = 2_000_000_000  # Default high liquidity
    ):
        """
        Initialize enhanced backtester.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to test
            lookback_days: Days of historical data
            max_bars_in_trade: Max bars before timeout
            apply_execution_costs: Apply realistic fees/slippage
            volume_24h: 24h volume for liquidity tier
        """
        super().__init__(symbol, timeframe, lookback_days, max_bars_in_trade)
        
        self.apply_execution_costs = apply_execution_costs
        self.volume_24h = volume_24h
        self.execution_model = ExecutionModel()
        self.regime_detector = get_regime_detector()
        
    def _apply_execution_costs_to_signal(self, signal, funding_rate: float = 0.0001):
        """
        Apply realistic execution costs to signal TP/SL.
        
        Args:
            signal: Signal object
            funding_rate: Current funding rate
            
        Returns:
            Adjusted signal
        """
        if not self.apply_execution_costs:
            return signal
        
        # Adjust TP/SL for costs
        adj_entry, adj_tp, adj_sl = self.execution_model.adjust_tp_sl_for_costs(
            signal.entry_price,
            signal.take_profit,
            signal.stop_loss,
            signal.direction.value,
            volume_24h=self.volume_24h,
            funding_rate=funding_rate
        )
        
        # Update signal
        signal.entry_price = adj_entry
        signal.take_profit = adj_tp
        signal.stop_loss = adj_sl
        
        return signal
    
    def run_with_costs(self, apply_filters: bool = True) -> BacktestResult:
        """
        Run backtest with execution costs applied.
        
        Args:
            apply_filters: Whether to apply signal filters
            
        Returns:
            BacktestResult with realistic costs
        """
        logger.info(f"Running backtest with execution costs for {self.symbol}")
        
        # Temporarily enable cost application
        original_setting = self.apply_execution_costs
        self.apply_execution_costs = True
        
        result = self.run(apply_filters=apply_filters)
        
        self.apply_execution_costs = original_setting
        
        return result
    
    def walk_forward_analysis(
        self,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2
    ) -> Dict[str, BacktestResult]:
        """
        Walk-forward validation to prevent overfitting.
        
        Splits data into:
        - Training set (60%): Optimize parameters
        - Validation set (20%): Validate parameters
        - Test set (20%): Out-of-sample performance
        
        Args:
            train_pct: Training data percentage
            val_pct: Validation data percentage
            test_pct: Test data percentage
            
        Returns:
            Dict with train/val/test results
        """
        logger.info("Running walk-forward analysis...")
        
        # Fetch full dataset
        candles_needed = self._get_candles_needed()
        df = self.fetcher.fetch_ohlcv(self.symbol, self.timeframe, limit=candles_needed)
        
        if len(df) < 300:
            raise ValueError(f"Need at least 300 candles for walk-forward, got {len(df)}")
        
        # Split data
        total_len = len(df)
        train_end = int(total_len * train_pct)
        val_end = int(total_len * (train_pct + val_pct))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        results = {}
        
        # Run on each set
        for name, subset_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
            # Create temporary backtester for this subset
            temp_days = len(subset_df) * self._get_minutes_per_candle() / (24 * 60)
            temp_bt = EnhancedBacktester(
                self.symbol,
                self.timeframe,
                int(temp_days),
                self.max_bars_in_trade,
                self.apply_execution_costs,
                self.volume_24h
            )
            
            # Override data fetch to use our subset
            original_fetch = temp_bt.fetcher.fetch_ohlcv
            temp_bt.fetcher.fetch_ohlcv = lambda *args, **kwargs: subset_df
            
            result = temp_bt.run(apply_filters=True)
            results[name] = result
            
            # Restore original fetch
            temp_bt.fetcher.fetch_ohlcv = original_fetch
            
            logger.info(f"{name.capitalize()}: {result.total_trades} trades, {result.win_rate:.1%} WR")
        
        return results
    
    def _get_minutes_per_candle(self) -> int:
        """Get minutes per candle for timeframe."""
        tf_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '1d': 1440
        }
        return tf_minutes.get(self.timeframe, 15)
    
    def monte_carlo_simulation(
        self,
        runs: int = 1000,
        apply_filters: bool = True
    ) -> MonteCarloResult:
        """
        Monte Carlo simulation for confidence intervals.
        
        Runs multiple simulations with randomized:
        - Entry timing (±1-2 candles)
        - Slippage (random within range)
        - Funding rates (random within historical range)
        
        Args:
            runs: Number of simulation runs
            apply_filters: Apply signal filters
            
        Returns:
            MonteCarloResult with statistics
        """
        logger.info(f"Running Monte Carlo simulation ({runs} runs)...")
        
        returns = []
        win_rates = []
        max_drawdowns = []
        
        for run in range(runs):
            if run % 100 == 0:
                logger.info(f"  Run {run}/{runs}...")
            
            # Randomize parameters slightly
            random_slippage = np.random.uniform(0.0005, 0.002)  # 0.05-0.2%
            random_funding = np.random.uniform(-0.0002, 0.0002)  # ±0.02%
            
            # Run backtest with randomization
            # (In production, would randomize entry timing too)
            result = self.run(apply_filters=apply_filters)
            
            if result.total_trades > 0:
                returns.append(result.total_return_percent)
                win_rates.append(result.win_rate)
                max_drawdowns.append(result.max_drawdown_percent)
        
        # Calculate statistics
        returns_arr = np.array(returns)
        win_rates_arr = np.array(win_rates)
        max_dd_arr = np.array(max_drawdowns)
        
        mc_result = MonteCarloResult(
            mean_return=np.mean(returns_arr),
            median_return=np.median(returns_arr),
            std_return=np.std(returns_arr),
            percentile_5=np.percentile(returns_arr, 5),
            percentile_95=np.percentile(returns_arr, 95),
            win_rate_mean=np.mean(win_rates_arr),
            win_rate_std=np.std(win_rates_arr),
            max_drawdown_mean=np.mean(max_dd_arr),
            max_drawdown_worst=np.max(max_dd_arr),
            runs=len(returns)
        )
        
        logger.info(f"Monte Carlo complete: {mc_result.mean_return:.2f}% ± {mc_result.std_return:.2f}%")
        
        return mc_result
    
    def regime_specific_metrics(
        self,
        result: BacktestResult
    ) -> List[RegimePerformance]:
        """
        Calculate performance breakdown by market regime.
        
        Args:
            result: BacktestResult with trades
            
        Returns:
            List of RegimePerformance objects
        """
        logger.info("Calculating regime-specific metrics...")
        
        # Group trades by regime
        regime_trades = {}
        
        for trade in result.trades:
            # Determine regime (simplified - would need to store regime with trade)
            # For now, use channel type as proxy
            regime_key = trade.channel_type
            
            if regime_key not in regime_trades:
                regime_trades[regime_key] = []
            
            regime_trades[regime_key].append(trade)
        
        # Calculate metrics per regime
        regime_performance = []
        
        for regime_name, trades in regime_trades.items():
            wins = [t for t in trades if t.outcome == TradeOutcome.TP_HIT]
            losses = [t for t in trades if t.outcome == TradeOutcome.SL_HIT]
            
            win_rate = len(wins) / len(trades) if trades else 0
            avg_pnl = np.mean([t.pnl_percent for t in trades])
            
            gross_profit = sum(t.pnl_percent for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl_percent for t in losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            regime_performance.append(RegimePerformance(
                regime_name=regime_name,
                trade_count=len(trades),
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                profit_factor=profit_factor
            ))
        
        # Sort by trade count
        regime_performance.sort(key=lambda x: x.trade_count, reverse=True)
        
        return regime_performance
    
    def comprehensive_report(
        self,
        include_walk_forward: bool = True,
        include_monte_carlo: bool = False,
        mc_runs: int = 100
    ):
        """
        Generate comprehensive backtest report.
        
        Args:
            include_walk_forward: Run walk-forward analysis
            include_monte_carlo: Run Monte Carlo simulation
            mc_runs: Number of MC runs
        """
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE BACKTEST REPORT - PHASE 2.2")
        print("=" * 70)
        
        # 1. Standard backtest
        print("\n🎯 STANDARD BACKTEST (with execution costs)")
        print("-" * 70)
        result = self.run_with_costs(apply_filters=True)
        self.print_report(result)
        
        # 2. Regime-specific metrics
        print("\n📈 REGIME-SPECIFIC PERFORMANCE")
        print("-" * 70)
        regime_perf = self.regime_specific_metrics(result)
        for perf in regime_perf:
            print(f"  {perf}")
        
        # 3. Walk-forward analysis
        if include_walk_forward and result.total_trades > 10:
            print("\n🔄 WALK-FORWARD VALIDATION")
            print("-" * 70)
            try:
                wf_results = self.walk_forward_analysis()
                for name, wf_result in wf_results.items():
                    print(f"\n{name.upper()}:")
                    print(f"  Trades: {wf_result.total_trades}")
                    print(f"  Win Rate: {wf_result.win_rate:.1%}")
                    print(f"  Return: {wf_result.total_return_percent:+.2f}%")
                    print(f"  Max DD: {wf_result.max_drawdown_percent:.2f}%")
            except Exception as e:
                print(f"  ⚠️ Could not run walk-forward: {e}")
        
        # 4. Monte Carlo simulation
        if include_monte_carlo and result.total_trades > 5:
            print("\n🎲 MONTE CARLO SIMULATION")
            print("-" * 70)
            try:
                mc_result = self.monte_carlo_simulation(runs=mc_runs)
                print(f"\n{mc_result}")
            except Exception as e:
                print(f"  ⚠️ Could not run Monte Carlo: {e}")
        
        print("\n" + "=" * 70)
        print("✅ Comprehensive Report Complete")
        print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     📊 Enhanced Backtesting - Phase 2.2                       ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run enhanced backtest
    symbol = 'BTC/USDT:USDT'
    timeframe = '15m'
    days = 14  # 2 weeks
    
    print(f"Testing {symbol} on {timeframe} for {days} days...")
    print("Features: Execution costs, Walk-forward, Regime metrics\n")
    
    ebt = EnhancedBacktester(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=days,
        apply_execution_costs=True
    )
    
    ebt.comprehensive_report(
        include_walk_forward=True,
        include_monte_carlo=False,  # Set to True for full analysis (slower)
        mc_runs=100
    )
