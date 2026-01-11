"""
Data Quality Validator
======================
Validates OHLCV data quality before use in signal generation.
Prevents garbage-in-garbage-out scenarios.

Checks:
- Missing data / gaps
- Price outliers
- Volume validation
- Timestamp continuity
- OHLC relationship validity
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from config import (
        MAX_ALLOWED_GAP_PERCENT, MAX_PRICE_JUMP_PERCENT,
        OUTLIER_STD_THRESHOLD, MIN_VOLUME_THRESHOLD,
        MAX_DATA_AGE_SECONDS
    )
except ImportError:
    MAX_ALLOWED_GAP_PERCENT = 0.05
    MAX_PRICE_JUMP_PERCENT = 0.10
    OUTLIER_STD_THRESHOLD = 5.0
    MIN_VOLUME_THRESHOLD = 1000
    MAX_DATA_AGE_SECONDS = 300


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    gap_percent: float
    outlier_count: int
    zero_volume_count: int
    
    def __str__(self):
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (f"{status} | Gaps: {self.gap_percent*100:.1f}% | "
                f"Outliers: {self.outlier_count} | "
                f"Zero Vol: {self.zero_volume_count}")


class DataValidator:
    """
    Validates OHLCV data quality.
    
    Institutional standard: Reject bad data rather than risk bad signals.
    """
    
    @staticmethod
    def validate_ohlcv(
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "15m",
        strict: bool = True
    ) -> ValidationResult:
        """
        Comprehensive OHLCV validation.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for logging
            timeframe: Timeframe for gap detection
            strict: If True, reject on any critical issue
            
        Returns:
            ValidationResult object
        """
        issues = []
        warnings = []
        
        # Check 1: Minimum data requirement
        if len(df) < 20:
            issues.append(f"Insufficient data: {len(df)} candles (need 20+)")
            return ValidationResult(
                is_valid=False,
                issues=issues,
                warnings=warnings,
                gap_percent=1.0,
                outlier_count=0,
                zero_volume_count=0
            )
        
        # Check 2: Required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return ValidationResult(
                is_valid=False,
                issues=issues,
                warnings=warnings,
                gap_percent=0.0,
                outlier_count=0,
                zero_volume_count=0
            )
        
        # Check 3: Gap detection
        gap_percent, gap_issues = DataValidator._check_gaps(df, timeframe)
        if gap_percent > MAX_ALLOWED_GAP_PERCENT:
            issues.append(f"Too many gaps: {gap_percent*100:.1f}% (max {MAX_ALLOWED_GAP_PERCENT*100:.1f}%)")
        if gap_issues:
            warnings.extend(gap_issues)
        
        # Check 4: OHLC relationship validity
        ohlc_issues = DataValidator._validate_ohlc_relationships(df)
        if ohlc_issues:
            issues.extend(ohlc_issues)
        
        # Check 5: Price outliers
        outlier_count, outlier_issues = DataValidator._detect_price_outliers(df)
        if outlier_issues:
            warnings.extend(outlier_issues)
        
        # Check 6: Volume validation
        zero_vol_count, vol_issues = DataValidator._validate_volume(df)
        if vol_issues:
            warnings.extend(vol_issues)
        
        # Check 7: Price jumps
        jump_issues = DataValidator._detect_price_jumps(df)
        if jump_issues:
            warnings.extend(jump_issues)
        
        # Check 8: Data freshness
        freshness_issue = DataValidator._check_freshness(df)
        if freshness_issue:
            warnings.append(freshness_issue)
        
        # Check 9: NaN/Inf values
        nan_issues = DataValidator._check_nan_inf(df)
        if nan_issues:
            issues.extend(nan_issues)
        
        # Determine validity
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Data validation FAILED for {symbol} {timeframe}: {issues}")
        elif warnings:
            logger.info(f"Data validation passed with warnings for {symbol} {timeframe}: {warnings}")
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            gap_percent=gap_percent,
            outlier_count=outlier_count,
            zero_volume_count=zero_vol_count
        )
    
    @staticmethod
    def _check_gaps(df: pd.DataFrame, timeframe: str) -> Tuple[float, List[str]]:
        """Check for missing candles (gaps in timestamps)."""
        issues = []
        
        # Expected interval based on timeframe
        interval_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        expected_interval = interval_map.get(timeframe, 900)
        
        # Calculate actual intervals
        timestamps = pd.to_datetime(df['timestamp'], unit='ms')
        intervals = timestamps.diff().dt.total_seconds()
        
        # Find gaps (intervals > 1.5x expected)
        gaps = intervals[intervals > expected_interval * 1.5]
        gap_percent = len(gaps) / len(df) if len(df) > 0 else 0
        
        if len(gaps) > 0:
            issues.append(f"Found {len(gaps)} gaps in data")
        
        return gap_percent, issues
    
    @staticmethod
    def _validate_ohlc_relationships(df: pd.DataFrame) -> List[str]:
        """Validate OHLC relationships (high >= low, etc.)."""
        issues = []
        
        # High must be >= Low
        invalid_hl = df[df['high'] < df['low']]
        if len(invalid_hl) > 0:
            issues.append(f"{len(invalid_hl)} candles with high < low")
        
        # High must be >= Open and Close
        invalid_ho = df[df['high'] < df['open']]
        invalid_hc = df[df['high'] < df['close']]
        if len(invalid_ho) > 0 or len(invalid_hc) > 0:
            issues.append(f"High not >= open/close in {len(invalid_ho) + len(invalid_hc)} candles")
        
        # Low must be <= Open and Close
        invalid_lo = df[df['low'] > df['open']]
        invalid_lc = df[df['low'] > df['close']]
        if len(invalid_lo) > 0 or len(invalid_lc) > 0:
            issues.append(f"Low not <= open/close in {len(invalid_lo) + len(invalid_lc)} candles")
        
        return issues
    
    @staticmethod
    def _detect_price_outliers(df: pd.DataFrame) -> Tuple[int, List[str]]:
        """Detect price outliers using z-score."""
        issues = []
        
        # Calculate z-scores for close prices
        close_mean = df['close'].mean()
        close_std = df['close'].std()
        
        if close_std == 0:
            return 0, issues
        
        z_scores = np.abs((df['close'] - close_mean) / close_std)
        outliers = z_scores > OUTLIER_STD_THRESHOLD
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            issues.append(f"{outlier_count} price outliers detected (>{OUTLIER_STD_THRESHOLD} std devs)")
        
        return outlier_count, issues
    
    @staticmethod
    def _validate_volume(df: pd.DataFrame) -> Tuple[int, List[str]]:
        """Validate volume data."""
        issues = []
        
        # Check for zero/negative volume
        zero_vol = df[df['volume'] <= 0]
        zero_count = len(zero_vol)
        
        if zero_count > 0:
            issues.append(f"{zero_count} candles with zero/negative volume")
        
        # Check for suspiciously low volume
        low_vol = df[df['volume'] < MIN_VOLUME_THRESHOLD]
        if len(low_vol) > len(df) * 0.1:  # >10% of candles
            issues.append(f"{len(low_vol)} candles with very low volume (<{MIN_VOLUME_THRESHOLD})")
        
        return zero_count, issues
    
    @staticmethod
    def _detect_price_jumps(df: pd.DataFrame) -> List[str]:
        """Detect suspicious price jumps."""
        issues = []
        
        # Calculate price changes
        price_changes = df['close'].pct_change().abs()
        
        # Find jumps > threshold
        jumps = price_changes[price_changes > MAX_PRICE_JUMP_PERCENT]
        
        if len(jumps) > 0:
            max_jump = jumps.max()
            issues.append(f"{len(jumps)} price jumps >{MAX_PRICE_JUMP_PERCENT*100:.0f}% (max: {max_jump*100:.1f}%)")
        
        return issues
    
    @staticmethod
    def _check_freshness(df: pd.DataFrame) -> Optional[str]:
        """Check if data is fresh (not stale)."""
        if len(df) == 0:
            return None
        
        # Get latest timestamp
        latest_ts = df['timestamp'].iloc[-1]
        latest_dt = datetime.fromtimestamp(latest_ts / 1000)
        
        # Check age
        age_seconds = (datetime.now() - latest_dt).total_seconds()
        
        if age_seconds > MAX_DATA_AGE_SECONDS:
            return f"Stale data: {age_seconds/60:.1f} minutes old (max {MAX_DATA_AGE_SECONDS/60:.1f})"
        
        return None
    
    @staticmethod
    def _check_nan_inf(df: pd.DataFrame) -> List[str]:
        """Check for NaN or Inf values."""
        issues = []
        
        # Check for NaN
        nan_counts = df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        
        if len(cols_with_nan) > 0:
            issues.append(f"NaN values in columns: {cols_with_nan.to_dict()}")
        
        # Check for Inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"{inf_count} Inf values in {col}")
        
        return issues


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    test_df = pd.DataFrame({
        'timestamp': [datetime.now().timestamp() * 1000 - i*900000 for i in range(100, 0, -1)],
        'open': np.random.uniform(95000, 96000, 100),
        'high': np.random.uniform(96000, 97000, 100),
        'low': np.random.uniform(94000, 95000, 100),
        'close': np.random.uniform(95000, 96000, 100),
        'volume': np.random.uniform(10000, 100000, 100)
    })
    
    print("="*60)
    print("Data Validator Test")
    print("="*60)
    
    result = DataValidator.validate_ohlcv(test_df, "BTC/USDT", "15m")
    print(f"\n{result}")
    
    if result.issues:
        print("\nIssues:")
        for issue in result.issues:
            print(f"  ❌ {issue}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠️ {warning}")
    
    print("\n✅ Data Validator working!")
