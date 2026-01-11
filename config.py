"""
Crypto Futures Signal Bot - Configuration Module
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
CHARTS_DIR = PROJECT_ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Exchange Configuration
EXCHANGE_ID = "binance"
USE_SANDBOX = False

# Scanning Configuration - DAY TRADING OPTIMIZED
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))  # 1 minute for maximum signals
TOP_COINS_COUNT = int(os.getenv("TOP_COINS_COUNT", "100"))  # More coins = more signals

# Timeframes optimized for DAY TRADING:
# - 5m: Quick scalp entries, very frequent signals
# - 15m: Day trading sweet spot
# - 1h: Swing trades, confluence
TIMEFRAMES = ["5m", "15m"]  # Multiple timeframes = more signals!

# Swing Detection Parameters
SWING_LOOKBACK = 5  # Bars on each side to confirm swing point
MIN_SWING_SIGNIFICANCE = 0.5  # Minimum ATR multiple for swing significance

# Channel Parameters
MIN_TOUCHES_PER_LINE = 2  # Minimum touch points for valid channel
CHANNEL_LOOKBACK_BARS = 100  # How many bars to analyze
MIN_R_SQUARED = 0.4  # Minimum R² for trendline quality (lowered for more signals)
SLOPE_THRESHOLD = 0.0001  # Slope threshold for horizontal classification

# Signal Parameters
TOUCH_TOLERANCE_ATR = 0.1  # How close to line counts as "touch" (ATR multiple)
SL_BUFFER_ATR = 0.5  # Stop loss buffer beyond channel line (ATR multiple)
MIN_RR_RATIO = 2.0  # Minimum risk/reward ratio to generate signal (raised for quality)

# Signal Filter Parameters (NEW)
VOLUME_MULTIPLIER = 1.5  # Require volume > 1.5x average (more signals)
RSI_PERIOD = 14  # RSI calculation period
RSI_OVERBOUGHT = 70  # Don't go LONG above this
RSI_OVERSOLD = 30  # Don't go SHORT below this
MIN_CONFIDENCE_SCORE = 65  # Minimum filter score (0-100) to send signal (balanced quality/quantity)

# Advanced Filter Parameters (NEW)
ADX_PERIOD = 14  # ADX calculation period
ADX_TRENDING_THRESHOLD = 25  # ADX above this = strong trend
ADX_RANGING_THRESHOLD = 25  # ADX below this = ranging market (good for channels)
ADX_MAX_THRESHOLD = 35  # ADX above this = too trending (avoid)
SR_LOOKBACK_PERIODS = 50  # Periods to look back for S/R levels
SR_TOUCH_TOLERANCE = 0.002  # 0.2% tolerance for S/R matching
FUNDING_RATE_THRESHOLD = 0.0005  # 0.05% funding rate considered "high"

# Chart Settings
CHART_BARS = 80  # Number of bars to show on chart
CHART_STYLE = "nightclouds"  # mplfinance style

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================
# INSTITUTIONAL RISK MANAGEMENT (Phase 1)
# ============================================================================

# Portfolio Risk Limits
MAX_PORTFOLIO_RISK = 0.02  # 2% maximum total portfolio risk
MAX_POSITIONS = 10  # Maximum concurrent positions
MAX_CORRELATED_EXPOSURE = 0.05  # 5% max exposure to correlated assets (correlation > 0.7)
MAX_DRAWDOWN_STOP = 0.15  # Circuit breaker at 15% drawdown

# Position Sizing
MIN_POSITION_SIZE = 0.001  # 0.1% minimum position size
MAX_POSITION_SIZE = 0.02  # 2% maximum position size
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))  # Starting capital in USD

# Kelly Criterion Parameters (for dynamic sizing)
USE_KELLY_SIZING = True  # Enable Kelly Criterion position sizing
KELLY_FRACTION = 0.5  # Use half-Kelly for safety

# ============================================================================
# EXECUTION COST MODELING (Phase 1)
# ============================================================================

# Trading Fees (Binance Futures)
MAKER_FEE = 0.0002  # 0.02% maker fee
TAKER_FEE = 0.0004  # 0.04% taker fee
ASSUME_TAKER = True  # Conservative: assume taker fees

# Slippage Modeling
SLIPPAGE_BPS = {
    'high_liquidity': 5,     # 0.05% - BTC, ETH
    'medium_liquidity': 15,  # 0.15% - Top 20 coins
    'low_liquidity': 50      # 0.5% - Others
}
DEFAULT_SLIPPAGE_BPS = 15  # Default to medium liquidity

# Liquidity Tiers (by 24h volume)
HIGH_LIQUIDITY_THRESHOLD = 1_000_000_000  # $1B+ daily volume
MEDIUM_LIQUIDITY_THRESHOLD = 100_000_000  # $100M+ daily volume

# Funding Rate Impact
FUNDING_INTERVAL_HOURS = 8  # Funding every 8 hours
EXPECTED_HOLDING_HOURS = 4  # Average position duration

# ============================================================================
# DATA QUALITY VALIDATION (Phase 1)
# ============================================================================

# Gap Detection
MAX_ALLOWED_GAP_PERCENT = 0.05  # Reject if >5% data missing
MAX_PRICE_JUMP_PERCENT = 0.10  # Flag if price jumps >10% in one candle

# Outlier Detection
OUTLIER_STD_THRESHOLD = 5.0  # Flag if price >5 std devs from mean

# Volume Validation
MIN_VOLUME_THRESHOLD = 1000  # Minimum volume per candle (USD)

# Data Freshness
MAX_DATA_AGE_SECONDS = 300  # Reject data older than 5 minutes
