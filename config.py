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

# Scanning Configuration
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "180"))  # 3 minutes for faster signals
TOP_COINS_COUNT = int(os.getenv("TOP_COINS_COUNT", "50"))

# Timeframes optimized for 15m trading:
# - 5m: Quick entries, more frequent signals
# - 15m: Primary trading timeframe  
# - 1h: Higher timeframe confluence
TIMEFRAMES = ["15m"]  # Only trade on 15m, 1h/4h for confluence

# Swing Detection Parameters
SWING_LOOKBACK = 5  # Bars on each side to confirm swing point
MIN_SWING_SIGNIFICANCE = 0.5  # Minimum ATR multiple for swing significance

# Channel Parameters
MIN_TOUCHES_PER_LINE = 2  # Minimum touch points for valid channel
CHANNEL_LOOKBACK_BARS = 100  # How many bars to analyze
MIN_R_SQUARED = 0.5  # Minimum R² for trendline quality (lowered for real markets)
SLOPE_THRESHOLD = 0.0001  # Slope threshold for horizontal classification

# Signal Parameters
TOUCH_TOLERANCE_ATR = 0.1  # How close to line counts as "touch" (ATR multiple)
SL_BUFFER_ATR = 0.5  # Stop loss buffer beyond channel line (ATR multiple)
MIN_RR_RATIO = 2.0  # Minimum risk/reward ratio to generate signal (raised for quality)

# Signal Filter Parameters (NEW)
VOLUME_MULTIPLIER = 2.0  # Require volume > 2.0x average (stronger confirmation)
RSI_PERIOD = 14  # RSI calculation period
RSI_OVERBOUGHT = 70  # Don't go LONG above this
RSI_OVERSOLD = 30  # Don't go SHORT below this
MIN_CONFIDENCE_SCORE = 75  # Minimum filter score (0-100) to send signal (high quality only)

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
