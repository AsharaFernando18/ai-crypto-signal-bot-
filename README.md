# 🚀 AI Crypto Futures Signal Bot

Advanced cryptocurrency futures trading signal bot with AI-powered channel detection, multi-timeframe analysis, and real-time Telegram notifications.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AWS](https://img.shields.io/badge/AWS-Deployed-orange.svg)](https://aws.amazon.com/)

## ✨ Features

### 🎯 Signal Generation
- **Channel Detection** - Automatically detects ascending, descending, and parallel channels
- **Multi-Timeframe Analysis** - 15m signals with 1h & 4h trend confirmation
- **Smart Entry Points** - Precise entry, take-profit, and stop-loss levels
- **High Win Rate** - Optimized filters for 40-50% win rate with 3.5:1 R:R ratio

### 📊 Advanced Filters
- Volume confirmation (2.0x average)
- RSI overbought/oversold detection
- ADX trend strength analysis
- Support/Resistance confluence
- CVD (Cumulative Volume Delta) analysis
- Funding rate monitoring

### 📱 Telegram Integration
- Real-time signal notifications with charts
- Interactive bot commands (`/stats`, `/positions`, `/status`)
- Mobile-optimized message formatting
- Position tracking with TP/SL updates

### ☁️ AWS Deployment
- 24/7 operation on AWS EC2
- Auto-restart on failure
- Systemd service integration
- Remote monitoring via SSH

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
TA-Lib
CCXT
Telegram Bot Token
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ai-crypto-signal-bot.git
cd ai-crypto-signal-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your Telegram credentials
```

4. **Run the bot**
```bash
python main.py
```

## ⚙️ Configuration

### Key Settings (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_COINS_COUNT` | 50 | Number of coins to scan |
| `SCAN_INTERVAL_SECONDS` | 180 | Scan interval (3 minutes) |
| `MIN_CONFIDENCE_SCORE` | 75 | Minimum signal quality |
| `MIN_RR_RATIO` | 2.0 | Minimum risk/reward ratio |
| `VOLUME_MULTIPLIER` | 2.0 | Volume confirmation threshold |

## 📱 Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Show welcome message |
| `/stats` | View trading statistics |
| `/positions` | Show active positions |
| `/status` | Display bot configuration |
| `/reset` | Close all positions & reset |
| `/help` | Show command help |

## 📊 Performance

- **Win Rate**: 40-50% (optimized)
- **Risk/Reward**: 3.5:1 average
- **Signals/Day**: 15-20 high-quality signals
- **Timeframes**: 15m (signals), 1h + 4h (confluence)

## 🏗️ Architecture

```
crypto-futures-signals/
├── main.py                 # Main bot entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── src/
│   ├── channel_builder.py    # Channel detection logic
│   ├── signal_generator.py   # Signal generation
│   ├── signal_filters.py     # Advanced filtering
│   ├── advanced_filters.py   # ADX, S/R, CVD analysis
│   ├── data_fetcher.py       # Market data fetching
│   ├── telegram_notifier.py  # Telegram integration
│   ├── telegram_commands.py  # Interactive commands
│   ├── position_tracker.py   # Position management
│   ├── chart_generator.py    # Chart visualization
│   └── cvd_analyzer.py       # Volume analysis
└── .env.example          # Environment template
```

## 🌟 Key Algorithms

### Channel Detection
- Linear regression for trendline calculation
- Parallel channel identification
- R-squared validation (>0.5)
- Minimum touch points (2+)

### Signal Validation
- Multi-timeframe trend alignment
- Volume spike confirmation
- RSI divergence detection
- Support/Resistance confluence
- CVD trend confirmation

### Risk Management
- Dynamic stop-loss placement
- ATR-based position sizing
- Risk/Reward optimization
- Position tracking with auto-close

## 🚀 AWS Deployment

### EC2 Setup
```bash
# Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv build-essential -y

# Install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib && ./configure --prefix=/usr && make && sudo make install

# Setup bot
cd ~/crypto-futures-signals
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Systemd Service
```bash
sudo systemctl enable crypto-bot
sudo systemctl start crypto-bot
sudo systemctl status crypto-bot
```

## 📈 Example Signals

### Signal Message Format
```
🟢 LONG SIGNAL 🟢
━━━━━━━━━━━━━━

🪙 BTC
⬆️ Ascending Channel
⏱ 15m

💵 Entry: $95,000.0000
✅ TP: $96,500.0000 (+1.58%)
❌ SL: $94,500.0000 (-0.53%)

📊 R:R Ratio: 1:2.50
🎯 Confidence: ⭐⭐⭐⭐ (85/100)
```

## 🛠️ Customization

### Adding New Filters
```python
# src/advanced_filters.py
def custom_filter(df, signal):
    # Your custom logic
    return passed, score
```

### Modifying Signal Criteria
```python
# config.py
MIN_CONFIDENCE_SCORE = 80  # Stricter filtering
VOLUME_MULTIPLIER = 2.5    # Higher volume requirement
```

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This bot is for educational purposes only. Cryptocurrency trading carries risk. Always do your own research and never invest more than you can afford to lose.

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Made with ❤️ for crypto traders**

🌟 Star this repo if you find it useful!
