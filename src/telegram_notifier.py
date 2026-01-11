"""
Telegram Notifier Module
=========================
Sends trading signals with chart images to Telegram.
Uses simple HTTP requests to Telegram Bot API.
"""
import requests
from pathlib import Path
from typing import Optional
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

from signal_generator import Signal, SignalDirection

logger = logging.getLogger(__name__)

# Telegram Bot API base URL
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"


class TelegramNotifier:
    """
    Telegram notification handler.
    Sends trading signals with chart images via Telegram Bot API.
    """
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize the Telegram notifier.
        
        Args:
            bot_token: Telegram Bot Token (from @BotFather)
            chat_id: Telegram Chat ID to send messages to
        """
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.api_base = TELEGRAM_API_BASE.format(token=self.bot_token)
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
    
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id and 
                    self.bot_token != "your_bot_token_here" and
                    self.chat_id != "your_chat_id_here")
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message to Telegram.
        
        Args:
            text: Message text
            parse_mode: Parse mode ("HTML" or "Markdown")
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("Telegram not configured, skipping message send")
            return False
        
        url = f"{self.api_base}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            logger.info("Telegram message sent successfully")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_photo(
        self, 
        image_path: str, 
        caption: str = "",
        parse_mode: str = "HTML"
    ) -> bool:
        """
        Send a photo with caption to Telegram.
        
        Args:
            image_path: Path to the image file
            caption: Optional caption for the image
            parse_mode: Parse mode for caption
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            logger.warning("Telegram not configured, skipping photo send")
            return False
        
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return False
        
        url = f"{self.api_base}/sendPhoto"
        
        try:
            with open(image_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": parse_mode
                }
                
                response = requests.post(url, data=data, files=files, timeout=60)
                response.raise_for_status()
                logger.info(f"Telegram photo sent successfully: {image_path.name}")
                return True
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return False
    
    def send_signal_alert(
        self, 
        signal: Signal, 
        chart_path: Optional[str] = None
    ) -> bool:
        """
        Send a complete signal alert with chart to Telegram.
        
        Args:
            signal: Signal object with trade details
            chart_path: Path to chart image (optional)
        
        Returns:
            True if successful, False otherwise
        """
        # Format the caption
        caption = self._format_signal_caption(signal)
        
        if chart_path and Path(chart_path).exists():
            # Send photo with caption
            return self.send_photo(chart_path, caption)
        else:
            # Send text only
            return self.send_message(caption)
    
    def _format_signal_caption(self, signal: Signal) -> str:
        """
        Format signal data into an attractive Telegram caption.
        
        Args:
            signal: Signal object
        
        Returns:
            Formatted caption string
        """
        # Direction styling
        if signal.direction.value == "long":
            direction_emoji = "🟢"
            direction_text = "LONG"
            header_style = "📈"
        else:
            direction_emoji = "🔴"
            direction_text = "SHORT"
            header_style = "📉"
        
        # Channel type emoji
        channel_emoji = {
            "ascending": "⬆️",
            "descending": "⬇️",
            "horizontal": "↔️",
            "converging": "🔺",
            "diverging": "🔻"
        }.get(signal.channel_type.value, "📊")
        
        # Calculate potential profit %
        if signal.direction.value == "long":
            profit_pct = ((signal.take_profit - signal.entry_price) / signal.entry_price) * 100
        else:
            profit_pct = ((signal.entry_price - signal.take_profit) / signal.entry_price) * 100
        
        # Confidence stars
        score = getattr(signal, 'confidence_score', 0) or 0
        if score >= 90:
            conf_display = "⭐⭐⭐⭐⭐"
        elif score >= 75:
            conf_display = "⭐⭐⭐⭐"
        elif score >= 60:
            conf_display = "⭐⭐⭐"
        else:
            conf_display = "⭐⭐"
        
        # Build the premium caption (DCA - no SL)
        caption = f"""
{direction_emoji} <b>{direction_text} SIGNAL</b> {direction_emoji}
━━━━━━━━━━━━━━

🪙 <b>{signal.symbol.split('/')[0]}</b>
{channel_emoji} <i>{signal.channel_type.value.capitalize()}</i>
⏱ <code>{signal.timeframe}</code>

<b>TRADE SETUP</b>
━━━━━━━━━━━━━━

<<<<<<< HEAD
💵 Entry: <code>${signal.entry_price:,.4f}</code>
✅ TP: <code>${signal.take_profit:,.4f}</code>
   (+{profit_pct:.2f}%)
❌ SL: <code>${signal.stop_loss:,.4f}</code>
   (-{risk_pct:.2f}%)
=======
   💵 Entry:  <code>${signal.entry_price:,.4f}</code>
   ✅ TP:      <code>${signal.take_profit:,.4f}</code>  (+{profit_pct:.2f}%)
>>>>>>> old-version

━━━━━━━━━━━━━━

📊 R:R: <code>1:{signal.rr_ratio:.2f}</code>
🎯 Score: {conf_display}
<code>({score:.0f}/100)</code>

<<<<<<< HEAD
<i>⚠️ Risk only what you can
afford to lose</i>
=======
<i>💡 Using DCA strategy</i>
>>>>>>> old-version
"""
        return caption.strip()


def send_dca_opportunity_alert(dca_opp: 'DCAOpportunity', chart_path: Optional[str] = None) -> bool:
    """
    Send DCA opportunity alert to Telegram.
    
    Args:
        dca_opp: DCAOpportunity object
        chart_path: Optional path to chart image
    
    Returns:
        True if successful
    """
    # Direction emoji
    if dca_opp.direction == "long":
        dir_emoji = "🟢"
        dir_text = "LONG"
    else:
        dir_emoji = "🔴"
        dir_text = "SHORT"
    
    # Confidence stars
    score = int(dca_opp.confidence_score)
    if score >= 90:
        stars = "⭐⭐⭐⭐⭐"
    elif score >= 75:
        stars = "⭐⭐⭐⭐"
    elif score >= 60:
        stars = "⭐⭐⭐"
    else:
        stars = "⭐⭐"
    
    # Calculate potential profit after DCA
    if dca_opp.direction == "long":
        potential_pct = ((dca_opp.take_profit - dca_opp.new_average) / dca_opp.new_average) * 100
    else:
        potential_pct = ((dca_opp.new_average - dca_opp.take_profit) / dca_opp.new_average) * 100
    
    caption = f"""
🔄 <b>DCA OPPORTUNITY!</b> 🔄
━━━━━━━━━━━━━━

🪙 <b>{dca_opp.symbol.split('/')[0]} {dir_text}</b>

📉 <b>Current Situation:</b>
   Initial Entry: <code>${dca_opp.original_entry:,.4f}</code>
   Current Price: <code>${dca_opp.current_price:,.4f}</code>
   Unrealized: <code>{dca_opp.unrealized_pct:+.2f}%</code>

✅ <b>NEW Channel Detected!</b>
   {dir_emoji} {dca_opp.channel_type.capitalize()} ({dca_opp.timeframe})
   DCA Entry: <code>${dca_opp.dca_entry:,.4f}</code>
   Confidence: {stars}

📊 <b>After DCA (Equal Size):</b>
   New Average: <code>${dca_opp.new_average:,.4f}</code>
   TP: <code>${dca_opp.take_profit:,.4f}</code>
   Potential: <code>+{potential_pct:.2f}%</code>

💡 <i>New channel formed - DCA opportunity!</i>
"""
    
    notifier = get_notifier()
    if chart_path:
        return notifier.send_photo(chart_path, caption.strip())
    else:
        return notifier.send_message(caption.strip())


def send_dca_confirmation(position: 'Position') -> bool:
    """
    Send DCA entry confirmation message.
    
    Args:
        position: Updated position with DCA entry
    
    Returns:
        True if successful
    """
    # Direction emoji
    dir_emoji = "📈" if position.direction == "long" else "📉"
    
    # Calculate potential profit
    if position.direction == "long":
        potential_pct = ((position.take_profit - position.average_entry) / position.average_entry) * 100
    else:
        potential_pct = ((position.average_entry - position.take_profit) / position.average_entry) * 100
    
    # Format entries
    entries_text = ""
    for i, entry in enumerate(position.entries or [], 1):
        label = "Initial" if i == 1 else f"DCA #{i-1}"
        entries_text += f"   {label}: <code>${entry['price']:,.4f}</code>\n"
    
    caption = f"""
✅ <b>DCA ENTRY ADDED</b>
━━━━━━━━━━━━━━

🪙 <b>{position.symbol.split('/')[0]} {position.direction.upper()}</b>

<b>📍 Entries:</b>
{entries_text}
<b>📊 Updated Position:</b>
   Average Entry: <code>${position.average_entry:,.4f}</code>
   Take Profit: <code>${position.take_profit:,.4f}</code>
   Potential: <code>+{potential_pct:.2f}%</code>
   
🎯 <b>DCA Count:</b> {position.dca_count}
"""
    
    return get_notifier().send_message(caption.strip())


# Default notifier instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the default TelegramNotifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def send_telegram_alert(signal: Signal, image_path: Optional[str] = None) -> bool:
    """
    Convenience function to send a signal alert.
    
    Args:
        signal: Signal object
        image_path: Path to chart image
    
    Returns:
        True if successful
    """
    return get_notifier().send_signal_alert(signal, image_path)


def send_startup_message() -> bool:
    """Send a startup notification to Telegram."""
    notifier = get_notifier()
    if not notifier.is_configured():
        return False
    
<<<<<<< HEAD
    from datetime import datetime
    from config import TOP_COINS_COUNT, TIMEFRAMES, MIN_CONFIDENCE_SCORE
    
    message = f"""
🚀 <b>SIGNAL BOT ACTIVATED</b> 🚀
━━━━━━━━━━━━━━━━━━━━

<b>⚡ System Grade: 11/10 LEGENDARY</b>

<b>📊 Monitoring:</b>
🪙 Top {TOP_COINS_COUNT} coins by volume
⏱ Timeframes: {', '.join(TIMEFRAMES)}
⭐ Min Score: {MIN_CONFIDENCE_SCORE}

<b>🎯 Features Active:</b>
✅ Market Regime Detection
✅ Dynamic Kelly Sizing
✅ ML Signal Scoring
✅ Trailing Stops
✅ Partial Profits
✅ Real-Time Correlation
✅ Risk Management

<b>🛡️ Protection:</b>
🔥 Portfolio Heat Monitor
📊 Correlation Matrix
🎯 Diversification Score
⚡ Circuit Breaker

━━━━━━━━━━━━━━━━━━━━
<i>🟢 Online | {datetime.now().strftime('%H:%M:%S')}</i>

<b>Ready to hunt alpha! 🎯</b>
=======
    from config import TOP_COINS_COUNT, TIMEFRAMES, SCAN_INTERVAL_SECONDS
    
    # Format timeframes nicely
    tf_display = ", ".join(TIMEFRAMES)
    scan_mins = SCAN_INTERVAL_SECONDS // 60
    
    message = f"""
╔══════════════╗
  🚀 <b>BOT LIVE</b> 🚀
╚══════════════╝

<b>📊 Signal Samurai</b>
<i>AI Crypto Signals</i>

<b>⚙️ Config:</b>
🪙 {TOP_COINS_COUNT} coins
⏱ {tf_display}
🔄 Every {scan_mins}min

<b>🎯 Strategy:</b>
• Channel entries
• MTF confluence
• DCA alerts
• Auto tracking

<b>🔔 Alerts:</b>
📈 Signals
🔄 DCA ops
✅ TP hits

━━━━━━━━━━━━━━
<i>🟢 Scanning...</i>
<i>💡 DCA mode</i>
>>>>>>> old-version
"""
    return notifier.send_message(message.strip())


def send_shutdown_message() -> bool:
    """Send a shutdown notification to Telegram."""
    notifier = get_notifier()
    if not notifier.is_configured():
        return False
    
    message = """
🔴 <b>Crypto Signal Bot Stopped</b>
━━━━━━━━━━━━━━━━━━━━━━

Bot has been shut down.
"""
    return notifier.send_message(message.strip())


def send_telegram_message(text: str) -> bool:
    """
    Convenience function to send a simple text message.
    
    Args:
        text: Message text
    
    Returns:
        True if successful
    """
    return get_notifier().send_message(text)


# Test the module when run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Telegram Notifier Module Test")
    print("=" * 60)
    
    notifier = TelegramNotifier()
    
    if notifier.is_configured():
        print("\n✅ Telegram is configured")
        print(f"   Bot Token: {notifier.bot_token[:10]}...{notifier.bot_token[-5:]}")
        print(f"   Chat ID: {notifier.chat_id}")
        
        # Test sending a message
        print("\n📤 Sending test message...")
        success = notifier.send_message("🧪 <b>Test Message</b>\n\nIf you see this, the bot is working!")
        
        if success:
            print("✅ Test message sent successfully!")
        else:
            print("❌ Failed to send test message")
    else:
        print("\n⚠️ Telegram is NOT configured")
        print("   Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
        print("\n   Example .env content:")
        print("   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ")
        print("   TELEGRAM_CHAT_ID=123456789")
    
    print("\n" + "=" * 60)
    print("✅ Telegram notifier test complete!")
    print("=" * 60)
