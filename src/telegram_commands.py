"""
Telegram Bot Commands Module
=============================
Handles interactive commands sent to the Telegram bot.
Allows users to check stats, status, and control the bot via Telegram.
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    from src.position_tracker import get_tracker
except ImportError as e:
    print(f"Warning: Could not import telegram modules: {e}")

logger = logging.getLogger(__name__)


class TelegramCommands:
    """Handles Telegram bot commands."""
    
    def __init__(self):
        """Initialize command handler."""
        self.authorized_chat_id = TELEGRAM_CHAT_ID
        
    def is_authorized(self, update: Update) -> bool:
        """Check if user is authorized to use commands."""
        return str(update.effective_chat.id) == str(self.authorized_chat_id)
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        message = """
🤖 <b>Signal Bot</b> 🤖
━━━━━━━━━━━━━━

<b>📱 Commands:</b>

/stats
📊 Performance

/positions
🔄 Active Trades

/status
⚙️ Bot Settings

/reset
🔄 Fresh Start

/help
📚 Help Guide

━━━━━━━━━━━━━━
<i>✅ Running 24/7</i>
"""
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - show trading statistics."""
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        try:
            tracker = get_tracker()
            stats = tracker.get_stats()
            
            total = stats['total_trades']
            wins = stats['wins']
            losses = stats['losses']
            win_rate = stats['win_rate']
            avg_pnl = stats['avg_pnl']
            active = stats['active_positions']
            
            # Win rate emoji
            if win_rate >= 50:
                wr_emoji = "🔥"
            elif win_rate >= 40:
                wr_emoji = "✅"
            elif win_rate >= 30:
                wr_emoji = "📊"
            else:
                wr_emoji = "⚠️"
            
            # PnL emoji
            pnl_emoji = "💰" if avg_pnl > 0 else "📉"
            
            message = f"""
📊 <b>Performance</b> 📊
━━━━━━━━━━━━━━

<b>Results:</b>
✅ Wins: <code>{wins}</code>
❌ Losses: <code>{losses}</code>
{wr_emoji} Win Rate: <code>{win_rate}%</code>
{pnl_emoji} Avg PnL: <code>{avg_pnl:+.2f}%</code>

<b>Summary:</b>
🔄 Active: <code>{active}</code>
📊 Total: <code>{total}</code>

━━━━━━━━━━━━━━
<i>⏰ {datetime.now().strftime('%H:%M')}</i>
"""
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in /stats command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command - show active positions."""
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        try:
            tracker = get_tracker()
            active = tracker.active_positions
            
            if not active:
                message = """
🔄 <b>Positions</b> 🔄
━━━━━━━━━━━━━━

📭 <i>No active positions</i>

━━━━━━━━━━━━━━
<i>Waiting for signals...</i>
"""
                await update.message.reply_text(message, parse_mode='HTML')
                return
            
            message = f"""
🔄 <b>Active ({len(active)})</b> 🔄
━━━━━━━━━━━━━━

"""
            
            for i, pos in enumerate(active, 1):
                dir_emoji = "📈" if pos.direction == "long" else "📉"
                entry_time = datetime.fromisoformat(pos.entry_time)
                duration = datetime.now() - entry_time
                hours = int(duration.total_seconds() / 3600)
                mins = int((duration.total_seconds() % 3600) / 60)
                
                symbol_short = pos.symbol.split('/')[0]
                
                # Confidence stars
                conf = int(pos.confidence_score)
                if conf >= 90:
                    stars = "⭐⭐⭐⭐⭐"
                elif conf >= 80:
                    stars = "⭐⭐⭐⭐"
                elif conf >= 70:
                    stars = "⭐⭐⭐"
                else:
                    stars = "⭐⭐"
                
                message += f"""<b>{i}. {dir_emoji} {symbol_short}</b>

💵 Entry: <code>${pos.entry_price:.4f}</code>
🎯 TP: <code>${pos.take_profit:.4f}</code>
🛑 SL: <code>${pos.stop_loss:.4f}</code>

⏱ {hours}h {mins}m  |  {stars}

"""
            
            message += """━━━━━━━━━━━━━━
<i>Monitoring TP/SL...</i>"""
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in /positions command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - show bot configuration."""
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        try:
            from config import (
                TOP_COINS_COUNT, SCAN_INTERVAL_SECONDS, TIMEFRAMES,
                MIN_CONFIDENCE_SCORE, VOLUME_MULTIPLIER, MIN_RR_RATIO
            )
            
            message = f"""
⚙️ <b>Bot Config</b> ⚙️
━━━━━━━━━━━━━━

<b>Scanning:</b>
🪙 Coins: <code>{TOP_COINS_COUNT}</code>
⏱ Interval: <code>{SCAN_INTERVAL_SECONDS // 60}min</code>
📊 TF: <code>{', '.join(TIMEFRAMES)}</code>

<b>Filters:</b>
⭐ Min Score: <code>{MIN_CONFIDENCE_SCORE}</code>
📈 Volume: <code>{VOLUME_MULTIPLIER}x</code>
💰 Min R:R: <code>{MIN_RR_RATIO}</code>

<b>Status:</b>
🟢 Running on AWS
🔄 Auto-restart ON
📱 Notifications ON

━━━━━━━━━━━━━━
<i>⏰ {datetime.now().strftime('%H:%M')}</i>
"""
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in /status command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        message = """
📚 <b>Help Guide</b> 📚
━━━━━━━━━━━━━━

<b>/stats</b> 📊
View win rate & PnL

<b>/positions</b> 🔄
Show active trades

<b>/status</b> ⚙️
Bot settings

<b>/reset</b> 🔄
Close all & delete

<b>/help</b> 📚
This help message

━━━━━━━━━━━━━━
<i>💡 Auto signals ON</i>
"""
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command - close all positions and delete charts."""
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        try:
            import shutil
            from pathlib import Path
            
            # Get tracker
            tracker = get_tracker()
            active_count = len(tracker.active_positions)
            
            # Close all active positions
            tracker.active_positions = []
            tracker.closed_positions = []
            tracker.total_wins = 0
            tracker.total_losses = 0
            tracker._save_data()
            
            # Delete all charts
            charts_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "charts"
            if charts_dir.exists():
                chart_count = len(list(charts_dir.glob("*.png")))
                for chart_file in charts_dir.glob("*.png"):
                    chart_file.unlink()
            else:
                chart_count = 0
            
            message = f"""
🔄 <b>Reset Done!</b> 🔄
━━━━━━━━━━━━━━

✅ Closed <code>{active_count}</code> positions
✅ Deleted <code>{chart_count}</code> charts
✅ Reset all statistics

━━━━━━━━━━━━━━
<i>🆕 Fresh start!</i>
"""
            await update.message.reply_text(message, parse_mode='HTML')
            logger.info(f"Reset executed: {active_count} positions closed, {chart_count} charts deleted")
            
        except Exception as e:
            logger.error(f"Error in /reset command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")


async def setup_commands(application: Application):
    """Setup command handlers."""
    commands = TelegramCommands()
    
    application.add_handler(CommandHandler("start", commands.cmd_start))
    application.add_handler(CommandHandler("stats", commands.cmd_stats))
    application.add_handler(CommandHandler("positions", commands.cmd_positions))
    application.add_handler(CommandHandler("status", commands.cmd_status))
    application.add_handler(CommandHandler("reset", commands.cmd_reset))
    application.add_handler(CommandHandler("help", commands.cmd_help))
    
    logger.info("Telegram commands registered")


def start_command_bot():
    """Start the Telegram bot command listener in background."""
    try:
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Setup commands
        import asyncio
        asyncio.run(setup_commands(application))
        
        # Start polling in background
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Error starting command bot: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting Telegram command bot...")
    start_command_bot()
