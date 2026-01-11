"""
Crypto Futures Signal Bot - Main Application
==============================================
Scans top crypto futures pairs for channel-based trading signals
and sends alerts via Telegram.

Enhanced with:
- Volume confirmation
- RSI filter
- Multi-timeframe confluence
- Candle pattern detection
"""
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Optional
import pandas as pd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    SCAN_INTERVAL_SECONDS, TOP_COINS_COUNT, TIMEFRAMES,
    CHARTS_DIR, LOG_LEVEL, MIN_CONFIDENCE_SCORE, INITIAL_CAPITAL
)
from src.data_fetcher import DataFetcher, get_top_coins, fetch_ohlcv
from src.channel_builder import detect_channel, Channel
from src.signal_generator import check_signal, Signal
from src.signal_filters import apply_signal_filters, FilterResult
from src.chart_generator import generate_signal_chart
from src.telegram_notifier import (
    send_telegram_alert, send_startup_message, 
    send_shutdown_message, get_notifier, send_telegram_message,
    send_dca_opportunity_alert, send_dca_confirmation
)
from src.position_tracker import get_tracker, PositionTracker
from src.dca_manager import get_dca_manager
from src.channel_builder import ChannelType

# Phase 1: Institutional Risk Management
from src.risk_manager import get_risk_manager, RiskManager
from src.execution_model import ExecutionModel
from src.data_validator import DataValidator

# Phase 3: Optimization
from src.position_sizer import get_position_sizer
from src.signal_strength import get_signal_classifier, SignalStrength
from src.trailing_stops import get_stop_manager
from src.partial_profits import get_profit_manager
from src.market_bias import get_bias_detector

# Phase 4: ML Enhancement
from src.ml_scorer import get_ml_scorer

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / 'bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Track sent signals to avoid duplicates
sent_signals = {}  # Key: (symbol, timeframe, signal_direction), Value: timestamp


def should_send_signal(symbol: str, timeframe: str, direction: str) -> bool:
    """
    Check if we should send this signal (avoid duplicates).
    
    Args:
        symbol: Trading symbol
        timeframe: Signal timeframe
        direction: Signal direction (long/short)
    
    Returns:
        True if signal should be sent
    """
    key = (symbol, timeframe, direction)
    now = datetime.now()
    
    # Smart Cooldowns 🧠
    # Dynamic cooldown based on timeframe
    if timeframe == '15m':
        cooldown_seconds = 1800  # 30 minutes (2 candles)
    elif timeframe == '1h':
        cooldown_seconds = 7200  # 2 hours (2 candles)
    elif timeframe == '5m':
        cooldown_seconds = 900   # 15 minutes (3 candles)
    else:
        cooldown_seconds = 3600  # Default 1 hour
    
    if key in sent_signals:
        last_sent = sent_signals[key]
        if (now - last_sent).total_seconds() < cooldown_seconds:
            return False
    
    sent_signals[key] = now
    return True


def cleanup_old_signals():
    """Remove old signals from tracking dict."""
    now = datetime.now()
    keys_to_remove = []
    
    for key, timestamp in sent_signals.items():
        # Remove signals older than 2 hours
        if (now - timestamp).total_seconds() > 7200:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del sent_signals[key]


def scan_symbol(
    symbol: str, 
    timeframe: str, 
    fetcher: DataFetcher,
    higher_tf_channel: Optional[Channel] = None,
    pre_fetched_df: Optional[pd.DataFrame] = None,
    funding_rate: Optional[float] = None
) -> Optional[Signal]:
    """
    Scan a single symbol on a single timeframe for signals.
    """
    try:
        logger.debug(f"Scanning {symbol} {timeframe}...")
        
        # Use pre-fetched data if available, otherwise fetch
        if pre_fetched_df is not None:
            df = pre_fetched_df
        else:
            df = fetcher.fetch_ohlcv(symbol, timeframe, limit=150)
        
        if len(df) < 50:
            logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(df)} candles")
            return None
        
        # PHASE 1: Data Quality Validation
        validation = DataValidator.validate_ohlcv(df, symbol, timeframe, strict=True)
        
        if not validation.is_valid:
            logger.warning(f"❌ Data validation failed for {symbol} {timeframe}: {validation.issues}")
            return None
        
        if validation.warnings:
            logger.debug(f"⚠️ Data warnings for {symbol} {timeframe}: {validation.warnings}")
        
        # Build channel
        channel = detect_channel(df)
        
        if not channel.is_valid:
            logger.debug(f"No valid channel for {symbol} {timeframe}")
            return None
        
        logger.debug(f"Valid {channel.channel_type.value} channel found for {symbol} {timeframe}")
        
        # Check for signal
        signal = check_signal(df, channel, symbol, timeframe, lookback_bars=2)
        
        if signal is None:
            return None
        
        # PHASE 2: Market Regime Detection
        from regime_detector import get_regime_detector
        
        regime_detector = get_regime_detector()
        regime = regime_detector.detect_regime(df, symbol)
        
        # DISABLED: Allow trading in all regimes for maximum signals
        # if regime and not regime.should_trade:
        #     logger.warning(f"⚠️ Regime unsuitable for trading {symbol}: {regime.trend}, {regime.volatility}, {regime.liquidity}")
        #     return None
        
        # PHASE 3.5: Market Bias Detection (DISABLED for maximum signals)
        # try:
        #     bias_detector = get_bias_detector()
        #     market_bias = bias_detector.detect_bias(df)
        #     
        #     # Reject shorts in strong bull markets
        #     if signal.direction.value == "short" and market_bias.bias == "bullish" and market_bias.strength > 0.6:
        #         logger.warning(f"🚫 SHORT rejected: Strong BULLISH bias ({market_bias.strength:.2f}) | Trend: {market_bias.price_trend:+.1f}%")
        #         return None
        #     
        #     # Reject longs in strong bear markets
        #     if signal.direction.value == "long" and market_bias.bias == "bearish" and market_bias.strength > 0.6:
        #         logger.warning(f"🚫 LONG rejected: Strong BEARISH bias ({market_bias.strength:.2f}) | Trend: {market_bias.price_trend:+.1f}%")
        #         return None
        #     
        #     logger.info(f"✅ Bias check passed: {market_bias.bias.upper()} market, {signal.direction.value.upper()} signal")
        # except Exception as e:
        #     logger.debug(f"Bias detection skipped: {e}")
        
        # Apply signal filters (Basic + Advanced)
        filter_result = apply_signal_filters(
            df, 
            signal.direction, 
            timeframe,
            higher_tf_channel,
            symbol=symbol,  # For funding rate logic
            channel=channel,  # For ADX and S/R
            funding_rate=funding_rate # Pass pre-fetched rate
        )
        
        # Check if signal passes filters
        if not filter_result.passed:
            logger.info(f"Signal filtered out for {symbol} {timeframe}: {filter_result.reasons}")
            return None
        
        # PHASE 2: Apply regime-based confidence adjustment
        base_confidence = filter_result.confidence_score
        
        if regime:
            adjusted_confidence = base_confidence + regime.confidence_adjustment
            logger.info(f"📊 Regime adjustment: {base_confidence} → {adjusted_confidence} ({regime.confidence_adjustment:+d}) | {regime.trend}")
            
            # Apply SL multiplier from regime
            if regime.sl_multiplier != 1.0:
                original_sl = signal.stop_loss
                if signal.direction.value == "long":
                    sl_distance = signal.entry_price - signal.stop_loss
                    signal.stop_loss = signal.entry_price - (sl_distance * regime.sl_multiplier)
                else:
                    sl_distance = signal.stop_loss - signal.entry_price
                    signal.stop_loss = signal.entry_price + (sl_distance * regime.sl_multiplier)
                
                logger.info(f"🛡️ SL adjusted for {regime.volatility}: {original_sl:.4f} → {signal.stop_loss:.4f} ({regime.sl_multiplier:.1f}x)")
        else:
            adjusted_confidence = base_confidence
        
        # PHASE 3: Signal Strength Classification
        try:
            classifier = get_signal_classifier()
            strength_result = classifier.classify_signal(signal, df, channel, higher_tf_channel)
            
            # Apply strength-based confidence boost
            adjusted_confidence += strength_result.confidence_boost
            signal_strength = strength_result.strength.name.lower()
            
            logger.info(f"💪 Signal Strength: {strength_result.strength.name} ({strength_result.confirmation_count} confirmations) | Boost: {strength_result.confidence_boost:+d}")
        except Exception as e:
            logger.warning(f"Signal strength classification failed: {e}")
            signal_strength = "moderate"
        
        # PHASE 4: ML Win Probability Prediction
        try:
            ml_scorer = get_ml_scorer()
            ml_prediction = ml_scorer.predict_win_probability(
                signal, channel, df, regime, 
                confirmation_count=strength_result.confirmation_count if 'strength_result' in locals() else 0
            )
            
            if ml_prediction:
                # Apply ML-based confidence boost
                adjusted_confidence += ml_prediction.confidence_boost
                
                logger.info(f"🤖 ML Prediction: {ml_prediction.win_probability:.1%} win probability | Boost: {ml_prediction.confidence_boost:+d}")
                
                # Store ML prediction with signal
                signal.ml_win_probability = ml_prediction.win_probability
        except Exception as e:
            logger.debug(f"ML prediction not available: {e}")
        
        # Check minimum confidence score (after all adjustments)
        if adjusted_confidence < MIN_CONFIDENCE_SCORE:
            logger.info(f"Signal confidence too low for {symbol} {timeframe}: {adjusted_confidence}")
            return None
        
        # Update signal with all adjustments
        signal.confidence_score = adjusted_confidence
        signal.filter_reasons = tuple(filter_result.reasons)
        
        # Check if we should send this signal (avoid duplicates)
        if not should_send_signal(symbol, timeframe, signal.direction.value):
            logger.info(f"Skipping duplicate signal for {symbol} {timeframe}")
            return None
        
        return (signal, df, channel)
        
    except Exception as e:
        logger.error(f"Error scanning {symbol} {timeframe}: {e}", exc_info=True)
        return None


def run_scan_cycle(fetcher: DataFetcher) -> int:
    """
    Run a complete scan cycle through all coins and timeframes.
    Uses multi-timeframe confluence by getting 1h channel first.
    
    Args:
        fetcher: DataFetcher instance
    
    Returns:
        Number of signals found
    """
    signals_found = 0
    
    try:
        # Get top coins
        logger.info(f"Fetching top {TOP_COINS_COUNT} coins by volume...")
        top_coins = fetcher.get_top_coins(TOP_COINS_COUNT)
        logger.info(f"Scanning {len(top_coins)} coins: {', '.join([c.split('/')[0] for c in top_coins[:5]])}...")
        
        # Scan each coin
        # Bulk Fetch Funding Rates 🚀
        logger.info("Fetching global funding rates...")
        funding_map = fetcher.fetch_all_funding_rates()
        logger.info(f"Fetched funding rates for {len(funding_map)} symbols")

        for symbol in top_coins:
            # Get funding rate for this symbol
            symbol_funding = funding_map.get(symbol.split('/')[0] + 'USDT', None)
            
            # MTF Confluence: Get 1h AND 4h channels for trend confirmation
            higher_tf_channel = None  # Will use the most relevant HTF channel
            channel_1h = None
            channel_4h = None
            
            # Fetch 1h channel
            try:
                df_1h = fetcher.fetch_ohlcv(symbol, '1h', limit=150)
                if len(df_1h) >= 50:
                    channel_1h = detect_channel(df_1h)
                    if not channel_1h.is_valid:
                        channel_1h = None
            except Exception as e:
                logger.debug(f"Could not get 1h channel for {symbol}: {e}")
            
            # Fetch 4h channel (stronger confluence)
            try:
                df_4h = fetcher.fetch_ohlcv(symbol, '4h', limit=150)
                if len(df_4h) >= 50:
                    channel_4h = detect_channel(df_4h)
                    if not channel_4h.is_valid:
                        channel_4h = None
                    else:
                        logger.debug(f"4h channel for {symbol}: {channel_4h.channel_type.value}")
            except Exception as e:
                logger.debug(f"Could not get 4h channel for {symbol}: {e}")
            
            # Use 4h channel if available (stronger), else use 1h
            higher_tf_channel = channel_4h if channel_4h else channel_1h
            
            # STRICTER MTF CONFLUENCE: Check if both 4h and 1h are aligned
            # Reject early if HTF trends conflict
            mtf_valid = True
            if channel_4h and channel_1h:
                # Both exist - check alignment
                if channel_4h.channel_type != channel_1h.channel_type:
                    # 4h and 1h trends conflict - skip this symbol
                    logger.debug(f"Skipping {symbol}: 4h ({channel_4h.channel_type.value}) conflicts with 1h ({channel_1h.channel_type.value})")
                    mtf_valid = False
            
            if not mtf_valid:
                continue  # Skip to next symbol
            
            # Scan each timeframe with MTF confluence
            # Scan each timeframe with MTF confluence
            symbol_signals = []
            
            for timeframe in TIMEFRAMES:
                # Use higher TF channel for confluence (except for 1h itself)
                mtf_channel = higher_tf_channel if timeframe != '1h' else None
                
                # OPTIMIZATION: If timeframe is 1h, use the already fetched df_1h
                current_df = df_1h if timeframe == '1h' else None
                
                result = scan_symbol(symbol, timeframe, fetcher, mtf_channel, pre_fetched_df=current_df, funding_rate=symbol_funding)
                if result is not None:
                    # ADDITIONAL CHECK: Verify signal aligns with HTF trend
                    signal, df, channel = result
                    if higher_tf_channel and higher_tf_channel.is_valid:
                        # Check if signal direction matches HTF trend
                        htf_type = higher_tf_channel.channel_type
                        signal_dir = signal.direction.value
                        
                        # Reject if signal conflicts with HTF trend
                        if (htf_type == ChannelType.ASCENDING and signal_dir == "short") or \
                           (htf_type == ChannelType.DESCENDING and signal_dir == "long"):
                            logger.debug(f"Rejecting {symbol} {signal_dir}: conflicts with HTF {htf_type.value}")
                            continue
                    
                    symbol_signals.append(result)
            
            # Conflict Resolution: Check if we have both LONG and SHORT
            directions = set(s[0].direction.value for s in symbol_signals)
            if len(directions) > 1:
                logger.info(f"⚠️ CONFLICT: Found {directions} for {symbol}. Resolving via Trend Hierarchy...")
                
                accepted_direction = None
                
                # 1. Check HTF Trend (1h Channel)
                if higher_tf_channel and higher_tf_channel.is_valid:
                    if higher_tf_channel.channel_type == ChannelType.ASCENDING:
                        accepted_direction = "long"
                        logger.info(f"   HTF Trend is UP 📈 -> Preferring LONG signals")
                    elif higher_tf_channel.channel_type == ChannelType.DESCENDING:
                        accepted_direction = "short"
                        logger.info(f"   HTF Trend is DOWN 📉 -> Preferring SHORT signals")
                
                # 2. Filter or Tie-Break
                if accepted_direction:
                    # Filter for trend direction
                    symbol_signals = [s for s in symbol_signals if s[0].direction.value == accepted_direction]
                    if not symbol_signals:
                        logger.warning("   No signals matched HTF trend. Skipping.")
                        continue
                else:
                    # Flat Trend or No Data -> Tie Break by Confidence Score
                    logger.info("   HTF Trend Neutral -> Resolving by Confidence Score")
                    # Sort desc by score
                    symbol_signals.sort(key=lambda x: x[0].confidence_score, reverse=True)
                    winner = symbol_signals[0]
                    # Keep only winner's direction
                    symbol_signals = [s for s in symbol_signals if s[0].direction.value == winner[0].direction.value]
                    logger.info(f"   Winner: {winner[0].direction.value.upper()} (Score: {winner[0].confidence_score})")
            
            # Process valid signals
            tracker = get_tracker()
            dca_manager = get_dca_manager()
            
            for signal, df, channel in symbol_signals:
                try:
                    logger.info(f"🚨 SIGNAL: {signal.direction.value.upper()} {symbol} {signal.timeframe} (Score: {signal.confidence_score:.0f})")
                    
<<<<<<< HEAD
                    # PHASE 1: Risk Management Validation
                    tracker = get_tracker()
                    risk_manager = get_risk_manager(initial_capital=INITIAL_CAPITAL)
                    
                    # Update capital with current P&L
                    stats = tracker.get_stats()
                    # Estimate current capital (simplified - will be more accurate with real trading)
                    total_pnl_percent = stats['avg_pnl'] * stats['total_trades'] if stats['total_trades'] > 0 else 0
                    current_capital = INITIAL_CAPITAL * (1 + total_pnl_percent / 100)
                    risk_manager.update_capital(current_capital)
                    
                    # Validate new position
                    is_valid, risk_metrics = risk_manager.validate_new_position(
                        signal,
                        tracker.active_positions,
                        fetcher
                    )
                    
                    if not is_valid:
                        logger.warning(f"🛡️ RISK REJECTED: {risk_metrics.rejection_reason}")
                        logger.info(f"   Portfolio Heat: {risk_metrics.portfolio_heat*100:.2f}% | "
                                  f"Positions: {risk_metrics.num_positions}/{risk_manager.max_positions} | "
                                  f"Drawdown: {risk_metrics.current_drawdown*100:.1f}%")
                        continue  # Skip this signal
                    
                    logger.info(f"✅ Risk Check Passed | Heat: {risk_metrics.portfolio_heat*100:.2f}% | "
                              f"Available: {risk_metrics.available_risk*100:.2f}%")
                    
                    # Generate chart and send alert
                    chart_path = generate_signal_chart(df, channel, signal, symbol, signal.timeframe)
                    
                    if send_telegram_alert(signal, chart_path):
                        logger.info(f"✅ Alert sent for {symbol} {signal.timeframe}")
                        signals_found += 1
                        
                        # Track position for TP/SL monitoring
                        tracker.add_position(signal)
=======
                    # CHECK FOR DCA OPPORTUNITY
                    dca_opportunity = None
                    for position in tracker.active_positions:
                        if position.symbol == symbol:
                            # Check if this signal is a DCA opportunity
                            dca_opp = dca_manager.detect_dca_opportunity(position, signal)
                            if dca_opp:
                                dca_opportunity = dca_opp
                                logger.info(f"🔄 DCA Opportunity detected for {symbol}!")
                                break
                    
                    # Generate chart
                    chart_path = generate_signal_chart(df, channel, signal, symbol, signal.timeframe)
                    
                    if dca_opportunity:
                        # Send DCA alert instead of regular signal
                        if send_dca_opportunity_alert(dca_opportunity, chart_path):
                            logger.info(f"✅ DCA Alert sent for {symbol}")
                            signals_found += 1
                            
                            # Add DCA entry to position
                            updated_pos = tracker.add_dca_entry(
                                symbol=symbol,
                                dca_price=dca_opportunity.dca_entry,
                                size=1.0
                            )
                            
                            if updated_pos:
                                # Send confirmation
                                send_dca_confirmation(updated_pos)
                        else:
                            logger.warning(f"⚠️ Failed to send DCA alert for {symbol}")
>>>>>>> old-version
                    else:
                        # Regular signal - send as normal
                        if send_telegram_alert(signal, chart_path):
                            logger.info(f"✅ Alert sent for {symbol} {signal.timeframe}")
                            signals_found += 1
                            
                            # Track position for TP/SL monitoring
                            tracker.add_position(signal)
                        else:
                            logger.warning(f"⚠️ Failed to send alert for {symbol}")
                except Exception as e:
                    logger.error(f"Error sending signal for {symbol}: {e}")
                
                # Small delay
                time.sleep(0.3)
        
        # Cleanup old signals
        cleanup_old_signals()
        
        # Check positions for TP/SL hits
        try:
            tracker = get_tracker()
            if tracker.active_positions:
                # Fetch current prices for tracked symbols
                current_prices = {}
                for pos in tracker.active_positions:
                    try:
                        df = fetcher.fetch_ohlcv(pos.symbol, '1m', limit=1)
                        if df is not None and len(df) > 0:
                            current_prices[pos.symbol] = df['close'].iloc[-1]
                    except:
                        pass
                
                # Check for closures
                closed = tracker.check_positions(current_prices)
                for position, outcome, exit_price in closed:
                    # Send outcome notification
                    outcome_msg = tracker.format_outcome_message(position)
                    send_telegram_message(outcome_msg)
                    logger.info(f"Position closed: {position.symbol} - {outcome.value}")
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
        
    except Exception as e:
        logger.error(f"Error during scan cycle: {e}", exc_info=True)
    
    return signals_found


def main():
    """Main bot entry point."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           🚀 Crypto Futures Signal Bot                        ║
║                                                               ║
║   Scanning for channel-based trading signals                  ║
║   Timeframes: 15m, 1h                                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("Starting Crypto Futures Signal Bot...")
    
    # Initialize Risk Manager
    risk_manager = get_risk_manager(initial_capital=INITIAL_CAPITAL)
    logger.info(f"🛡️ Risk Manager initialized:")
    logger.info(f"   Capital: ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"   Max Portfolio Risk: {risk_manager.max_portfolio_risk*100}%")
    logger.info(f"   Max Positions: {risk_manager.max_positions}")
    logger.info(f"   Max Correlated Exposure: {risk_manager.max_correlated_exposure*100}%")
    logger.info(f"   Drawdown Circuit Breaker: {risk_manager.max_drawdown*100}%")
    
    # Check Telegram configuration
    notifier = get_notifier()
    if notifier.is_configured():
        logger.info("✅ Telegram configured")
        send_startup_message()
        
        # Start Telegram command bot in separate process
        try:
            import multiprocessing
            from config import TELEGRAM_BOT_TOKEN
            
            def run_command_bot():
                """Run command bot in separate process."""
                try:
                    import asyncio
                    from telegram.ext import Application
                    from src.telegram_commands import setup_commands
                    
                    async def start_bot():
                        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
                        await setup_commands(application)
                        await application.initialize()
                        await application.start()
                        await application.updater.start_polling(allowed_updates=['message'])
                        # Keep running
                        await asyncio.Event().wait()
                    
                    asyncio.run(start_bot())
                except Exception as e:
                    print(f"Command bot error: {e}")
            
            # Start in separate process
            command_process = multiprocessing.Process(target=run_command_bot, daemon=True)
            command_process.start()
            logger.info("📱 Telegram command listener started")
            
        except ImportError:
            logger.info("ℹ️ Telegram commands not available (missing dependencies)")
        except Exception as e:
            logger.warning(f"Could not start command bot: {e}")
    else:
        logger.warning("⚠️ Telegram not configured - signals will only be logged")
        print("\n⚠️  WARNING: Telegram not configured!")
        print("    Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env file")
        print("    Signals will be generated but not sent to Telegram.\n")
    
    # Initialize data fetcher
    fetcher = DataFetcher()
    logger.info(f"✅ Exchange connected: {fetcher.exchange_id}")
    
    # Create charts directory
    CHARTS_DIR.mkdir(exist_ok=True)
    
    scan_count = 0
    total_signals = 0
    
    try:
        while True:
            scan_count += 1
            start_time = datetime.now()
            
            logger.info(f"\n{'='*50}")
            logger.info(f"📊 Starting Scan #{scan_count} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*50}")
            
            # Run scan cycle
            signals_found = run_scan_cycle(fetcher)
            total_signals += signals_found
            
            # Log summary
            scan_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"\n✅ Scan #{scan_count} complete in {scan_duration:.1f}s")
            logger.info(f"   Signals this scan: {signals_found}")
            logger.info(f"   Total signals: {total_signals}")
            
            # Wait for next scan
            logger.info(f"\n⏳ Next scan in {SCAN_INTERVAL_SECONDS // 60} minutes...")
            time.sleep(SCAN_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Bot stopped by user")
        send_shutdown_message()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        send_shutdown_message()
        raise


if __name__ == "__main__":
    main()
