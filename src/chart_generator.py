"""
Chart Generator Module
=======================
Creates professional candlestick charts with channel overlays
and signal highlighting using mplfinance.
"""
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import CHARTS_DIR, CHART_BARS, CHART_STYLE
except ImportError:
    CHARTS_DIR = Path(__file__).parent.parent / "charts"
    CHART_BARS = 80
    CHART_STYLE = "nightclouds"

from channel_builder import Channel, ChannelType
from signal_generator import Signal, SignalDirection
from swing_detector import SwingPoint

logger = logging.getLogger(__name__)

# Ensure charts directory exists
CHARTS_DIR = Path(CHARTS_DIR)
CHARTS_DIR.mkdir(exist_ok=True)


def create_custom_style():
    """Create a custom dark theme style for charts."""
    # Base on nightclouds but customize
    custom_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mpf.make_marketcolors(
            up='#26a69a',      # Green for up candles
            down='#ef5350',    # Red for down candles
            edge='inherit',
            wick='inherit',
            volume='in',
        ),
        figcolor='#1e222d',
        facecolor='#1e222d',
        edgecolor='#363a45',
        gridcolor='#363a45',
        gridstyle='--',
        gridaxis='both',
        y_on_right=True,
        rc={
            'font.size': 10,
            'axes.labelcolor': '#b2b5be',
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'xtick.color': '#b2b5be',
            'ytick.color': '#b2b5be',
        }
    )
    return custom_style


def prepare_channel_lines(
    df: pd.DataFrame,
    channel: Channel,
    display_bars: int = CHART_BARS
) -> Tuple[List[float], List[float]]:
    """
    Prepare channel line data for plotting.
    
    Args:
        df: DataFrame with OHLCV data
        channel: Channel object with trendlines
        display_bars: Number of bars being displayed
    
    Returns:
        Tuple of (upper_line_prices, lower_line_prices)
    """
    # Calculate line prices for each bar
    start_idx = len(df) - display_bars
    
    # Offset Correction: Align plot index with channel logic index
    # Channel indices are 0 to channel.df_length-1
    # Plot indices are start_idx to len(df)-1
    # We must shift plot index by (len(df) - channel.df_length) to get channel index
    channel_offset = len(df) - channel.df_length
    
    upper_prices = []
    lower_prices = []
    
    for i in range(display_bars):
        plot_idx = start_idx + i
        channel_idx = plot_idx - channel_offset
        
        # Ensure index is within channel bounds (should be, but safety first)
        if 0 <= channel_idx < channel.df_length:
            upper_prices.append(channel.upper_line.get_price_at(channel_idx))
            lower_prices.append(channel.lower_line.get_price_at(channel_idx))
        else:
            upper_prices.append(np.nan)
            lower_prices.append(np.nan)
    
    return upper_prices, lower_prices


def prepare_swing_markers(
    df: pd.DataFrame,
    channel: Channel,
    display_bars: int = CHART_BARS
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Prepare swing point markers for plotting.
    
    Args:
        df: DataFrame used for channel detection
        channel: Channel object with trendlines
        display_bars: Number of bars being displayed
    
    Returns:
        Tuple of (swing_high_markers, swing_low_markers)
    """
    start_idx = len(df) - display_bars
    
    # Initialize with NaN (no marker)
    high_markers = [np.nan] * display_bars
    low_markers = [np.nan] * display_bars
    
    # Mark swing highs
    for sp in channel.upper_line.touch_points:
        display_idx = sp.index - start_idx
        if 0 <= display_idx < display_bars:
            high_markers[display_idx] = sp.price
    
    # Mark swing lows
    for sp in channel.lower_line.touch_points:
        display_idx = sp.index - start_idx
        if 0 <= display_idx < display_bars:
            low_markers[display_idx] = sp.price
    
    return high_markers, low_markers


def generate_signal_chart(
    df: pd.DataFrame,
    channel: Channel,
    signal: Optional[Signal] = None,
    symbol: str = "UNKNOWN",
    timeframe: str = "15m",
    display_bars: int = CHART_BARS,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a chart image with channel and signal visualization.
    
    Args:
        df: DataFrame with OHLCV data
        channel: Channel object with trendlines
        signal: Optional Signal object to highlight
        symbol: Trading symbol for title
        timeframe: Timeframe for title
        display_bars: Number of candles to display
        save_path: Optional custom save path
    
    Returns:
        Path to saved chart image
    """
    # Limit to display bars
    if len(df) > display_bars:
        plot_df = df.iloc[-display_bars:].copy()
    else:
        plot_df = df.copy()
        display_bars = len(df)
    
    # Prepare channel lines
    upper_prices, lower_prices = prepare_channel_lines(df, channel, display_bars)
    
    # Prepare swing point markers
    high_markers, low_markers = prepare_swing_markers(df, channel, display_bars)
    
    # Create additional plots for channel lines
    additional_plots = []
    
    # Upper channel line (red)
    upper_line_plot = mpf.make_addplot(
        upper_prices,
        type='line',
        color='#ff5252',
        width=2,
        linestyle='--',
        alpha=0.8,
        panel=0
    )
    additional_plots.append(upper_line_plot)
    
    # Lower channel line (green)
    lower_line_plot = mpf.make_addplot(
        lower_prices,
        type='line',
        color='#4caf50',
        width=2,
        linestyle='--',
        alpha=0.8,
        panel=0
    )
    additional_plots.append(lower_line_plot)
    
    # Swing high markers (red dots)
    high_marker_plot = mpf.make_addplot(
        high_markers,
        type='scatter',
        markersize=80,
        marker='v',
        color='#ff5252',
        alpha=0.9,
        panel=0
    )
    additional_plots.append(high_marker_plot)
    
    # Swing low markers (green dots)
    low_marker_plot = mpf.make_addplot(
        low_markers,
        type='scatter',
        markersize=80,
        marker='^',
        color='#4caf50',
        alpha=0.9,
        panel=0
    )
    additional_plots.append(low_marker_plot)
    
    # Highlight signal candle if present
    if signal:
        signal_markers = [np.nan] * display_bars
        signal_idx = display_bars - 1  # Default to last candle
        
        # Find the signal candle by timestamp
        for i, ts in enumerate(plot_df.index):
            if hasattr(signal.timestamp, 'tz_localize'):
                signal_ts = signal.timestamp
            else:
                signal_ts = pd.Timestamp(signal.timestamp)
            
            if ts == signal_ts or (hasattr(ts, 'to_pydatetime') and 
                ts.to_pydatetime().replace(tzinfo=None) == signal_ts.replace(tzinfo=None) if hasattr(signal_ts, 'replace') else False):
                signal_idx = i
                break
        
        # Mark the signal candle
        signal_price = plot_df.iloc[signal_idx]['high'] if signal.direction.value == "short" else plot_df.iloc[signal_idx]['low']
        signal_markers[signal_idx] = signal_price
        
        signal_color = '#ffeb3b' if signal.direction.value == "long" else '#ff9800'
        signal_marker = 'D'  # Diamond marker (matplotlib standard)
        
        signal_marker_plot = mpf.make_addplot(
            signal_markers,
            type='scatter',
            markersize=200,
            marker=signal_marker,
            color=signal_color,
            alpha=1.0,
            panel=0
        )
        additional_plots.append(signal_marker_plot)
    
    # Create custom style
    custom_style = create_custom_style()
    
    # Generate title
    channel_emoji = {
        ChannelType.ASCENDING: "📈",
        ChannelType.DESCENDING: "📉", 
        ChannelType.HORIZONTAL: "➡️",
        ChannelType.CONVERGING: "🔺",
        ChannelType.DIVERGING: "🔻",
        ChannelType.INVALID: "❓"
    }
    
    title_parts = [f"{symbol} | {timeframe}"]
    title_parts.append(f"{channel_emoji.get(channel.channel_type, '')} {channel.channel_type.value.capitalize()} Channel")
    
    if signal:
        signal_emoji = "🟢 LONG" if signal.direction.value == "long" else "🔴 SHORT"
        title_parts.append(f"| {signal_emoji}")
    
    title = " ".join(title_parts)
    
    # Generate filename
    if save_path:
        output_path = Path(save_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_clean = symbol.replace("/", "_").replace(":", "_")
        signal_type = ""
        if signal:
            signal_type = f"_{signal.direction.value}"
        filename = f"{symbol_clean}_{timeframe}{signal_type}_{timestamp}.png"
        output_path = CHARTS_DIR / filename
    
    # Calculate Y-axis limits to ensure TP/SL are visible
    y_min = plot_df['low'].min()
    y_max = plot_df['high'].max()
    
    if signal:
        # Extend range to include TP/SL
        targets = [signal.take_profit, signal.stop_loss, signal.entry_price]
        y_min = min(y_min, min(targets))
        y_max = max(y_max, max(targets))
    
    # Add padding (5%)
    y_range = y_max - y_min
    if y_range == 0: y_range = y_max * 0.1
    y_min -= y_range * 0.05
    y_max += y_range * 0.05
    
    # Create the chart
    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style=custom_style,
        title=title,
        volume=False,
        addplot=additional_plots,
        figsize=(14, 8),
        returnfig=True,
        tight_layout=True,
        scale_padding={'left': 0.5, 'right': 1.0, 'top': 0.6, 'bottom': 0.5},
        ylim=(y_min, y_max)
    )
    
    # Add text annotations with TP/SL levels if signal present
    if signal:
        ax = axes[0]
        
        # Add TP line
        ax.axhline(y=signal.take_profit, color='#4caf50', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.text(0.02, signal.take_profit, f'TP: ${signal.take_profit:,.2f}', 
                transform=ax.get_yaxis_transform(), fontsize=9, color='#4caf50',
                verticalalignment='center', fontweight='bold')
        
        # Add SL line
        ax.axhline(y=signal.stop_loss, color='#ef5350', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.text(0.02, signal.stop_loss, f'SL: ${signal.stop_loss:,.2f}',
                transform=ax.get_yaxis_transform(), fontsize=9, color='#ef5350',
                verticalalignment='center', fontweight='bold')
        
        # Add Entry line
        ax.axhline(y=signal.entry_price, color='#ffeb3b', linestyle='-', alpha=0.5, linewidth=1)
        ax.text(0.02, signal.entry_price, f'Entry: ${signal.entry_price:,.2f}',
                transform=ax.get_yaxis_transform(), fontsize=9, color='#ffeb3b',
                verticalalignment='center', fontweight='bold')
    
    # Save the chart
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e222d')
    plt.close(fig)
    
    logger.info(f"Chart saved to: {output_path}")
    return str(output_path)


def generate_analysis_chart(
    df: pd.DataFrame,
    channel: Channel,
    symbol: str = "UNKNOWN",
    timeframe: str = "15m",
    display_bars: int = CHART_BARS
) -> str:
    """
    Generate an analysis chart without a specific signal.
    Useful for debugging and visualization.
    
    Args:
        df: DataFrame with OHLCV data
        channel: Channel object
        symbol: Trading symbol
        timeframe: Timeframe
        display_bars: Number of candles to display
    
    Returns:
        Path to saved chart image
    """
    return generate_signal_chart(df, channel, None, symbol, timeframe, display_bars)


# Test the module when run directly
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_fetcher import fetch_ohlcv
    from channel_builder import detect_channel
    from signal_generator import check_signal, Signal, SignalDirection
    
    print("=" * 60)
    print("Chart Generator Module Test")
    print("=" * 60)
    
    # Test with BTC
    symbol = 'BTC/USDT:USDT'
    timeframe = '1h'
    
    print(f"\n📈 Fetching {symbol} {timeframe} data...")
    df = fetch_ohlcv(symbol, timeframe, 150)
    print(f"✅ Fetched {len(df)} candles")
    
    # Build channel
    print("\n🔧 Building channel...")
    channel = detect_channel(df)
    
    if channel.is_valid:
        print(f"✅ Valid {channel.channel_type.value} channel detected")
        
        # Generate analysis chart (no signal)
        print("\n📊 Generating analysis chart...")
        chart_path = generate_analysis_chart(df, channel, symbol, timeframe)
        print(f"✅ Chart saved to: {chart_path}")
        
        # Create a mock signal for testing visualization
        print("\n🎯 Creating mock signal for visualization test...")
        from signal_generator import Signal, SignalDirection
        mock_signal = Signal(
            direction=SignalDirection.SHORT,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=df.index[-1].to_pydatetime(),
            entry_price=df['close'].iloc[-1],
            take_profit=channel.get_current_lower(),
            stop_loss=channel.get_current_upper() + 200,
            channel_type=channel.channel_type,
            upper_line_price=channel.get_current_upper(),
            lower_line_price=channel.get_current_lower(),
            touch_type='wick_rejection',
            rr_ratio=2.5,
            channel_width_pct=2.5
        )
        
        print("\n📊 Generating signal chart...")
        signal_chart_path = generate_signal_chart(df, channel, mock_signal, symbol, timeframe)
        print(f"✅ Signal chart saved to: {signal_chart_path}")
    else:
        print("❌ No valid channel detected")
    
    print("\n" + "=" * 60)
    print("✅ Chart generator test complete!")
    print("=" * 60)
