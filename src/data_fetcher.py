"""
Data Fetcher Module
====================
Handles all exchange data fetching using CCXT library.
Supports Binance Futures and Bybit Futures.
"""
import ccxt
import pandas as pd
import logging
from typing import List, Optional
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import EXCHANGE_ID, TOP_COINS_COUNT
except ImportError:
    # Default values if config not available
    EXCHANGE_ID = 'binance'
    TOP_COINS_COUNT = 50

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Exchange data fetcher for crypto futures markets.
    Supports Binance USDT-M Futures and Bybit Linear Futures.
    """
    
    def __init__(self, exchange_id: str = EXCHANGE_ID):
        """
        Initialize the exchange client in futures mode.
        
        Args:
            exchange_id: 'binance' or 'bybit'
        """
        self.exchange_id = exchange_id.lower()
        self.exchange = self._init_exchange()
        logger.info(f"Initialized {self.exchange_id} futures exchange")
    
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize the exchange with futures/linear settings."""
        
        if self.exchange_id == 'binance':
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use USDT-M Futures
                    'adjustForTimeDifference': True,
                }
            })
        elif self.exchange_id == 'bybit':
            exchange = ccxt.bybit({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'linear',  # Use USDT Linear Futures
                }
            })
        else:
            raise ValueError(f"Unsupported exchange: {self.exchange_id}. Use 'binance' or 'bybit'")
        
        return exchange
    
    def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '15m', 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT' for futures)
            timeframe: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max 1000 for Binance)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Fetch raw OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching {symbol}: {e}")
            raise
    
    def get_top_coins(self, n: int = TOP_COINS_COUNT) -> List[str]:
        """
        Get top N USDT futures pairs by 24h trading volume.
        
        Args:
            n: Number of top coins to return (default from config)
        
        Returns:
            List of symbol strings (e.g., ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...])
        """
        try:
            # Load markets if not already loaded
            if not self.exchange.markets:
                self.exchange.load_markets()
            
            # Fetch 24h tickers for all symbols
            tickers = self.exchange.fetch_tickers()
            
            # Filter for USDT futures pairs and extract volume data
            volume_data = []
            
            for symbol, ticker in tickers.items():
                # Filter criteria:
                # 1. Must be USDT-margined futures (symbol ends with :USDT)
                # 2. Must have quote volume data
                # 3. Exclude stablecoins and leveraged tokens
                
                if not symbol.endswith(':USDT'):
                    continue
                    
                base = symbol.split('/')[0]
                
                # Skip stablecoins and leveraged tokens
                skip_tokens = ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'UP', 'DOWN', 'BULL', 'BEAR']
                if any(skip in base for skip in skip_tokens):
                    continue
                
                # Get quote volume (volume in USDT terms)
                quote_volume = ticker.get('quoteVolume', 0) or 0
                
                if quote_volume > 0:
                    volume_data.append({
                        'symbol': symbol,
                        'quoteVolume': quote_volume
                    })
            
            # Sort by volume descending and take top N
            volume_data.sort(key=lambda x: x['quoteVolume'], reverse=True)
            top_symbols = [item['symbol'] for item in volume_data[:n]]
            
            logger.info(f"Found top {len(top_symbols)} coins by volume")
            logger.debug(f"Top coins: {top_symbols}")
            
            return top_symbols
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching tickers: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching tickers: {e}")
            raise
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Get market information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Market info dict or None if not found
        """
        try:
            if not self.exchange.markets:
                self.exchange.load_markets()
            return self.exchange.markets.get(symbol)
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def fetch_all_funding_rates(self) -> dict:
        """
        Fetch funding rates for ALL symbols in one request.
        
        Returns:
            Dict mapping symbol -> funding_rate (float)
        """
        try:
            # CCXT fetch_funding_rates returns dict of funding info for all symbols
            rates = self.exchange.fetch_funding_rates()
            
            # Map symbol -> Rate
            funding_map = {}
            for symbol, data in rates.items():
                funding_map[symbol] = data.get('fundingRate', 0.0)
                
            return funding_map
            
        except Exception as e:
            logger.error(f"Error fetching global funding rates: {e}")
            return {}


# Standalone functions for convenience
_default_fetcher: Optional[DataFetcher] = None

def get_fetcher(exchange_id: str = EXCHANGE_ID) -> DataFetcher:
    """Get or create the default DataFetcher instance."""
    global _default_fetcher
    if _default_fetcher is None or _default_fetcher.exchange_id != exchange_id:
        _default_fetcher = DataFetcher(exchange_id)
    return _default_fetcher


def fetch_ohlcv(symbol: str, timeframe: str = '15m', limit: int = 100) -> pd.DataFrame:
    """Convenience function to fetch OHLCV data."""
    return get_fetcher().fetch_ohlcv(symbol, timeframe, limit)


def get_top_coins(n: int = TOP_COINS_COUNT) -> List[str]:
    """Convenience function to get top coins by volume."""
    return get_fetcher().get_top_coins(n)


# Test the module when run directly
if __name__ == "__main__":
    import sys
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Data Fetcher Module Test")
    print("=" * 60)
    
    # Initialize fetcher
    fetcher = DataFetcher('binance')
    
    # Test 1: Get top coins
    print("\n📊 Test 1: Fetching Top 10 Coins by Volume...")
    try:
        top_coins = fetcher.get_top_coins(10)
        print(f"✅ Top 10 coins: {top_coins}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Test 2: Fetch OHLCV for BTC
    print("\n📈 Test 2: Fetching BTC/USDT 15m OHLCV...")
    try:
        btc_symbol = 'BTC/USDT:USDT'
        df = fetcher.fetch_ohlcv(btc_symbol, '15m', 50)
        print(f"✅ Fetched {len(df)} candles")
        print(f"\nLatest 5 candles:")
        print(df.tail())
        print(f"\nDataFrame dtypes:")
        print(df.dtypes)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Test 3: Fetch OHLCV for 1h timeframe
    print("\n📈 Test 3: Fetching ETH/USDT 1h OHLCV...")
    try:
        eth_symbol = 'ETH/USDT:USDT'
        df_1h = fetcher.fetch_ohlcv(eth_symbol, '1h', 50)
        print(f"✅ Fetched {len(df_1h)} candles")
        print(f"\nLatest 3 candles:")
        print(df_1h.tail(3))
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Data fetcher is working correctly.")
    print("=" * 60)
