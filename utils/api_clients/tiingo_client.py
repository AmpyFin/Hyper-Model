import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from tiingo import TiingoClient as BaseTiingoClient
from .api_client import APIClient

logger = logging.getLogger(__name__)

class TiingoClient(APIClient):
    """
    Tiingo API client implementation.
    """
    
    def __init__(self):
        self._client = None
    
    def load_key_from_config(self) -> str | None:
        """Load API key from config file."""
        try:
            from config import TIINGO_API_KEY
            logger.info("Loaded API key from config")
            return TIINGO_API_KEY
        except ImportError:
            logger.warning("Could not load API key from config")
            return None
    
    def load_key_from_env(self) -> str | None:
        """Load API key from environment variables."""
        key = os.getenv("TIINGO_API_KEY")
        if key:
            logger.info("Loaded API key from environment")
        else:
            logger.warning("Could not load API key from environment")
        return key
    
    def get_key(self) -> str | None:
        """Get API key first from env, then from config."""
        key = self.load_key_from_env()
        if not key:
            key = self.load_key_from_config()
        if not key:
            raise ValueError("Tiingo API key not found in environment variables or config file")
        return key
    
    def get_client(self) -> BaseTiingoClient:
        """Get a reusable Tiingo client instance."""
        if self._client is None:
            config = {
                'api_key': self.get_key(),
                'session': True  # Reuse the same session for better performance
            }
            self._client = BaseTiingoClient(config)
        return self._client
    
    def get_price(self, ticker: str, client: BaseTiingoClient = None) -> float:
        """
        Get the current price for a given ticker symbol.
        
        Args:
            ticker (str): The ticker symbol of the asset
            client (BaseTiingoClient, optional): A reusable client instance. If None, one will be created.
            
        Returns:
            float: The current price of the asset
        """
        if client is None:
            client = self.get_client()
        
        # Get latest price data
        try:
            # Get the latest daily data
            today = datetime.now()
            ticker_price = client.get_ticker_price(
                ticker,
                startDate=(today - timedelta(days=1)).strftime("%Y-%m-%d"),
                endDate=today.strftime("%Y-%m-%d")
            )
            
            if not ticker_price:
                raise ValueError(f"No price data found for ticker {ticker}")
            
            # Return the latest closing price
            return float(ticker_price[-1]['close'])
        except Exception as e:
            raise ValueError(f"Error fetching price for ticker {ticker}: {str(e)}")
    
    def _convert_frequency(self, frequency: str) -> str:
        """
        Convert our frequency format to Tiingo's format.
        Our format: {number}{unit} where unit is:
        - m: minutes (e.g., 1m, 5m, 15m)
        - h: hours (e.g., 1h, 2h)
        - d: days (e.g., 1d)
        - w: weeks (e.g., 1w)
        - M: months (e.g., 1M)
        - y: years (e.g., 1y)
        
        Args:
            frequency (str): Frequency in our format
            
        Returns:
            str: Frequency in Tiingo's format
            
        Raises:
            ValueError: If frequency format is invalid
        """
        if not frequency:
            return "1min"  # Default to 1-minute data
            
        # Parse the frequency string
        import re
        match = re.match(r"(\d+)([mhdwMy])", frequency)
        if not match:
            raise ValueError(
                f"Invalid frequency format: {frequency}. "
                "Expected format: {number}{unit} where unit is m (minutes), "
                "h (hours), d (days), w (weeks), M (months), or y (years)"
            )
            
        number, unit = match.groups()
        number = int(number)
        
        # Convert to Tiingo format
        if unit == 'm':
            if number not in [1, 5, 15, 30]:
                raise ValueError("Minute frequency must be 1, 5, 15, or 30")
            return f"{number}min"
        elif unit == 'h':
            if number not in [1, 2, 3, 4, 6, 8, 12]:
                raise ValueError("Hour frequency must be 1, 2, 3, 4, 6, 8, or 12")
            return f"{number}hour"
        elif unit == 'd':
            if number != 1:
                raise ValueError("Day frequency must be 1")
            return "daily"
        elif unit == 'w':
            if number != 1:
                raise ValueError("Week frequency must be 1")
            return "weekly"
        elif unit == 'M':
            if number != 1:
                raise ValueError("Month frequency must be 1")
            return "monthly"
        elif unit == 'y':
            if number != 1:
                raise ValueError("Year frequency must be 1")
            return "annually"
        else:
            raise ValueError(f"Unsupported frequency unit: {unit}")

    def get_historical_data(
        self,
        ticker: str,
        frequency: str = "1d",
        start_date: datetime = None,
        end_date: Optional[datetime] = None,
        client: BaseTiingoClient = None
    ) -> dict:
        """
        Get historical price data for a given ticker symbol.
        
        Args:
            ticker (str): The ticker symbol of the asset
            frequency (str): The frequency of the data. Supported formats:
                - {number}m: minutes (e.g., 1m, 5m, 15m, 30m)
                - {number}h: hours (e.g., 1h, 2h, 3h, 4h, 6h, 8h, 12h)
                - 1d: daily
                - 1w: weekly
                - 1M: monthly
                - 1y: yearly
            start_date (datetime): The start date for historical data
            end_date (datetime, optional): The end date for historical data
            client (BaseTiingoClient, optional): A reusable client instance. If None, one will be created.
            
        Returns:
            dict: Historical price data containing OHLCV data
            
        Raises:
            ValueError: If frequency format is invalid or unsupported
        """
        if client is None:
            client = self.get_client()
        
        try:
            # Convert frequency to Tiingo format
            tiingo_frequency = self._convert_frequency(frequency)
            
            # Get historical data using get_ticker_price
            historical_prices = client.get_ticker_price(
                ticker,
                startDate=start_date.strftime("%Y-%m-%d") if start_date else "2020-01-01",
                endDate=end_date.strftime("%Y-%m-%d") if end_date else datetime.now().strftime("%Y-%m-%d"),
                frequency=tiingo_frequency,
                fmt='json',  # Ensure we get JSON format for consistent handling
                columns="open,high,low,close,volume"  # Explicitly request all OHLCV data
            )
            
            if not historical_prices:
                raise ValueError(f"No historical data found for ticker {ticker}")
            
            # Convert historical prices to the expected format
            prices = []
            for price_data in historical_prices:
                date = price_data.get('date')
                price_dict = {
                    'date': date,
                    'open': float(price_data.get('open') or price_data.get('openPrice') or 0),
                    'high': float(price_data.get('high') or price_data.get('highPrice') or 0),
                    'low': float(price_data.get('low') or price_data.get('lowPrice') or 0),
                    'close': float(price_data.get('close') or price_data.get('closePrice') or 0),
                    'volume': float(price_data.get('volume') or price_data.get('tradeVolume') or 0)
                }
                prices.append(price_dict)
            
            # Calculate summary statistics
            if prices:
                summary = {
                    'open': sum(p['open'] for p in prices) / len(prices),
                    'high': max(p['high'] for p in prices),
                    'low': min(p['low'] for p in prices),
                    'close': sum(p['close'] for p in prices) / len(prices),
                    'volume': sum(p['volume'] for p in prices) / len(prices)
                }
            else:
                summary = {
                    'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0
                }
            
            result = {
                'prices': prices,
                'summary': summary
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error fetching historical data for ticker {ticker}: {str(e)}")
