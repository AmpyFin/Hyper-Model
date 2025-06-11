import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Any

import databento as db
import pandas as pd
from .api_client import APIClient

logger = logging.getLogger(__name__)

class DatabentoClient(APIClient):
    """
    Databento API client implementation for US Equities dataset.
    Standard plan with:
    - 1 year of L1 history
    - 1 month of L2 and L3 history
    - Live data access requires additional license
    """
    
    def __init__(self):
        self._historical_client = None
        self._live_client = None
    
    def load_key_from_config(self) -> str | None:
        """Load API key from config file."""
        try:
            from config import DATABENTO_API_KEY
            logger.info("Loaded API key from config")
            return DATABENTO_API_KEY
        except ImportError:
            logger.warning("Could not load API key from config")
            return None
    
    def load_key_from_env(self) -> str | None:
        """Load API key from environment variables."""
        key = os.getenv("DATABENTO_API_KEY")
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
            raise ValueError("Databento API key not found in environment variables or config file")
        return key
    
    def get_client(self) -> db.Historical:
        """
        Get a reusable Databento client instance.
        This is the abstract method implementation required by APIClient.
        For specific client types, use get_historical_client() or get_live_client().
        """
        return self.get_historical_client()
    
    def get_historical_client(self) -> db.Historical:
        """Get a reusable Databento Historical client instance."""
        if self._historical_client is None:
            self._historical_client = db.Historical(self.get_key())
        return self._historical_client
    
    def get_live_client(self) -> db.Live:
        """Get a reusable Databento Live client instance."""
        if self._live_client is None:
            self._live_client = db.Live(self.get_key())
        return self._live_client
    
    def _get_symbol(self, ticker: str) -> str:
        """
        Convert standard ticker symbol to Databento format.
        For stocks, we'll use XNAS (Nasdaq) for common tickers.
        """
        return ticker  # For XNAS.BASIC dataset, we use plain ticker symbols
    
    def get_price(self, ticker: str, client: db.Historical = None) -> float:
        """
        Get the current price for a given ticker symbol.
        Uses historical data from the most recent trading day.
        
        Args:
            ticker (str): The trading symbol to get data for
            client (db.Historical, optional): A reusable client instance. If None, one will be created.
            
        Returns:
            float: The current price of the asset
        """
        raise NotImplementedError("Databento does not support getting current price data")
    
    def get_historical_data(
        self,
        ticker: str,
        frequency: str = "1d",
        start_date: datetime = None,
        end_date: Optional[datetime] = None,
        client: db.Historical = None
    ) -> dict:
        """
        Get historical price data for a given ticker symbol.
        Note: Databento only supports historical data requests with end date at least 24 hours in the past.
        
        Args:
            ticker (str): The trading symbol to get data for
            frequency (str, optional): The frequency of the data. Defaults to "1d" (daily)
            start_date (datetime): The start date for historical data
            end_date (datetime, optional): The end date for historical data
            client (db.Historical, optional): A reusable client instance. If None, one will be created.
            
        Returns:
            dict: Historical price data containing OHLCV data
            
        Raises:
            ValueError: If end_date is within 24 hours of current time
        """
        if client is None:
            client = self.get_historical_client()
            
        symbol = self._get_symbol(ticker)
        now = datetime.now()
        
            
        # Convert to naive datetime if needed
        if start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
            
        # Check if end_date is within 24 hours
        if (now - end_date) < timedelta(hours=24):
            raise ValueError(
                "Databento only supports historical data requests with end date at least 24 hours in the past. "
                "For current price data, use get_price() method instead."
            )
            
        # Format dates for API call
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Validate start_date is before end_date
        if start_date >= end_date:
            raise ValueError(
                f"Invalid date range: start_date ({start_date}) must be before end_date ({end_date})"
            )

        try:
            # Get historical data
            historical_data = client.timeseries.get_range(
                dataset="XNAS.BASIC",
                symbols=symbol,
                schema="trades",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d")
            )
            df = historical_data.to_df() if historical_data else pd.DataFrame()
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Set timestamp as index (ts_event = event timestamp)
            df.set_index('ts_event', inplace=True)
            df.index = pd.to_datetime(df.index, unit='ns')
            
            # Resample based on frequency
            freq_map = {
                "1d": "D",
                "1h": "h",
                "1m": "min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
            }
            pd_freq = freq_map.get(frequency, "D")
            
            # Group by time period and calculate OHLCV
            grouped = df.resample(pd_freq).agg({
                'price': ['first', 'max', 'min', 'last'],
                'size': 'sum'
            })
            
            # Flatten column names
            grouped.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Convert to list of dictionaries with dates
            prices = []
            for timestamp, row in grouped.iterrows():
                prices.append({
                    'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                })
            
            # Format the response with a 'prices' key
            result = {
                'prices': prices,
                'summary': {
                    'symbol': ticker,
                    'frequency': frequency,
                    'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error fetching data for ticker {ticker}: {str(e)}") 