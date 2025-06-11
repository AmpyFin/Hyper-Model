import os
from datetime import datetime
from typing import Optional

import yfinance as yf
from .api_client import APIClient

class YFinanceClient(APIClient):
    """
    Yahoo Finance API client implementation using yfinance.
    Note: YFinance doesn't require an API key, but we'll keep the key-related
    methods to maintain consistency with the abstract base class.
    This client only supports daily frequency.
    """
    
    def __init__(self):
        self._tickers = {}  # Cache for Ticker objects
    
    def load_key_from_config(self) -> str | None:
        """
        YFinance doesn't require an API key, but we implement this
        to maintain consistency with the abstract base class.
        """
        return None
    
    def load_key_from_env(self) -> str | None:
        """
        YFinance doesn't require an API key, but we implement this
        to maintain consistency with the abstract base class.
        """
        return None
    
    def get_key(self) -> str | None:
        """
        YFinance doesn't require an API key, but we implement this
        to maintain consistency with the abstract base class.
        """
        return None
    
    def get_client(self):
        """
        Get a reusable YFinance Ticker instance.
        YFinance uses per-ticker clients, so we cache them by ticker symbol.
        
        Args:
            ticker (str, optional): The ticker symbol to get a client for
            
        Returns:
            yf.Ticker: The Ticker instance for the given symbol
        """
        return None
    
    def get_price(self, ticker: str, client: yf.Ticker = None) -> float:
        """
        Get the current price for a given ticker symbol.
        
        Args:
            ticker (str): The ticker symbol of the asset
            client (yf.Ticker, optional): A reusable client instance. If None, one will be created.
            
        Returns:
            float: The current price of the asset
        """
        try:
            client = yf.Ticker(ticker)
            
            # Get the current price (last closing price)
            current_price = client.history(period='1d')['Close'].iloc[-1]
            
            if not current_price or current_price == 0:
                raise ValueError(f"Invalid price data for ticker {ticker}")
                
            return float(current_price)
            
        except Exception as e:
            raise ValueError(f"Error fetching price for ticker {ticker}: {str(e)}")
    
    def get_historical_data(
        self,
        ticker: str,
        frequency: str = "1d",
        start_date: datetime = None,
        end_date: Optional[datetime] = None,
        client: yf.Ticker = None
    ) -> dict:
        """
        Get historical price data for a given ticker symbol.
        Only supports daily frequency.
        
        Args:
            ticker (str): The ticker symbol of the asset
            frequency (str, optional): The frequency of the data. Must be "1d"
            start_date (datetime): The start date for historical data
            end_date (datetime, optional): The end date for historical data
            client (yf.Ticker, optional): A reusable client instance. If None, one will be created.
            
        Returns:
            dict: Historical price data containing OHLCV data
            
        Raises:
            ValueError: If frequency is not "1d"
        """
        if frequency != "1d":
            raise ValueError("YFinance client only supports daily frequency ('1d')")
            
        try:
            # Get or create Ticker object
            client = yf.Ticker(ticker)
            
            # Get historical data
            df = client.history(
                interval="1d",
                start=start_date.strftime("%Y-%m-%d") if start_date else "2020-01-01",
                end=end_date.strftime("%Y-%m-%d") if end_date else None
            )
            
            if df.empty:
                raise ValueError(f"No historical data found for ticker {ticker}")
            
            # Convert DataFrame to list of dictionaries with the expected format
            prices = []
            for timestamp, row in df.iterrows():
                prices.append({
                    'date': timestamp.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            # Format the response with a 'prices' key
            result = {
                'prices': prices,
                'summary': {
                    'symbol': ticker,
                    'frequency': frequency,
                    'start_date': start_date.strftime('%Y-%m-%d') if start_date else "2020-01-01",
                    'end_date': end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
                }
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error fetching historical data for ticker {ticker}: {str(e)}")
