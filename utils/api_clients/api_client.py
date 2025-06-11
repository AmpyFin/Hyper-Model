from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any

class APIClient(ABC):
    """
    Abstract base class defining the interface for API clients.
    All concrete API client implementations should inherit from this class.
    """
    
    @abstractmethod
    def load_key_from_config(self) -> str | None:
        """Load API key from config file."""
        pass
    
    @abstractmethod
    def load_key_from_env(self) -> str | None:
        """Load API key from environment variables."""
        pass

    @abstractmethod
    def get_key(self) -> str | None:
        """Get API key."""
        pass
    
    @abstractmethod
    def get_client(self) -> Any:
        """
        Get a reusable client instance for making API calls.
        
        Returns:
            Any: The client instance specific to the API implementation
        """
        pass
    
    @abstractmethod
    def get_price(self, ticker: str, client: Any = None) -> float:
        """
        Get the current price for a given ticker symbol.
        
        Args:
            ticker (str): The ticker symbol of the asset
            client (Any, optional): A reusable client instance. If None, one will be created.
            
        Returns:
            float: The current price of the asset
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        ticker: str,
        frequency: str = "1d",
        start_date: datetime = None,
        end_date: Optional[datetime] = None,
        client: Any = None
    ) -> dict:
        """
        Get historical price data for a given ticker symbol.
        
        Args:
            ticker (str): The ticker symbol of the asset
            frequency (str, optional): The frequency of the data. Defaults to "1d" (daily)
            start_date (datetime): The start date for historical data
            end_date (datetime, optional): The end date for historical data. Defaults to current time
            client (Any, optional): A reusable client instance. If None, one will be created.
            
        Returns:
            dict: Historical price data containing at minimum OHLCV data
        """
        pass