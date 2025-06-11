from typing import Dict, Type
from utils.api_clients.api_client import APIClient
from utils.api_clients.tiingo_client import TiingoClient
from utils.api_clients.databento_client import DatabentoClient
from utils.api_clients.yfinance_client import YFinanceClient

# Registry of available latest price clients
registry: Dict[str, Type[APIClient]] = {
    "tiingo_client": TiingoClient,
    "databento_client": DatabentoClient,
    "yfinance_client": YFinanceClient,
} 