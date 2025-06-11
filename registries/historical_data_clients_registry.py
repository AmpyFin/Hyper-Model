"""
Registry for API clients.
"""
from utils.api_clients.api_client import APIClient
from utils.api_clients.databento_client import DatabentoClient
from utils.api_clients.tiingo_client import TiingoClient
from utils.api_clients.yfinance_client import YFinanceClient

# Registry of available API client factories
registry = {
    "databento_client": lambda: DatabentoClient(),
    "tiingo_client": lambda: TiingoClient(),
    "yfinance_client": lambda: YFinanceClient()
}