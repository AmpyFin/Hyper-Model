"""
utils/clients/full_tiingo_client.py
Comprehensive Tiingo API client with retry logic, DataFrame helpers, and flexible
alias exports.  Drop this file into `utils/clients/` (alongside `__init__.py`).

Key features
------------
* Reads the API key from **config.py** (preferred) or the `TIINGO_API_KEY` env var.
* Exponential-back-off on HTTP ≥500 errors to respect Tiingo rate-limits.
* High-level `get_dataframe()` helper that returns tidy OHLCV DataFrames.
* Low-level `get_json()` helper for arbitrary endpoints (fundamentals, news, etc.).
* Implemented as a **singleton** so any import in your project shares the
  same underlying session & quota.
* Exposes convenient aliases:
    • `TiingoClientSingleton`   – canonical class name
    • `tiingoClient`            – generic alias
"""

from __future__ import annotations

import importlib
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration discovery
# -----------------------------------------------------------------------------
_CONFIG_MODULE_NAME = "config"            # root-level config.py
_TIINGO_ENV_VAR = "TIINGO_API_KEY"       # alternative env var
_BASE_URL = "https://api.tiingo.com"      # Tiingo REST root

# Default retry parameters
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SEC = 0.7
_DEFAULT_TIMEOUT = 30  # seconds


def _load_key_from_config() -> str | None:  # noqa: D401 – plain helper
    """Return TIINGO_API_KEY from config.py if that file exists and contains it."""
    cfg_path = Path.cwd() / f"{_CONFIG_MODULE_NAME}.py"
    if cfg_path.is_file():
        cfg = importlib.import_module(_CONFIG_MODULE_NAME)
        return getattr(cfg, _TIINGO_ENV_VAR, None)
    return None


# -----------------------------------------------------------------------------
# Networking helper with exponential back-off
# -----------------------------------------------------------------------------

def _request_with_retries(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    max_attempts: int = _MAX_ATTEMPTS,
    backoff_sec: float = _BACKOFF_BASE_SEC,
) -> Response:
    """Perform an HTTP request with automatic retries on server (5xx) errors."""
    for attempt in range(1, max_attempts + 1):
        resp = requests.request(method, url, params=params, timeout=_DEFAULT_TIMEOUT)
        # Non-5xx responses are considered final (200-499)
        if resp.status_code < 500:
            resp.raise_for_status()
            return resp
        # Otherwise back-off and retry
        wait_time = backoff_sec * 2 ** (attempt - 1)
        logger.warning(f"Request failed with 5xx error, retrying in {wait_time:.1f}s (attempt {attempt}/{max_attempts})")
        time.sleep(wait_time)
    # Last attempt still failed ➜ raise the error
    resp.raise_for_status()
    return resp  # makes mypy happy; never executed


# -----------------------------------------------------------------------------
# Singleton client class
# -----------------------------------------------------------------------------

class TiingoClientSingleton:  # pylint: disable=too-few-public-methods
    """A singleton wrapper around the Tiingo REST API."""

    _instance: "TiingoClientSingleton | None" = None

    # ------------------------------------------------------------------ #
    def __new__(cls) -> "TiingoClientSingleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()  # type: ignore[attr-defined] – run once
        return cls._instance

    # ------------------------------------------------------------------ #
    def _init(self) -> None:
        """Internal one-time initialiser."""
        self.api_key: str | None = (
            _load_key_from_config() or os.getenv(_TIINGO_ENV_VAR)
        )
        if not self.api_key:
            raise EnvironmentError(
                "Tiingo API key not found. Set TIINGO_API_KEY in config.py "
                "or export it as an environment variable."
            )
        logger.debug("Initialized Tiingo client singleton")

    # ------------------------------------------------------------------ #
    # Low-level helper
    # ------------------------------------------------------------------ #
    def get_json(self, endpoint: str, **params: Any) -> Any:
        """Return parsed JSON from any Tiingo endpoint."""
        params["token"] = self.api_key
        url = f"{_BASE_URL}/{endpoint.lstrip('/')}"
        logger.debug(f"Making request to {url}")
        return _request_with_retries("GET", url, params=params).json()

    # ------------------------------------------------------------------ #
    # High-level helper
    # ------------------------------------------------------------------ #
    def get_dataframe(
        self,
        symbol: str,
        *,
        frequency: str = "daily",          # 'daily', '1min', '5min', …
        start_date: str | None = None,
        end_date:   str | None = None,
        tz_convert: bool = True,
    ) -> pd.DataFrame:
        """Get historical data as a DataFrame."""
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Determine endpoint based on frequency
        if frequency == "daily":
            endpoint = f"tiingo/daily/{symbol}/prices"
            params = {}
        else:
            endpoint = f"iex/{symbol}/prices"
            params = {
                "resampleFreq": frequency,
                "columns": "open,high,low,close,volume,date",  # Explicitly request volume
                "format": "json"
            }

        # Query parameters
        params["startDate"] = start_date
        params["endDate"] = end_date

        try:
            data = self.get_json(endpoint, **params)
            if not data:
                logger.warning(f"No data returned from {endpoint}")
                return pd.DataFrame()  # Return empty DataFrame instead of None
                
            df = pd.DataFrame(data)
            # Handle IEX endpoint data which has different column names
            if frequency != "daily":
                # Map IEX column names to standard names
                column_map = {
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'date': 'date'
                }
                # Rename columns if they exist
                df = df.rename(columns=column_map)
                
                # If volume is missing but we have totalVolume, use that
                if 'volume' not in df.columns and 'totalVolume' in df.columns:
                    df['volume'] = df['totalVolume']
                    df = df.drop('totalVolume', axis=1)

            if tz_convert and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_convert("UTC")
                df.set_index("date", inplace=True)
                
            logger.info(f"Successfully retrieved {len(df)} rows for {symbol} from {start_date} to {end_date}")
            
            # Ensure all required columns are present
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in response: {missing_columns}")
                return pd.DataFrame()
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {endpoint}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error



# -----------------------------------------------------------------------------
# Convenience alias so callers can `import tiingoClient`
# -----------------------------------------------------------------------------

# Both names point to the same singleton class; pick whichever import style you
# prefer without duplicating sessions or API quota.
tiingoClient = TiingoClientSingleton  # type: ignore[assignment]

# Explicit re-exports when using `from ... import *`
__all__ = [
    "TiingoClientSingleton",
    "tiingoClient",
]
