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
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests import Response

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
        time.sleep(backoff_sec * 2 ** (attempt - 1))
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

    # ------------------------------------------------------------------ #
    # Low-level helper
    # ------------------------------------------------------------------ #
    def get_json(self, endpoint: str, **params: Any) -> Any:
        """Return parsed JSON from any Tiingo endpoint."""
        params["token"] = self.api_key
        url = f"{_BASE_URL}/{endpoint.lstrip('/')}"
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

        # ── choose correct REST endpoint ───────────────────────────────
        if frequency == "daily":
            endpoint = f"tiingo/daily/{symbol}/prices"
        else:                                       # intraday bars
            endpoint = f"iex/{symbol}/prices"
            params = {"resampleFreq": frequency,
              "columns": "open,high,low,close,volume"}   # <-- add this

        # ── query parameters ───────────────────────────────────────────
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        # ── fetch & frame ──────────────────────────────────────────────
        data = self.get_json(endpoint, **params)
        df = pd.DataFrame(data)

        if tz_convert and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert("UTC")
            df.set_index("date", inplace=True)
        return df



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
