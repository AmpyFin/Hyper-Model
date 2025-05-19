"""
utils/get_historical_data.py
Download OHLCV history to Parquet.

Example:
    python -m utils.get_historical_data MSFT 1min 2025-05-01 2025-05-19
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.clients.tiingo_client import tiingoClient as historical_data_client


def pull_and_save(symbol: str, frequency: str, start_date: str,
                  end_date: str | None) -> Path:
    client = historical_data_client()
    df = client.get_dataframe(symbol, frequency=frequency,
                              start_date=start_date, end_date=end_date)

    return df


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("symbol")
    a.add_argument("frequency", choices=["daily", "1min", "5min", "15min", "1hour"])
    a.add_argument("start_date")
    a.add_argument("end_date", nargs="?", default=None)
    args = a.parse_args()
    print(pull_and_save(args.symbol, args.frequency, args.start_date, args.end_date))
