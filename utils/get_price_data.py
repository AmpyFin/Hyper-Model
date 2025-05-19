"""
utils/get_price_data.py
Fetch latest Tiingo IEX prices.

Usage:
    python -m utils.get_price_data AAPL MSFT TSLA
"""

from __future__ import annotations

import argparse
from typing import List

# flexible alias for future provider swaps
from utils.clients.tiingo_client import tiingoClient as price_client


def fetch_latest(symbols: List[str]) -> None:
    client = price_client()  # singleton
    for sym in symbols:
        quote = client.get_json(f"iex/{sym}")[0]  # list with one dict
        price = quote.get("last") or quote.get("prevClose")
        if price is None:
            print(f"{sym:<6}  -- no price yet (market closed) --")
        else:
            print(f"{sym:<6}  ${price:,.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch latest Tiingo prices")
    p.add_argument("symbols", nargs="+")
    fetch_latest(p.parse_args().symbols)
