#!/usr/bin/env python3
"""
setup.py – one-shot repository initialiser for the *new* AmpyFin stack.

Tasks performed
---------------
1.  rank_to_coefficient       – geometric decay table (1 … N strategies)
2.  algorithm_holdings        – starting cash / empty positions per strategy
3.  points_tally              – zeroed scorecard per strategy
4.  time_delta                – initial time-step for ranking feedback (0.01)
5.  market_status             – default to "closed" at launch
6.  portfolio_values          – baseline P&L vs QQQ/SPY (Tiingo quotes)
7.  IndicatorsDatabase        – `(strategy, ideal_period)` from registry
8.  HistoricalDatabase        – collection created (empty) for caching

No external brokers (e.g. Alpaca) are contacted – all trades will be paper
tracked inside MongoDB only.

Edit `MONGO_URL` or `STARTING_CASH` below if you need custom values.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import math
from datetime import datetime, timezone
from typing import Dict, Any

from pymongo import MongoClient, errors

# --------------------------------------------------------------------------- #
# Config – adjust here if needed                                              #
# --------------------------------------------------------------------------- #
try:
    from config import MONGO_URL                      # Updated to match actual config.py
except ImportError as exc:
    raise SystemExit("config.py must define `MONGO_URL`") from exc

STARTING_CASH: float = 50_000.0                       # per-strategy seed cash
TIME_DELTA_START: float = 0.01                        # feedback increment
BENCHMARKS = ("QQQ", "SPY")                           # baseline indices
# --------------------------------------------------------------------------- #

# ---- registry of <<strategy_name → ideal_period(days)>> ------------------- #
from registries.ideal_periods_registry import registry as STRATEGY_REGISTRY

# ---- Tiingo helper (quotations only) -------------------------------------- #
try:
    from utils.clients.tiingo_client import tiingoClient as _TClient
except ImportError:
    _TClient = None                                   # price fetch not critical


def _latest_price(ticker: str) -> float | None:
    """Get last/prevClose from Tiingo.  Returns None on failure."""
    if _TClient is None:
        return None
    cli = _TClient()
    try:
        quote: Dict[str, Any] = cli.get_json(f"iex/{ticker}")[0]
        return quote.get("last") or quote.get("prevClose")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Mongo initialisation helpers                                                #
# --------------------------------------------------------------------------- #
def insert_rank_to_coefficient(client: MongoClient) -> None:
    """Populate trading_simulator.rank_to_coefficient   (1 → len(registry))."""
    n = len(STRATEGY_REGISTRY)
    col = client.trading_simulator.rank_to_coefficient
    col.delete_many({})
    e = math.e
    rate = (e ** e) / (e ** 2) - 1
    docs = [
        {"rank": r, "coefficient": rate ** (2 * r)}
        for r in range(1, n + 1)
    ]
    col.insert_many(docs)
    print(f"✓ rank_to_coefficient      ({n} rows)")


def initialize_rank_and_points(client: MongoClient) -> None:
    """algorithm_holdings  +  points_tally   (one doc per strategy)."""
    algo_col = client.trading_simulator.algorithm_holdings
    pts_col  = client.trading_simulator.points_tally

    initialization_date = datetime.now(tz=timezone.utc)
    algo_col.delete_many({})
    pts_col.delete_many({})

    for strat in STRATEGY_REGISTRY:
        algo_col.insert_one(
            {
                "strategy": strat,
                "holdings": {},
                "amount_cash": STARTING_CASH,
                "portfolio_value": STARTING_CASH,
                "initialized_date": initialization_date,
                "total_trades": 0,
                "successful_trades": 0,
                "neutral_trades": 0,
                "failed_trades": 0,
                "last_updated": initialization_date,
            }
        )
        pts_col.insert_one(
            {
                "strategy": strat,
                "total_points": 0,
                "initialized_date": initialization_date,
                "last_updated": initialization_date,
            }
        )
    print(f"✓ algorithm_holdings + points_tally ({len(STRATEGY_REGISTRY)} strats)")


def initialize_time_delta(client: MongoClient) -> None:
    client.trading_simulator.time_delta.delete_many({})
    client.trading_simulator.time_delta.insert_one({"time_delta": TIME_DELTA_START})
    print("✓ time_delta              (0.01)")


def initialize_market_setup(client: MongoClient) -> None:
    client.market_data.market_status.delete_many({})
    client.market_data.market_status.insert_one({"market_status": "CLOSED"})
    print("✓ market_status           (CLOSED)")


def initialize_portfolio_percentages(client: MongoClient) -> None:
    """Seed portfolio vs benchmarks (QQQ/SPY) – relies on Tiingo quotes."""
    col = client.trades.portfolio_values
    col.delete_many({})

    for name, base_price in {"portfolio": STARTING_CASH,
                             "ndaq": _latest_price("QQQ"),
                             "spy":  _latest_price("SPY")}.items():
        col.insert_one(
            {
                "name": f"{name}_percentage",
                "portfolio_value": 0.0 if base_price is None else 0.0,
            }
        )
    print("✓ portfolio_values        (placeholders)")


def initialize_indicator_setup(client: MongoClient) -> None:
    """Insert (strategy, ideal_period) pairs into IndicatorsDatabase.Indicators."""
    col = client.IndicatorsDatabase.Indicators
    col.delete_many({})
    col.insert_many(
        [{"indicator": strat, "ideal_period": period}
         for strat, period in STRATEGY_REGISTRY.items()]
    )
    print(f"✓ IndicatorsDatabase      ({len(STRATEGY_REGISTRY)} rows)")


def initialize_historical_database_cache(client: MongoClient) -> None:
    # Just ensure collection exists – no docs inserted here
    _ = client.HistoricalDatabase.HistoricalDatabase
    print("✓ HistoricalDatabase      (collection created)")


# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        # Fix SSL certificate issues with MongoDB Atlas
        import certifi
        client = MongoClient(MONGO_URL, tlsCAFile=certifi.where())
    except errors.ConnectionFailure as exc:
        raise SystemExit(f"Cannot connect to MongoDB → {exc}") from exc

    insert_rank_to_coefficient(client)
    initialize_rank_and_points(client)
    initialize_time_delta(client)
    initialize_market_setup(client)
    initialize_portfolio_percentages(client)
    initialize_indicator_setup(client)
    initialize_historical_database_cache(client)

    client.close()
    print("\n🎉  Repository initialised – ready to trade!")


if __name__ == "__main__":
    main()