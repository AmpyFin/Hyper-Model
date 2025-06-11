"""
utils/get_historical_data.py
Download OHLCV history and manage MongoDB cache.

Example:
    python -m utils.get_historical_data MSFT 1min 2025-05-01 2025-05-19
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pymongo import MongoClient
import pandas as pd
import logging

from utils.clients.tiingo_client import tiingoClient as historical_data_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_historical_data(
    symbol: str,
    period_days: int,
    frequency: str = "1min",
    mongo_client: Optional[MongoClient] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Get historical OHLCV data as a DataFrame, optionally using MongoDB cache.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    period_days : int
        Number of days of historical data to retrieve
    frequency : str, default="1min"
        Data frequency (1min, 5min, 15min, 1hour, daily)
    mongo_client : Optional[MongoClient], default=None
        MongoDB client instance for caching. If None, caching is disabled.
    use_cache : bool, default=True
        Whether to check MongoDB cache before fetching from API
        
    Returns
    -------
    pd.DataFrame
        Historical OHLCV data as DataFrame. Empty DataFrame if error.
    """
    try:
        # Calculate date range using current date
        end_date = datetime.now()
        buffer_days = int(period_days * (0.4 if period_days <= 30 else 0.6))
        start_date = end_date - timedelta(days=period_days + buffer_days)
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching {frequency} data for {symbol} from {start_str} to {end_str}")
        
        # Check MongoDB cache if enabled
        if mongo_client is not None and use_cache:
            df = _get_from_cache(mongo_client, symbol, period_days, frequency, start_date)
            if df is not None:
                return df
        
        # Switch to daily frequency for longer periods to respect rate limits
        if period_days > 30:
            frequency = "daily"
            logger.debug(f"Switching to daily frequency for {period_days} day period")

        # Fetch from API
        client = historical_data_client()
        df = client.get_dataframe(symbol, frequency=frequency, 
                                start_date=start_str, end_date=end_str)
        
        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol} in date range {start_str} to {end_str}")
            return pd.DataFrame()
            
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns for {symbol}: {missing_columns}")
            if 'volume' in missing_columns:
                df = _fetch_missing_volume(client, symbol, start_str, end_str, frequency, df)
            
            # Don't cache if still missing required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Still missing required columns after recovery attempt: {missing_columns}")
                return df
        
        # Cache the complete data if MongoDB client provided
        if mongo_client is not None:
            _cache_dataframe(mongo_client, symbol, period_days, frequency, df)
        logger.info(f"Successfully retrieved {len(df)} rows of data")
        return df
            
    except Exception as e:
        logger.error(f"Error fetching {frequency} data for {symbol}: {e}")
        return pd.DataFrame()

def _get_from_cache(
    mongo_client: MongoClient,
    symbol: str,
    period_days: int,
    frequency: str,
    start_date: datetime
) -> Optional[pd.DataFrame]:
    """Try to get data from MongoDB cache."""
    try:
        db = mongo_client.HistoricalDatabase
        collection = db.HistoricalDatabase
        
        cached_data = collection.find_one({
            "ticker": symbol,
            "period": period_days,
            "frequency": frequency,
            "date": {"$gte": start_date.replace(hour=0, minute=0, second=0, microsecond=0)}
        })
        
        if cached_data and "data" in cached_data:
            logger.info(f"Found cached data for {symbol}")
            return pd.DataFrame(cached_data["data"])
            
    except Exception as e:
        logger.error(f"Error reading from cache: {e}")
    
    return None

def _cache_dataframe(
    mongo_client: MongoClient,
    symbol: str,
    period_days: int,
    frequency: str,
    df: pd.DataFrame
) -> None:
    """Cache DataFrame in MongoDB."""
    try:
        # Reset index to get date as a column
        df_reset = df.reset_index()
        
        # Convert to list of dictionaries
        historical_data = []
        for _, row in df_reset.iterrows():
            data_point = {
                "date": row.get("date", row.name).isoformat() if hasattr(row.get("date", row.name), 'isoformat') else str(row.get("date", row.name)),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0))
            }
            historical_data.append(data_point)
        
        # Cache the data
        cache_doc = {
            "ticker": symbol,
            "period": period_days,
            "frequency": frequency,
            "date": datetime.now(),
            "data": historical_data
        }
        
        db = mongo_client.HistoricalDatabase
        collection = db.HistoricalDatabase
        
        collection.update_one(
            {"ticker": symbol, "period": period_days, "frequency": frequency},
            {"$set": cache_doc},
            upsert=True
        )
        logger.info(f"Successfully cached {len(historical_data)} records for {symbol}")
        
    except Exception as e:
        logger.error(f"Error caching data: {e}")

def _fetch_missing_volume(
    client: historical_data_client,
    symbol: str,
    start_str: str,
    end_str: str,
    frequency: str,
    df: pd.DataFrame
) -> pd.DataFrame:
    """Attempt to fetch missing volume data from daily endpoint."""
    try:
        logger.info(f"Attempting to fetch volume data for {symbol}")
        daily_df = client.get_dataframe(symbol, frequency='daily',
                                      start_date=start_str, end_date=end_str)
        if 'volume' in daily_df.columns:
            # Resample daily volume to match frequency
            daily_volume = daily_df['volume'].resample(frequency).ffill()
            df['volume'] = daily_volume
            logger.info(f"Successfully added volume data for {symbol}")
    except Exception as e:
        logger.error(f"Failed to fetch volume data: {e}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical OHLCV data")
    parser.add_argument("symbol", help="Stock ticker symbol")
    parser.add_argument("frequency", choices=["daily", "1min", "5min", "15min", "1hour"],
                      help="Data frequency")
    parser.add_argument("period_days", type=int, help="Number of days of data to retrieve")
    args = parser.parse_args()
    
    df = get_historical_data(args.symbol, args.period_days, args.frequency)
    if not df.empty:
        logger.info(f"Successfully retrieved {len(df)} rows of data")
    else:
        logger.error("Failed to retrieve data")
