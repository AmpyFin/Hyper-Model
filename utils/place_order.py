"""
utils/place_order.py
Simulate order placement and log trades to MongoDB without actual trading.

This module provides functionality to:
- Log simulated trades to MongoDB
- Track asset quantities and positions
- Manage stop loss and take profit limits
- All without actually placing real trades
"""

from __future__ import annotations

import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pymongo import MongoClient

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clients.tiingo_client import tiingoClient
except ImportError:
    # If running from root directory, try utils.clients path
    from utils.clients.tiingo_client import tiingoClient

# Configure logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize price client
price_client = tiingoClient()
logger.debug("Initialized Tiingo price client")

# Default risk management parameters
DEFAULT_STOP_LOSS = 0.03  # 3% stop loss
DEFAULT_TAKE_PROFIT = 0.05  # 5% take profit


def get_latest_price(symbol: str) -> float:
    """
    Get the latest price for a ticker using Tiingo API.
    
    :param symbol: Stock ticker symbol
    :return: Latest price
    """
    try:
        logger.debug(f"Fetching latest price for {symbol}")
        quote = price_client.get_json(f"iex/{symbol}")[0]
        price = quote.get("last") or quote.get("prevClose")
        if price is not None:
            logger.debug(f"Latest price for {symbol}: ${float(price):,.2f}")
            return float(price)
        else:
            logger.warning(f"No price available for {symbol} (market may be closed)")
            return None
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return None


def place_order(symbol: str, side: str, quantity: float, mongo_client: MongoClient,
                stop_loss: float = DEFAULT_STOP_LOSS, 
                take_profit: float = DEFAULT_TAKE_PROFIT) -> Dict[str, Any]:
    """
    Simulate placing a market order and log the order to MongoDB.
    Updates asset quantities and manages stop loss/take profit limits.

    :param symbol: The stock symbol to trade
    :param side: Order side ("BUY" or "SELL")
    :param quantity: Quantity to trade
    :param mongo_client: MongoDB client instance
    :param stop_loss: Stop loss percentage (default 3%)
    :param take_profit: Take profit percentage (default 5%)
    :return: Dictionary containing order simulation results
    """
    logger.info(f"Processing {side} order for {quantity} {symbol}")
    
    # Validate inputs
    if side.upper() not in ["BUY", "SELL"]:
        error_msg = "Side must be 'BUY' or 'SELL'"
        logger.error(f"{error_msg} (got '{side}')")
        raise ValueError(error_msg)
    
    if quantity <= 0:
        error_msg = "Quantity must be positive"
        logger.error(f"{error_msg} (got {quantity})")
        raise ValueError(error_msg)
    
    # Get current price
    current_price = get_latest_price(symbol)
    if current_price is None:
        error_msg = f"Could not fetch current price for {symbol}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Round quantity to 3 decimal places
    qty = round(quantity, 3)
    
    # Calculate stop loss and take profit prices
    stop_loss_price = round(current_price * (1 - stop_loss), 2)
    take_profit_price = round(current_price * (1 + take_profit), 2)
    
    logger.debug(f"Order details: Price=${current_price:,.2f}, Stop=${stop_loss_price:,.2f}, Target=${take_profit_price:,.2f}")
    
    # Create order timestamp
    order_time = datetime.now(tz=timezone.utc)
    
    # Log trade details to MongoDB
    db = mongo_client.trades
    trade_doc = {
        "symbol": symbol,
        "qty": qty,
        "side": side.upper(),
        "price": current_price,
        "time_in_force": "DAY",
        "time": order_time,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "order_type": "MARKET",
        "status": "FILLED"  # Simulate immediate fill
    }
    
    try:
        db.paper.insert_one(trade_doc)
        logger.info(f"Logged {side} trade: {qty} {symbol} @ ${current_price:,.2f}")
    except Exception as e:
        logger.error(f"Failed to log trade to MongoDB: {e}")
        raise
    
    # Track assets and quantities
    assets = db.assets_quantities
    limits = db.assets_limit
    
    try:
        if side.upper() == "BUY":
            # Update asset quantity (add to position)
            assets.update_one(
                {"symbol": symbol}, 
                {"$inc": {"quantity": qty}}, 
                upsert=True
            )
            
            # Set or update stop loss and take profit limits
            limits.update_one(
                {"symbol": symbol},
                {
                    "$set": {
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                        "last_updated": order_time
                    }
                },
                upsert=True,
            )
            
            logger.info(f"Updated long position: +{qty} {symbol}, Stop=${stop_loss_price:,.2f}, Target=${take_profit_price:,.2f}")
            
        elif side.upper() == "SELL":
            # Update asset quantity (subtract from position)
            assets.update_one(
                {"symbol": symbol}, 
                {"$inc": {"quantity": -qty}}, 
                upsert=True
            )
            
            # Check if position is fully closed
            asset_doc = assets.find_one({"symbol": symbol})
            if asset_doc and asset_doc.get("quantity", 0) <= 0:
                # Remove asset if quantity is zero or negative
                assets.delete_one({"symbol": symbol})
                limits.delete_one({"symbol": symbol})
                logger.info(f"Position closed: {symbol}")
            else:
                remaining_qty = asset_doc.get("quantity", 0) if asset_doc else 0
                logger.info(f"Reduced position: -{qty} {symbol} (remaining: {remaining_qty:,.3f})")
    except Exception as e:
        logger.error(f"Failed to update position tracking: {e}")
        raise
    
    # Create order result dictionary (simulating Alpaca response structure)
    order_result = {
        "id": f"simulated_{symbol}_{int(order_time.timestamp())}",
        "symbol": symbol,
        "qty": qty,
        "side": side.upper(),
        "order_type": "market",
        "time_in_force": "day",
        "status": "filled",
        "filled_at": order_time.isoformat(),
        "filled_qty": qty,
        "filled_avg_price": current_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price
    }
    
    logger.debug(f"Order complete: {order_result['id']}")
    return order_result


def get_current_positions(mongo_client: MongoClient) -> Dict[str, Dict[str, Any]]:
    """
    Get current positions from MongoDB.
    
    :param mongo_client: MongoDB client instance
    :return: Dictionary of current positions
    """
    logger.debug("Fetching current positions from MongoDB")
    db = mongo_client.trades
    assets = db.assets_quantities
    limits = db.assets_limit
    
    positions = {}
    
    try:
        for asset_doc in assets.find({}):
            symbol = asset_doc["symbol"]
            quantity = asset_doc["quantity"]
            
            # Get stop loss and take profit info
            limit_doc = limits.find_one({"symbol": symbol})
            
            positions[symbol] = {
                "quantity": quantity,
                "stop_loss_price": limit_doc.get("stop_loss_price") if limit_doc else None,
                "take_profit_price": limit_doc.get("take_profit_price") if limit_doc else None,
                "last_updated": limit_doc.get("last_updated") if limit_doc else None
            }
        
        logger.info(f"Found {len(positions)} active positions")
        return positions
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return {}


def get_trade_history(mongo_client: MongoClient, symbol: Optional[str] = None, 
                     limit: int = 100) -> list:
    """
    Get trade history from MongoDB.
    
    :param mongo_client: MongoDB client instance
    :param symbol: Optional symbol filter
    :param limit: Maximum number of trades to return
    :return: List of trade documents
    """
    logger.debug(f"Fetching trade history (symbol={symbol}, limit={limit})")
    db = mongo_client.trades
    
    try:
        query = {"symbol": symbol} if symbol else {}
        trades = list(db.paper.find(query).sort("time", -1).limit(limit))
        logger.info(f"Retrieved {len(trades)} trades from history")
        return trades
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
        return []

