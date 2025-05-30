"""
Functions to check if US stock market is currently open.
"""
import finnhub
import datetime
import pytz
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path to make the import work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import FINNHUB_API_KEY

def get_market_status():
    """
    Get the current US market status using Finnhub API.
    
    Returns:
        dict: Market status information including:
            - open: Whether regular market hours are active
            - exchange: Exchange code (US)
            - stockExchangeName: US Stock Markets
            - session: Current market session (pre-market, regular, post-market, closed)
            - next_open: Next market open time (if available)
            - next_close: Next market close time (if available)
            - raw: Raw Finnhub response
    """
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    try:
        logger.debug("Fetching market status from Finnhub API")
        # Get basic market status
        status_data = finnhub_client.market_status(exchange='US')
        
        # Current time in ET for reference
        eastern = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(eastern)
        logger.debug(f"Current time (ET): {now}")
        
        # Default status
        status = {
            "open": status_data.get('isOpen', False),
            "exchange": "US",
            "stockExchangeName": "US Stock Markets",
            "session": status_data.get('session', 'closed'),  # Original session from API
            "raw": status_data
        }
        
        # Set unified market_status field based on session
        if status["session"] == "regular":
            status["market_status"] = "OPEN"
        elif status["session"] == "pre-market":
            status["market_status"] = "EARLY_HOURS"
        elif status["session"] == "post-market":
            status["market_status"] = "CLOSED"
        else:
            status["market_status"] = "CLOSED"
            
        # For backward compatibility
        status["pre_market"] = status["session"] == "pre-market"
        status["after_hours"] = status["session"] == "post-market"
        
        logger.info(f"Market status: {status['market_status']} (Session: {status['session']})")
        
        # Try to get next market open/close
        try:
            logger.debug("Fetching market calendar information")
            # Get calendar information
            calendar = finnhub_client.stock_market_status()
            if "marketHoliday" in calendar:
                for holiday in calendar["marketHoliday"]:
                    if holiday.get("exchange") == "US":
                        status["next_open"] = holiday.get("open")
                        status["next_close"] = holiday.get("close")
                        logger.info(f"Next market open: {status['next_open']}, close: {status['next_close']}")
                        break
        except Exception as e:
            logger.warning(f"Failed to fetch market calendar: {e}")
            pass
            
        return status
        
    except Exception as e:
        logger.error(f"Error fetching market status: {e}")
        # Return error info but with default values
        return {
            "open": False,
            "session": "closed",
            "market_status": "CLOSED",
            "exchange": "US",
            "error": str(e)
        }

# ------------------------------------------------------------
# Convenience Functions
# ------------------------------------------------------------
def is_market_open():
    """Check if US market is in regular trading hours"""
    status = get_market_status()
    is_open = status.get("market_status") == "OPEN"
    logger.debug(f"Market open check: {is_open}")
    return is_open

def is_extended_hours():
    """Check if US market is in extended trading hours (pre-market or post-market)"""
    status = get_market_status()
    is_extended = status.get("market_status") in ["PRE_MARKET", "POST_MARKET"]
    logger.debug(f"Extended hours check: {is_extended}")
    return is_extended

def is_pre_market():
    """Check if US market is in pre-market session"""
    status = get_market_status()
    is_pre = status.get("market_status") == "PRE_MARKET"
    logger.debug(f"Pre-market check: {is_pre}")
    return is_pre

def is_after_hours():
    """Check if US market is in post-market session"""
    status = get_market_status()
    is_after = status.get("market_status") == "POST_MARKET"
    logger.debug(f"After-hours check: {is_after}")
    return is_after

def get_market_session():
    """Get current market session (pre-market, regular, post-market, closed)"""
    status = get_market_status()
    session = status.get("session", "closed")
    logger.debug(f"Market session: {session}")
    return session

def get_market_status_simple():
    """Get simplified market status (OPEN, EARLY_HOURS, CLOSED)"""
    status = get_market_status()
    market_status = status.get("market_status", "CLOSED")
    logger.debug(f"Simple market status: {market_status}")
    return market_status

# For backward compatibility
is_us_market_open = is_market_open
