"""
Fetch NASDAQ-100 holdings using Financial Modeling Prep API.
"""
import json
import logging
from urllib.request import urlopen
import ssl
from typing import List

from config import FMP_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_jsonparsed_data(url: str) -> dict:
    """
    Parses the JSON response from the provided URL.

    Args:
        url: The API endpoint to retrieve data from.
    Returns:
        Parsed JSON data as a dictionary.
    """
    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    logger.debug(f"Making request to {url}")
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    return json.loads(data)

def get_ndaq_100_holdings() -> List[str]:
    """
    Fetches the list of NASDAQ 100 holdings using the Financial Modeling Prep API.

    Returns:
        List of NASDAQ-100 ticker symbols.
    """
    logger.info("Fetching NASDAQ-100 holdings from Financial Modeling Prep API")

    try:
        # API URL for fetching NASDAQ 100 holdings
        ndaq_url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey={FMP_API_KEY}"
        ndaq_stocks = get_jsonparsed_data(ndaq_url)
        
        if not ndaq_stocks:
            logger.warning("No holdings data returned from API")
            return []
            
        # Extract just the symbols from the response
        holdings = [stock["symbol"] for stock in ndaq_stocks]
        logger.info(f"Successfully retrieved {len(holdings)} NASDAQ-100 holdings")
        return holdings
            
    except Exception as e:
        logger.error(f"Error fetching NASDAQ-100 holdings from API: {e}")
        return []
