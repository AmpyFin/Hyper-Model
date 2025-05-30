"""
Fetch and cache NASDAQ-100 holdings using Financial Modeling Prep API.
"""
import json
import logging
from urllib.request import urlopen
import sys
import os
import ssl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FMP_API_KEY


def get_ndaq_100_holdings(mongo_client):
    """
    Connects to MongoDB and retrieves NASDAQ-100 holdings.

    :param mongo_client: MongoDB client instance
    :return: List of NASDAQ-100 ticker symbols.
    """

    def call_ndaq_100():
        """
        Fetches the list of NASDAQ 100 holdings using the Financial Modeling Prep API and stores it in a MongoDB collection.
        The MongoDB collection is cleared before inserting the updated list of holdings.
        """
        logger.info("Fetching NASDAQ-100 holdings from Financial Modeling Prep API")

        def get_jsonparsed_data(url):
            """
            Parses the JSON response from the provided URL.

            :param url: The API endpoint to retrieve data from.
            :return: Parsed JSON data as a dictionary.
            """
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            logger.debug(f"Making request to {url}")
            response = urlopen(url, context=ssl_context)
            data = response.read().decode("utf-8")
            return json.loads(data)

        try:
            # API URL for fetching NASDAQ 100 holdings
            ndaq_url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey={FMP_API_KEY}"  # noqa: E231
            ndaq_stocks = get_jsonparsed_data(ndaq_url)
            logger.info(f"Successfully retrieved {len(ndaq_stocks)} NASDAQ-100 holdings")
            
            if not ndaq_stocks:
                logger.warning("No holdings data returned from API")
                return
                
        except Exception as e:
            logger.error(f"Error fetching NASDAQ-100 holdings from API: {e}")
            return
            
        try:
            # MongoDB connection details
            db = mongo_client.stock_list
            ndaq100_holdings = db.ndaq100_holdings

            # Clear existing data
            delete_result = ndaq100_holdings.delete_many({})
            logger.debug(f"Cleared {delete_result.deleted_count} existing holdings from MongoDB")
            
            # Insert new data
            insert_result = ndaq100_holdings.insert_many(ndaq_stocks)
            logger.info(f"Successfully inserted {len(insert_result.inserted_ids)} holdings into MongoDB")
            
        except Exception as e:
            logger.error(f"Error updating MongoDB with holdings: {e}")

    # Fetch and update holdings
    call_ndaq_100()

    # Retrieve updated holdings list
    try:
        holdings = [
            stock["symbol"] for stock in mongo_client.stock_list.ndaq100_holdings.find()
        ]
        logger.info(f"Retrieved {len(holdings)} holdings from MongoDB")
        return holdings
    except Exception as e:
        logger.error(f"Error retrieving holdings from MongoDB: {e}")
        return []
