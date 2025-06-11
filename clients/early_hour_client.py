from datetime import datetime, timezone
from typing import List, Set
from clients.client import Client
from utils.db_clients.mongodb_client import MongoDBClient
from control import tickers

# Priority for Early Hour Client is to process information that will be used in trading and ranking client
# as well as to store necessary information in DuckDB

class EarlyHourClient(Client):
    def __init__(self):
        """Initialize the EarlyHourClient."""
        super().__init__()
        self.mongodb = MongoDBClient()
        self.client = self.mongodb.get_client()
        self.trades_db = self.client.trades
        self.return_documents = {
            # used for both trading and ranking clients
            "tickers": [],

            # used by trading client primarily
            "tickers_to_buy_at_open": {},
            "tickers_to_sell_at_open": {},
            "strategies_to_coefficients": {},

            # used by ranking client primarily
            "strategies_to_portfolio": {},
            "strategies_to_points": {},
            
        }
        
    def get_db_holdings(self) -> Set[str]:
        """Get current holdings from MongoDB trades.current_holdings collection."""
        try:
            # Get all documents from the current_holdings collection and extract tickers
            holdings = set(doc['ticker'] for doc in self.trades_db.current_holdings.find({}, {'ticker': 1}))
            self.logger.info(f"Retrieved {len(holdings)} holdings from database")
            return holdings
        except Exception as e:
            self.logger.error(f"Error retrieving holdings from database: {e}")
            return set()

    def find_mismatched_tickers(self) -> None:
        """Compare control tickers with current holdings and identify mismatches."""
        try:
            # Get tickers from control
            control_tickers = set(tickers)
            self.logger.info(f"Retrieved {len(control_tickers)} tickers from control")
            
            # Get current holdings from database
            db_holdings = self.get_db_holdings()
            
            # Find tickers that should be added (in control but not in holdings)
            tickers_to_add = control_tickers - db_holdings
            if tickers_to_add:
                self.logger.info(f"Tickers to add: {sorted(tickers_to_add)}")
            
            # Find tickers that should be removed (in holdings but not in control)
            tickers_to_remove = db_holdings - control_tickers
            if tickers_to_remove:
                self.logger.info(f"Tickers to remove: {sorted(tickers_to_remove)}")
                
            if not tickers_to_add and not tickers_to_remove:
                self.logger.info("Holdings are up to date with control tickers")
                
        except Exception as e:
            self.logger.error(f"Error comparing tickers: {e}")
            raise

    def run(self) -> None:
        """Execute the early hour processing tasks."""
        self.logger.info("Early hour client started")
        
        try:
            self.find_mismatched_tickers()
            self.logger.info("Early hour client operations completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during early hour processing: {e}")
            raise
        finally:
            self.mongodb.close()

def main():
    client = EarlyHourClient()
    client.run()

if __name__ == "__main__":
    main()
