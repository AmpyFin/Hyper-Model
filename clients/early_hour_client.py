from datetime import datetime, timezone
from typing import List, Set, Dict
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
        
    def get_db_holdings(self) -> Dict[str, float]:
        """Get current holdings from MongoDB trades.current_holdings collection."""
        try:
            # Get all documents from the current_holdings collection with ticker and amount
            holdings = {
                doc['ticker']: doc['amount'] 
                for doc in self.trades_db.current_holdings.find({}, {'ticker': 1, 'amount': 1, '_id': 0})
            }
            self.logger.info(f"Retrieved {len(holdings)} holdings from database")
            return holdings
        except Exception as e:
            self.logger.error(f"Error retrieving holdings from database: {e}")
            return {}

    def find_unknown_tickers(self) -> Dict[str, float]:
        """
        Find tickers in our database that are not in our control list (unknown tickers).
        These are positions we hold but shouldn't, as they're not in our approved ticker list.
        
        Returns:
            Dict[str, float]: Dictionary of unknown tickers and their quantities to be sold.
                             Key: ticker symbol
                             Value: current holding amount that needs to be sold
        """
        try:
            # Get approved tickers from control
            approved_tickers = set(tickers)
            self.logger.info(f"Retrieved {len(approved_tickers)} approved tickers from control")
            
            # Store approved tickers for other clients to use
            self.return_documents["tickers"] = list(approved_tickers)
            
            # Get current holdings from database with quantities
            db_holdings = self.get_db_holdings()
            
            # Find unknown tickers (in database but not in approved list)
            unknown_tickers = {}
            unapproved_holdings = set(db_holdings.keys()) - approved_tickers
            
            if unapproved_holdings:
                self.logger.info(f"Found unknown tickers with quantities: {[(t, db_holdings[t]) for t in sorted(unapproved_holdings)]}")
                # Add to unknown with their current quantities (to sell)
                for ticker in unapproved_holdings:
                    unknown_tickers[ticker] = db_holdings[ticker]
                # Store in return documents for trading client
                self.return_documents["tickers_to_sell_at_open"] = {
                    ticker: {"amount": db_holdings[ticker]} for ticker in unapproved_holdings
                }
            
            # Find missing approved tickers (in approved list but not in holdings)
            missing_approved = approved_tickers - set(db_holdings.keys())
            if missing_approved:
                self.logger.info(f"Missing approved tickers to potentially buy: {sorted(missing_approved)}")
                # Store in return documents for trading client
                self.return_documents["tickers_to_buy_at_open"] = {
                    ticker: {"amount": 0} for ticker in missing_approved
                }
                
            if not unknown_tickers:
                self.logger.info("No unknown tickers found in database")
                
            return unknown_tickers
                
        except Exception as e:
            self.logger.error(f"Error finding unknown tickers: {e}")
            raise

    def get_tickers_to_buy_at_open(self) -> List[str]:
        """Get tickers to buy at open."""
        return self.return_documents["tickers_to_buy_at_open"]
    
    def get_tickers_to_sell_at_open(self) -> List[str]:
        """Get tickers to sell at open."""
        raise NotImplementedError("Not implemented")
    
    def get_strategies_to_coefficients(self) -> Dict[str, float]:
        """Get strategies to coefficients."""
        raise NotImplementedError("Not implemented")
    
    def get_strategies_to_portfolio(self) -> Dict[str, float]:
        """Get strategies to portfolio."""
        raise NotImplementedError("Not implemented")

    def get_strategies_to_points(self) -> Dict[str, float]:
        """Get strategies to points."""
        raise NotImplementedError("Not implemented")

    def get_tickers(self) -> List[str]:
        """Get tickers."""
        return tickers
    def save_tickers(self, tickers: List[str]) -> None:
        """Save tickers to MongoDB."""
        self.return_documents["tickers"] = tickers

    def get_strategies_to_portfolio(self) -> Dict[str, float]:
        """Get strategies to portfolio."""
        raise NotImplementedError("Not implemented")

    def run(self) -> None:
        """Execute the early hour processing tasks."""
        self.logger.info("Early hour client started")
        
        try:
            unknown_tickers = self.find_unknown_tickers()
            if unknown_tickers:
                self.logger.info(f"Found unknown tickers to sell: {unknown_tickers}")
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
