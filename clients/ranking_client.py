import heapq
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any

import sys
import certifi
from pymongo import MongoClient
import pandas as pd

# Updated imports using new structure
from config import FMP_API_KEY, MONGO_URL
from utils.get_ndaq_100_holdings import get_ndaq_100_holdings
from utils.get_market_status import get_market_status_simple
from utils.get_price_data import fetch_latest
from utils.get_historical_data import get_data
from utils.clients.tiingo_client import tiingoClient
from strategies import discover
from registries.ideal_periods_registry import registry

# Import control parameters (assuming they exist in a separate control module or will be defined here)
# For now, I'll define them as constants - you can move these to a separate control module if needed
loss_price_change_ratio_d1 = 0.95
loss_price_change_ratio_d2 = 0.90
loss_profit_time_d1 = 1.0
loss_profit_time_d2 = 1.5
loss_profit_time_else = 2.0
profit_price_change_ratio_d1 = 1.05
profit_price_change_ratio_d2 = 1.10
profit_profit_time_d1 = 1.0
profit_profit_time_d2 = 1.5
profit_profit_time_else = 2.0
rank_asset_limit = 0.1
rank_liquidity_limit = 1000
time_delta_balanced = 0.01
time_delta_increment = 1.0
time_delta_mode = "additive"
time_delta_multiplicative = 1.01

ca = certifi.where()

# Set up robust logging configuration that works even if basicConfig was already called
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers to prevent duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create formatters
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Create file handler
file_handler = logging.FileHandler("rank_system.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize Tiingo client for price data
price_client = tiingoClient()

# Discover all available strategies
strategies = discover()
logging.info(f"Discovered {len(strategies)} strategies: {list(strategies.keys())}")


def get_latest_price(ticker: str) -> float:
    """
    Get the latest price for a ticker using Tiingo API.
    
    :param ticker: Stock ticker symbol
    :return: Latest price
    """
    try:
        quote = price_client.get_json(f"iex/{ticker}")[0]
        price = quote.get("last") or quote.get("prevClose")
        return float(price) if price is not None else None
    except Exception as e:
        logging.error(f"Error fetching price for {ticker}: {e}")
        return None


def simulate_strategy(strategy_class, ticker: str, current_price: float, 
                     historical_data: List[Dict], account_cash: float, 
                     portfolio_qty: int, total_portfolio_value: float):
    """
    Simulate trading strategy and return action and quantity.
    
    :param strategy_class: Strategy class with fit and predict methods
    :param ticker: Stock ticker symbol
    :param current_price: Current stock price
    :param historical_data: Historical price data
    :param account_cash: Available cash
    :param portfolio_qty: Current quantity held
    :param total_portfolio_value: Total portfolio value
    :return: Tuple of (action, quantity)
    """
    try:
        # Create strategy instance
        strategy = strategy_class()
        
        # Convert historical data to DataFrame if needed
        if historical_data and isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        else:
            df = historical_data
        
        # Fit the strategy with historical data
        strategy.fit(df)
        
        # Predict using the standard parameters that most strategies expect
        prediction = strategy.predict(
            current_price=current_price,
            historical_df=df
        )
        
        # Handle different prediction return types
        if isinstance(prediction, (int, float)):
            # Most strategies return a signal value between -1 and 1
            signal = float(prediction)
            print(f"Signal: {signal}")
            if signal > 0.1:
                action = 'buy'
                # Calculate quantity based on signal strength and available cash
                max_investment = min(account_cash * 0.1, total_portfolio_value * 0.05)  # Max 10% of cash or 5% of portfolio
                quantity = int((max_investment * abs(signal)) / current_price)
            elif signal < -0.1:
                action = 'sell'
                # Sell based on signal strength
                quantity = int(portfolio_qty * abs(signal) * 0.5)  # Sell up to 50% of holdings
            else:
                action = 'hold'
                quantity = 0
        elif isinstance(prediction, dict):
            action = prediction.get('action', 'hold')
            quantity = prediction.get('quantity', 0)
        elif isinstance(prediction, tuple):
            action, quantity = prediction
        else:
            action, quantity = 'hold', 0
            
        return action, quantity
        
    except Exception as e:
        logging.error(f"Error simulating strategy {strategy_class.__name__} for {ticker}: {e}")
        return 'hold', 0


def process_ticker(ticker: str, mongo_client: MongoClient):
    """
    Process a single ticker for all strategies.
    """
    try:
        current_price = None
        while current_price is None:
            try:
                current_price = get_latest_price(ticker)
                if current_price is None:
                    logging.warning(f"No price available for {ticker}, retrying...")
                    time.sleep(10)
                    continue
            except Exception as fetch_error:
                logging.warning(
                    f"Error fetching price for {ticker}. Retrying... {fetch_error}"
                )
                time.sleep(10)
                return

        indicator_tb = mongo_client.IndicatorsDatabase
        indicator_collection = indicator_tb.Indicators
        
        for strategy_name, strategy_class in strategies.items():
            historical_data = None
            while historical_data is None:
                try:
                    # First ensure the strategy exists in indicators collection
                    strategy_indicator = indicator_collection.find_one({"indicator": strategy_name})
                    
                    if not strategy_indicator:
                        # Get ideal period from registry and store in MongoDB
                        ideal_period_from_registry = registry.get(strategy_name, 14)  # Default to 14 if not found
                        indicator_collection.insert_one({
                            "indicator": strategy_name,
                            "ideal_period": ideal_period_from_registry
                        })
                        ideal_period = ideal_period_from_registry
                    else:
                        # Get ideal period from MongoDB
                        ideal_period = strategy_indicator["ideal_period"]
                    
                    historical_data = get_data(ticker, mongo_client, ideal_period)
                except Exception as fetch_error:
                    logging.warning(
                        f"Error fetching historical data for {ticker}. Retrying... {fetch_error}"
                    )
                    time.sleep(10)
                    
            db = mongo_client.trading_simulator
            holdings_collection = db.algorithm_holdings
            print(f"Processing {strategy_name} for {ticker}")
            
            strategy_doc = holdings_collection.find_one({"strategy": strategy_name})
            if not strategy_doc:
                # Initialize strategy document if it doesn't exist
                initial_doc = {
                    "strategy": strategy_name,
                    "amount_cash": 100000.0,  # Start with $100k
                    "portfolio_value": 100000.0,
                    "holdings": {},
                    "total_trades": 0,
                    "successful_trades": 0,
                    "failed_trades": 0,
                    "neutral_trades": 0,
                    "last_updated": datetime.now()
                }
                holdings_collection.insert_one(initial_doc)
                strategy_doc = initial_doc
                logging.info(f"Initialized strategy {strategy_name} with default values")

            account_cash = strategy_doc["amount_cash"]
            total_portfolio_value = strategy_doc["portfolio_value"]
            portfolio_qty = strategy_doc["holdings"].get(ticker, {}).get("quantity", 0)

            simulate_trade(
                ticker,
                strategy_class,
                historical_data,
                current_price,
                account_cash,
                portfolio_qty,
                total_portfolio_value,
                mongo_client,
            )

        print(f"{ticker} processing completed.")
    except Exception as e:
        logging.error(f"Error in thread for {ticker}: {e}")


def simulate_trade(
    ticker: str,
    strategy_class,
    historical_data: List[Dict],
    current_price: float,
    account_cash: float,
    portfolio_qty: int,
    total_portfolio_value: float,
    mongo_client: MongoClient,
):
    """
    Simulates a trade based on the given strategy and updates MongoDB.
    """
    strategy_name = strategy_class.__name__
    
    # Simulate trading action from strategy
    print(
        f"Simulating trade for {ticker} with strategy {strategy_name} and quantity of {portfolio_qty}"
    )
    action, quantity = simulate_strategy(
        strategy_class,
        ticker,
        current_price,
        historical_data,
        account_cash,
        portfolio_qty,
        total_portfolio_value,
    )

    # MongoDB setup
    db = mongo_client.trading_simulator
    holdings_collection = db.algorithm_holdings
    points_collection = db.points_tally

    # Find the strategy document in MongoDB
    strategy_doc = holdings_collection.find_one({"strategy": strategy_name})
    holdings_doc = strategy_doc.get("holdings", {})
    
    # Get or initialize time_delta
    time_delta_doc = db.time_delta.find_one({})
    if not time_delta_doc:
        db.time_delta.insert_one({"time_delta": 1.0})
        time_delta = 1.0
    else:
        time_delta = time_delta_doc["time_delta"]

    # Update holdings and cash based on trade action
    if (
        action in ["buy"]
        and strategy_doc["amount_cash"] - quantity * current_price > rank_liquidity_limit
        and quantity > 0
        and ((portfolio_qty + quantity) * current_price) / total_portfolio_value < rank_asset_limit
    ):
        logging.info(
            f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}"
        )
        # Calculate average price if already holding some shares of the ticker
        if ticker in holdings_doc:
            current_qty = holdings_doc[ticker]["quantity"]
            new_qty = current_qty + quantity
            average_price = (
                holdings_doc[ticker]["price"] * current_qty + current_price * quantity
            ) / new_qty
        else:
            new_qty = quantity
            average_price = current_price

        # Update the holdings document for the ticker.
        holdings_doc[ticker] = {"quantity": new_qty, "price": average_price}

        # Deduct the cash used for buying and increment total trades
        holdings_collection.update_one(
            {"strategy": strategy_name},
            {
                "$set": {
                    "holdings": holdings_doc,
                    "amount_cash": strategy_doc["amount_cash"] - quantity * current_price,
                    "last_updated": datetime.now(),
                },
                "$inc": {"total_trades": 1},
            },
            upsert=True,
        )

    elif (
        action in ["sell"]
        and str(ticker) in holdings_doc
        and holdings_doc[str(ticker)]["quantity"] > 0
    ):
        logging.info(
            f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}"
        )
        current_qty = holdings_doc[ticker]["quantity"]

        # Ensure we do not sell more than we have
        sell_qty = min(quantity, current_qty)
        holdings_doc[ticker]["quantity"] = current_qty - sell_qty

        price_change_ratio = (
            current_price / holdings_doc[ticker]["price"]
            if ticker in holdings_doc
            else 1
        )

        if current_price > holdings_doc[ticker]["price"]:
            # increment successful trades
            holdings_collection.update_one(
                {"strategy": strategy_name},
                {"$inc": {"successful_trades": 1}},
                upsert=True,
            )

            # Calculate points to add if the current price is higher than the purchase price
            if price_change_ratio < profit_price_change_ratio_d1:
                points = time_delta * profit_profit_time_d1
            elif price_change_ratio < profit_price_change_ratio_d2:
                points = time_delta * profit_profit_time_d2
            else:
                points = time_delta * profit_profit_time_else

        else:
            # Calculate points to deduct if the current price is lower than the purchase price
            if holdings_doc[ticker]["price"] == current_price:
                holdings_collection.update_one(
                    {"strategy": strategy_name}, 
                    {"$inc": {"neutral_trades": 1}}
                )
            else:
                holdings_collection.update_one(
                    {"strategy": strategy_name},
                    {"$inc": {"failed_trades": 1}},
                    upsert=True,
                )

            if price_change_ratio > loss_price_change_ratio_d1:
                points = -time_delta * loss_profit_time_d1
            elif price_change_ratio > loss_price_change_ratio_d2:
                points = -time_delta * loss_profit_time_d2
            else:
                points = -time_delta * loss_profit_time_else

        # Update the points tally
        points_collection.update_one(
            {"strategy": strategy_name},
            {
                "$set": {"last_updated": datetime.now()},
                "$inc": {"total_points": points},
            },
            upsert=True,
        )
        
        if holdings_doc[ticker]["quantity"] == 0:
            del holdings_doc[ticker]
            
        # Update cash after selling
        holdings_collection.update_one(
            {"strategy": strategy_name},
            {
                "$set": {
                    "holdings": holdings_doc,
                    "amount_cash": strategy_doc["amount_cash"] + sell_qty * current_price,
                    "last_updated": datetime.now(),
                },
                "$inc": {"total_trades": 1},
            },
            upsert=True,
        )

    else:
        logging.info(
            f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}"
        )
    print(
        f"Action: {action} | Ticker: {ticker} | Quantity: {quantity} | Price: {current_price}"
    )


def update_portfolio_values(client: MongoClient):
    """
    Update portfolio values for all strategies by calculating cash + holdings value.
    """
    db = client.trading_simulator
    holdings_collection = db.algorithm_holdings
    
    # Update portfolio values
    for strategy_doc in holdings_collection.find({}):
        # Calculate the portfolio value for the strategy
        portfolio_value = strategy_doc["amount_cash"]

        for ticker, holding in strategy_doc["holdings"].items():
            current_price = None
            while current_price is None:
                try:
                    current_price = get_latest_price(ticker)
                except Exception as e:
                    print(f"Error fetching price for {ticker} due to: {e}. Retrying...")
                    break

            print(f"Current price of {ticker}: {current_price}")
            if current_price is None:
                current_price = 0
                
            # Calculate the value of the holding
            holding_value = holding["quantity"] * current_price
            if current_price == 0:
                holding_value = 5000  # Default value for error cases
                
            # Add the holding value to the portfolio value
            portfolio_value += holding_value

        # Update the portfolio value in the strategy document
        holdings_collection.update_one(
            {"strategy": strategy_doc["strategy"]},
            {"$set": {"portfolio_value": portfolio_value}},
            upsert=True,
        )


def update_ranks(client: MongoClient):
    """
    Based on portfolio values and points, rank the strategies for actual trading.
    """
    db = client.trading_simulator
    points_collection = db.points_tally
    rank_collection = db.rank
    algo_holdings = db.algorithm_holdings
    
    # Delete all documents in rank collection first
    rank_collection.delete_many({})
    
    # Create ranking queue
    q = []
    for strategy_doc in algo_holdings.find({}):
        strategy_name = strategy_doc["strategy"]
        if strategy_name == "test" or strategy_name == "test_strategy":
            continue
            
        # Get or initialize points
        points_doc = points_collection.find_one({"strategy": strategy_name})
        if not points_doc:
            points_collection.insert_one({"strategy": strategy_name, "total_points": 0})
            total_points = 0
        else:
            total_points = points_doc["total_points"]
            
        if total_points > 0:
            heapq.heappush(
                q,
                (
                    total_points * 2 + strategy_doc["portfolio_value"],
                    strategy_doc["successful_trades"] - strategy_doc["failed_trades"],
                    strategy_doc["amount_cash"],
                    strategy_doc["strategy"],
                ),
            )
        else:
            heapq.heappush(
                q,
                (
                    strategy_doc["portfolio_value"],
                    strategy_doc["successful_trades"] - strategy_doc["failed_trades"],
                    strategy_doc["amount_cash"],
                    strategy_doc["strategy"],
                ),
            )
            
    rank = 1
    while q:
        _, _, _, strategy_name = heapq.heappop(q)
        rank_collection.insert_one({"strategy": strategy_name, "rank": rank})
        rank += 1

    # Delete historical database so new one can be used tomorrow
    db = client.HistoricalDatabase
    collection = db.HistoricalDatabase
    collection.delete_many({})
    print("Successfully updated ranks")
    print("Successfully deleted historical database")


def main():
    """
    Main function to control the workflow based on the market's status.
    """
    ndaq_tickers = []
    early_hour_first_iteration = True
    post_market_hour_first_iteration = True

    while True:
        mongo_client = MongoClient(MONGO_URL, tlsAllowInvalidCertificates=True)

        
        status = mongo_client.market_data.market_status.find_one({})["market_status"]
        

        if status == "OPEN":
            if not ndaq_tickers:
                logging.info("Market is open. Processing strategies.")
                ndaq_tickers = get_ndaq_100_holdings(mongo_client)

            threads = []

            for ticker in ndaq_tickers:
                thread = threading.Thread(
                    target=process_ticker, args=(ticker, mongo_client)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            logging.info("Finished processing all strategies. Waiting for 30 seconds.")
            # clear the historical database
            db = mongo_client.HistoricalDatabase
            collection = db.HistoricalDatabase
            collection.delete_many({})
            print("Successfully deleted historical database")
            time.sleep(300)

        elif status == "EARLY_HOURS":
            if early_hour_first_iteration is True:
                ndaq_tickers = get_ndaq_100_holdings(mongo_client)
                early_hour_first_iteration = False
                post_market_hour_first_iteration = True
                logging.info("Market is in early hours. Waiting for 30 seconds.")
            time.sleep(30)

        elif status == "CLOSED":
            if post_market_hour_first_iteration is True:
                early_hour_first_iteration = True
                logging.info("Market is closed. Performing post-market analysis.")
                post_market_hour_first_iteration = False
                
                # Update time delta based on the mode
                if time_delta_mode == "additive":
                    mongo_client.trading_simulator.time_delta.update_one(
                        {}, {"$inc": {"time_delta": time_delta_increment}}
                    )
                elif time_delta_mode == "multiplicative":
                    mongo_client.trading_simulator.time_delta.update_one(
                        {}, {"$mul": {"time_delta": time_delta_multiplicative}}
                    )
                elif time_delta_mode == "balanced":
                    time_delta_doc = mongo_client.trading_simulator.time_delta.find_one({})
                    if time_delta_doc:
                        time_delta = time_delta_doc["time_delta"]
                        mongo_client.trading_simulator.time_delta.update_one(
                            {}, {"$inc": {"time_delta": time_delta_balanced * time_delta}}
                        )

                # Update ranks
                update_portfolio_values(mongo_client)
                update_ranks(mongo_client)
            time.sleep(30)
        else:
            logging.error("An error occurred while checking market status. Invalid status: " + status)
            time.sleep(60)
            
        mongo_client.close()


if __name__ == "__main__":
    main()
