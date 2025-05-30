import heapq
import logging
import threading
import time
from statistics import median
from datetime import datetime
from typing import Dict, List, Tuple, Any

import certifi
from pymongo import MongoClient

# Updated imports using new structure
from config import MONGO_URL
from utils.get_ndaq_100_holdings import get_ndaq_100_holdings
from utils.get_market_status import get_market_status_simple
from utils.get_historical_data import get_data
from utils.place_order import place_order, get_latest_price, get_current_positions
from strategies import discover
from registries.ideal_periods_registry import registry

# Import control parameters - you can move these to a separate control module
suggestion_heap_limit = 5.0
trade_asset_limit = 0.1  # 10% max per asset
trade_liquidity_limit = 1000  # Min cash to keep

# Global variables for trading state
buy_heap = []
suggestion_heap = []
sold = False

ca = certifi.where()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('trading_system.log'),  # Log messages to a file
        logging.StreamHandler()             # Log messages to the console
    ]
)

# Discover all available strategies
strategies = discover()
logging.info(f"Discovered {len(strategies)} strategies for trading")


def simulate_strategy(strategy_class, ticker: str, current_price: float, 
                     historical_data: List[Dict], buying_power: float, 
                     portfolio_qty: float, portfolio_value: float):
    """
    Simulate trading strategy and return decision and quantity.
    """
    try:
        # Create strategy instance
        strategy = strategy_class()
        
        # Fit the strategy with historical data
        strategy.fit(historical_data)
        
        # Predict action and quantity
        prediction = strategy.predict(
            ticker=ticker,
            current_price=current_price,
            account_cash=buying_power,
            portfolio_qty=portfolio_qty,
            total_portfolio_value=portfolio_value
        )
        
        # Parse prediction - assuming it returns a dict with action and quantity
        if isinstance(prediction, dict):
            decision = prediction.get('action', 'hold')
            quantity = prediction.get('quantity', 0)
        elif isinstance(prediction, tuple):
            decision, quantity = prediction
        else:
            decision, quantity = 'hold', 0
            
        return decision, quantity
        
    except Exception as e:
        logging.error(f"Error simulating strategy {strategy_class.__name__} for {ticker}: {e}")
        return 'hold', 0


def weighted_majority_decision_and_median_quantity(decisions_and_quantities):
    """
    Determines the majority decision (buy, sell, or hold) and returns the weighted median quantity for the chosen action.
    Groups 'strong buy' with 'buy' and 'strong sell' with 'sell'.
    Applies weights to quantities based on strategy coefficients.
    """
    buy_decisions = ["buy", "strong buy"]
    sell_decisions = ["sell", "strong sell"]

    weighted_buy_quantities = []
    weighted_sell_quantities = []
    buy_weight = 0
    sell_weight = 0
    hold_weight = 0

    # Process decisions with weights
    for decision, quantity, weight in decisions_and_quantities:
        if decision in buy_decisions:
            weighted_buy_quantities.extend([quantity])
            buy_weight += weight
        elif decision in sell_decisions:
            weighted_sell_quantities.extend([quantity])
            sell_weight += weight
        elif decision == "hold":
            hold_weight += weight

    # Determine the majority decision based on the highest accumulated weight
    if buy_weight > sell_weight and buy_weight > hold_weight:
        return (
            "buy",
            median(weighted_buy_quantities) if weighted_buy_quantities else 0,
            buy_weight,
            sell_weight,
            hold_weight,
        )
    elif sell_weight > buy_weight and sell_weight > hold_weight:
        return (
            "sell",
            median(weighted_sell_quantities) if weighted_sell_quantities else 0,
            buy_weight,
            sell_weight,
            hold_weight,
        )
    else:
        return "hold", 0, buy_weight, sell_weight, hold_weight


def get_portfolio_info(mongo_client: MongoClient) -> Tuple[float, float, float]:
    """
    Get portfolio information from MongoDB simulation.
    """
    try:
        # Get simulated portfolio value from trades database
        db = mongo_client.trades
        
        # Calculate total cash and portfolio value from positions
        positions = get_current_positions(mongo_client)
        total_position_value = 0
        
        for symbol, position_info in positions.items():
            current_price = get_latest_price(symbol)
            if current_price:
                total_position_value += position_info['quantity'] * current_price
        
        # For simulation, assume starting cash (you can adjust this)
        starting_cash = 100000.0
        current_cash = starting_cash - total_position_value  # Simplified calculation
        portfolio_value = current_cash + total_position_value
        
        return current_cash, portfolio_value, 0.0  # cash, portfolio_value, portfolio_qty
        
    except Exception as e:
        logging.error(f"Error getting portfolio info: {e}")
        return 50000.0, 100000.0, 0.0  # Default values


def process_ticker(ticker: str, mongo_client: MongoClient, strategy_to_coefficient: Dict[str, float]):
    """
    Process a single ticker for trading decisions.
    """
    global buy_heap
    global suggestion_heap
    global sold
    
    if sold is True:
        print("Sold boolean is True. Exiting process_ticker function.")
        return
        
    try:
        decisions_and_quantities = []
        current_price = None

        while current_price is None:
            try:
                current_price = get_latest_price(ticker)
            except Exception as fetch_error:
                logging.warning(
                    f"Error fetching price for {ticker}. Retrying... {fetch_error}"
                )
                break
                
        if current_price is None:
            return
            
        print(f"Current price of {ticker}: {current_price}")

        # Get portfolio information
        buying_power, portfolio_value, _ = get_portfolio_info(mongo_client)
        
        # Get current position for this ticker
        positions = get_current_positions(mongo_client)
        portfolio_qty = positions.get(ticker, {}).get('quantity', 0.0)
        print(f"Portfolio quantity for {ticker}: {portfolio_qty}")

        # Check stop loss and take profit conditions
        if ticker in positions:
            position_info = positions[ticker]
            stop_loss_price = position_info.get('stop_loss_price')
            take_profit_price = position_info.get('take_profit_price')
            
            if stop_loss_price and take_profit_price:
                if current_price <= stop_loss_price or current_price >= take_profit_price:
                    sold = True
                    print(f"Executing SELL order for {ticker} due to stop-loss or take-profit condition")
                    quantity = portfolio_qty
                    order = place_order(ticker, "SELL", quantity, mongo_client)
                    logging.info(f"Executed SELL order for {ticker}: {order}")
                    return

        # Get historical data and run strategies
        indicator_tb = mongo_client.IndicatorsDatabase
        indicator_collection = indicator_tb.Indicators

        for strategy_name, strategy_class in strategies.items():
            historical_data = None
            while historical_data is None:
                try:
                    # Get strategy info from MongoDB
                    strategy_indicator = indicator_collection.find_one({"indicator": strategy_name})
                    
                    if not strategy_indicator:
                        # Get ideal period from registry and store in MongoDB
                        ideal_period_from_registry = registry.get(strategy_name, 14)
                        indicator_collection.insert_one({
                            "indicator": strategy_name,
                            "ideal_period": ideal_period_from_registry
                        })
                        ideal_period = ideal_period_from_registry
                    else:
                        ideal_period = strategy_indicator["ideal_period"]
                    
                    historical_data = get_data(ticker, mongo_client, ideal_period)
                except Exception as fetch_error:
                    logging.warning(
                        f"Error fetching historical data for {ticker}. Retrying... {fetch_error}"
                    )
                    time.sleep(10)

            decision, quantity = simulate_strategy(
                strategy_class,
                ticker,
                current_price,
                historical_data,
                buying_power,
                portfolio_qty,
                portfolio_value,
            )
            
            print(f"Strategy: {strategy_name}, Decision: {decision}, Quantity: {quantity} for {ticker}")
            
            weight = strategy_to_coefficient.get(strategy_name, 1.0)
            decisions_and_quantities.append((decision, quantity, weight))

        # Make weighted decision
        (
            decision,
            quantity,
            buy_weight,
            sell_weight,
            hold_weight,
        ) = weighted_majority_decision_and_median_quantity(decisions_and_quantities)
        
        print(f"Ticker: {ticker}, Decision: {decision}, Quantity: {quantity}, "
              f"Weights: Buy: {buy_weight}, Sell: {sell_weight}, Hold: {hold_weight}")

        # Execute trading logic
        if (
            decision == "buy"
            and buying_power > trade_liquidity_limit
            and (((quantity + portfolio_qty) * current_price) / portfolio_value) < trade_asset_limit
        ):
            heapq.heappush(
                buy_heap,
                (
                    -(buy_weight - (sell_weight + (hold_weight * 0.5))),
                    quantity,
                    ticker,
                ),
            )
        elif decision == "sell" and portfolio_qty > 0:
            print(f"Executing SELL order for {ticker}")
            print(f"Executing quantity of {quantity} for {ticker}")
            sold = True
            quantity = max(quantity, 1)
            order = place_order(ticker, "SELL", quantity, mongo_client)
            logging.info(f"Executed SELL order for {ticker}: {order}")
        elif (
            portfolio_qty == 0.0
            and buy_weight > sell_weight
            and (((quantity + portfolio_qty) * current_price) / portfolio_value) < trade_asset_limit
            and buying_power > trade_liquidity_limit
        ):
            max_investment = portfolio_value * trade_asset_limit
            buy_quantity = min(
                int(max_investment // current_price),
                int(buying_power // current_price),
            )
            if buy_weight > suggestion_heap_limit:
                buy_quantity = max(buy_quantity, 2)
                buy_quantity = buy_quantity // 2
                print(f"Suggestions for buying for {ticker} with a weight of {buy_weight} and quantity of {buy_quantity}")
                heapq.heappush(
                    suggestion_heap,
                    (-(buy_weight - sell_weight), buy_quantity, ticker),
                )
            else:
                logging.info(f"Holding for {ticker}, no action taken.")
        else:
            logging.info(f"Holding for {ticker}, no action taken.")

    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")


def main():
    """
    Main function to control the workflow based on the market's status.
    """
    logging.info("Trading mode is live.")
    global buy_heap
    global suggestion_heap
    global sold
    
    ndaq_tickers = []
    early_hour_first_iteration = True
    post_hour_first_iteration = True
    mongo_client = MongoClient(MONGO_URL, tlsAllowInvalidCertificates=True)
    strategy_to_coefficient = {}
    sold = False
    
    while True:
        status = get_market_status_simple().lower()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Market status check at {current_time}: {status.upper()}")
        
        # Update market status in MongoDB for compatibility
        market_db = mongo_client.market_data
        market_collection = market_db.market_status
        market_collection.update_one({}, {"$set": {"market_status": status}}, upsert=True)

        if status == "open":
            if not ndaq_tickers:
                logging.info("Market is open - initializing trading session")
                ndaq_tickers = get_ndaq_100_holdings(mongo_client)
                
                # Get strategy coefficients from ranking system
                sim_db = mongo_client.trading_simulator
                rank_collection = sim_db.rank
                r_t_c_collection = sim_db.rank_to_coefficient
                
                for strategy_name in strategies.keys():
                    try:
                        rank_doc = rank_collection.find_one({"strategy": strategy_name})
                        if rank_doc:
                            rank = rank_doc["rank"]
                            coeff_doc = r_t_c_collection.find_one({"rank": rank})
                            coefficient = coeff_doc["coefficient"] if coeff_doc else 1.0
                        else:
                            coefficient = 1.0  # Default coefficient
                        strategy_to_coefficient[strategy_name] = coefficient
                    except Exception as e:
                        logging.warning(f"Could not get coefficient for {strategy_name}: {e}")
                        strategy_to_coefficient[strategy_name] = 1.0
                
                early_hour_first_iteration = False
                post_hour_first_iteration = True
            else:
                logging.info("Market is open - continuing trading operations")
            
            # Get portfolio metrics for tracking
            buying_power, portfolio_value, _ = get_portfolio_info(mongo_client)
            qqq_latest = get_latest_price("QQQ")
            spy_latest = get_latest_price("SPY")
            
            buy_heap = []
            suggestion_heap = []

            # Update portfolio tracking
            trades_db = mongo_client.trades
            portfolio_collection = trades_db.portfolio_values

            # You can adjust these baseline values
            baseline_portfolio = 50491.13
            baseline_qqq = 518.58
            baseline_spy = 591.95

            if qqq_latest and spy_latest:
                portfolio_collection.update_one(
                    {"name": "portfolio_percentage"},
                    {"$set": {"portfolio_value": (portfolio_value - baseline_portfolio) / baseline_portfolio}},
                    upsert=True
                )
                portfolio_collection.update_one(
                    {"name": "ndaq_percentage"},
                    {"$set": {"portfolio_value": (qqq_latest - baseline_qqq) / baseline_qqq}},
                    upsert=True
                )
                portfolio_collection.update_one(
                    {"name": "spy_percentage"},
                    {"$set": {"portfolio_value": (spy_latest - baseline_spy) / baseline_spy}},
                    upsert=True
                )

            # Process all tickers with threading
            threads = []
            for ticker in ndaq_tickers:
                thread = threading.Thread(
                    target=process_ticker,
                    args=(ticker, mongo_client, strategy_to_coefficient),
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Execute buy orders from heaps
            while (buy_heap or suggestion_heap) and not sold:
                try:
                    buying_power, _, _ = get_portfolio_info(mongo_client)
                    print(f"Available buying power: ${buying_power}")
                    
                    if buy_heap and buying_power > trade_liquidity_limit:
                        _, quantity, ticker = heapq.heappop(buy_heap)
                        print(f"Executing BUY order for {ticker} of quantity {quantity}")
                        order = place_order(ticker, "BUY", quantity, mongo_client)
                        logging.info(f"Executed BUY order for {ticker}: {order}")

                    elif suggestion_heap and buying_power > trade_liquidity_limit:
                        _, quantity, ticker = heapq.heappop(suggestion_heap)
                        print(f"Executing BUY order for {ticker} of quantity {quantity}")
                        order = place_order(ticker, "BUY", quantity, mongo_client)
                        logging.info(f"Executed BUY order for {ticker}: {order}")

                    time.sleep(5)  # Allow order to propagate
                    
                except Exception as e:
                    print(f"Error occurred while executing buy order due to {e}. Continuing...")
                    break

            buy_heap = []
            suggestion_heap = []
            sold = False
            print("Sleeping for 30 seconds...")
            time.sleep(30)

        elif status in ["pre_market", "early_hours"]:
            if early_hour_first_iteration:
                logging.info("Market is in pre-market/early hours - initializing")
                ndaq_tickers = get_ndaq_100_holdings(mongo_client)
                
                # Get strategy coefficients
                sim_db = mongo_client.trading_simulator
                rank_collection = sim_db.rank
                r_t_c_collection = sim_db.rank_to_coefficient
                
                for strategy_name in strategies.keys():
                    try:
                        rank_doc = rank_collection.find_one({"strategy": strategy_name})
                        if rank_doc:
                            rank = rank_doc["rank"]
                            coeff_doc = r_t_c_collection.find_one({"rank": rank})
                            coefficient = coeff_doc["coefficient"] if coeff_doc else 1.0
                        else:
                            coefficient = 1.0
                        strategy_to_coefficient[strategy_name] = coefficient
                    except:
                        strategy_to_coefficient[strategy_name] = 1.0
                
                early_hour_first_iteration = False
                post_hour_first_iteration = True
                logging.info("Market is in early hours. Waiting for 30 seconds.")
            else:
                logging.info("Market is in pre-market/early hours - waiting for market open")
            time.sleep(30)

        elif status == "closed":
            if post_hour_first_iteration:
                early_hour_first_iteration = True
                post_hour_first_iteration = False
                logging.info("Market is closed - performing post-market operations")
            else:
                logging.info("Market is closed - waiting for next trading session")
            time.sleep(30)
        else:
            logging.error(f"Unknown market status received: {status}")
            time.sleep(60)


if __name__ == "__main__":
    main()
