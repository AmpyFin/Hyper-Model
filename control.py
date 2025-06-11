import os
import sys

# Import using absolute path
from registries.historical_data_clients_registry import registry as historical_data_clients_registry
from registries.latest_price_clients_registry import registry as latest_price_clients_registry
from utils.get_ndaq_100_holdings import get_ndaq_100_holdings

#Client registry for historical data
# Please note that for databento, we need to get historical data with end date at least 24 hours in the past.
trading_historical_data_client = historical_data_clients_registry["databento_client"]()

#Client registry for latest prices
trading_price_client = latest_price_clients_registry["tiingo_client"]()

# Early Hour Client - This is the client that will be used to run the early hour logic
# If tickers is not specified, we can use the NDAQ 100 tickers. Tickers will always be a list of strings.
tickers = get_ndaq_100_holdings() 

# Trading Client - This is the main client that will be used to trade.

# Ranking Client - This is the client that will be used to rank the agents

# Training Client - This is the client that will be used to train the overall model

# Testing Client - This is the client that will be used to test the overall model at certain time period

# Log Pushing Client - This is the client that will be used to push logs to the database and delete ones that are over certain period of time

# System Client - This is the client that will be used to run the overall system

# Parameter Pushing Client - This is the client that will be used to push parameters to the database if there needs to be an update

# Log pushing client
# Log expiration days - this is the number of days that logs are stored in the database. Afterwards, they are deleted.
log_expiration_days = 30








