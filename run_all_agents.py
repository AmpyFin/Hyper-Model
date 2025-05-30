#!/usr/bin/env python3
"""
run_all_agents.py
~~~~~~~~~~~~~~~~~
Comprehensive script to run all trading agents.

This script:
1. Imports all agents from strategies package or specific class module
2. Gets ideal periods from the strategies registry
3. Fetches historical data using Tiingo client
4. Gets current prices for each ticker
5. Runs each agent's strategy method
6. Catches and reports all errors encountered

Usage:
    python run_all_agents.py [key=value ...]
    Example: python run_all_agents.py class=computer_science_agents
"""

import sys
import traceback
import logging
import importlib
import os
import glob
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Type
import pandas as pd
import numpy as np
from pathlib import Path
import inspect

def setup_logging():
    """Set up logging to console only."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Set up logging early
logger = setup_logging()

# Add the current directory to Python path for imports
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    logger.info(f"Adding current directory to Python path: {current_dir}")
    sys.path.append(current_dir)
else:
    logger.debug(f"Current directory already in Python path: {current_dir}")

# Import required modules
try:
    import strategies
    logger.info("Successfully imported strategies package")
    logger.debug(f"Strategies package location: {strategies.__file__}")
except ImportError as e:
    logger.error(f"Failed to import strategies package: {e}")
    sys.exit(1)

from registries.ideal_periods_registry import registry as STRATEGY_REGISTRY
from utils.get_historical_data import get_historical_data
from utils.get_price_data import get_current_price
from config import TIINGO_API_KEY

# Test tickers as specified
TICKERS = ["GOOGL", "AAPL", "MSFT", "REGN", "GME"]

def collect_agents(agent_class: Optional[str] = None) -> Dict[str, Type]:
    """
    Dynamically collect trading agents either from a specific class or all available agents.
    
    Parameters
    ----------
    agent_class : str, optional
        Specific class of agents to collect (e.g., 'computer_science_agents')
        If None, collects all agents using strategies.discover()
    
    Returns
    -------
    dict
        Dictionary mapping agent names to agent classes
    """
    agents = {}
    
    if agent_class:
        # Import specific class module
        try:
            # Import the module
            module = importlib.import_module(f"strategies.{agent_class}")
            logger.info(f"Loading agents from {agent_class}")
            
            # Get all Python files in the module directory
            module_dir = Path(module.__file__).parent
            agent_files = list(module_dir.glob("*_agent.py"))
            logger.info(f"Found {len(agent_files)} agent files")
            
            # Import each agent file
            for agent_file in agent_files:
                if agent_file.stem == "__init__":
                    continue
                    
                try:
                    # Import the agent module
                    agent_module = importlib.import_module(f"strategies.{agent_class}.{agent_file.stem}")
                    
                    # Find all classes in the module that have strategy() method
                    for name, obj in inspect.getmembers(agent_module, inspect.isclass):
                        if (obj.__module__ == agent_module.__name__ and 
                            hasattr(obj, 'strategy')):
                            agents[name] = obj
                            logger.info(f"Successfully loaded {name}")
                            
                except Exception as e:
                    logger.error(f"Failed to load {agent_file.stem}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error loading {agent_class}: {str(e)}")
            logger.debug(traceback.format_exc())
    else:
        # Use discover() to get all agents
        try:
            agents = strategies.discover()
            logger.info(f"Discovered {len(agents)} total agents")
        except Exception as e:
            logger.error(f"Error discovering agents: {e}")
            
    return agents

def get_orderflow_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance OHLCV data with estimated orderflow metrics when bid/ask data is not available.
    This is a fallback when we don't have actual bid/ask data.
    """
    # Estimate bid/ask spread based on volatility
    volatility = df['close'].pct_change().std()
    avg_spread = df['high'] - df['low']
    typical_spread = avg_spread.median()
    
    # Estimate bid/ask prices
    df['bid'] = df['close'] - (typical_spread / 2)
    df['ask'] = df['close'] + (typical_spread / 2)
    
    # Estimate trade direction and size
    df['trade_direction'] = np.where(df['close'] > df['open'], 1,  # Buy
                                   np.where(df['close'] < df['open'], -1,  # Sell
                                          0))  # Neutral
    
    # Classify trade sizes
    volume_quantiles = df['volume'].quantile([0.5, 0.9]).values
    df['trade_size'] = np.where(df['volume'] > volume_quantiles[1], 'large',
                               np.where(df['volume'] > volume_quantiles[0], 'medium',
                                      'small'))
    
    # Calculate price levels and volume at each level
    df['price_level'] = df['close'].round(2)
    df['level_volume'] = df.groupby('price_level')['volume'].transform('sum')
    
    # Calculate cumulative metrics
    df['buy_volume'] = df['volume'] * (df['trade_direction'] == 1)
    df['sell_volume'] = df['volume'] * (df['trade_direction'] == -1)
    df['neutral_volume'] = df['volume'] * (df['trade_direction'] == 0)
    
    # Calculate deltas
    df['volume_delta'] = df['buy_volume'] - df['sell_volume']
    df['cumulative_delta'] = df['volume_delta'].cumsum()
    
    # Calculate VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Add aggressive flow metrics
    df['aggressive_buys'] = df['volume'] * ((df['close'] >= df['ask']) & (df['trade_direction'] > 0))
    df['aggressive_sells'] = df['volume'] * ((df['close'] <= df['bid']) & (df['trade_direction'] < 0))
    df['flow_ratio'] = (df['aggressive_buys'] - df['aggressive_sells']) / df['volume']
    
    return df

class AgentTestRunner:
    """Class to run and test all trading agents."""
    
    def __init__(self, logger):
        self.errors = {}  # {agent_name: {ticker: error_info}}
        self.successful_runs = {}  # {agent_name: {ticker: result}}
        self.logger = logger
        self.agent_predictions = {}  # {agent_name: {ticker: prediction}}
        
    def get_historical_data(self, ticker: str, period_days: int) -> Optional[pd.DataFrame]:
        """Get historical data for a ticker."""
        try:
            df = get_historical_data(ticker, period_days)
            if df is not None and not df.empty:
                return get_orderflow_data(df)
            return None
        except Exception as e:
            self.logger.error(f"Error getting historical data for {ticker}: {e}")
            return None
    
    def get_current_price_for_ticker(self, ticker: str) -> Optional[float]:
        """
        Get current price for a ticker using the get_price_data utility.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
            
        Returns
        -------
        float or None
            Current price or None if error
        """
        try:
            price = get_current_price(ticker)
            if price is None:
                self.logger.warning(f"No current price available for {ticker}")
            return price
        except Exception as e:
            self.logger.error(f"Error fetching current price for {ticker}: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def _cleanup_agent_state(self):
        """Reset agent state between runs"""
        self.agent_predictions.clear()

    def run_agent_on_ticker(self, agent_name: str, agent_class: type, ticker: str) -> Dict[str, Any]:
        """
        Run a single agent on a single ticker.
        
        Parameters
        ----------
        agent_name : str
            Name of the agent
        agent_class : type
            Agent class to instantiate
        ticker : str
            Stock ticker to test
            
        Returns
        -------
        dict
            Result dictionary with success/error information
        """
        result = {
            "agent": agent_name,
            "ticker": ticker,
            "success": False,
            "error": None,
            "prediction": None,
            "time": None
        }
        
        try:
            # Get ideal period from registry
            ideal_period = STRATEGY_REGISTRY.get(agent_name, 21)  # Default to 21 days if not found
            
            self.logger.info(f"Running {agent_name} on {ticker} (period: {ideal_period} days)")
            
            # Get historical data
            historical_df = self.get_historical_data(ticker, ideal_period)
            if historical_df is None or len(historical_df) < 5:
                error_msg = f"Insufficient historical data (got {len(historical_df) if historical_df is not None else 0} rows, need at least 5)"
                result["error"] = error_msg
                self.logger.error(f"{ticker}: {error_msg}")
                return result
            
            # Instantiate agent
            start_time = datetime.now()
            agent = agent_class()
            
            # Run strategy
            prediction = agent.strategy(historical_df)
            
            # Store prediction for comparison
            if agent_name not in self.agent_predictions:
                self.agent_predictions[agent_name] = {}
            self.agent_predictions[agent_name][ticker] = prediction
            
            result["prediction"] = float(prediction) if prediction is not None else None
            result["success"] = True
            
            total_time = (datetime.now() - start_time).total_seconds()
            result["time"] = total_time
            self.logger.info(f"{ticker}: SUCCESS - prediction={prediction:.4f}, time={total_time:.2f}s")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            result["error"] = error_msg
            result["traceback"] = traceback.format_exc()
            self.logger.error(f"{ticker}: ERROR - {error_msg}")
            self.logger.debug(traceback.format_exc())
            
        return result
    
    def run_single_agent(self, agent_name: str, agent_class: type) -> Dict[str, Any]:
        """
        Run a single agent on all tickers.
        
        Parameters
        ----------
        agent_name : str
            Name of the agent
        agent_class : type
            Agent class to instantiate
            
        Returns
        -------
        dict
            Results for this agent
        """
        self.logger.info(f"\nTesting agent: {agent_name}")
        self.logger.info("=" * 60)
        
        # Reset agent state
        self._cleanup_agent_state()
        
        agent_results = {
            "agent_name": agent_name,
            "results": {},
            "successful_tickers": 0,
            "failed_tickers": 0,
            "total_time": 0
        }
        
        start_time = datetime.now()
        
        for ticker in TICKERS:
            result = self.run_agent_on_ticker(agent_name, agent_class, ticker)
            agent_results["results"][ticker] = result
            
            if result["success"]:
                agent_results["successful_tickers"] += 1
                if agent_name not in self.successful_runs:
                    self.successful_runs[agent_name] = {}
                self.successful_runs[agent_name][ticker] = result
            else:
                agent_results["failed_tickers"] += 1
                if agent_name not in self.errors:
                    self.errors[agent_name] = {}
                self.errors[agent_name][ticker] = result
        
        agent_results["total_time"] = (datetime.now() - start_time).total_seconds()
        
        # Print summary for this agent
        success_rate = (agent_results["successful_tickers"] / len(TICKERS)) * 100
        self.logger.info("\nAgent Summary:")
        self.logger.info(f"Success rate: {success_rate:.1f}% ({agent_results['successful_tickers']}/{len(TICKERS)} tickers)")
        self.logger.info(f"Total runtime: {agent_results['total_time']:.2f}s")
        
        # Show predictions if any successful
        if agent_results["successful_tickers"] > 0:
            predictions = [result["prediction"] for result in agent_results["results"].values() 
                          if result["success"] and result["prediction"] is not None]
            if predictions:
                min_pred = min(predictions)
                max_pred = max(predictions)
                avg_pred = sum(predictions) / len(predictions)
                
                # Check if all predictions are identical - only after we have all predictions
                if len(predictions) > 1:
                    first_pred = predictions[0]
                    all_identical = all(abs(p - first_pred) < 1e-6 for p in predictions[1:])
                    if all_identical:
                        error_msg = f"All predictions are identical ({first_pred:.4f}) - Agent not generating varied signals"
                        self.logger.error(error_msg)
                        
                        # Mark all runs as failed
                        for ticker, result in agent_results["results"].items():
                            if result["success"]:
                                result["success"] = False
                                result["error"] = error_msg
                                # Move from successful to errors
                                if agent_name not in self.errors:
                                    self.errors[agent_name] = {}
                                self.errors[agent_name][ticker] = result
                                # Remove from successful runs
                                if agent_name in self.successful_runs and ticker in self.successful_runs[agent_name]:
                                    del self.successful_runs[agent_name][ticker]
                        
                        # Update counters
                        agent_results["successful_tickers"] = 0
                        agent_results["failed_tickers"] = len(agent_results["results"])
                        
                        # Clean up empty successful_runs entry
                        if agent_name in self.successful_runs:
                            del self.successful_runs[agent_name]
                        
                        self.logger.error("Status updated: All runs marked as FAILED")
                        self.logger.error(f"Success rate: 0.0% (0/{len(TICKERS)} tickers)")
                    else:
                        self.logger.info(f"Prediction range: {min_pred:.4f} to {max_pred:.4f} (avg: {avg_pred:.4f})")
                else:
                    self.logger.info(f"Single prediction: {min_pred:.4f}")
        
        # Show errors if any
        if agent_results["failed_tickers"] > 0:
            self.logger.error("\nErrors:")
            for ticker, result in agent_results["results"].items():
                if not result["success"]:
                    self.logger.error(f"{ticker}: {result['error']}")
        
        return agent_results
    
    def run_all_agents(self, agent_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Run all trading agents automatically, one at a time.
        
        Parameters
        ----------
        agent_class : str, optional
            Specific class of agents to run (e.g., 'computer_science_agents')
            If None, runs all available agents
            
        Returns
        -------
        dict
            Summary of all runs
        """
        # Collect agents based on class specification
        agents = collect_agents(agent_class)
        
        class_desc = f"from {agent_class}" if agent_class else "total"
        self.logger.info(f"Found {len(agents)} agents {class_desc}")
        
        # Log which agents we found and their ideal periods
        self.logger.info("\nTrading Agents and their ideal periods:")
        for agent_name in agents.keys():
            ideal_period = STRATEGY_REGISTRY.get(agent_name, 21)
            self.logger.info(f"{agent_name}: {ideal_period} days")
        
        self.logger.info(f"\nTesting on {len(TICKERS)} tickers: {', '.join(TICKERS)}")
        
        total_runs = 0
        successful_runs = 0
        
        for agent_name, agent_class in agents.items():
            # Run the agent
            agent_results = self.run_single_agent(agent_name, agent_class)
            
            # Update totals
            total_runs += len(TICKERS)
            successful_runs += agent_results["successful_tickers"]
        
        return self._generate_summary(total_runs, successful_runs)
    
    def _generate_summary(self, total_runs: int, successful_runs: int) -> Dict[str, Any]:
        """Generate summary statistics."""
        self.logger.info("\nFINAL SUMMARY")
        self.logger.info("=" * 80)
        
        if total_runs > 0:
            success_rate = successful_runs/total_runs*100
            self.logger.info(f"Total agent-ticker combinations: {total_runs}")
            self.logger.info(f"Successful runs: {successful_runs}")
            self.logger.info(f"Failed runs: {total_runs - successful_runs}")
            self.logger.info(f"Overall success rate: {success_rate:.1f}%")
        else:
            self.logger.warning("No agents were run")
        
        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": total_runs - successful_runs,
            "success_rate": successful_runs/total_runs*100 if total_runs > 0 else 0,
            "errors": self.errors,
            "successful": self.successful_runs
        }
    
    def print_error_report(self):
        """Print detailed error report."""
        if not self.errors:
            self.logger.info("\nNo errors encountered")
            return
        
        self.logger.error(f"\nERROR REPORT ({len(self.errors)} agents with errors)")
        self.logger.error("=" * 80)
        
        # First report agents with identical predictions
        identical_pred_agents = []
        other_error_agents = []
        
        for agent_name, ticker_errors in self.errors.items():
            # Check if this is an identical prediction error
            is_identical_pred = any("identical" in result["error"].lower() 
                                  for result in ticker_errors.values())
            if is_identical_pred:
                identical_pred_agents.append(agent_name)
            else:
                other_error_agents.append(agent_name)
        
        if identical_pred_agents:
            self.logger.error("\nCRITICAL: Agents returning identical predictions:")
            for agent_name in identical_pred_agents:
                ticker_errors = self.errors[agent_name]
                pred_value = None
                for result in ticker_errors.values():
                    if result.get("prediction") is not None:
                        pred_value = result["prediction"]
                        break
                if pred_value is not None:
                    self.logger.error(f"{agent_name}: All tickers returning {pred_value:.4f}")
                else:
                    self.logger.error(f"{agent_name}: All tickers returning identical predictions (value unknown)")
            self.logger.error("These agents need immediate attention as they are not functioning properly.")
        
        if other_error_agents:
            self.logger.error("\nOther Errors:")
            for agent_name in other_error_agents:
                ticker_errors = self.errors[agent_name]
                self.logger.error(f"\n{agent_name} - {len(ticker_errors)} failed tickers")
                
                # Group errors by type
                error_groups = {}
                for ticker, result in ticker_errors.items():
                    error_type = result["error"].split(":")[0] if result["error"] else "Unknown"
                    if error_type not in error_groups:
                        error_groups[error_type] = []
                    error_groups[error_type].append((ticker, result["error"]))
                
                for error_type, ticker_list in error_groups.items():
                    self.logger.error(f"{error_type}: {len(ticker_list)} tickers")
                    for ticker, error_msg in ticker_list:
                        self.logger.error(f"  {ticker}: {error_msg}")
    
    def print_success_summary(self):
        """Print summary of successful runs."""
        if not self.successful_runs:
            self.logger.error("\nNo successful runs")
            return
        
        self.logger.info(f"\nSUCCESS SUMMARY ({len(self.successful_runs)} agents)")
        self.logger.info("=" * 80)
        
        for agent_name, ticker_results in self.successful_runs.items():
            successful_tickers = len(ticker_results)
            total_tickers = len(TICKERS)
            success_rate = successful_tickers / total_tickers * 100
            
            self.logger.info(f"\n{agent_name}:")
            self.logger.info(f"Success rate: {success_rate:.1f}% ({successful_tickers}/{total_tickers} tickers)")
            
            # Show prediction range
            predictions = [result["prediction"] for result in ticker_results.values() 
                          if result["prediction"] is not None]
            if predictions:
                min_pred = min(predictions)
                max_pred = max(predictions)
                avg_pred = sum(predictions) / len(predictions)
                
                # Add warning if prediction range is very narrow
                if len(predictions) > 1 and (max_pred - min_pred) < 0.1:
                    self.logger.warning(f"WARNING: Very narrow prediction range ({max_pred - min_pred:.4f})")
                
                self.logger.info(f"Predictions: min={min_pred:.4f}, max={max_pred:.4f}, avg={avg_pred:.4f}")
            else:
                self.logger.warning("WARNING: No valid predictions available")
            
            # Show timing info
            times = [result["time"] for result in ticker_results.values() 
                    if result["time"] is not None]
            
            if times:
                avg_time = sum(times) / len(times)
                self.logger.info(f"Average runtime: {avg_time:.3f}s")
            else:
                self.logger.warning("WARNING: Timing information not available")


def main():
    """Main function to run all trading agents."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run trading agents', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('args', nargs='*', help='Arguments in key=value format\nExample: class=computer_science_agents')
    args = parser.parse_args()
    
    # Parse key=value arguments
    arg_dict = {}
    for arg in args.args:
        try:
            key, value = arg.split('=', 1)
            arg_dict[key] = value
        except ValueError:
            logger.warning(f"Ignoring invalid argument format: {arg}")
    
    # Get agent class if specified
    agent_class = arg_dict.get('class')
    
    logger.info("AmpyFin Trading Agent Test Runner")
    logger.info("=" * 80)
    
    if agent_class:
        logger.info(f"Running agents from class: {agent_class}")
    else:
        logger.info("Running all available agents")
    
    runner = AgentTestRunner(logger)
    
    try:
        summary = runner.run_all_agents(agent_class)
        runner.print_success_summary()
        runner.print_error_report()
        
    except KeyboardInterrupt:
        logger.warning("\nTest interrupted by user")
        runner.print_error_report()
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main() 