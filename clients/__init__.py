"""
Clients Package

This package contains high-level client implementations that use the API clients.
"""

import logging
import os
from typing import Optional

def setup_client_logger(client_name: str) -> logging.Logger:
    """
    Sets up a logger for a specific client that writes to logs/{client_name}.log
    
    Args:
        client_name: Name of the client (e.g., 'trading_client', 'training_client')
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure the logger
    logger = logging.getLogger(client_name)
    
    # Only add handler if the logger doesn't already have handlers
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Create file handler
        log_file = os.path.join(log_dir, f'{client_name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    return logger

from .trading_client import TradingClient

__all__ = ["TradingClient", "setup_client_logger"]

"""
Trading client implementations.
""" 