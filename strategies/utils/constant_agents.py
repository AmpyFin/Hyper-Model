"""
Constant Agents
~~~~~~~~~~~~~
Utility agents that provide baseline and testing signals regardless of input.
These can be useful for testing, benchmarking, and creating baseline models.

The agents:
1. AlwaysBuyAgent - Always returns maximum buy signal (1.0)
2. AlwaysSellAgent - Always returns maximum sell signal (-1.0)
3. NeutralAgent - Always returns neutral signal (0.0)
4. RandomAgent - Returns random signal between -1.0 and 1.0

Input: Any DataFrame. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class AlwaysBuyAgent:
    """Agent that always returns maximum buy signal (1.0)"""
    
    def __init__(self, **kwargs):
        """Initialize agent with any kwargs for compatibility"""
        pass
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """No-op fit method for compatibility"""
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Always return maximum buy signal"""
        return 1.0
        
    def __str__(self) -> str:
        return "Always Buy Agent"


class AlwaysSellAgent:
    """Agent that always returns maximum sell signal (-1.0)"""
    
    def __init__(self, **kwargs):
        """Initialize agent with any kwargs for compatibility"""
        pass
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """No-op fit method for compatibility"""
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Always return maximum sell signal"""
        return -1.0
        
    def __str__(self) -> str:
        return "Always Sell Agent"


class NeutralAgent:
    """Agent that always returns neutral signal (0.0)"""
    
    def __init__(self, **kwargs):
        """Initialize agent with any kwargs for compatibility"""
        pass
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """No-op fit method for compatibility"""
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Always return neutral signal"""
        return 0.0
        
    def __str__(self) -> str:
        return "Neutral Agent"


class RandomAgent:
    """Agent that returns random signal between -1.0 and 1.0"""
    
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """Initialize agent with optional random seed"""
        self.rng = np.random.RandomState(seed)
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """No-op fit method for compatibility"""
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Return random signal between -1.0 and 1.0"""
        return float(self.rng.uniform(-1.0, 1.0))
        
    def __str__(self) -> str:
        return "Random Agent" 