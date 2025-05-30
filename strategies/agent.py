"""
Agent Interface
~~~~~~~~~~~~~
Base interface for all trading agents.

All agents must implement the strategy method which takes historical OHLCV data
and returns a trading signal in the range [-1.0000, 1.0000].

Signal interpretation:
  -1.0000 = Strong sell signal
  -0.5000 = Weak sell signal
   0.0000 = Neutral signal
   0.5000 = Weak buy signal
   1.0000 = Strong buy signal
"""

from abc import ABC, abstractmethod
import pandas as pd

class Agent(ABC):
    """
    Abstract base class defining the interface for all trading agents.
    
    All agents must implement the strategy method which takes historical OHLCV data
    and returns a normalized trading signal.
    """
    
    @abstractmethod
    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal based on historical OHLCV data.
        
        Parameters
        ----------
        historical_df : pd.DataFrame
            Historical OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns
        -------
        float
            Trading signal in range [-1.0000, 1.0000]
            -1.0000 = Strong sell
            -0.5000 = Weak sell
             0.0000 = Neutral
             0.5000 = Weak buy
             1.0000 = Strong buy
        """
        pass
