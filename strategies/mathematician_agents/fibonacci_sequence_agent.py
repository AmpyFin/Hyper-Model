"""
Fibonacci Agent
~~~~~~~~~~~~~
Agent implementing trading strategies based on Leonardo Fibonacci's principles
of the Fibonacci sequence, the golden ratio, and related mathematical patterns.

This agent detects key Fibonacci retracement and extension levels in price action,
which often act as support/resistance in market psychology. It also uses the
Fibonacci sequence as a basis for cycle timing analysis.

Concepts employed:
1. Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
2. Fibonacci extension levels (127.2%, 161.8%, 261.8%)
3. Fibonacci time cycles for swing period analysis
4. Golden ratio (phi = 1.618...) and its inverse (0.618...) for price projections
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

class FibonacciAgent:
    """
    Trading agent based on Fibonacci levels and ratios.
    
    Parameters
    ----------
    lookback_window : int, default=100
        Window size for pattern detection
    retracement_levels : list, default=[0.236, 0.382, 0.5, 0.618, 0.786]
        Fibonacci retracement levels to monitor
    extension_levels : list, default=[1.0, 1.272, 1.618, 2.618]
        Fibonacci extension levels to monitor
    peak_prominence : float, default=0.05
        Minimum prominence for peak detection (as fraction of price)
    signal_threshold : float, default=0.15
        Minimum distance to a Fibonacci level (as fraction of level) for signal
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        retracement_levels: List[float] = [0.236, 0.382, 0.5, 0.618, 0.786],
        extension_levels: List[float] = [1.0, 1.272, 1.618, 2.618],
        peak_prominence: float = 0.05,
        signal_threshold: float = 0.15
    ):
        self.lookback_window = lookback_window
        self.retracement_levels = retracement_levels
        self.extension_levels = extension_levels
        self.peak_prominence = peak_prominence
        self.signal_threshold = signal_threshold
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Track identified swing points
        self.swing_highs = []
        self.swing_lows = []
        self.fib_levels = {}
        
    def _identify_swing_points(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify significant swing highs and lows using peak detection
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with 'high' and 'low' columns
            
        Returns
        -------
        tuple
            (swing_high_indices, swing_low_indices)
        """
        # Make a copy to avoid warnings
        df_copy = df.copy()
        
        # Ensure we have required columns
        high_col = 'high' if 'high' in df_copy.columns else 'close'
        low_col = 'low' if 'low' in df_copy.columns else 'close'
        
        # Get high and low series
        highs = df_copy[high_col].values
        lows = df_copy[low_col].values
        
        # Calculate prominence threshold based on price range
        price_range = np.max(highs) - np.min(lows)
        prominence = price_range * self.peak_prominence
        
        # Find swing highs
        high_peaks, _ = find_peaks(highs, prominence=prominence)
        
        # Find swing lows (invert the values to find troughs as peaks)
        low_peaks, _ = find_peaks(-lows, prominence=prominence)
        
        return high_peaks, low_peaks
    
    def _calculate_fibonacci_levels(
        self, 
        df: pd.DataFrame, 
        high_peaks: np.ndarray, 
        low_peaks: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data
        high_peaks : numpy.ndarray
            Indices of swing highs
        low_peaks : numpy.ndarray
            Indices of swing lows
            
        Returns
        -------
        dict
            Dictionary of Fibonacci levels for different swing patterns
        """
        levels = {}
        
        # Not enough swing points for analysis
        if len(high_peaks) < 1 or len(low_peaks) < 1:
            return levels
            
        # Get prices at swing points
        highs = df.iloc[high_peaks]['close' if 'close' in df.columns else 'high'].values
        lows = df.iloc[low_peaks]['close' if 'close' in df.columns else 'low'].values
        
        # Find the most recent swing high and low
        latest_idx = max(high_peaks.max() if len(high_peaks) > 0 else 0, 
                         low_peaks.max() if len(low_peaks) > 0 else 0)
        
        # Determine trend direction based on most recent swing (up or down)
        is_uptrend = False
        if latest_idx in high_peaks:
            is_uptrend = True
            
        # Calculate the most relevant Fibonacci levels based on recent swings
        # For uptrends: most recent swing low to most recent swing high
        # For downtrends: most recent swing high to most recent swing low
        
        if is_uptrend and len(high_peaks) > 0 and len(low_peaks) > 0:
            # Get the most recent swing high
            high_idx = np.max(high_peaks)
            high_price = df.iloc[high_idx]['close' if 'close' in df.columns else 'high']
            
            # Find the most recent swing low that comes before the high
            valid_lows = low_peaks[low_peaks < high_idx]
            if len(valid_lows) > 0:
                low_idx = np.max(valid_lows)
                low_price = df.iloc[low_idx]['close' if 'close' in df.columns else 'low']
                
                # Calculate retracement levels (downward from high)
                price_range = high_price - low_price
                retracements = {
                    f'retracement_{level:.3f}': high_price - level * price_range
                    for level in self.retracement_levels
                }
                
                # Calculate extension levels (upward from range)
                extensions = {
                    f'extension_{level:.3f}': high_price + (level - 1) * price_range
                    for level in self.extension_levels
                }
                
                levels['uptrend'] = {**retracements, **extensions, 
                                   'swing_high': high_price, 'swing_low': low_price}
                
        elif not is_uptrend and len(high_peaks) > 0 and len(low_peaks) > 0:
            # Get the most recent swing low
            low_idx = np.max(low_peaks)
            low_price = df.iloc[low_idx]['close' if 'close' in df.columns else 'low']
            
            # Find the most recent swing high that comes before the low
            valid_highs = high_peaks[high_peaks < low_idx]
            if len(valid_highs) > 0:
                high_idx = np.max(valid_highs)
                high_price = df.iloc[high_idx]['close' if 'close' in df.columns else 'high']
                
                # Calculate retracement levels (upward from low)
                price_range = high_price - low_price
                retracements = {
                    f'retracement_{level:.3f}': low_price + level * price_range
                    for level in self.retracement_levels
                }
                
                # Calculate extension levels (downward from range)
                extensions = {
                    f'extension_{level:.3f}': low_price - (level - 1) * price_range
                    for level in self.extension_levels
                }
                
                levels['downtrend'] = {**retracements, **extensions, 
                                     'swing_high': high_price, 'swing_low': low_price}
                
        return levels
    
    def _analyze_fibonacci_time_cycles(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze Fibonacci time cycles in price data
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data
            
        Returns
        -------
        dict
            Dictionary of detected cycle lengths
        """
        # Use the Fibonacci sequence for time cycle analysis
        fib_sequence = [5, 8, 13, 21, 34, 55, 89]
        
        # Initialize results
        cycle_strengths = {}
        
        # Not enough data for cycle analysis
        if len(df) < max(fib_sequence) * 2:
            return cycle_strengths
            
        # Calculate price returns
        returns = df['close'].pct_change().dropna().values
        
        # Test each Fibonacci cycle length
        for cycle_len in fib_sequence:
            if len(returns) < cycle_len * 2:
                continue
                
            # Calculate autocorrelation at this lag
            n = len(returns)
            lag_prod = returns[cycle_len:n] * returns[:n-cycle_len]
            autocorr = np.mean(lag_prod) / (np.std(returns[cycle_len:n]) * np.std(returns[:n-cycle_len]))
            
            # Store autocorrelation strength
            cycle_strengths[str(cycle_len)] = autocorr
            
        return cycle_strengths
    
    def _generate_signal_from_levels(
        self, 
        current_price: float, 
        levels: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Generate trading signal based on proximity to Fibonacci levels
        
        Parameters
        ----------
        current_price : float
            Current price
        levels : dict
            Dictionary of Fibonacci levels
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        if not levels:
            return 0.0
            
        # Initialize signal components
        retracement_signal = 0.0
        extension_signal = 0.0
        
        # Determine which trend type (uptrend or downtrend) to use
        trend_type = 'uptrend' if 'uptrend' in levels else 'downtrend'
        
        if trend_type in levels:
            trend_levels = levels[trend_type]
            
            # Extract retracement and extension levels separately
            retracement_keys = [k for k in trend_levels.keys() if 'retracement' in k]
            extension_keys = [k for k in trend_levels.keys() if 'extension' in k]
            
            # Check proximity to retracement levels
            nearest_retracement = None
            min_retracement_dist = float('inf')
            
            for key in retracement_keys:
                level_value = trend_levels[key]
                dist = abs(current_price - level_value) / level_value
                
                if dist < min_retracement_dist:
                    min_retracement_dist = dist
                    nearest_retracement = key
            
            # Check proximity to extension levels
            nearest_extension = None
            min_extension_dist = float('inf')
            
            for key in extension_keys:
                level_value = trend_levels[key]
                dist = abs(current_price - level_value) / level_value
                
                if dist < min_extension_dist:
                    min_extension_dist = dist
                    nearest_extension = key
            
            # Generate signals based on proximity to levels
            if nearest_retracement and min_retracement_dist < self.signal_threshold:
                # Extract the retracement percentage from the key name
                level_str = nearest_retracement.split('_')[1]
                level = float(level_str)
                
                # Higher retracement levels are stronger reversal signals
                # Lower retracement levels are weaker reversal signals
                strength = min(1.0, (level + 0.2) / 0.8)  # Scale to reasonable range
                
                # Direction depends on trend: in uptrend, retracements are bearish
                if trend_type == 'uptrend':
                    retracement_signal = -strength
                else:
                    retracement_signal = strength
                    
                # Scale by proximity (closer = stronger)
                proximity_factor = 1.0 - (min_retracement_dist / self.signal_threshold)
                retracement_signal *= proximity_factor
            
            # Extension levels suggest trend continuation
            if nearest_extension and min_extension_dist < self.signal_threshold:
                # Extract the extension percentage from the key name
                level_str = nearest_extension.split('_')[1]
                level = float(level_str)
                
                # Higher extension levels are stronger continuation signals
                strength = min(1.0, level / 2.0)  # Scale to reasonable range
                
                # Direction depends on trend
                if trend_type == 'uptrend':
                    extension_signal = strength
                else:
                    extension_signal = -strength
                    
                # Scale by proximity
                proximity_factor = 1.0 - (min_extension_dist / self.signal_threshold)
                extension_signal *= proximity_factor
                
        # Combine signals with some weighting
        # Retracements often have more immediate impact, so weighted higher
        combined_signal = 0.6 * retracement_signal + 0.4 * extension_signal
        
        return combined_signal
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.lookback_window:
            self.is_fitted = False
            return
            
        try:
            # Get recent data
            df_copy = historical_df.copy()
            
            # Identify swing points
            high_peaks, low_peaks = self._identify_swing_points(df_copy)
            
            # Calculate Fibonacci levels
            self.fib_levels = self._calculate_fibonacci_levels(df_copy, high_peaks, low_peaks)
            
            # Analyze Fibonacci time cycles
            time_cycles = self._analyze_fibonacci_time_cycles(df_copy)
            
            # Store swing points
            self.swing_highs = high_peaks
            self.swing_lows = low_peaks
            
            # Mark as fitted
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Fibonacci Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Fibonacci levels and patterns
        
        Parameters
        ----------
        current_price : float
            Current asset price
        historical_df : pandas.DataFrame
            Historical price data
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        # Process the data
        self.fit(historical_df)
        
        if not self.is_fitted:
            return 0.0
            
        # Generate signal from Fibonacci levels
        signal = self._generate_signal_from_levels(current_price, self.fib_levels)
        
        # Clip signal to valid range
        self.latest_signal = np.clip(signal, -1.0, 1.0)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        return "Fibonacci Agent" 