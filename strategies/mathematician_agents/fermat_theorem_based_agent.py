"""
Fermat Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Pierre de Fermat's principles of
number theory, analytic geometry, and method of finding maxima and minima.

The agent uses Fermat's principles to:
1. Identify optimal entry/exit points (maxima/minima) using tangent methods
2. Discover patterns in price movements using modular arithmetic and congruences
3. Apply principles from Fermat's Last Theorem to detect market extremes
4. Use Fermat's Little Theorem for prime cycle detection in market movements

The agent combines these approaches to identify points of equilibrium shift in markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal
import logging
import math

logger = logging.getLogger(__name__)

class FermatAgent:
    """
    Trading agent based on Fermat's mathematical principles.
    
    Parameters
    ----------
    extrema_window : int, default=14
        Window size for detecting local maxima and minima
    tangent_period : int, default=5
        Period for calculating tangent lines at extrema
    prime_cycle_test : bool, default=True
        Whether to test for prime-based market cycles
    congruence_mod : int, default=7
        Modulus for congruence analysis
    confidence_threshold : float, default=0.6
        Threshold for signal generation
    """
    
    def __init__(
        self,
        extrema_window: int = 14,
        tangent_period: int = 5,
        prime_cycle_test: bool = True,
        congruence_mod: int = 7,
        confidence_threshold: float = 0.6
    ):
        self.extrema_window = extrema_window
        self.tangent_period = tangent_period
        self.prime_cycle_test = prime_cycle_test
        self.congruence_mod = congruence_mod
        self.confidence_threshold = confidence_threshold
        self.latest_signal = 0.0
        self.is_fitted = False
        
    def _find_local_extrema(self, prices: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Find local maxima and minima using Fermat's method of tangents
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        tuple
            (maxima_indices, minima_indices)
        """
        n = len(prices)
        if n < self.extrema_window:
            return [], []
            
        # Apply Fermat's method: find where "tangent" is horizontal (derivative = 0)
        # In a discrete setting, this is where the sign of the difference changes
        diff = np.diff(prices)
        
        # Find local maxima (diff changes from positive to negative)
        maxima = []
        for i in range(1, len(diff)):
            if diff[i-1] > 0 and diff[i] <= 0:
                maxima.append(i)
                
        # Find local minima (diff changes from negative to positive)
        minima = []
        for i in range(1, len(diff)):
            if diff[i-1] < 0 and diff[i] >= 0:
                minima.append(i)
                
        return maxima, minima
    
    def _fermat_tangent_projection(self, prices: np.ndarray) -> float:
        """
        Project future price using tangent at current price
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Projected directional signal
        """
        n = len(prices)
        if n < self.tangent_period + 1:
            return 0.0
            
        # Calculate slope of tangent line over tangent_period
        recent_prices = prices[-self.tangent_period:]
        x = np.arange(len(recent_prices))
        
        # Fit a line to approximate tangent
        if len(x) > 1:
            slope, _ = np.polyfit(x, recent_prices, 1)
            
            # Normalize the slope relative to current price
            normalized_slope = slope / prices[-1] if prices[-1] != 0 else 0
            
            # Convert to a signal in [-1, 1] range
            signal = np.clip(normalized_slope * 10, -1.0, 1.0)  # Scale for reasonable range
            
            return signal
        
        return 0.0
    
    def _fermat_little_theorem_cycles(self, prices: np.ndarray) -> Dict[int, float]:
        """
        Detect cycles based on prime number modular arithmetic
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Dictionary of prime cycles with correlation scores
        """
        n = len(prices)
        if n < 20:  # Need reasonable amount of data
            return {}
            
        # Test prime number cycles according to Fermat's Little Theorem
        # a^p ≡ a (mod p) for prime p
        prime_periods = [5, 7, 11, 13, 17, 19]
        correlations = {}
        
        for p in prime_periods:
            if p >= n // 2:
                continue
                
            # Calculate how well prices repeat with period p
            correlation = 0.0
            count = 0
            
            for i in range(n - p - 1):  # Ensure we don't go out of bounds
                # Verify both index i+1 and i+p+1 are valid
                if i + 1 < n and i + p + 1 < n:
                    # Use change direction for comparison (up or down)
                    dir1 = 1 if prices[i+1] > prices[i] else (-1 if prices[i+1] < prices[i] else 0)
                    dir2 = 1 if prices[i+p+1] > prices[i+p] else (-1 if prices[i+p+1] < prices[i+p] else 0)
                    
                    # Correlation is higher when directions match
                    if dir1 == dir2:
                        correlation += 1
                    count += 1
            
            if count > 0:
                correlations[p] = (correlation / count) * 2 - 1  # Scale to [-1, 1]
            
        return correlations
    
    def _fermat_congruence_analysis(self, prices: np.ndarray) -> float:
        """
        Analyze price patterns using modular congruences
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Congruence-based signal
        """
        n = len(prices)
        if n < self.congruence_mod * 3:
            return 0.0
            
        # Analyze price changes mod congruence_mod
        changes = np.diff(prices)
        
        # Discretize the changes based on sign
        signs = np.sign(changes).astype(int)
        
        # Check if current pattern matches previous patterns at same mod position
        mod_positions = {}
        
        for i in range(len(signs)):
            mod_pos = i % self.congruence_mod
            
            if mod_pos not in mod_positions:
                mod_positions[mod_pos] = []
                
            mod_positions[mod_pos].append(signs[i])
            
        # Get current mod position - ensure it's in bounds
        # We use n-2 because we're working with the changes array which is one element shorter
        current_mod = ((n - 2) % self.congruence_mod) if n > 1 else 0
        
        # Check if there are enough samples for this mod position
        if current_mod in mod_positions and len(mod_positions[current_mod]) >= 3:
            # Get most common direction at this mod position
            directions = mod_positions[current_mod]
            pos_count = sum(1 for d in directions if d > 0)
            neg_count = sum(1 for d in directions if d < 0)
            
            total = len(directions)
            
            if total > 0:
                if pos_count > neg_count:
                    confidence = pos_count / total
                    return confidence - 0.5  # Scale to a reasonable signal
                elif neg_count > pos_count:
                    confidence = neg_count / total
                    return 0.5 - confidence  # Negative signal
            
        return 0.0
    
    def _fermat_last_theorem_extremes(self, prices: np.ndarray) -> float:
        """
        Use principles inspired by Fermat's Last Theorem to detect market extremes
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Extreme-point signal
        """
        n = len(prices)
        if n < 20:
            return 0.0
            
        # Calculate a sequence of price "powers" (inspired by Fermat's x^n + y^n = z^n)
        power_sequence = []
        for exponent in [2, 3, 4]:
            # Use normalized prices to avoid overflow
            normalized = prices / np.mean(prices)
            power_sequence.append(normalized ** exponent)
            
        # Calculate volatility ratio between different powers
        # This gets higher during market extremes
        window_size = min(10, n // 2)  # Ensure window size doesn't exceed half the data length
        volatility_2 = np.std(power_sequence[0][-window_size:])
        volatility_3 = np.std(power_sequence[1][-window_size:])
        volatility_4 = np.std(power_sequence[2][-window_size:])
        
        # Near market extremes, higher powers show increasingly more volatility
        if volatility_2 > 0:
            ratio_3_2 = volatility_3 / volatility_2
            ratio_4_3 = volatility_4 / volatility_3
            
            # When volatility grows exponentially with power, it indicates extreme
            extreme_indicator = (ratio_4_3 / ratio_3_2) - 1.0 if ratio_3_2 > 0 else 0.0
            
            # Convert to a signal: extreme_indicator > 0 suggests potential reversal
            # Direction of reversal depends on recent trend
            if abs(extreme_indicator) > 0.1:
                # Use min to ensure we don't exceed array bounds
                recent_window = min(5, n - 1)
                recent_trend = np.mean(np.diff(prices[-recent_window:]))
                return -np.sign(recent_trend) * min(abs(extreme_indicator), 1.0)
                
        return 0.0
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        min_required_bars = max(self.extrema_window, self.tangent_period, self.congruence_mod * 3)
        if len(historical_df) < min_required_bars:
            self.is_fitted = False
            return
            
        try:
            # Extract closing prices
            prices = historical_df['close'].values
            
            # 1. Find local extrema
            maxima, minima = self._find_local_extrema(prices)
            
            # 2. Calculate tangent projection
            tangent_signal = self._fermat_tangent_projection(prices)
            
            # 3. Analyze prime cycles using Fermat's Little Theorem
            cycle_signals = {}
            if self.prime_cycle_test:
                cycle_signals = self._fermat_little_theorem_cycles(prices)
            
            # 4. Congruence analysis
            congruence_signal = self._fermat_congruence_analysis(prices)
            
            # 5. Detect market extremes
            extreme_signal = self._fermat_last_theorem_extremes(prices)
            
            # Combine signals
            combined_signal = 0.0
            weight_sum = 0.0
            
            # Tangent projection (30%)
            combined_signal += tangent_signal * 0.3
            weight_sum += 0.3
            
            # Prime cycles (20%)
            if cycle_signals:
                cycle_avg_signal = sum(cycle_signals.values()) / len(cycle_signals)
                combined_signal += cycle_avg_signal * 0.2
                weight_sum += 0.2
                
            # Congruence analysis (25%)
            combined_signal += congruence_signal * 0.25
            weight_sum += 0.25
            
            # Extreme detection (25%)
            combined_signal += extreme_signal * 0.25
            weight_sum += 0.25
            
            # Normalize final signal
            if weight_sum > 0:
                self.latest_signal = combined_signal / weight_sum
                
            # Apply confidence threshold
            if abs(self.latest_signal) < self.confidence_threshold:
                self.latest_signal *= abs(self.latest_signal) / self.confidence_threshold
                
            # Ensure signal is in [-1, 1] range
            self.latest_signal = np.clip(self.latest_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Fermat Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Fermat's mathematical principles
        
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
            
        return self.latest_signal
    
    def __str__(self) -> str:
        return "Fermat Agent" 