"""
Ramanujan Agent
~~~~~~~~~~~~~
Agent implementing trading strategies based on Srinivasa Ramanujan's work on
number theory, mathematical analysis, infinite series, and continued fractions.

This agent uses Ramanujan's insights on number patterns, highly accurate approximations,
and special functions to identify hidden mathematical structures in price movements.

Key concepts:
1. Ramanujan Theta Functions: Detecting cyclic patterns and periodicities
2. Modular Forms: Identifying invariance across different market scales
3. Continued Fractions: Finding rational approximations to market behaviors
4. Mock Theta Functions: Capturing market oscillations with special functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import special
import logging
import math

logger = logging.getLogger(__name__)

class RamanujanAgent:
    """
    Trading agent based on Ramanujan's mathematical principles.
    
    Parameters
    ----------
    theta_periods : list, default=[5, 13, 21, 34, 55, 89]
        Periods for theta function cycle detection (Fibonacci-related values)
    partition_levels : int, default=5
        Number of levels for integer partition analysis
    mock_theta_order : int, default=3
        Order of mock theta function approximation
    smoothing_window : int, default=7
        Window size for smoothing indicators
    normalize_returns : bool, default=True
        Whether to normalize returns for calculations
    """
    
    def __init__(
        self,
        theta_periods: List[int] = [5, 13, 21, 34, 55, 89],
        partition_levels: int = 5,
        mock_theta_order: int = 3,
        smoothing_window: int = 7,
        normalize_returns: bool = True
    ):
        self.theta_periods = theta_periods
        self.partition_levels = partition_levels
        self.mock_theta_order = mock_theta_order
        self.smoothing_window = smoothing_window
        self.normalize_returns = normalize_returns
        self.latest_signal = 0.0
        self.is_fitted = False
        
    def _ramanujan_theta(self, q: float, n: int = 10) -> float:
        """
        Calculate Ramanujan's theta function approximation
        
        Parameters
        ----------
        q : float
            Value (0 < q < 1) representing the phase
        n : int
            Number of terms in the series
            
        Returns
        -------
        float
            Theta function value
        """
        if q >= 1.0 or q <= 0.0:
            return 0.0
        
        # Limit to reasonable value of n
        n = min(n, 50)
        
        # Ramanujan's general theta function
        # f(q) = 1 + Σ(-1)^n * (2n+1) * q^(n(n+1)/2)
        result = 1.0
        for i in range(1, n+1):
            exponent = i * (i + 1) / 2
            term = pow(-1, i) * (2 * i + 1) * pow(q, exponent)
            result += term
            
        return result
    
    def _calculate_continued_fraction(self, value: float, max_iterations: int = 20) -> List[int]:
        """
        Calculate the continued fraction expansion of a number
        
        Parameters
        ----------
        value : float
            Number to expand
        max_iterations : int
            Maximum number of iterations
            
        Returns
        -------
        list
            Coefficients of the continued fraction
        """
        if not np.isfinite(value):
            return [0]
            
        coefficients = []
        remainder = abs(value)
        
        for _ in range(max_iterations):
            integer_part = int(remainder)
            coefficients.append(integer_part)
            
            remainder = remainder - integer_part
            if abs(remainder) < 1e-10:
                break
                
            remainder = 1.0 / remainder
            
        return coefficients
    
    def _mockTheta(self, q: float, order: int = 3, terms: int = 10) -> float:
        """
        Calculate an approximation of Ramanujan's mock theta function
        
        Parameters
        ----------
        q : float
            Value (0 < q < 1)
        order : int
            Order of the mock theta function
        terms : int
            Number of terms in the series
            
        Returns
        -------
        float
            Mock theta function value
        """
        if q >= 1.0 or q <= 0.0:
            return 0.0
        
        # Limit terms to prevent overflow
        terms = min(terms, 30)
        
        # Implementation of third order mock theta function
        # f(q) = Σ q^(n^2) / (1 + q + q^2 + ... + q^n)
        if order == 3:
            result = 0.0
            for n in range(terms):
                numerator = pow(q, n*n)
                
                # Calculate denominator
                denominator = 0.0
                for i in range(n+1):
                    denominator += pow(q, i)
                
                result += numerator / (denominator + 1e-10)
            return result
        
        # Default to a simpler function for other orders
        result = 0.0
        for n in range(terms):
            result += pow(q, n*order) / (1 - pow(q, n+1))
        return result
    
    def _multiscale_analysis(self, prices: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Multiscale analysis using Ramanujan's methods
        
        Parameters
        ----------
        prices : numpy.ndarray
            Price series
            
        Returns
        -------
        dict
            Dictionary of theta functions at different scales
        """
        n = len(prices)
        result = {}
        
        # Calculate returns
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        
        # Calculate normalized returns if needed
        if self.normalize_returns:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret > 0:
                returns = (returns - mean_ret) / std_ret
        
        for period in self.theta_periods:
            if n < period * 2:
                continue
                
            # Theta function values for each window
            theta_values = np.zeros(n)
            
            # For each point, calculate theta function
            for i in range(period, n):
                # Use return series for calculation
                window = returns[i-period:i]
                
                # Map returns to q-value (0 < q < 1)
                # Use an exponential mapping to ensure valid q range
                q_values = np.array([0.5 * (1 + np.tanh(ret)) for ret in window])
                
                # Average q-value for stability
                q = np.mean(q_values)
                
                # Calculate theta function
                theta_values[i] = self._ramanujan_theta(q, n=min(period, 15))
            
            # Smooth values
            theta_values = pd.Series(theta_values).rolling(self.smoothing_window).mean().values
            
            result[period] = theta_values
            
        return result
    
    def _mock_theta_indicators(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate mock theta based indicators
        
        Parameters
        ----------
        returns : numpy.ndarray
            Return series
            
        Returns
        -------
        dict
            Dictionary of mock theta indicators
        """
        n = len(returns)
        mock_values = np.zeros(n)
        
        # Number of terms to use in mock theta function
        terms = min(20, self.mock_theta_order * 3)
        
        for i in range(self.mock_theta_order * 3, n):
            # Get the window of returns
            window = returns[i - self.mock_theta_order * 3:i]
            
            # Map returns to q values (0 < q < 1)
            q_values = 0.5 * (1 + np.tanh(window))
            
            # Average q for stability
            q = np.mean(q_values)
            
            # Calculate mock theta function
            mock_values[i] = self._mockTheta(q, self.mock_theta_order, terms)
        
        # Smooth values
        mock_values = pd.Series(mock_values).rolling(self.smoothing_window).mean().values
        
        # Calculate differentials
        mock_diff = np.zeros(n)
        mock_diff[1:] = np.diff(mock_values)
        
        # Smooth differentials
        mock_diff = pd.Series(mock_diff).rolling(self.smoothing_window).mean().values
        
        return {
            'mock_theta': mock_values,
            'mock_theta_diff': mock_diff
        }
    
    def _calculate_partitions(self, price_seq: np.ndarray) -> np.ndarray:
        """
        Calculate a sequence based on Ramanujan's partition function
        
        Parameters
        ----------
        price_seq : numpy.ndarray
            Discretized price sequence
            
        Returns
        -------
        numpy.ndarray
            Partition-based indicator
        """
        n = len(price_seq)
        partitions = np.zeros(n)
        
        # Create a discrete version of the price sequence
        # Map to integers 0 to partition_levels-1
        min_price = np.min(price_seq)
        max_price = np.max(price_seq)
        
        if max_price == min_price:
            return partitions
            
        # Discretize to integers
        discrete_prices = np.round((price_seq - min_price) / (max_price - min_price) * (self.partition_levels - 1)).astype(int)
        
        # For each point, calculate a Ramanujan-inspired partition value
        for i in range(self.partition_levels, n):
            window = discrete_prices[i - self.partition_levels:i]
            
            # Count unique partitions in the window
            unique_vals = np.unique(window)
            num_partitions = len(unique_vals)
            
            # Calculate the "partition function" P(n)
            # Approximate using Ramanujan's asymptotic formula
            x = num_partitions / self.partition_levels
            
            # Hardy-Ramanujan formula inspired calculation
            # P(n) ~ (1 / 4n√3) * e^(π√(2n/3))
            if x > 0:
                partitions[i] = (1 / (4 * x * np.sqrt(3))) * np.exp(np.pi * np.sqrt(2 * x / 3))
            else:
                partitions[i] = 0
        
        # Normalize and smooth
        if np.max(partitions) > np.min(partitions):
            partitions = (partitions - np.min(partitions)) / (np.max(partitions) - np.min(partitions))
        
        partitions = pd.Series(partitions).rolling(self.smoothing_window).mean().values
        
        return partitions
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate Ramanujan-inspired indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        max_period = max(self.theta_periods) if self.theta_periods else 0
        min_required = max(max_period * 2, self.mock_theta_order * 5, self.partition_levels * 2)
        
        if len(historical_df) < min_required:
            self.is_fitted = False
            return
            
        try:
            # Make a copy to avoid warnings
            df_copy = historical_df.copy()
            
            # Calculate returns
            df_copy['returns'] = df_copy['close'].pct_change()
            returns = df_copy['returns'].values
            
            # Replace NaN/inf with zeros
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize returns if needed
            if self.normalize_returns:
                std_ret = np.std(returns)
                if std_ret > 0:
                    returns = returns / std_ret
            
            # Multiscale theta analysis
            theta_indicators = self._multiscale_analysis(df_copy['close'].values)
            
            # Mock theta indicators
            mock_indicators = self._mock_theta_indicators(returns)
            
            # Partition function analysis
            partition_indicator = self._calculate_partitions(df_copy['close'].values)
            
            # Calculate continued fraction for recent trend
            recent_returns = returns[-min(len(returns), 10):]
            avg_recent_return = np.mean(recent_returns)
            cf_coefs = self._calculate_continued_fraction(avg_recent_return, 5)
            
            # Generate signal
            theta_signal = 0.0
            mock_signal = 0.0
            partition_signal = 0.0
            
            # Process theta indicators for signal
            for period, theta_vals in theta_indicators.items():
                if not np.isnan(theta_vals[-1]):
                    # Detect inflection points
                    recent_theta = theta_vals[-min(period, len(theta_vals)):]
                    if len(recent_theta) > 3:
                        # Calculate derivatives
                        deriv1 = np.diff(recent_theta)
                        deriv2 = np.diff(deriv1)
                        
                        # Check for inflection (second derivative sign change)
                        if len(deriv2) > 1 and deriv2[-1] * deriv2[-2] < 0:
                            # Direction depends on the first derivative
                            theta_signal += np.sign(deriv1[-1]) * (1.0 / np.sqrt(period))
            
            # Process mock theta indicators
            if not np.isnan(mock_indicators['mock_theta'][-1]):
                mock_val = mock_indicators['mock_theta'][-1]
                mock_diff = mock_indicators['mock_theta_diff'][-1]
                
                # Generate signal from mock theta
                mock_signal = np.sign(mock_diff) * min(1.0, abs(mock_diff) * 20)
            
            # Process partition indicator
            if not np.isnan(partition_indicator[-1]):
                # Compare current partition value to recent average
                recent_avg = np.mean(partition_indicator[-self.smoothing_window:])
                current = partition_indicator[-1]
                
                # Signal direction based on current vs average
                partition_signal = np.sign(current - recent_avg) * min(1.0, abs(current - recent_avg) * 5)
            
            # Combine signals
            # Weight by the fibonacci sequence inspired ratio (golden ratio approximation)
            phi = (1 + np.sqrt(5)) / 2
            combined_signal = (theta_signal / (1 + abs(theta_signal))) * (1/phi) + \
                             mock_signal * (1/phi**2) + \
                             partition_signal * (1/phi**3)
            
            # Final adjustment using continued fraction insight
            # If fraction is simple (few terms with small values), strengthen the signal
            cf_simplicity = 1.0 / (1.0 + sum(abs(c) for c in cf_coefs))
            
            # Scale the signal by continued fraction simplicity
            self.latest_signal = np.clip(combined_signal * (1 + cf_simplicity), -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Ramanujan Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Ramanujan's mathematical principles
        
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
        return "Ramanujan Agent" 