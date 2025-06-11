"""
Riemann Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Bernhard Riemann's work in 
complex analysis, analytic number theory, and the famous Riemann zeta function.

This agent uses complex analytic methods to identify critical points in price series,
applying Riemann's insights about the distribution of zeros and the behavior of
complex-valued functions to market data.

Key concepts:
1. Riemann Zeta Function: Analyzing price distributions and spectral properties
2. Complex Analysis: Identifying critical points in price trajectories
3. Analytic Continuation: Extending price patterns beyond observed data
4. Distribution of Zeros: Finding significant turning points in market behavior
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import special
import logging
import cmath

logger = logging.getLogger(__name__)

class RiemannAgent:
    """
    Trading agent based on Riemann's complex analytical principles.
    
    Parameters
    ----------
    critical_strip_width : float, default=0.5
        Width parameter for critical strip analysis
    zeta_terms : int, default=100
        Number of terms to use in zeta function approximation
    complex_window : int, default=20
        Window size for complex mapping of price data
    critical_point_threshold : float, default=0.2
        Threshold for identifying critical points
    smoothing_window : int, default=5
        Window size for smoothing indicators
    """
    
    def __init__(
        self,
        critical_strip_width: float = 0.5,
        zeta_terms: int = 100,
        complex_window: int = 20,
        critical_point_threshold: float = 0.2,
        smoothing_window: int = 5
    ):
        self.critical_strip_width = critical_strip_width
        self.zeta_terms = zeta_terms
        self.complex_window = complex_window
        self.critical_point_threshold = critical_point_threshold
        self.smoothing_window = smoothing_window
        self.latest_signal = 0.0
        self.is_fitted = False
        
    def _approximate_zeta(self, s: complex, terms: int = 100) -> complex:
        """
        Calculate an approximation of the Riemann zeta function
        
        Parameters
        ----------
        s : complex
            Complex argument
        terms : int
            Number of terms in the series
            
        Returns
        -------
        complex
            Approximated zeta function value
        """
        # Cap number of terms for performance
        terms = min(terms, 1000)
        
        # For Re(s) <= 0, we'd need to use the functional equation
        # We'll simply return a default value for simplicity
        if s.real <= 0:
            return complex(0, 0)
            
        # Use direct summation for Re(s) > 1
        if s.real > 1:
            result = complex(0, 0)
            for n in range(1, terms + 1):
                result += 1 / pow(complex(n), s)
            return result
            
        # For 0 < Re(s) <= 1, use Riemann's formula with analytic continuation
        # This is a simplified approximation
        result = complex(0, 0)
        for n in range(1, terms + 1):
            result += 1 / pow(complex(n), s)
            
        # Adjust with simple correction term
        # This is not the exact analytic continuation but serves our purpose
        correction = complex(1, 0) / (s - complex(1, 0))
        return result - correction
    
    def _map_to_complex(self, prices: np.ndarray, returns: np.ndarray) -> List[complex]:
        """
        Map price and return data to the complex plane
        
        Parameters
        ----------
        prices : numpy.ndarray
            Price series
        returns : numpy.ndarray
            Return series
            
        Returns
        -------
        list
            Complex numbers representing the market state
        """
        # Ensure arrays are the same length
        min_length = min(len(prices), len(returns))
        prices = prices[-min_length:]
        returns = returns[-min_length:]
        
        # Normalize price and returns
        if min_length > 1:
            norm_prices = (prices - np.mean(prices)) / (np.std(prices) + 1e-10)
            norm_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        else:
            norm_prices = prices
            norm_returns = returns
        
        # Map to complex plane: price as real part, returns as imaginary part
        complex_points = [complex(norm_prices[i], norm_returns[i]) for i in range(min_length)]
        
        return complex_points
    
    def _find_critical_points(self, complex_points: List[complex]) -> List[Tuple[int, float]]:
        """
        Find critical points in the complex representation of market data
        
        Parameters
        ----------
        complex_points : list
            Series of complex numbers representing the market
            
        Returns
        -------
        list
            Indices and strengths of critical points
        """
        n = len(complex_points)
        critical_points = []
        
        # Need at least window + 2 points
        if n < self.complex_window + 2:
            return critical_points
            
        for i in range(self.complex_window, n - 1):
            # Analyze local behavior
            window = complex_points[i-self.complex_window:i+1]
            
            # Calculate derivatives in the complex plane
            derivatives = []
            for j in range(1, len(window)):
                derivatives.append(window[j] - window[j-1])
            
            # Check for near-zero of first derivative (critical point)
            if len(derivatives) > 1:
                # Last derivative in window
                last_deriv = abs(derivatives[-1])
                
                # Average derivative magnitude in window
                avg_deriv = np.mean([abs(d) for d in derivatives[:-1]])
                
                # If last derivative is much smaller than average,
                # we might have a critical point
                if avg_deriv > 0 and last_deriv / avg_deriv < self.critical_point_threshold:
                    # Check second derivative for minimum/maximum
                    if len(derivatives) > 2:
                        # Simple second derivative
                        second_deriv = derivatives[-1] - derivatives[-2]
                        
                        # Store index, strength, and direction
                        direction = 1 if second_deriv.real > 0 else -1
                        strength = min(1.0, avg_deriv / (last_deriv + 1e-10))
                        critical_points.append((i, direction * strength))
        
        return critical_points
    
    def _critical_strip_analysis(self, returns: np.ndarray) -> float:
        """
        Analyze return distribution using Riemann's critical strip concept
        
        Parameters
        ----------
        returns : numpy.ndarray
            Series of returns
            
        Returns
        -------
        float
            Indicator value based on critical strip analysis
        """
        if len(returns) < self.complex_window:
            return 0.0
            
        # Use only recent returns
        recent_returns = returns[-self.complex_window:]
        
        # Clean NaN values
        recent_returns = recent_returns[~np.isnan(recent_returns)]
        
        if len(recent_returns) < 3:
            return 0.0
        
        # Normalize returns
        norm_returns = (recent_returns - np.mean(recent_returns)) / (np.std(recent_returns) + 1e-10)
        
        # Calculate spectral density
        # Use FFT as analogy to analyzing zeros in critical strip
        fft_values = np.fft.fft(norm_returns)
        magnitudes = np.abs(fft_values)
        phases = np.angle(fft_values)
        
        # Analyze concentration in the critical strip
        # Define critical strip as mid-frequencies
        n = len(magnitudes)
        strip_start = int(n * (0.5 - self.critical_strip_width/2))
        strip_end = int(n * (0.5 + self.critical_strip_width/2))
        
        # Ensure valid indices
        strip_start = max(1, strip_start)  # Skip DC component
        strip_end = min(n-1, strip_end)
        
        # Calculate energy in critical strip vs total
        strip_energy = np.sum(magnitudes[strip_start:strip_end+1]**2)
        total_energy = np.sum(magnitudes[1:]**2)  # Skip DC component
        
        # Calculate concentration ratio
        concentration = strip_energy / (total_energy + 1e-10)
        
        # Calculate "zero-like" behavior in the phase
        phase_changes = np.diff(phases)
        # Normalize to [-pi, pi]
        phase_changes = ((phase_changes + np.pi) % (2*np.pi)) - np.pi
        zero_like_score = np.sum(np.abs(phase_changes)) / (np.pi * len(phase_changes))
        
        # Combine metrics
        indicator = (2 * concentration - 1) * zero_like_score
        
        return np.clip(indicator, -1.0, 1.0)
    
    def _zeta_transform(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Apply a transformation inspired by the Riemann zeta function
        
        Parameters
        ----------
        prices : numpy.ndarray
            Price series
        window : int
            Window size for calculation
            
        Returns
        -------
        numpy.ndarray
            Transformed series
        """
        n = len(prices)
        result = np.zeros(n, dtype=complex)
        
        # Need sufficient data
        if n < window:
            return np.abs(result)
            
        # Calculate returns
        returns = np.zeros(n)
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-10)
            
        # For each point with sufficient history
        for i in range(window, n):
            # Get window of normalized returns
            window_returns = returns[i-window:i]
            norm_returns = (window_returns - np.mean(window_returns)) / (np.std(window_returns) + 1e-10)
            
            # Map to complex s value in critical strip (0.5 + it)
            s_value = complex(0.5, np.mean(norm_returns) * 10)
            
            # Calculate zeta function
            result[i] = self._approximate_zeta(s_value, self.zeta_terms)
        
        # Use magnitude of result
        return np.abs(result)
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate Riemann-inspired indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.complex_window * 2:
            self.is_fitted = False
            return
            
        try:
            # Make a copy to avoid warnings
            df_copy = historical_df.copy()
            
            # Extract price and calculate returns
            prices = df_copy['close'].values
            df_copy['returns'] = df_copy['close'].pct_change()
            returns = df_copy['returns'].values
            
            # Clean NaN/inf values
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Map price and returns to complex plane
            complex_points = self._map_to_complex(prices, returns)
            
            # Find critical points
            critical_points = self._find_critical_points(complex_points)
            
            # Perform critical strip analysis
            strip_indicator = self._critical_strip_analysis(returns)
            
            # Calculate zeta transform
            zeta_values = self._zeta_transform(prices, self.complex_window)
            
            # Extract recent zeta trend
            recent_zeta = zeta_values[-self.smoothing_window:]
            zeta_trend = np.mean(np.diff(recent_zeta)) if len(recent_zeta) > 1 else 0
            
            # Generate signal
            # Start with critical strip indicator
            signal = strip_indicator
            
            # Add influence from recent critical points
            for idx, strength in critical_points:
                # Only consider very recent critical points
                if idx >= len(complex_points) - self.smoothing_window:
                    # Weight by recency
                    recency_weight = 1.0 - (len(complex_points) - 1 - idx) / self.smoothing_window
                    signal += strength * recency_weight * 0.5
            
            # Add influence from zeta transform trend
            signal += np.sign(zeta_trend) * min(0.5, abs(zeta_trend) * 10)
            
            # Scale final signal to [-1, 1]
            self.latest_signal = np.clip(signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Riemann Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Riemann's complex analysis principles
        
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
        return "Riemann Agent" 