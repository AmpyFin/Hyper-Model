"""
Euler Agent
~~~~~~~~~~
Agent implementing trading strategies based on Leonhard Euler's principles of 
differential equations, complex analysis, and the famous Euler's formula.

The agent models price movements as solutions to ordinary differential equations,
particularly focusing on oscillatory systems described by e^(ix) = cos(x) + i*sin(x).
This allows decomposition of price action into cyclical components with varying periods.

Concepts employed:
1. Euler's method for numerical integration of differential equations
2. Euler's identity and complex exponentials for cycle detection
3. Exponential moving averages inspired by e^x properties
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal
import logging

logger = logging.getLogger(__name__)

class EulerAgent:
    """
    Trading agent based on Euler's mathematical principles.
    
    Parameters
    ----------
    cycle_periods : list, default=[5, 10, 21, 42]
        Periods for cycle analysis in trading days
    alpha : float, default=0.1
        Learning rate for differential equation solver
    signal_smoothing : int, default=5
        Smoothing window for final signal
    use_complex_analysis : bool, default=True
        Whether to use complex analysis or simple cyclical decomposition
    """
    
    def __init__(
        self,
        cycle_periods: List[int] = [5, 10, 21, 42],
        alpha: float = 0.1,
        signal_smoothing: int = 5,
        use_complex_analysis: bool = True
    ):
        self.cycle_periods = cycle_periods
        self.alpha = alpha
        self.signal_smoothing = signal_smoothing
        self.use_complex_analysis = use_complex_analysis
        self.latest_signal = 0.0
        self.is_fitted = False
        self.cycle_components = {}
        
    def _euler_ode_solver(self, prices: np.ndarray) -> np.ndarray:
        """
        Apply Euler's method to solve the differential equation representing price movement
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        numpy.ndarray
            Projected price trajectory
        """
        n = len(prices)
        derivatives = np.zeros(n)
        projections = np.zeros(n)
        
        # Calculate first derivatives
        derivatives[1:] = prices[1:] - prices[:-1]
        
        # Initial condition
        projections[0] = prices[0]
        
        # Apply Euler's method: y_{n+1} = y_n + h * f(y_n)
        for i in range(1, n):
            projections[i] = projections[i-1] + self.alpha * derivatives[i-1]
            
        return projections
        
    def _complex_cycle_analysis(self, prices: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        Decompose price into cyclical components using Euler's formula e^(ix) = cos(x) + i*sin(x)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Dictionary of cycle components with their amplitude and phase
        """
        n = len(prices)
        result = {}
        
        # Detrend the data
        detrended = signal.detrend(prices)
        
        for period in self.cycle_periods:
            if n < period * 2:
                continue
                
            # Create complex exponential with this period
            t = np.arange(n)
            freq = 2 * np.pi / period
            complex_exp = np.exp(1j * freq * t)
            
            # Convolve with the price series to extract this frequency component
            component = np.convolve(detrended, complex_exp, mode='same') / n
            
            # Extract amplitude and phase
            amplitude = np.abs(component)
            phase = np.angle(component)
            
            # Store the real projection and its current strength
            real_component = amplitude * np.cos(freq * t + phase)
            current_strength = real_component[-1] / np.std(real_component) if np.std(real_component) > 0 else 0
            
            result[period] = (real_component, current_strength)
            
        return result
    
    def _simple_cycle_analysis(self, prices: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        Simplified cycle analysis without complex numbers
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Dictionary of cycle components with their amplitude and phase
        """
        n = len(prices)
        result = {}
        
        # Detrend the data
        detrended = signal.detrend(prices)
        
        for period in self.cycle_periods:
            if n < period * 2:
                continue
                
            # Use Euler's identity to create sine and cosine components
            t = np.arange(n)
            freq = 2 * np.pi / period
            
            # Find best fit sine wave for this period
            sin_wave = np.sin(freq * t)
            cos_wave = np.cos(freq * t)
            
            # Project price onto these components (like a simplified Fourier analysis)
            sin_coef = np.sum(detrended * sin_wave) / np.sum(sin_wave**2) if np.sum(sin_wave**2) > 0 else 0
            cos_coef = np.sum(detrended * cos_wave) / np.sum(cos_wave**2) if np.sum(cos_wave**2) > 0 else 0
            
            # Reconstruct the component
            component = sin_coef * sin_wave + cos_coef * cos_wave
            
            # Calculate current strength
            amplitude = np.sqrt(sin_coef**2 + cos_coef**2)
            current_strength = component[-1] / amplitude if amplitude > 0 else 0
            
            result[period] = (component, current_strength)
            
        return result
    
    def _calculate_euler_ema(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Euler-inspired exponential moving averages
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Dictionary of EMAs with different decay rates
        """
        result = {}
        n = len(prices)
        
        # Euler's number e as the base for decay rates
        for k in [0.5, 1.0, 2.0]:
            # Calculate exponential decay factor alpha
            alpha = 1 - np.exp(-k / 10)  # Maps k to a reasonable alpha range
            
            # Initialize EMA array
            ema = np.zeros(n)
            ema[0] = prices[0]
            
            # Apply EMA formula
            for i in range(1, n):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
                
            result[f'ema_{k}'] = ema
            
        return result
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        min_required_bars = max(self.cycle_periods) * 2
        if len(historical_df) < min_required_bars:
            self.is_fitted = False
            return
            
        try:
            # Use closing prices for analysis
            prices = historical_df['close'].values
            
            # Apply Euler's ODE solver
            projections = self._euler_ode_solver(prices)
            
            # Calculate cycle components
            if self.use_complex_analysis:
                self.cycle_components = self._complex_cycle_analysis(prices)
            else:
                self.cycle_components = self._simple_cycle_analysis(prices)
                
            # Calculate exponential moving averages
            emas = self._calculate_euler_ema(prices)
            
            # Generate trading signal based on all analyses
            signal_val = 0.0
            
            # 1. ODE projection signal
            price_diff = projections[-1] - prices[-1]
            ode_signal = np.sign(price_diff) * min(1.0, abs(price_diff) / (np.std(prices) * 0.1))
            signal_val += ode_signal * 0.4  # 40% weight to ODE projection
            
            # 2. Cycle component signals
            if self.cycle_components:
                cycle_signal = sum(strength for _, strength in self.cycle_components.values()) 
                cycle_signal /= len(self.cycle_components)  # Average across all cycles
                signal_val += cycle_signal * 0.4  # 40% weight to cycle analysis
            
            # 3. EMA cross signals
            if len(emas) >= 2:
                ema_slow = emas['ema_0.5'][-1]
                ema_fast = emas['ema_2.0'][-1]
                ema_signal = np.sign(ema_fast - ema_slow) * 0.3  # Scaled signal from EMA cross
                signal_val += ema_signal * 0.2  # 20% weight to EMA cross
                
            # Clip final signal to [-1, 1] range
            self.latest_signal = np.clip(signal_val, -1.0, 1.0)
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Euler Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Euler's mathematical principles
        
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
        return "Euler Agent" 