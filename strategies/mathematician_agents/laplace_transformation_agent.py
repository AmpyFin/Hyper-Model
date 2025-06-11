"""
Laplace Agent
~~~~~~~~~~~~
Agent implementing trading strategies based on Pierre-Simon Laplace's work in
probability theory, differential equations, and the famous Laplace transform.

This agent models market prices using Laplace's deterministic view of the universe
where future states can be predicted given sufficient information about current conditions.
It uses transforms to convert time-domain signals to frequency domain for analysis.

Key concepts:
1. Laplace Transform: Converting price series to frequency domain for analysis
2. Probability Theory: Bayesian-like updating of market beliefs
3. Differential Equations: Modeling market dynamics as continuous systems
4. Laplacian Smoothing: Noise reduction using Laplace's principles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal, stats
import logging

logger = logging.getLogger(__name__)

class LaplaceAgent:
    """
    Trading agent based on Laplace's probability and transform principles.
    
    Parameters
    ----------
    transform_window : int, default=40
        Window size for Laplace transform approximation
    prior_weight : float, default=0.3
        Weight given to prior beliefs in probability updates
    differential_order : int, default=2
        Order of differential equation to fit
    smoothing_factor : float, default=0.5
        Factor for Laplacian smoothing intensity
    smoothing_window : int, default=5
        Window size for smoothing indicators
    """
    
    def __init__(
        self,
        transform_window: int = 40,
        prior_weight: float = 0.3,
        differential_order: int = 2,
        smoothing_factor: float = 0.5,
        smoothing_window: int = 5
    ):
        self.transform_window = transform_window
        self.prior_weight = prior_weight
        self.differential_order = differential_order
        self.smoothing_factor = smoothing_factor
        self.smoothing_window = smoothing_window
        self.latest_signal = 0.0
        self.is_fitted = False
        self.market_prior = 0.5  # Prior probability of market going up (neutral)
        
    def _approximate_laplace_transform(self, time_series: np.ndarray, s_values: np.ndarray) -> np.ndarray:
        """
        Calculate numerical approximation of the Laplace transform
        
        Parameters
        ----------
        time_series : numpy.ndarray
            Input time series
        s_values : numpy.ndarray
            Values of s in the Laplace domain
            
        Returns
        -------
        numpy.ndarray
            Laplace transform values at the specified s points
        """
        n = len(time_series)
        result = np.zeros(len(s_values), dtype=complex)
        
        # Define time values (normalized)
        t_values = np.linspace(0, 1, n)
        
        # For each s, calculate L{f(t)} = ∫(0→∞) f(t)e^(-st)dt
        for i, s in enumerate(s_values):
            # Calculate e^(-st)
            exp_term = np.exp(-s * t_values)
            
            # Multiply by function and integrate (using simple trapezoidal rule)
            integrand = time_series * exp_term
            result[i] = np.trapz(integrand, t_values)
            
        return result
    
    def _laplacian_smoothing(self, series: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Apply Laplacian smoothing to a time series
        
        Parameters
        ----------
        series : numpy.ndarray
            Input time series
        alpha : float
            Smoothing parameter (0 to 1)
            
        Returns
        -------
        numpy.ndarray
            Smoothed time series
        """
        n = len(series)
        
        if n < 3:
            return series.copy()
            
        # Create the Laplacian matrix
        laplacian = np.zeros((n, n))
        
        # Fill diagonal
        np.fill_diagonal(laplacian, 1.0)
        
        # Fill off-diagonal elements for neighbors
        for i in range(n-1):
            laplacian[i, i+1] = -alpha/2
            laplacian[i+1, i] = -alpha/2
            
        # Solve the linear system
        smoothed = np.linalg.solve(laplacian, series)
        
        return smoothed
    
    def _solve_differential_equation(self, prices: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Fit a differential equation to price data and project forward
        
        Parameters
        ----------
        prices : numpy.ndarray
            Price time series
            
        Returns
        -------
        tuple
            (projected prices, coefficients of differential equation)
        """
        n = len(prices)
        
        if n < self.differential_order + 2:
            return prices.copy(), [0.0] * (self.differential_order + 1)
            
        # Calculate derivatives
        derivatives = [prices]
        for i in range(1, self.differential_order + 1):
            # Calculate i-th derivative
            deriv = np.zeros(n)
            if i == 1:
                deriv[1:] = np.diff(prices)
            else:
                deriv[i:] = np.diff(derivatives[i-1], n=1)[:-i+1]
                
            derivatives.append(deriv)
        
        # Set up linear system to fit coefficients
        X = np.zeros((n - self.differential_order, self.differential_order + 1))
        y = np.zeros(n - self.differential_order)
        
        for i in range(self.differential_order, n):
            # Fill row with derivatives in reverse order
            for j in range(self.differential_order + 1):
                X[i - self.differential_order, j] = derivatives[self.differential_order - j][i]
                
            # Target is the price
            y[i - self.differential_order] = prices[i]
        
        # Solve for coefficients
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            # If the system is ill-conditioned, use simple coefficients
            coeffs = np.zeros(self.differential_order + 1)
            coeffs[0] = 1.0
        
        # Project forward using the differential equation
        projected = np.zeros(n)
        projected[:self.differential_order] = prices[:self.differential_order]
        
        for i in range(self.differential_order, n):
            # Calculate projection using the differential equation
            projection = 0.0
            for j in range(self.differential_order + 1):
                if j == 0:
                    # Constant term
                    projection += coeffs[j]
                else:
                    # Derivative terms
                    projection += coeffs[j] * derivatives[self.differential_order - j][i]
                    
            projected[i] = projection
        
        return projected, coeffs.tolist()
    
    def _update_probability(self, prior: float, evidence: float, likelihood_up: float, likelihood_down: float) -> float:
        """
        Update probability using Laplace's rule of succession (a form of Bayesian updating)
        
        Parameters
        ----------
        prior : float
            Prior probability
        evidence : float
            Evidence value (normalized to [-1, 1])
        likelihood_up : float
            Likelihood of evidence given upward movement
        likelihood_down : float
            Likelihood of evidence given downward movement
            
        Returns
        -------
        float
            Updated probability
        """
        # Convert evidence to a likelihood ratio
        if evidence > 0:
            # Positive evidence favors upward movement
            evidence_strength = abs(evidence)
            likelihood_ratio = (likelihood_up / likelihood_down) * (1 + evidence_strength)
        else:
            # Negative evidence favors downward movement
            evidence_strength = abs(evidence)
            likelihood_ratio = (likelihood_up / likelihood_down) / (1 + evidence_strength)
        
        # Bayes' rule
        posterior_odds = (prior / (1 - prior)) * likelihood_ratio
        posterior = posterior_odds / (1 + posterior_odds)
        
        # Laplace's rule of succession: Add a small weight to move towards 0.5
        succession_factor = 0.98  # Slightly less than 1 to allow for mean reversion
        posterior = succession_factor * posterior + (1 - succession_factor) * 0.5
        
        return posterior
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate Laplace-inspired indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.transform_window:
            self.is_fitted = False
            return
            
        try:
            # Make a copy to avoid warnings
            df_copy = historical_df.copy()
            
            # Extract prices and calculate returns
            prices = df_copy['close'].values
            df_copy['returns'] = df_copy['close'].pct_change()
            returns = df_copy['returns'].fillna(0).values
            
            # Apply Laplacian smoothing to price data
            smoothed_prices = self._laplacian_smoothing(prices, self.smoothing_factor)
            
            # Calculate Laplace transform for recent window
            if len(prices) >= self.transform_window:
                recent_prices = smoothed_prices[-self.transform_window:]
                
                # Normalize to [0, 1] for transform calculation
                norm_prices = (recent_prices - np.min(recent_prices)) / (np.max(recent_prices) - np.min(recent_prices) + 1e-10)
                
                # Define s values in complex plane for transform
                s_values = np.array([complex(0.1 * i, 0) for i in range(1, 5)])
                
                # Calculate approximate Laplace transform
                transform_values = self._approximate_laplace_transform(norm_prices, s_values)
                
                # Extract features from transform
                transform_magnitudes = np.abs(transform_values)
                transform_phases = np.angle(transform_values)
                
                # Differential equation approach
                projected_prices, de_coeffs = self._solve_differential_equation(smoothed_prices)
                
                # Calculate projected direction
                if len(projected_prices) > 2:
                    projection_direction = projected_prices[-1] - projected_prices[-2]
                else:
                    projection_direction = 0
                
                # Calculate different evidence terms
                
                # 1. Transform-based evidence: compare magnitudes at different frequencies
                if len(transform_magnitudes) > 1:
                    freq_ratio = transform_magnitudes[0] / (transform_magnitudes[-1] + 1e-10)
                    transform_evidence = (freq_ratio - 1.0) / (freq_ratio + 1.0)
                else:
                    transform_evidence = 0.0
                
                # 2. Differential equation evidence: based on the slope of projection
                de_evidence = np.sign(projection_direction) * min(1.0, abs(projection_direction) / (np.std(prices[-10:]) + 1e-10))
                
                # 3. Recent momentum evidence
                recent_returns = returns[-min(10, len(returns)):]
                momentum_evidence = np.tanh(5 * np.mean(recent_returns))
                
                # Update market prior using Laplace's approach
                
                # First update with transform evidence
                self.market_prior = self._update_probability(
                    self.market_prior,
                    transform_evidence,
                    0.6,  # Likelihood if market going up
                    0.4   # Likelihood if market going down
                )
                
                # Then update with differential equation evidence
                self.market_prior = self._update_probability(
                    self.market_prior,
                    de_evidence,
                    0.7,  # Likelihood if market going up
                    0.3   # Likelihood if market going down
                )
                
                # Finally update with momentum evidence
                self.market_prior = self._update_probability(
                    self.market_prior,
                    momentum_evidence,
                    0.65,  # Likelihood if market going up
                    0.35   # Likelihood if market going down
                )
                
                # Calculate final signal by converting probability to [-1, 1]
                signal = 2 * self.market_prior - 1
                
                # Modulate signal strength by conviction from differential equation
                conviction = min(1.0, abs(de_evidence) * 2)
                
                self.latest_signal = signal * conviction
            else:
                self.latest_signal = 0.0
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Laplace Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Laplace's principles
        
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
        return "Laplace Agent" 