"""
Lagrange Agent
~~~~~~~~~~~~~
Agent implementing trading strategies based on Joseph-Louis Lagrange's principles
of calculus of variations, Lagrangian mechanics, and interpolation polynomials.

This agent models price movements as trajectories seeking paths of least action,
using Lagrangian mechanics concepts to identify potential energy-like accumulations
and kinetic energy-like momentum in price action.

Concepts employed:
1. Lagrange interpolation for price curve fitting and prediction
2. Lagrangian mechanics to model market forces as energy conservation systems
3. Calculus of variations for identifying optimal trading paths
4. Lagrange multipliers for constrained optimization of risk/reward
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.interpolate import lagrange
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class LagrangeAgent:
    """
    Trading agent based on Lagrangian principles of mechanics and optimization.
    
    Parameters
    ----------
    lookback_window : int, default=60
        Window size for trajectory calculations
    polynomial_degree : int, default=4
        Degree of Lagrange interpolation polynomial
    smoothing_factor : float, default=0.3
        Smoothing factor for energy calculations (0 to 1)
    prediction_horizon : int, default=5
        Number of periods to forecast ahead
    action_threshold : float, default=0.5
        Minimum threshold for signal significance
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        polynomial_degree: int = 4,
        smoothing_factor: float = 0.3,
        prediction_horizon: int = 5,
        action_threshold: float = 0.5
    ):
        self.lookback_window = lookback_window
        self.polynomial_degree = polynomial_degree
        self.smoothing_factor = smoothing_factor
        self.prediction_horizon = prediction_horizon
        self.action_threshold = action_threshold
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Store calculated trajectories and energy values
        self.interpolation_poly = None
        self.kinetic_energy = None
        self.potential_energy = None
        self.total_energy = None
        
    def _lagrange_interpolation(self, x: np.ndarray, y: np.ndarray) -> np.polynomial.polynomial.Polynomial:
        """
        Perform Lagrange polynomial interpolation
        
        Parameters
        ----------
        x : numpy.ndarray
            X-coordinates (time/index values)
        y : numpy.ndarray
            Y-coordinates (price values)
            
        Returns
        -------
        numpy.polynomial.polynomial.Polynomial
            Lagrange interpolation polynomial
        """
        # If data length exceeds polynomial degree, select points by uniform sampling
        if len(x) > self.polynomial_degree + 1:
            # Select deg+1 points evenly spaced
            indices = np.linspace(0, len(x) - 1, self.polynomial_degree + 1, dtype=int)
            x_subset = x[indices]
            y_subset = y[indices]
        else:
            x_subset = x
            y_subset = y
            
        # Create Lagrange polynomial
        poly = lagrange(x_subset, y_subset)
        
        return poly
    
    def _calculate_energy_components(
        self, 
        prices: np.ndarray, 
        polynomial: np.polynomial.polynomial.Polynomial
    ) -> Tuple[float, float, float]:
        """
        Calculate kinetic and potential energy analogs for price trajectory
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        polynomial : numpy.polynomial.polynomial.Polynomial
            Lagrange interpolation polynomial
            
        Returns
        -------
        tuple
            (kinetic_energy, potential_energy, total_energy)
        """
        # Convert polynomial to numpy polynomial for evaluation
        x = np.arange(len(prices))
        
        # Evaluate polynomial at each point (smooth trajectory)
        y_smooth = polynomial(x)
        
        # Calculate "velocity" (first derivative - price change)
        velocity = np.diff(prices)
        velocity_smooth = np.diff(y_smooth)
        
        # Calculate "acceleration" (second derivative - change in price change)
        acceleration = np.diff(velocity)
        acceleration_smooth = np.diff(velocity_smooth)
        
        # Kinetic energy: analogous to 0.5 * m * v^2 in physics
        # Using squared velocity (squared returns) as kinetic energy
        kinetic_energy = 0.5 * np.sum(velocity**2)
        kinetic_energy_smooth = 0.5 * np.sum(velocity_smooth**2)
        
        # Blend raw and smoothed kinetic energy
        blended_kinetic = ((1 - self.smoothing_factor) * kinetic_energy + 
                          self.smoothing_factor * kinetic_energy_smooth)
        
        # Potential energy: analogous to m * g * h in physics
        # Using deviation from trend as proxy for "height"/potential
        trend = np.linspace(prices[0], prices[-1], len(prices))
        deviation = prices - trend
        potential_energy = np.sum(deviation**2)
        
        # Calculate deviation from smooth trajectory as another potential energy component
        deviation_from_smooth = prices - y_smooth
        potential_energy_smooth = np.sum(deviation_from_smooth**2)
        
        # Blend raw and smoothed potential energy
        blended_potential = ((1 - self.smoothing_factor) * potential_energy + 
                            self.smoothing_factor * potential_energy_smooth)
        
        # Total energy is the sum of kinetic and potential
        total_energy = blended_kinetic + blended_potential
        
        # Normalize to make comparable across different price scales
        scale_factor = np.mean(prices)**2
        if scale_factor > 0:
            blended_kinetic /= scale_factor
            blended_potential /= scale_factor
            total_energy /= scale_factor
            
        return blended_kinetic, blended_potential, total_energy
    
    def _lagrangian_optimization(self, prices: np.ndarray) -> float:
        """
        Perform Lagrangian optimization to find optimal position
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Optimal position value [-1, 1]
        """
        # Define the objective function (negative of expected utility)
        def objective(position):
            # Constraint: position in [-1, 1]
            if position < -1 or position > 1:
                return float('inf')
                
            # Calculate expected return
            returns = np.diff(prices) / prices[:-1]
            expected_return = np.mean(returns) * position
            
            # Calculate risk (variance of position-weighted returns)
            variance = np.var(returns * position)
            
            # Sharpe-like ratio (return / risk)
            if variance == 0:
                utility = expected_return * 10  # Avoid division by zero
            else:
                utility = expected_return / np.sqrt(variance)
                
            # We want to maximize utility, so minimize negative utility
            return -utility
            
        # Optimize using scipy.optimize.minimize
        result = minimize(objective, 0, method='BFGS')
        
        # Return optimal position
        if result.success:
            return np.clip(result.x[0], -1, 1)
        else:
            return 0.0
    
    def _predict_future_trajectory(
        self, 
        polynomial: np.polynomial.polynomial.Polynomial, 
        x_current: int
    ) -> np.ndarray:
        """
        Predict future price trajectory using Lagrange interpolation
        
        Parameters
        ----------
        polynomial : numpy.polynomial.polynomial.Polynomial
            Lagrange interpolation polynomial
        x_current : int
            Current x position (index)
            
        Returns
        -------
        numpy.ndarray
            Array of predicted future values
        """
        # Generate x values for future points
        x_future = np.arange(x_current + 1, x_current + self.prediction_horizon + 1)
        
        # Evaluate polynomial at future points
        future_values = polynomial(x_future)
        
        return future_values
    
    def _analyze_trajectory_action(
        self, 
        kinetic: float, 
        potential: float, 
        predicted_values: np.ndarray, 
        current_price: float
    ) -> float:
        """
        Analyze trajectory using principles of least action
        
        Parameters
        ----------
        kinetic : float
            Kinetic energy component
        potential : float
            Potential energy component
        predicted_values : numpy.ndarray
            Predicted future values
        current_price : float
            Current price
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        # Calculate energy ratio (K/P)
        if potential == 0:
            energy_ratio = 10.0  # Arbitrary large value
        else:
            energy_ratio = kinetic / potential
            
        # Determine whether system is more kinetic (trending) or potential (mean-reverting)
        is_kinetic_dominant = energy_ratio > 1.0
        
        # Calculate predicted return
        if len(predicted_values) > 0:
            predicted_return = (predicted_values[-1] / current_price) - 1
        else:
            predicted_return = 0.0
            
        # Generate signal based on energy state and predicted trajectory
        if is_kinetic_dominant:
            # In kinetic-dominant regime, follow the trajectory momentum
            signal = np.sign(predicted_return) * min(1.0, abs(predicted_return) * 10)
        else:
            # In potential-dominant regime, expect mean reversion
            # Higher potential energy = stronger mean reversion signal
            mean_reversion_strength = min(1.0, potential)
            signal = -np.sign(predicted_return) * mean_reversion_strength
            
        # Scale by confidence based on energy magnitude
        energy_magnitude = kinetic + potential
        confidence = min(1.0, energy_magnitude / self.action_threshold)
        
        return signal * confidence
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate Lagrangian trajectory
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.lookback_window:
            self.is_fitted = False
            return
            
        try:
            # Extract recent price data
            prices = historical_df['close'].values[-self.lookback_window:]
            
            # Create x values (time index)
            x = np.arange(len(prices))
            
            # Perform Lagrange interpolation
            self.interpolation_poly = self._lagrange_interpolation(x, prices)
            
            # Calculate energy components
            self.kinetic_energy, self.potential_energy, self.total_energy = self._calculate_energy_components(
                prices, self.interpolation_poly
            )
            
            # Predict future trajectory
            predicted_values = self._predict_future_trajectory(self.interpolation_poly, len(prices)-1)
            
            # Analyze trajectory using least action principle
            self.latest_signal = self._analyze_trajectory_action(
                self.kinetic_energy, 
                self.potential_energy, 
                predicted_values, 
                prices[-1]
            )
            
            # Mark as fitted
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Lagrange Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Lagrangian trajectory analysis
        
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
        return "Lagrange Agent" 