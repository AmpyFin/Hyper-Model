"""
Newton Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Isaac Newton's principles of calculus
and laws of motion.

This agent models price movements as if they follow Newton's laws of motion:
1. Law of Inertia: A trend in motion continues unless acted upon by an external force
2. Law of Force: Acceleration/deceleration of price is proportional to the force (volume)
3. Law of Action/Reaction: For every price action, there is an equal and opposite reaction

The agent uses calculus concepts (derivatives) to measure rate of change,
acceleration, and jerk (rate of change of acceleration) to predict price reversals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class NewtonAgent:
    """
    Trading agent based on Newton's calculus and physics principles.
    
    Parameters
    ----------
    velocity_window : int, default=10
        Window size for calculating price velocity (1st derivative)
    acceleration_window : int, default=5
        Window size for calculating price acceleration (2nd derivative)
    jerk_window : int, default=3
        Window size for calculating price jerk (3rd derivative)
    force_factor : float, default=0.01
        Factor to weight the impact of volume on price "force"
    signal_threshold : float, default=0.5
        Minimum threshold for signal significance
    """
    
    def __init__(
        self,
        velocity_window: int = 10,
        acceleration_window: int = 5,
        jerk_window: int = 3,
        force_factor: float = 0.01,
        signal_threshold: float = 0.5
    ):
        self.velocity_window = velocity_window
        self.acceleration_window = acceleration_window
        self.jerk_window = jerk_window
        self.force_factor = force_factor
        self.signal_threshold = signal_threshold
        self.latest_signal = 0.0
        self.is_fitted = False
        
    def _calculate_derivatives(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate the derivatives of price movement (velocity, acceleration, jerk)
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with 'close' column
            
        Returns
        -------
        dict
            Dictionary with arrays for velocity, acceleration, jerk
        """
        # Make a copy to avoid warnings
        df_copy = df.copy()
        
        # Calculate velocity (first derivative - rate of change)
        df_copy['velocity'] = df_copy['close'].diff(periods=1)
        
        # Calculate acceleration (second derivative - rate of change of velocity)
        df_copy['acceleration'] = df_copy['velocity'].diff(periods=1)
        
        # Calculate jerk (third derivative - rate of change of acceleration)
        df_copy['jerk'] = df_copy['acceleration'].diff(periods=1)
        
        # Calculate force (volume-weighted acceleration)
        if 'volume' in df_copy.columns:
            # Normalize volume
            mean_volume = df_copy['volume'].rolling(window=self.velocity_window).mean()
            relative_volume = df_copy['volume'] / mean_volume
            
            # Force = mass (volume) * acceleration
            df_copy['force'] = relative_volume * df_copy['acceleration'] * self.force_factor
        else:
            df_copy['force'] = df_copy['acceleration']
        
        # Calculate smoothed derivatives
        df_copy['smooth_velocity'] = df_copy['velocity'].rolling(window=self.velocity_window).mean()
        df_copy['smooth_accel'] = df_copy['acceleration'].rolling(window=self.acceleration_window).mean()
        df_copy['smooth_jerk'] = df_copy['jerk'].rolling(window=self.jerk_window).mean()
        
        return {
            'velocity': df_copy['smooth_velocity'].values,
            'acceleration': df_copy['smooth_accel'].values,
            'jerk': df_copy['smooth_jerk'].values,
            'force': df_copy['force'].values
        }
    
    def _detect_turning_points(self, derivatives: Dict[str, np.ndarray]) -> Tuple[bool, bool, float]:
        """
        Detect potential turning points using Newton's laws of motion
        
        Parameters
        ----------
        derivatives : dict
            Dictionary with calculated derivatives
            
        Returns
        -------
        tuple
            (is_reversal_up, is_reversal_down, confidence)
        """
        # Get the latest values
        velocity = derivatives['velocity'][-1] if len(derivatives['velocity']) > 0 else 0
        acceleration = derivatives['acceleration'][-1] if len(derivatives['acceleration']) > 0 else 0
        jerk = derivatives['jerk'][-1] if len(derivatives['jerk']) > 0 else 0
        force = derivatives['force'][-1] if len(derivatives['force']) > 0 else 0
        
        # Check for potential upward reversal (negative velocity, positive acceleration)
        is_reversal_up = velocity < 0 and acceleration > 0 and jerk > 0
        
        # Check for potential downward reversal (positive velocity, negative acceleration)
        is_reversal_down = velocity > 0 and acceleration < 0 and jerk < 0
        
        # Calculate confidence based on magnitudes
        confidence = min(1.0, abs(force) / self.signal_threshold)
        
        return is_reversal_up, is_reversal_down, confidence
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < max(self.velocity_window, self.acceleration_window, self.jerk_window) + 5:
            self.is_fitted = False
            return
            
        try:
            # Calculate all derivatives
            derivatives = self._calculate_derivatives(historical_df)
            
            # Detect turning points
            is_reversal_up, is_reversal_down, confidence = self._detect_turning_points(derivatives)
            
            # Generate signal
            if is_reversal_up:
                self.latest_signal = confidence
            elif is_reversal_down:
                self.latest_signal = -confidence
            else:
                # Law of Inertia: continue with the trend
                self.latest_signal = np.sign(derivatives['velocity'][-1]) * 0.3
                
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Newton Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Newton's physics principles
        
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
        return "Newton Agent" 