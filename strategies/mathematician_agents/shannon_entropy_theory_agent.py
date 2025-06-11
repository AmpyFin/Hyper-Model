"""
Shannon Agent
~~~~~~~~~~~~
Agent implementing trading strategies based on Claude Shannon's information theory
and communication principles.

This agent uses information entropy to quantify market uncertainty and predictability.
It models price movements as signals with varying degrees of noise, applying concepts
from Shannon's information theory to extract tradable patterns.

Key concepts:
1. Information Entropy: Measuring market uncertainty
2. Mutual Information: Quantifying relationships between price and volume/other metrics
3. Channel Capacity: Determining maximum predictable information in price movements
4. Signal-to-Noise Ratio: Separating meaningful signals from market noise
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

class ShannonAgent:
    """
    Trading agent based on Shannon's information theory principles.
    
    Parameters
    ----------
    entropy_window : int, default=20
        Window size for entropy calculations
    min_bins : int, default=5
        Minimum number of bins for histogram-based entropy calculation
    max_bins : int, default=20
        Maximum number of bins for histogram-based entropy calculation
    adapt_bins : bool, default=True
        Whether to adaptively adjust bin count based on volatility
    smoothing_window : int, default=5
        Window for smoothing entropy and information metrics
    """
    
    def __init__(
        self,
        entropy_window: int = 20,
        min_bins: int = 5,
        max_bins: int = 20,
        adapt_bins: bool = True,
        smoothing_window: int = 5
    ):
        self.entropy_window = entropy_window
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.adapt_bins = adapt_bins
        self.smoothing_window = smoothing_window
        self.latest_signal = 0.0
        self.is_fitted = False
        
    def _calculate_entropy(self, values: np.ndarray, num_bins: Optional[int] = None) -> float:
        """
        Calculate Shannon entropy of a time series
        
        Parameters
        ----------
        values : numpy.ndarray
            Time series values
        num_bins : int, optional
            Number of bins for histogram. If None, it's determined adaptively
            
        Returns
        -------
        float
            Entropy value in bits
        """
        if len(values) < 5:
            return 0.0
            
        # Filter out NaN values
        valid_values = values[~np.isnan(values)]
        if len(valid_values) < 5:
            return 0.0
            
        # Determine number of bins if not specified
        if num_bins is None:
            if self.adapt_bins:
                # Freedman-Diaconis rule for bin width
                iqr = np.percentile(valid_values, 75) - np.percentile(valid_values, 25)
                bin_width = 2 * iqr * (len(valid_values) ** (-1/3)) if iqr > 0 else 1
                
                data_range = np.max(valid_values) - np.min(valid_values)
                if data_range > 0 and bin_width > 0:
                    # Calculate number of bins, ensure it's integer and within reasonable bounds
                    num_bins = int(np.ceil(data_range / bin_width))
                    num_bins = max(self.min_bins, min(self.max_bins, num_bins))
                else:
                    num_bins = self.min_bins
            else:
                num_bins = self.min_bins
                
        # Compute histogram and probability distribution
        hist, _ = np.histogram(valid_values, bins=num_bins)
        
        # Convert counts to probabilities
        hist = hist.astype(float)
        total = np.sum(hist)
        
        if total > 0:
            probs = hist / total
            
            # Calculate entropy using only non-zero probabilities
            # to avoid log(0) which would give -inf
            non_zero_probs = probs[probs > 0]
            entropy_value = -np.sum(non_zero_probs * np.log2(non_zero_probs))
            
            return entropy_value
        else:
            return 0.0
    
    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray, num_bins: int) -> float:
        """
        Calculate mutual information between two time series
        
        Parameters
        ----------
        x : numpy.ndarray
            First time series
        y : numpy.ndarray
            Second time series
        num_bins : int
            Number of bins for histograms
            
        Returns
        -------
        float
            Mutual information in bits
        """
        if len(x) != len(y) or len(x) < 5:
            return 0.0
            
        # Filter out positions where either series has NaN
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Check if we have enough valid data points
        if len(x_valid) < 5:
            return 0.0
            
        # Calculate entropy of X and Y
        h_x = self._calculate_entropy(x_valid, num_bins)
        h_y = self._calculate_entropy(y_valid, num_bins)
        
        # Calculate joint entropy
        hist_2d, _, _ = np.histogram2d(x_valid, y_valid, bins=num_bins)
        hist_2d = hist_2d + 1e-10
        hist_2d = hist_2d / np.sum(hist_2d)
        h_xy = entropy(hist_2d.flatten(), base=2)
        
        # Mutual information
        return max(0, h_x + h_y - h_xy)
    
    def _information_metrics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate various information-theoretic metrics for price series
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with at minimum 'close' column
            
        Returns
        -------
        dict
            Dictionary of information-related metrics
        """
        # Make a copy to avoid warnings
        df_copy = df.copy()
        n = len(df_copy)
        
        # Price returns
        df_copy['returns'] = df_copy['close'].pct_change()
        
        # Arrays to store results
        price_entropy = np.zeros(n)
        return_entropy = np.zeros(n)
        volume_entropy = np.zeros(n) if 'volume' in df_copy.columns else None
        price_vol_mi = np.zeros(n) if 'volume' in df_copy.columns else None
        
        # Calculate rolling entropy
        for i in range(self.entropy_window, n):
            window = slice(i - self.entropy_window, i)
            
            # Price entropy
            price_entropy[i] = self._calculate_entropy(df_copy['close'].values[window])
            
            # Return entropy
            return_entropy[i] = self._calculate_entropy(df_copy['returns'].values[window])
            
            # Volume entropy and mutual information if available
            if 'volume' in df_copy.columns:
                volume_entropy[i] = self._calculate_entropy(df_copy['volume'].values[window])
                price_vol_mi[i] = self._calculate_mutual_information(
                    df_copy['returns'].values[window],
                    df_copy['volume'].values[window],
                    num_bins=min(self.min_bins, self.entropy_window // 4)
                )
        
        # Smooth metrics
        price_entropy = pd.Series(price_entropy).rolling(self.smoothing_window).mean().values
        return_entropy = pd.Series(return_entropy).rolling(self.smoothing_window).mean().values
        
        if 'volume' in df_copy.columns:
            volume_entropy = pd.Series(volume_entropy).rolling(self.smoothing_window).mean().values
            price_vol_mi = pd.Series(price_vol_mi).rolling(self.smoothing_window).mean().values
        
        result = {
            'price_entropy': price_entropy,
            'return_entropy': return_entropy
        }
        
        if 'volume' in df_copy.columns:
            result.update({
                'volume_entropy': volume_entropy,
                'price_vol_mi': price_vol_mi
            })
            
        return result
    
    def _calculate_signal_noise_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate signal-to-noise ratio for recent price movements
        
        Parameters
        ----------
        returns : numpy.ndarray
            Array of price returns
            
        Returns
        -------
        float
            Estimated signal-to-noise ratio
        """
        if len(returns) < self.entropy_window:
            return 0.0
            
        # Filter out NaN values first
        valid_returns = returns[~np.isnan(returns)]
        if len(valid_returns) < self.entropy_window // 2:
            return 0.0
        
        # Use moving average as proxy for signal
        window = min(len(valid_returns) // 4, 5)
        window = max(window, 1)  # Ensure window is at least 1
        
        signal = pd.Series(valid_returns).rolling(window=window).mean().values
        
        # Noise is the residual from signal
        noise = valid_returns - signal
        
        # Remove any NaN values from signal and noise
        valid_indices = ~np.isnan(signal)
        signal_clean = signal[valid_indices]
        noise_clean = noise[valid_indices]
        
        if len(signal_clean) == 0 or len(noise_clean) == 0:
            return 0.0
        
        # Calculate SNR
        signal_power = np.var(signal_clean)
        noise_power = np.var(noise_clean)
        
        if noise_power == 0 or np.isnan(noise_power) or np.isnan(signal_power):
            return 0.0  # Return neutral instead of a potentially extreme value
            
        snr = signal_power / noise_power
        return min(snr, 10.0)  # Cap at reasonable maximum
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate information theory metrics
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.entropy_window + self.smoothing_window:
            self.is_fitted = False
            return
            
        try:
            # Calculate returns
            df_copy = historical_df.copy()
            df_copy['returns'] = df_copy['close'].pct_change()
            
            # Calculate information metrics
            info_metrics = self._information_metrics(df_copy)
            
            # Get the latest values
            current_price_entropy = info_metrics['price_entropy'][-1]
            current_return_entropy = info_metrics['return_entropy'][-1]
            
            # Calculate signal-to-noise ratio
            # Filter NaN values before passing to the function
            valid_returns = df_copy['returns'].dropna().values
            snr = self._calculate_signal_noise_ratio(valid_returns)
            
            # Generate signal
            if 'volume' in df_copy.columns and 'price_vol_mi' in info_metrics:
                current_price_vol_mi = info_metrics['price_vol_mi'][-1]
                
                # When mutual information is high and return entropy is low, market is more predictable
                # Avoid division by zero or NaN
                if np.isnan(current_return_entropy) or current_return_entropy <= 0:
                    predictability = 0.0
                else:
                    predictability = current_price_vol_mi / (current_return_entropy + 0.001)
            else:
                # Without volume, use inverse of return entropy as proxy for predictability
                # Avoid division by zero or NaN
                if np.isnan(current_return_entropy) or current_return_entropy <= 0:
                    predictability = 0.0
                else:
                    predictability = 1.0 / (current_return_entropy + 0.001)
            
            # Ensure predictability is not NaN
            if np.isnan(predictability):
                predictability = 0.0
            
            # Combine predictability with recent trend and SNR
            recent_returns = df_copy['returns'].dropna().tail(5).values
            if len(recent_returns) > 0:
                recent_trend = np.mean(recent_returns)
            else:
                recent_trend = 0.0
            
            # Ensure we don't multiply by NaN
            if np.isnan(snr):
                snr = 0.0
            if np.isnan(recent_trend):
                recent_trend = 0.0
            
            # Scale trend by SNR and predictability
            trend_confidence = recent_trend * min(5, snr) * min(5, predictability)
            
            # Normalize signal to [-1, 1]
            self.latest_signal = np.clip(trend_confidence * 5, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Shannon Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Shannon's information theory
        
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
        return "Shannon Agent" 