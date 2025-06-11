"""
Gauss Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Carl Friedrich Gauss's principles
of normal distribution, statistical analysis, and the least squares method.

The agent models price movements using Gaussian statistics, measuring deviations
from normality, performing regression analysis, and detecting statistical anomalies
that may signal trading opportunities.

Concepts employed:
1. Normal distribution and standard deviations for price movement analysis
2. Method of least squares for trend detection and projection
3. Gaussian bell curve for identifying overbought/oversold conditions
4. Statistical significance testing for trade signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)

class GaussAgent:
    """
    Trading agent based on Gaussian statistical principles.
    
    Parameters
    ----------
    lookback_window : int, default=60
        Window size for statistical calculations (minimum 30 recommended)
    std_dev_threshold : float, default=2.0
        Number of standard deviations for signal generation
    regression_degree : int, default=2
        Polynomial degree for least squares regression fitting
    normalize_returns : bool, default=True
        Whether to normalize returns before statistical analysis
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        std_dev_threshold: float = 2.0,
        regression_degree: int = 2,
        normalize_returns: bool = True
    ):
        self.lookback_window = lookback_window
        self.std_dev_threshold = std_dev_threshold
        self.regression_degree = regression_degree
        self.normalize_returns = normalize_returns
        self.latest_signal = 0.0
        self.is_fitted = False
        self.gaussian_stats = {}
        
    def _calculate_gaussian_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Gaussian statistical measures from price data
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with 'close' and other columns
            
        Returns
        -------
        dict
            Dictionary with statistical measures
        """
        # Make a copy to avoid warnings
        df_copy = df.copy()
        
        # Calculate returns
        if 'returns' not in df_copy.columns:
            df_copy['returns'] = df_copy['close'].pct_change()
            
        # Get recent data based on lookback window
        recent_data = df_copy.iloc[-self.lookback_window:]
        recent_returns = recent_data['returns'].dropna().values
        
        if len(recent_returns) < self.lookback_window / 2:
            return {}
            
        # Normalize returns if requested (convert to z-scores)
        if self.normalize_returns:
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            if std_return > 0:
                recent_returns = (recent_returns - mean_return) / std_return
                
        # Basic Gaussian statistics
        mean = np.mean(recent_returns)
        median = np.median(recent_returns)
        std_dev = np.std(recent_returns)
        skew = stats.skew(recent_returns)
        kurtosis = stats.kurtosis(recent_returns)
        
        # Calculate latest z-score (how many standard deviations from mean)
        latest_return = recent_returns[-1]
        z_score = (latest_return - mean) / std_dev if std_dev > 0 else 0
        
        # Test for normality (Shapiro-Wilk test)
        shapiro_test = stats.shapiro(recent_returns)
        is_normal = shapiro_test.pvalue > 0.05  # p > 0.05 suggests normality
        
        # Percentile of latest return
        percentile = stats.percentileofscore(recent_returns, latest_return)
        
        # Probability based on cumulative density function
        cdf = stats.norm.cdf(z_score)
        
        return {
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'skew': skew,
            'kurtosis': kurtosis,
            'z_score': z_score,
            'is_normal': is_normal,
            'latest_return': latest_return,
            'percentile': percentile,
            'cdf': cdf
        }
    
    def _least_squares_regression(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform polynomial least squares regression on price data
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with 'close' column
            
        Returns
        -------
        tuple
            (x_values, predicted_prices, price_projection)
        """
        # Get recent close prices
        recent_data = df.iloc[-self.lookback_window:]
        y = recent_data['close'].values
        x = np.arange(len(y))
        
        # Fit polynomial using least squares method
        coeffs = np.polyfit(x, y, self.regression_degree)
        poly = np.poly1d(coeffs)
        
        # Generate model predictions
        y_pred = poly(x)
        
        # Project next value
        next_x = len(y)
        next_price = poly(next_x)
        
        return x, y_pred, next_price
    
    def _calculate_deviation_signal(self, stats: Dict[str, float]) -> float:
        """
        Generate signal based on deviation from normal distribution
        
        Parameters
        ----------
        stats : dict
            Dictionary with Gaussian statistics
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        if not stats or 'z_score' not in stats:
            return 0.0
            
        z_score = stats['z_score']
        
        # If z-score exceeds threshold, generate mean-reversion signal
        if abs(z_score) > self.std_dev_threshold:
            # Deeper into the tails of the distribution = stronger signal
            signal_strength = min(1.0, abs(z_score) / (self.std_dev_threshold * 1.5))
            # Negative z-score (below mean) = buy signal, positive = sell signal
            return -np.sign(z_score) * signal_strength
            
        return 0.0
    
    def _calculate_regression_signal(self, current_price: float, regression_projection: float) -> float:
        """
        Generate signal based on least squares regression projection
        
        Parameters
        ----------
        current_price : float
            Current asset price
        regression_projection : float
            Projected price from regression
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        if current_price <= 0 or regression_projection <= 0:
            return 0.0
            
        # Calculate percent difference
        pct_diff = (regression_projection / current_price) - 1
        
        # Scale to signal range
        signal = np.clip(pct_diff * 5, -1.0, 1.0)  # Scaling factor of 5 for reasonable values
        
        return signal
    
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
            # Calculate Gaussian statistics
            self.gaussian_stats = self._calculate_gaussian_stats(historical_df)
            
            # Perform least squares regression
            _, _, price_projection = self._least_squares_regression(historical_df)
            
            # Calculate signals from different methods
            deviation_signal = self._calculate_deviation_signal(self.gaussian_stats)
            
            current_price = historical_df['close'].iloc[-1]
            regression_signal = self._calculate_regression_signal(current_price, price_projection)
            
            # Combine signals (70% weight to deviation, 30% to regression)
            combined_signal = 0.7 * deviation_signal + 0.3 * regression_signal
            
            # Special case: if distribution doesn't look normal, reduce signal strength
            if 'is_normal' in self.gaussian_stats and not self.gaussian_stats['is_normal']:
                combined_signal *= 0.7  # Reduce confidence when not normally distributed
                
            # Special case: if high kurtosis (fat tails), we might be in a regime change
            if 'kurtosis' in self.gaussian_stats and abs(self.gaussian_stats['kurtosis']) > 3.0:
                # Emphasize regression over mean reversion during potential regime changes
                combined_signal = 0.3 * deviation_signal + 0.7 * regression_signal
                
            # Special case: high skew may indicate trending market
            if 'skew' in self.gaussian_stats and abs(self.gaussian_stats['skew']) > 1.0:
                # Skewed distribution - reduce mean reversion strength
                combined_signal = 0.5 * deviation_signal + 0.5 * regression_signal
                
            # Clip final signal to [-1, 1]
            self.latest_signal = np.clip(combined_signal, -1.0, 1.0)
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Gauss Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Gaussian statistical principles
        
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
        return "Gauss Agent" 