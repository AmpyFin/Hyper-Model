"""
Bernoulli Agent
~~~~~~~~~~~~~
Agent implementing trading strategies based on Jacob Bernoulli's work on 
probability theory, particularly the Bernoulli distribution, law of large numbers,
and applications of combinatorial analysis.

This agent models market returns as a series of Bernoulli trials, with strategies
that adapt to changing probabilities of success and failure over time.

Key concepts:
1. Bernoulli Distribution: Modeling binary market outcomes (up/down)
2. Law of Large Numbers: Using statistical convergence for edge detection
3. Combinatorial Analysis: Finding probable patterns in price series
4. Golden Ratio: Applying Bernoulli's work on logarithmic spirals to market cycles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)

class BernoulliAgent:
    """
    Trading agent based on Jacob Bernoulli's probability principles.
    
    Parameters
    ----------
    bernoulli_window : int, default=50
        Window size for probability estimation
    confidence_level : float, default=0.95
        Statistical confidence level for signal generation
    pattern_length : int, default=5
        Length of patterns to analyze for combinatorial analysis
    golden_ratio_factor : float, default=0.618
        Weighting factor based on golden ratio (phi^-1)
    smoothing_window : int, default=8
        Window size for smoothing indicators
    """
    
    def __init__(
        self,
        bernoulli_window: int = 50,
        confidence_level: float = 0.95,
        pattern_length: int = 5,
        golden_ratio_factor: float = 0.618,
        smoothing_window: int = 8
    ):
        self.bernoulli_window = bernoulli_window
        self.confidence_level = confidence_level
        self.pattern_length = pattern_length
        self.golden_ratio_factor = golden_ratio_factor
        self.smoothing_window = smoothing_window
        self.latest_signal = 0.0
        self.is_fitted = False
        
    def _bernoulli_probability(self, binary_series: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Bernoulli success probability and confidence interval
        
        Parameters
        ----------
        binary_series : numpy.ndarray
            Binary series (0s and 1s) of market outcomes
            
        Returns
        -------
        tuple
            (probability, margin of error)
        """
        # Count successes
        successes = np.sum(binary_series)
        n = len(binary_series)
        
        if n == 0:
            return 0.5, 1.0
            
        # Calculate probability
        p = successes / n
        
        # Calculate confidence interval using normal approximation
        # Valid when n*p*(1-p) > 5
        if n * p * (1 - p) > 5:
            z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            margin_of_error = z * math.sqrt(p * (1 - p) / n)
        else:
            # Use wider margin when approximation isn't valid
            margin_of_error = 0.5
            
        return p, margin_of_error
    
    def _binomial_test(self, binary_series: np.ndarray, expected_p: float = 0.5) -> Tuple[float, float]:
        """
        Perform binomial test to detect deviations from expected probability
        
        Parameters
        ----------
        binary_series : numpy.ndarray
            Binary series (0s and 1s) of market outcomes
        expected_p : float, default=0.5
            Expected probability of success
            
        Returns
        -------
        tuple
            (p-value, effect direction)
        """
        successes = np.sum(binary_series)
        n = len(binary_series)
        
        if n == 0:
            return 1.0, 0.0
            
        # Observe probability
        observed_p = successes / n
        
        # Calculate p-value for two-sided test
        # Custom implementation of binomial test since stats.binom_test may not be available in all scipy versions
        
        # Use cumulative distribution function to compute p-value
        if observed_p <= expected_p:
            p_value = 2.0 * stats.binom.cdf(successes, n, expected_p)
        else:
            p_value = 2.0 * (1.0 - stats.binom.cdf(successes - 1, n, expected_p))
            
        # Ensure p-value is in [0, 1]
        p_value = np.clip(p_value, 0.0, 1.0)
        
        # Direction of effect
        effect_direction = 1.0 if observed_p > expected_p else -1.0
        
        return p_value, effect_direction
    
    def _pattern_analysis(self, binary_series: np.ndarray) -> Dict[str, float]:
        """
        Analyze patterns in binary series using combinatorial analysis
        
        Parameters
        ----------
        binary_series : numpy.ndarray
            Binary series (0s and 1s) of market outcomes
            
        Returns
        -------
        dict
            Dictionary of pattern probabilities
        """
        n = len(binary_series)
        
        if n < self.pattern_length + 1:
            return {'': 0.0}
            
        # Extract all patterns of specified length
        patterns = []
        for i in range(n - self.pattern_length + 1):
            pattern = ''.join(map(str, binary_series[i:i+self.pattern_length].astype(int)))
            patterns.append(pattern)
            
        # Count pattern occurrences
        pattern_counts = Counter(patterns)
        
        # Calculate conditional probabilities
        conditional_probs = {}
        
        for pattern in pattern_counts:
            if len(pattern) > 0:
                # For each pattern, what's the probability of 1 following it
                next_is_one = 0
                pattern_occurrences = 0
                
                for i in range(n - self.pattern_length):
                    current_pattern = ''.join(map(str, binary_series[i:i+self.pattern_length].astype(int)))
                    if current_pattern == pattern:
                        pattern_occurrences += 1
                        if i + self.pattern_length < n and binary_series[i + self.pattern_length] == 1:
                            next_is_one += 1
                
                if pattern_occurrences > 0:
                    conditional_probs[pattern] = next_is_one / pattern_occurrences
                else:
                    conditional_probs[pattern] = 0.5
        
        return conditional_probs
    
    def _golden_ratio_cycles(self, series: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Detect cycles based on golden ratio (from Bernoulli's spiral work)
        
        Parameters
        ----------
        series : numpy.ndarray
            Input time series
            
        Returns
        -------
        dict
            Dictionary of cycle indicators at different golden-ratio scales
        """
        n = len(series)
        result = {}
        
        # Golden ratio is approximately 1.618
        # We'll use powers of the golden ratio to define cycles
        phi = (1 + math.sqrt(5)) / 2
        
        # Base cycle lengths
        base_periods = [5, 8, 13, 21, 34]  # Fibonacci sequence
        
        for period in base_periods:
            if n < period * 2:
                continue
                
            # Calculate offset to test golden ratio property
            offset = int(period * phi) % period
            if offset == 0:
                offset = 1
                
            # Calculate correlation between points separated by this period
            correlation = np.zeros(n)
            
            for i in range(period, n):
                # Correlation between current point and one period ago
                if i >= period:
                    correlation[i] = np.sign(series[i] - series[i-period])
                    
                # Golden ratio property: Check correlation with offset
                if i >= period + offset:
                    # If golden ratio property holds, correlation at offset should match
                    correlation[i] *= np.sign(series[i-offset] - series[i-period-offset])
            
            # Smooth correlation signal
            smoothed = pd.Series(correlation).rolling(self.smoothing_window).mean().values
            
            result[period] = smoothed
            
        return result
    
    def _bernoulli_significance(self, p: float, margin: float, threshold: float = 0.5) -> float:
        """
        Calculate significance of deviation from threshold probability
        
        Parameters
        ----------
        p : float
            Estimated probability
        margin : float
            Margin of error
        threshold : float, default=0.5
            Threshold probability (usually 0.5 for coin flip)
            
        Returns
        -------
        float
            Significance score in range [-1, 1]
        """
        # No signal if p is within margin of threshold
        if abs(p - threshold) <= margin:
            return 0.0
            
        # Calculate how many "margin" units we are away from threshold
        distance = (p - threshold) / margin if margin > 0 else 0
        
        # Scale to [-1, 1] range with sigmoid-like function
        significance = 2 / (1 + math.exp(-distance)) - 1
        
        return significance
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate Bernoulli-inspired indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.bernoulli_window:
            self.is_fitted = False
            return
            
        try:
            # Make a copy to avoid warnings
            df_copy = historical_df.copy()
            
            # Calculate returns
            df_copy['returns'] = df_copy['close'].pct_change()
            
            # Create binary series (1 for positive return, 0 for negative or zero)
            df_copy['binary'] = (df_copy['returns'] > 0).astype(int)
            
            # Get the recent window for Bernoulli analysis
            recent_binary = df_copy['binary'].values[-self.bernoulli_window:]
            
            # Calculate current pattern
            current_pattern = ''
            if len(recent_binary) >= self.pattern_length:
                current_pattern = ''.join(map(str, recent_binary[-self.pattern_length:].astype(int)))
            
            # Bernoulli probability analysis
            p, margin = self._bernoulli_probability(recent_binary)
            
            # Binomial test
            p_value, effect_dir = self._binomial_test(recent_binary)
            
            # Pattern analysis
            pattern_probs = self._pattern_analysis(recent_binary)
            
            # Golden ratio cycle analysis
            prices = df_copy['close'].values
            cycle_indicators = self._golden_ratio_cycles(prices)
            
            # Generate signal components
            
            # 1. Bernoulli probability-based signal
            bernoulli_signal = self._bernoulli_significance(p, margin)
            
            # 2. Statistical significance signal from binomial test
            # Convert p-value to signal strength
            stat_significance = 0.0
            if p_value < 1 - self.confidence_level:
                # Statistically significant deviation
                stat_significance = effect_dir * (1.0 - p_value) * 2
            
            # 3. Pattern-based signal
            pattern_signal = 0.0
            if current_pattern in pattern_probs:
                pattern_p = pattern_probs[current_pattern]
                # Convert to [-1, 1] signal
                pattern_signal = 2 * pattern_p - 1
            
            # 4. Golden ratio cycle signal
            cycle_signal = 0.0
            phi_weights = {
                period: self.golden_ratio_factor ** i  # Weight by powers of golden ratio
                for i, period in enumerate(sorted(cycle_indicators.keys()))
            }
            weight_sum = sum(phi_weights.values())
            
            # Combine cycle signals with golden ratio weighting
            for period, indicator in cycle_indicators.items():
                if len(indicator) > 0 and not np.isnan(indicator[-1]):
                    # Weight by golden ratio factor
                    cycle_signal += (indicator[-1] * phi_weights[period]) / weight_sum
            
            # Combine signals using Bernoulli-inspired weighting
            # More weight to components with higher confidence
            combined_signal = (
                bernoulli_signal * (1 - margin) * 0.3 +
                stat_significance * (1 - p_value) * 0.2 +
                pattern_signal * (1 - self.golden_ratio_factor) * 0.2 +
                cycle_signal * self.golden_ratio_factor * 0.3
            )
            
            # Scale the final signal to [-1, 1]
            self.latest_signal = np.clip(combined_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Bernoulli Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Bernoulli's probability principles
        
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
        return "Bernoulli Agent" 