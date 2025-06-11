"""
Pascal Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Blaise Pascal's principles of
probability theory, Pascal's Triangle, and combinatorial analysis.

This agent uses probabilistic methods to estimate likely market outcomes,
modeling price distributions and calculating expected values of different
scenarios. It leverages Pascal's Triangle properties for binomial distributions
and statistical significance testing.

Concepts employed:
1. Pascal's Triangle for calculating binomial coefficients and probability distributions
2. Expected value calculations for trading decisions
3. Combinatorial analysis of market patterns
4. Application of Pascal's Wager concept to position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import binom
from scipy.signal import find_peaks, peak_prominences
import logging

logger = logging.getLogger(__name__)

class PascalAgent:
    """
    Trading agent based on Pascal's probability and combinatorial principles.
    
    Parameters
    ----------
    lookback_window : int, default=60
        Window size for statistical calculations
    prediction_horizon : int, default=10
        How far into the future to generate predictions
    confidence_level : float, default=0.95
        Confidence level for probability estimates (0 to 1)
    pattern_length : int, default=5
        Length of price patterns to analyze
    risk_reward_threshold : float, default=0.8
        Minimum risk/reward ratio to generate signals
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        prediction_horizon: int = 10,
        confidence_level: float = 0.95,
        pattern_length: int = 5,
        risk_reward_threshold: float = 0.8
    ):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.confidence_level = confidence_level
        self.pattern_length = pattern_length
        self.risk_reward_threshold = risk_reward_threshold
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Store calculated probabilities and patterns
        self.up_probability = 0.5
        self.binomial_distribution = None
        self.pattern_matches = {}
        
    def _calculate_pascal_triangle(self, n: int) -> List[List[int]]:
        """
        Calculate Pascal's Triangle up to row n
        
        Parameters
        ----------
        n : int
            Number of rows to calculate
            
        Returns
        -------
        list
            List of lists representing Pascal's Triangle
        """
        triangle = []
        for i in range(n+1):
            row = [1]
            for j in range(1, i):
                # Each element is the sum of the two above it
                row.append(triangle[i-1][j-1] + triangle[i-1][j])
            if i > 0:
                row.append(1)
            triangle.append(row)
        return triangle
    
    def _estimate_binomial_probabilities(self, price_series: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Estimate binomial probability distribution of price movements
        
        Parameters
        ----------
        price_series : numpy.ndarray
            Array of price values
            
        Returns
        -------
        tuple
            (probability of up move, binomial distribution)
        """
        # Calculate returns
        returns = np.diff(price_series) / price_series[:-1]
        
        # Calculate probability of up move
        up_probability = np.mean(returns > 0)
        
        # Ensure probability is never exactly 0 or 1 (would break binomial)
        up_probability = max(0.01, min(0.99, up_probability))
        
        # Calculate binomial distribution for prediction_horizon trials
        n = self.prediction_horizon
        k = np.arange(n + 1)  # 0 to n up-moves
        
        # Binomial PMF: P(X = k) = nCk * p^k * (1-p)^(n-k)
        # nCk is the binomial coefficient from Pascal's Triangle
        binomial_distribution = binom.pmf(k, n, up_probability)
        
        return up_probability, binomial_distribution
    
    def _identify_price_patterns(self, price_series: np.ndarray) -> Dict[str, List[int]]:
        """
        Identify recurring price patterns using combinatorial analysis
        
        Parameters
        ----------
        price_series : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Dictionary mapping patterns to their occurrences
        """
        # Convert price to direction (up=1, down=0)
        directions = (np.diff(price_series) > 0).astype(int)
        
        # Not enough data for pattern analysis
        if len(directions) < self.pattern_length:
            return {}
            
        # Find all patterns of length pattern_length
        patterns = {}
        for i in range(len(directions) - self.pattern_length):
            # Extract pattern and convert to string for dict key
            pattern = directions[i:i+self.pattern_length]
            pattern_key = ''.join([str(d) for d in pattern])
            
            # Record the position where this pattern was found
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            patterns[pattern_key].append(i)
            
        return patterns
    
    def _analyze_pattern_outcomes(
        self, 
        patterns: Dict[str, List[int]], 
        directions: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze historical outcomes following each pattern
        
        Parameters
        ----------
        patterns : dict
            Dictionary mapping patterns to their occurrences
        directions : numpy.ndarray
            Array of price directions (1=up, 0=down)
            
        Returns
        -------
        dict
            Dictionary mapping patterns to their probability of an up move
        """
        pattern_probabilities = {}
        
        # For each pattern, analyze what happened after it occurred
        for pattern_key, positions in patterns.items():
            up_count = 0
            valid_positions = 0
            
            for pos in positions:
                # Check if there's data after this pattern
                future_pos = pos + self.pattern_length
                if future_pos < len(directions):
                    # Count up moves
                    if directions[future_pos] == 1:
                        up_count += 1
                    valid_positions += 1
            
            # Calculate probability if we have enough occurrences
            if valid_positions >= 3:  # Minimum sample size
                pattern_probabilities[pattern_key] = up_count / valid_positions
                
        return pattern_probabilities
    
    def _calculate_expected_value(
        self, 
        up_probability: float, 
        current_price: float, 
        historical_volatility: float
    ) -> Tuple[float, float, float]:
        """
        Calculate expected value using Pascal's decision theory
        
        Parameters
        ----------
        up_probability : float
            Probability of upward price movement
        current_price : float
            Current asset price
        historical_volatility : float
            Historical price volatility
            
        Returns
        -------
        tuple
            (expected_value, potential_gain, potential_loss)
        """
        # Estimate potential moves based on historical volatility
        avg_move = historical_volatility * current_price
        
        # Potential gain and loss (asymmetric to reflect market behavior)
        potential_gain = avg_move * 1.0  # Baseline move
        potential_loss = avg_move * 1.2  # Losses often exceed gains
        
        # Adjust based on probability skew (if up_probability deviates from 0.5)
        probability_skew = (up_probability - 0.5) * 2  # Range: -1 to 1
        
        if probability_skew > 0:
            # Higher chance of up, increase potential gain
            potential_gain *= (1 + probability_skew * 0.5)
        else:
            # Higher chance of down, increase potential loss
            potential_loss *= (1 - probability_skew * 0.5)
            
        # Calculate expected value: EV = p(gain) * gain - p(loss) * loss
        expected_value = up_probability * potential_gain - (1 - up_probability) * potential_loss
        
        return expected_value, potential_gain, potential_loss
    
    def _generate_pascal_signal(
        self, 
        expected_value: float, 
        potential_gain: float, 
        potential_loss: float, 
        confidence: float
    ) -> float:
        """
        Generate trading signal using Pascal's Wager concept
        
        Parameters
        ----------
        expected_value : float
            Expected value of the trade
        potential_gain : float
            Potential gain amount
        potential_loss : float
            Potential loss amount
        confidence : float
            Confidence in the probability estimate (0 to 1)
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        # Calculate risk/reward ratio
        if potential_loss == 0:
            risk_reward = float('inf')
        else:
            risk_reward = potential_gain / potential_loss
            
        # Generate signal based on expected value, but scale by risk/reward
        if abs(expected_value) < 1e-10:  # Very small expected value
            return 0.0
            
        # Base signal on expected value
        # Scale by potential_loss to normalize the signal
        if potential_loss > 0:
            base_signal = np.sign(expected_value) * min(1.0, abs(expected_value) / potential_loss)
        else:
            base_signal = np.sign(expected_value) * 0.1  # Small signal if no loss potential
        
        # Apply risk/reward scaling
        if risk_reward < self.risk_reward_threshold:
            # Scale down signal for poor risk/reward, but don't eliminate it completely
            risk_reward_factor = max(0.1, risk_reward / self.risk_reward_threshold)
            base_signal *= risk_reward_factor
            
        # Apply Pascal's Wager concept: take positive EV bets, even with low confidence
        # but scale by confidence level
        signal = base_signal * max(0.3, confidence)  # Minimum confidence factor of 0.3
        
        return np.clip(signal, -1.0, 1.0)
    
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
            # Extract recent price data
            prices = historical_df['close'].values[-self.lookback_window:]
            
            # Ensure we have enough data
            if len(prices) < self.lookback_window / 2:
                self.is_fitted = False
                return
                
            # Estimate binomial probability distribution
            self.up_probability, self.binomial_distribution = self._estimate_binomial_probabilities(prices)
            
            # Create direction series (1=up, 0=down)
            directions = (np.diff(prices) > 0).astype(int)
            
            # Identify price patterns
            self.pattern_matches = self._identify_price_patterns(prices)
            
            # Analyze pattern outcomes
            pattern_probabilities = self._analyze_pattern_outcomes(self.pattern_matches, directions)
            
            # Current pattern (most recent n days)
            if len(directions) >= self.pattern_length:
                current_pattern = directions[-self.pattern_length:]
                current_pattern_key = ''.join([str(d) for d in current_pattern])
                
                # If we have stats for this pattern, use that probability
                # Otherwise, use the general up_probability
                if current_pattern_key in pattern_probabilities:
                    pattern_probability = pattern_probabilities[current_pattern_key]
                    pattern_occurrences = len(self.pattern_matches.get(current_pattern_key, []))
                    
                    # Calculate confidence based on number of occurrences
                    # More occurrences = higher confidence
                    confidence = min(1.0, pattern_occurrences / 20)  # Max confidence at 20 occurrences
                    
                    # Blend with general probability (weighted by confidence)
                    blended_probability = (confidence * pattern_probability + 
                                          (1 - confidence) * self.up_probability)
                else:
                    blended_probability = self.up_probability
                    confidence = 0.7  # Default confidence in overall stats
                    
                # Calculate historical volatility (standard deviation of returns)
                returns = np.diff(prices) / prices[:-1]
                historical_volatility = np.std(returns)
                
                # Calculate expected value
                expected_value, potential_gain, potential_loss = self._calculate_expected_value(
                    blended_probability, prices[-1], historical_volatility
                )
                
                # Generate signal
                self.latest_signal = self._generate_pascal_signal(
                    expected_value, potential_gain, potential_loss, confidence
                )
                
            # Mark as fitted
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Pascal Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Pascal's probability principles
        
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
        return "Pascal Agent" 