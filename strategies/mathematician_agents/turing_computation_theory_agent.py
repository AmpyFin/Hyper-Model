"""
Turing Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Alan Turing's principles of
computational theory, pattern recognition, and algorithmic information theory.

This agent searches for computational patterns in market data, attempting to
identify algorithmic regularities that might indicate non-random price movements.
It applies concepts from computability theory to detect predictable vs. random
market regimes.

Concepts employed:
1. Algorithmic complexity measures for pattern detection
2. Finite state machines for market regime classification
3. Turing test concepts applied to distinguishing random vs. non-random price action
4. Cryptanalytic techniques for market 'code-breaking'
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.cluster import KMeans
import zlib  # For compression-based complexity measures
import logging

logger = logging.getLogger(__name__)

class TuringAgent:
    """
    Trading agent based on computational theory and pattern recognition.
    
    Parameters
    ----------
    lookback_window : int, default=100
        Window size for pattern analysis
    n_states : int, default=4
        Number of market states to model in the finite state machine
    min_pattern_length : int, default=3
        Minimum pattern length to search for
    max_pattern_length : int, default=10
        Maximum pattern length to search for
    randomness_threshold : float, default=0.2
        Threshold to distinguish random from non-random (0 to 1)
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        n_states: int = 4,
        min_pattern_length: int = 3,
        max_pattern_length: int = 10,
        randomness_threshold: float = 0.2
    ):
        self.lookback_window = lookback_window
        self.n_states = n_states
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.randomness_threshold = randomness_threshold
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Store computed patterns and states
        self.state_sequence = []
        self.transition_matrix = None
        self.current_state = None
        self.algorithmic_complexity = None
        self.detected_patterns = {}
        
    def _symbolize_price_series(self, prices: np.ndarray) -> np.ndarray:
        """
        Convert continuous price series to discrete symbols using clustering
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        numpy.ndarray
            Array of state symbols (integers)
        """
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Prepare features for clustering: return and volatility
        features = np.zeros((len(returns)-5, 2))
        features[:, 0] = returns[5:]  # Current return
        features[:, 1] = np.std([returns[i:i+5] for i in range(len(returns)-5)], axis=1)  # Local volatility
        
        # Normalize features
        features_norm = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Handle NaN values
        features_norm = np.nan_to_num(features_norm)
        
        # Cluster into states
        kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
        states = kmeans.fit_predict(features_norm)
        
        return states
    
    def _calculate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Calculate state transition probabilities (Markov model)
        
        Parameters
        ----------
        states : numpy.ndarray
            Array of state symbols
            
        Returns
        -------
        numpy.ndarray
            Transition probability matrix
        """
        n = self.n_states
        transitions = np.zeros((n, n))
        
        # Count transitions
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i+1]
            transitions[from_state, to_state] += 1
            
        # Convert to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        transition_matrix = transitions / row_sums
        
        return transition_matrix
    
    def _estimate_algorithmic_complexity(self, data: np.ndarray) -> float:
        """
        Estimate algorithmic (Kolmogorov) complexity using compression ratio
        
        Parameters
        ----------
        data : numpy.ndarray
            Data sequence to analyze
            
        Returns
        -------
        float
            Normalized complexity measure (0 to 1)
        """
        # Convert to string representation
        data_str = ','.join(data.astype(str)).encode()
        
        # Calculate compressed size
        compressed_size = len(zlib.compress(data_str))
        
        # Normalize by original size to get complexity ratio
        complexity = compressed_size / len(data_str)
        
        # Scale to 0-1 range (empirical scaling)
        normalized_complexity = min(1.0, complexity / 0.7)
        
        return normalized_complexity
    
    def _detect_repeating_patterns(self, states: np.ndarray) -> Dict[str, List[int]]:
        """
        Find repeating patterns in the state sequence
        
        Parameters
        ----------
        states : numpy.ndarray
            Array of state symbols
            
        Returns
        -------
        dict
            Dictionary mapping patterns to their occurrences
        """
        patterns = {}
        
        # Search for patterns of different lengths
        for pattern_len in range(self.min_pattern_length, min(self.max_pattern_length, len(states)//2)):
            for i in range(len(states) - pattern_len):
                # Extract potential pattern
                pattern = states[i:i+pattern_len]
                pattern_key = ','.join(pattern.astype(str))
                
                # Search for occurrences of this pattern
                occurrences = []
                for j in range(len(states) - pattern_len):
                    if np.array_equal(states[j:j+pattern_len], pattern):
                        occurrences.append(j)
                        
                # Only keep patterns that repeat
                if len(occurrences) > 1:
                    patterns[pattern_key] = occurrences
                    
        return patterns
    
    def _predict_next_state(self, states: np.ndarray, transition_matrix: np.ndarray) -> int:
        """
        Predict the next state using the transition matrix
        
        Parameters
        ----------
        states : numpy.ndarray
            Array of state symbols
        transition_matrix : numpy.ndarray
            State transition probability matrix
            
        Returns
        -------
        int
            Predicted next state
        """
        if len(states) == 0:
            return 0
            
        current_state = states[-1]
        probabilities = transition_matrix[current_state]
        
        # Return most likely next state
        return np.argmax(probabilities)
    
    def _generate_signal_from_patterns(
        self,
        states: np.ndarray,
        patterns: Dict[str, List[int]],
        returns_by_state: Dict[int, float],
        complexity: float
    ) -> float:
        """
        Generate trading signal based on detected patterns
        
        Parameters
        ----------
        states : numpy.ndarray
            Array of state symbols
        patterns : dict
            Dictionary of detected patterns
        returns_by_state : dict
            Average returns for each state
        complexity : float
            Algorithmic complexity measure
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        # Adjust randomness threshold to be less restrictive
        adjusted_randomness_threshold = max(0.1, self.randomness_threshold)
        
        # If high complexity (random), generate weaker signal but don't eliminate it
        if complexity > 1 - adjusted_randomness_threshold:
            randomness_factor = (complexity - (1 - adjusted_randomness_threshold)) / adjusted_randomness_threshold
            randomness_factor = min(1.0, max(0.0, randomness_factor))
            
            # Scale down signal based on randomness, but maintain minimum signal strength
            scale_factor = max(0.2, 1.0 - randomness_factor * 0.8)  # Minimum 20% signal strength
        else:
            scale_factor = 1.0
            
        # Initialize signal components
        pattern_signal = 0.0
        state_signal = 0.0
        
        # Generate signal from current state expected return
        if len(states) > 0 and states[-1] in returns_by_state:
            expected_return = returns_by_state[states[-1]]
            if abs(expected_return) > 1e-10:  # Only if return is meaningful
                state_signal = np.sign(expected_return) * min(1.0, abs(expected_return) * 20)
            
        # Check for active patterns
        active_patterns = []
        for pattern_key, occurrences in patterns.items():
            pattern = np.array([int(s) for s in pattern_key.split(',')])
            pattern_len = len(pattern)
            
            # Check if we're at the end of this pattern
            if len(states) >= pattern_len and np.array_equal(states[-pattern_len:], pattern):
                active_patterns.append(pattern_key)
                
        # If we have active patterns, look at what typically follows them
        pattern_signal_sum = 0.0
        pattern_count = 0
        
        for pattern_key in active_patterns:
            pattern = np.array([int(s) for s in pattern_key.split(',')])
            occurrences = patterns[pattern_key]
            
            # For each occurrence (except the last one), check what state followed
            next_states = []
            for pos in occurrences[:-1]:  # Exclude the last occurrence
                if pos + len(pattern) < len(states):
                    next_states.append(states[pos + len(pattern)])
                    
            if next_states:
                # Count frequency of each next state
                unique_next, counts = np.unique(next_states, return_counts=True)
                most_common_next = unique_next[np.argmax(counts)]
                probability = np.max(counts) / len(next_states)
                
                # Get expected return for this next state
                if most_common_next in returns_by_state:
                    pattern_expected_return = returns_by_state[most_common_next]
                    if abs(pattern_expected_return) > 1e-10:  # Only if return is meaningful
                        # Weight by probability and pattern length (longer patterns are more significant)
                        pattern_signal = np.sign(pattern_expected_return) * min(1.0, abs(pattern_expected_return) * 20)
                        pattern_signal *= probability * min(1.0, len(pattern) / 5)
                        
                        pattern_signal_sum += pattern_signal
                        pattern_count += 1
                    
        # Combine pattern signals if we have any
        if pattern_count > 0:
            pattern_signal = pattern_signal_sum / pattern_count
        
        # Blend state and pattern signals
        # Pattern signals are more important when complexity is low (more predictable)
        # State signals dominate when complexity is high (more random)
        combined_signal = (pattern_signal * (1 - complexity) + 
                          state_signal * complexity)
        
        # If combined signal is too weak, generate a small random-walk based signal
        if abs(combined_signal) < 0.01:
            # Use transition matrix to generate a weak directional bias
            if len(states) > 0 and self.transition_matrix is not None:
                current_state = states[-1]
                next_state_probs = self.transition_matrix[current_state]
                
                # Calculate expected state change
                expected_state_change = np.sum(next_state_probs * np.arange(len(next_state_probs))) - current_state
                
                # Convert to weak signal
                if abs(expected_state_change) > 0.1:
                    combined_signal = np.sign(expected_state_change) * min(0.3, abs(expected_state_change) / self.n_states)
        
        # Apply overall scaling based on randomness
        final_signal = combined_signal * scale_factor
        
        return np.clip(final_signal, -1.0, 1.0)
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and extract computational patterns
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.lookback_window:
            self.is_fitted = False
            return
            
        try:
            # Extract price data
            prices = historical_df['close'].values[-self.lookback_window:]
            returns = np.diff(prices) / prices[:-1]
            
            # Symbolize price series into discrete states
            states = self._symbolize_price_series(prices)
            
            # Store state sequence
            self.state_sequence = states
            
            # Calculate state transition matrix
            self.transition_matrix = self._calculate_transition_matrix(states)
            
            # Estimate algorithmic complexity
            self.algorithmic_complexity = self._estimate_algorithmic_complexity(states)
            
            # Detect repeating patterns
            self.detected_patterns = self._detect_repeating_patterns(states)
            
            # Calculate average returns by state
            returns_by_state = {}
            
            # Ensure states and returns arrays have the same length
            # states_with_returns should be the same length as returns
            if len(states) > len(returns):
                states_with_returns = states[:-1]  # Remove the last state
            else:
                states_with_returns = states  # Keep all states if lengths already match
            
            # Verify lengths match before proceeding
            if len(states_with_returns) == len(returns):
                for state in range(self.n_states):
                    # Use boolean indexing with matching dimensions
                    state_mask = (states_with_returns == state)
                    state_returns = returns[state_mask]
                    
                    if len(state_returns) > 0:
                        returns_by_state[state] = np.mean(state_returns)
            else:
                # If dimensions still don't match, initialize with empty values
                for state in range(self.n_states):
                    returns_by_state[state] = 0.0
            
            # Generate signal
            self.latest_signal = self._generate_signal_from_patterns(
                states, self.detected_patterns, returns_by_state, self.algorithmic_complexity
            )
            
            # Store current state
            self.current_state = states[-1] if len(states) > 0 else 0
            
            # Set as fitted
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Turing Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on computational pattern recognition
        
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
        return "Turing Agent" 