"""
Shannon Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Claude Shannon's pioneering
work in information theory, encryption, and digital circuit design.

Claude Shannon is known as "the father of information theory" and established
fundamental concepts including:
1. Information entropy as a measure of uncertainty
2. Channel capacity and noise reduction
3. Communication in presence of noise
4. Bit as fundamental unit of information
5. Mathematical theory of cryptography

This agent models market behavior using information-theoretic principles:
1. Entropy analysis to measure market predictability
2. Channel capacity to optimize signal-to-noise ratio in price data
3. Source coding for compression of price information
4. Cryptanalysis for detecting hidden patterns

Input: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
Output: Signal ∈ [-1.0000, 1.0000] where:
  -1.0000 = Strong sell signal (strong downward trend detected)
  -0.5000 = Weak sell signal (weak downward trend detected)
   0.0000 = Neutral signal (no clear trend)
   0.5000 = Weak buy signal (weak upward trend detected)
   1.0000 = Strong buy signal (strong upward trend detected)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import math
from scipy import stats
from collections import Counter, deque, defaultdict

from ..agent import Agent

logger = logging.getLogger(__name__)

class ShannonAgent(Agent):
    """
    Trading agent based on Claude Shannon's information theory.
    
    Parameters
    ----------
    entropy_window : int, default=30
        Window size for entropy calculations
    channel_memory : int, default=5
        Memory length for market state encoding
    noise_threshold : float, default=0.6
        Threshold for filtering market noise
    information_bits : int, default=3
        Number of bits for state discretization
    redundancy_factor : float, default=0.7
        Factor for redundant signal confirmation
    """
    
    def __init__(
        self,
        entropy_window: int = 30,
        channel_memory: int = 5,
        noise_threshold: float = 0.6,
        information_bits: int = 3,
        redundancy_factor: float = 0.7
    ):
        self.entropy_window = entropy_window
        self.channel_memory = channel_memory
        self.noise_threshold = noise_threshold
        self.information_bits = information_bits
        self.redundancy_factor = redundancy_factor
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.market_states = deque(maxlen=30)
        self.entropy_history = deque(maxlen=30)
        self.transition_matrix = {}
        
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """
        Calculate Shannon entropy of a data series
        
        Parameters
        ----------
        data : numpy.ndarray
            Data array for entropy calculation
            
        Returns
        -------
        float
            Shannon entropy value
        """
        # Handle empty arrays or arrays with NaN values
        if len(data) == 0:
            return 0.0
            
        # Clean data - remove NaN and infinite values
        clean_data = data[np.isfinite(data)]
        
        if len(clean_data) == 0:
            return 0.0
        
        # For continuous data, discretize into bins
        if len(clean_data) < 5:
            return 0.0
            
        # Use quantile-based binning for better distribution
        try:
            # Use 8 bins for reasonable discretization
            bins = min(8, len(clean_data) // 2)
            if bins < 2:
                return 0.0
                
            # Create bins based on quantiles to ensure reasonable distribution
            if np.max(clean_data) == np.min(clean_data):
                # All values are the same - maximum predictability
                return 0.0
                
            _, bin_edges = np.histogram(clean_data, bins=bins)
            digitized = np.digitize(clean_data, bin_edges[:-1])
            
            # Count occurrences of each bin
            counter = Counter(digitized)
            
        except Exception:
            # Fallback: simple sign-based discretization
            counter = Counter(np.sign(clean_data).astype(int))
        
        if len(counter) <= 1:
            return 0.0
            
        # Calculate probabilities
        total = sum(counter.values())
        probs = [count / total for count in counter.values()]
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(counter))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Ensure result is valid
        if not np.isfinite(normalized_entropy):
            return 0.0
            
        return normalized_entropy
    
    def _encode_market_state(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> List[int]:
        """
        Encode market state into discrete symbols
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        list
            Discretized market state
        """
        if len(prices) < 2:
            return []
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Get recent returns for state encoding
        recent_returns = returns[-self.channel_memory:]
        
        # Encode returns into discrete states based on information_bits
        # For 3 bits, we have 8 possible states
        num_states = 2 ** self.information_bits
        
        # Calculate quantiles for discretization
        quantiles = np.linspace(0, 1, num_states + 1)[1:-1]
        breakpoints = np.nanquantile(returns, quantiles) if len(returns) > num_states else np.linspace(-0.01, 0.01, num_states-1)
        
        # Encode each return value
        encoded_state = []
        for ret in recent_returns:
            if not np.isfinite(ret):  # Handle NaN or inf values
                state = num_states // 2  # Middle state as default
            else:
                state = 0
                for i, bp in enumerate(breakpoints):
                    if ret > bp:
                        state = i + 1
            encoded_state.append(state)
            
        # Include volume information if available
        if volumes is not None and len(volumes) >= 2:
            vol_changes = np.diff(volumes) / volumes[:-1]
            recent_vol = vol_changes[-self.channel_memory:] if len(vol_changes) >= self.channel_memory else []
            
            # Encode volume changes (high/low)
            for vol in recent_vol:
                if not np.isfinite(vol):  # Handle NaN or inf
                    encoded_state.append(0)
                elif vol > np.nanmedian(vol_changes):
                    encoded_state.append(1)
                else:
                    encoded_state.append(0)
        
        return encoded_state
    
    def _build_transition_matrix(self, states: List[List[int]]) -> Dict:
        """
        Build transition probability matrix from market states
        
        Parameters
        ----------
        states : list
            List of encoded market states
            
        Returns
        -------
        dict
            Transition probability matrix
        """
        if not states or len(states) < 2:
            return {}
            
        # Convert states to strings for dictionary keys
        state_strings = [''.join(map(str, state)) for state in states]
        
        # Count transitions
        transitions = {}
        
        for i in range(len(state_strings) - 1):
            current = state_strings[i]
            next_state = state_strings[i + 1]
            
            if current not in transitions:
                transitions[current] = {}
                
            if next_state not in transitions[current]:
                transitions[current][next_state] = 0
                
            transitions[current][next_state] += 1
            
        # Convert counts to probabilities
        for current, next_states in transitions.items():
            total = sum(next_states.values())
            for next_state in next_states:
                transitions[current][next_state] /= total
                
        return transitions
    
    def _calculate_information_gain(self, prices: np.ndarray) -> float:
        """
        Calculate information gain from recent price movement
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Information gain value
        """
        if len(prices) < self.entropy_window:
            return 0.0
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Clean NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 2:
            return 0.0
            
        # Calculate entropy of entire window
        full_entropy = self._calculate_entropy(returns[-self.entropy_window:])
        
        # Calculate entropy of first and second half
        mid = len(returns[-self.entropy_window:]) // 2
        first_half = returns[-self.entropy_window:-mid] if mid > 0 else returns[-1:]
        second_half = returns[-mid:] if mid > 0 else returns[-1:]
        
        first_entropy = self._calculate_entropy(first_half)
        second_entropy = self._calculate_entropy(second_half)
        
        # Information gain: reduction in entropy
        # Positive = becoming more predictable, Negative = becoming less predictable
        entropy_change = first_entropy - second_entropy
        
        # Store entropy history
        self.entropy_history.append(second_entropy)
        
        return entropy_change
    
    def _channel_capacity_filter(self, signal: float, prices: np.ndarray) -> float:
        """
        Apply noise filtering based on channel capacity principles
        
        Parameters
        ----------
        signal : float
            Raw signal value
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Filtered signal
        """
        if len(prices) < 30:
            return signal
            
        # Calculate recent volatility as a proxy for noise
        returns = np.diff(prices) / prices[:-1]
        volatility = np.nanstd(returns[-20:])  # Handle NaN values
        
        # Estimate signal-to-noise ratio (SNR)
        trend_strength = abs(np.nanmean(returns[-10:])) / (volatility + 1e-10)
        
        # Calculate the theoretical channel capacity
        # Shannon's formula: C = B * log2(1 + S/N)
        # Where B is bandwidth (fixed at 1 for our purposes)
        capacity = np.log2(1 + trend_strength)
        
        # Normalize to [0, 1] range for practical use
        normalized_capacity = min(1.0, capacity / 3.0)  # 3.0 is a reasonable max capacity in this context
        
        # Apply channel capacity as confidence factor
        # If capacity is low, reduce signal strength (more noise than signal)
        filtered_signal = signal * normalized_capacity
        
        return filtered_signal
    
    def _source_coding_analysis(self, prices: np.ndarray) -> float:
        """
        Apply source coding principles to compress and analyze price information
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Source coding signal
        """
        # Reduce minimum requirement
        min_required = min(self.entropy_window, 20)
        if len(prices) < min_required:
            return 0.0
            
        # Get price changes direction as the simplest form of encoding
        price_changes = np.diff(prices)
        
        # Handle case where all price changes are zero
        if np.all(price_changes == 0):
            return 0.0
            
        directions = np.sign(price_changes).astype(int)
        
        # Remove zeros (unchanged prices) for cleaner encoding
        directions = directions[directions != 0]
        
        if len(directions) < 5:  # Reduced from 10 to 5
            return 0.0
            
        # Run-length encoding (RLE) of directions
        rle = []
        current_val = directions[0]
        current_run = 1
        
        for i in range(1, len(directions)):
            if directions[i] == current_val:
                current_run += 1
            else:
                rle.append((current_val, current_run))
                current_val = directions[i]
                current_run = 1
                
        rle.append((current_val, current_run))  # Add the last run
        
        if len(rle) == 0:
            return 0.0
        
        # Analyze run length distribution
        run_lengths_up = [run for val, run in rle if val == 1]
        run_lengths_down = [run for val, run in rle if val == -1]
        
        # Calculate compression ratio
        compression_ratio = len(directions) / (2 * len(rle)) if len(rle) > 0 else 0.0
        
        # Analyze average run lengths
        avg_up = np.mean(run_lengths_up) if run_lengths_up else 0
        avg_down = np.mean(run_lengths_down) if run_lengths_down else 0
        
        # Generate signal based on run length comparison
        if avg_up == 0 and avg_down == 0:
            return 0.0
            
        total_avg = avg_up + avg_down
        if total_avg == 0:
            return 0.0
            
        run_ratio = avg_up / total_avg
        
        # Convert to [-1, 1] signal (0.5 -> 0, 0 -> -1, 1 -> 1)
        signal = 2 * run_ratio - 1
        
        # Weight by compression ratio (more compressible = stronger pattern)
        # Ensure compression ratio is reasonable
        compression_ratio = max(0.1, min(2.0, compression_ratio))
        weighted_signal = signal * min(1.0, compression_ratio)
        
        # Ensure result is finite
        if not np.isfinite(weighted_signal):
            return 0.0
        
        return weighted_signal
    
    def _cryptanalysis(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        Apply cryptanalysis techniques to detect hidden patterns
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        float
            Cryptanalysis signal
        """
        if len(prices) < 30:
            return 0.0
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Clean NaN and infinite values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 20:
            return 0.0
            
        # Frequency analysis (analogous to letter frequency in cryptanalysis)
        # Discretize returns into bins
        bins = 10
        
        # Handle case when all returns are the same
        if np.max(returns) == np.min(returns):
            return 0.0
            
        try:
            hist, _ = np.histogram(returns, bins=bins)
            frequencies = hist / np.sum(hist)
        except Exception:
            # Fallback if histogram fails
            return 0.0
            
        # Calculate uniformity of the distribution
        # In cryptanalysis, non-uniform distributions suggest patterns
        expected_freq = 1.0 / bins
        deviation = np.sum(np.abs(frequencies - expected_freq)) / 2.0
        
        # Autocorrelation analysis (looking for repeating patterns)
        # Like finding the key length in cryptanalysis
        max_lag = min(10, len(returns) // 3)
        auto_corr = []
        
        for lag in range(1, max_lag + 1):
            if lag >= len(returns):
                break
            corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            if np.isfinite(corr):  # Handle NaN correlation
                auto_corr.append((lag, abs(corr)))
        
        # Find the lag with highest correlation
        best_lag, best_corr = max(auto_corr, key=lambda x: x[1]) if auto_corr else (1, 0)
        
        # Combine frequency analysis and autocorrelation
        # Direction is determined by the most recent "key" (pattern)
        if best_lag < len(returns):
            recent_pattern = returns[-best_lag:]
            pattern_direction = np.sign(np.mean(recent_pattern))
        else:
            pattern_direction = 0
            
        # Signal strength based on deviation from uniformity and autocorrelation
        signal_strength = (deviation * 0.5 + best_corr * 0.5) * self.redundancy_factor
        signal = pattern_direction * signal_strength
        
        return np.clip(signal, -1.0, 1.0)
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and extract information-theoretic patterns
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        # Reduce minimum requirement to be more flexible
        min_required = max(self.entropy_window, 25)
        if len(historical_df) < min_required:
            self.is_fitted = False
            return
            
        try:
            # Extract price and volume data
            prices = historical_df['close'].values
            volumes = historical_df['volume'].values if 'volume' in historical_df.columns else None
            
            # Ensure data is clean (no NaN or infinite values)
            prices = np.array(prices)
            if volumes is not None:
                volumes = np.array(volumes)
            
            # Information theory analysis
            
            # 1. Encode market state
            market_state = self._encode_market_state(prices, volumes)
            if market_state:
                self.market_states.append(market_state)
                
            # 2. Build transition matrix if we have enough states
            if len(self.market_states) >= 2:
                self.transition_matrix = self._build_transition_matrix(list(self.market_states))
                
            # 3. Calculate information theoretic metrics
            entropy_gain = self._calculate_information_gain(prices)
            source_coding_signal = self._source_coding_analysis(prices)
            crypto_signal = self._cryptanalysis(prices, volumes)
            
            # 4. Predict based on transition matrix
            prediction_signal = 0.0
            if self.transition_matrix and market_state:
                current_state = ''.join(map(str, market_state))
                if current_state in self.transition_matrix:
                    # Find most likely next state
                    next_state = max(self.transition_matrix[current_state].items(), 
                                    key=lambda x: x[1])[0]
                    
                    # Extract direction from state transition
                    # Simple approach: compare first digit of current and next state
                    if len(current_state) > 0 and len(next_state) > 0:
                        try:
                            current_direction = int(current_state[0])
                            next_direction = int(next_state[0])
                            
                            # Generate signal based on predicted direction change
                            # Direction increasing = positive signal
                            prediction_signal = np.sign(next_direction - current_direction)
                            
                            # Weight by transition probability
                            prediction_signal *= self.transition_matrix[current_state][next_state]
                        except (ValueError, IndexError):
                            prediction_signal = 0.0
                        
            # 5. Combine signals with equal weights initially
            raw_signal = (
                entropy_gain * 0.25 +
                source_coding_signal * 0.25 +
                crypto_signal * 0.25 +
                prediction_signal * 0.25
            )
            
            # Ensure raw signal is finite
            if not np.isfinite(raw_signal):
                raw_signal = 0.0
            
            # 6. Apply noise filtering
            filtered_signal = self._channel_capacity_filter(raw_signal, prices)
            
            # Ensure filtered signal is finite
            if not np.isfinite(filtered_signal):
                filtered_signal = 0.0
            
            # 7. Apply entropy-based confidence scaling
            # Only generate signal if entropy is below noise threshold
            # (Shannon's insight: low entropy = more predictable)
            if self.entropy_history:
                current_entropy = self.entropy_history[-1]
                if np.isfinite(current_entropy) and current_entropy < self.noise_threshold:
                    signal_strength = 1.0 - (current_entropy / self.noise_threshold)
                    self.latest_signal = filtered_signal * signal_strength
                else:
                    # Market too noisy for reliable signal, but still allow some signal
                    self.latest_signal = filtered_signal * 0.3  # Reduced confidence
            else:
                # No entropy history yet, use filtered signal with moderate confidence
                self.latest_signal = filtered_signal * 0.5
            
            # If all components are zero, try a simple fallback signal
            if abs(self.latest_signal) < 1e-10:
                # Simple momentum-based fallback
                if len(prices) >= 10:
                    recent_returns = np.diff(prices[-10:]) / prices[-11:-1]
                    clean_returns = recent_returns[np.isfinite(recent_returns)]
                    if len(clean_returns) > 0:
                        momentum = np.mean(clean_returns)
                        if np.isfinite(momentum):
                            self.latest_signal = np.sign(momentum) * min(0.3, abs(momentum) * 10)
                
            # Ensure signal is in range [-1, 1] and finite
            if np.isfinite(self.latest_signal):
                self.latest_signal = np.clip(self.latest_signal, -1.0, 1.0)
            else:
                self.latest_signal = 0.0
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Shannon Agent fit: {e}")
            self.latest_signal = 0.0
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on information theory principles
        
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

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Shannon's information theory principles.
        
        Parameters
        ----------
        historical_df : pd.DataFrame
            Historical OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns
        -------
        float
            Trading signal in range [-1.0000, 1.0000]
            -1.0000 = Strong sell
            -0.5000 = Weak sell
             0.0000 = Neutral
             0.5000 = Weak buy
             1.0000 = Strong buy
        """
        try:
            # Process the data using existing workflow
            self.fit(historical_df)
            
            if not self.is_fitted:
                return 0.0000
                
            # Get current price for prediction
            current_price = historical_df['close'].iloc[-1]
            
            # Generate signal using existing predict method
            signal = self.predict(current_price=current_price, historical_df=historical_df)
            
            # Ensure signal is properly formatted to 4 decimal places
            return float(round(signal, 4))
            
        except ValueError as e:
            # Handle the case where there's not enough data
            logger.error(f"ValueError in Shannon strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Shannon strategy: {str(e)}")
            return 0.0000 