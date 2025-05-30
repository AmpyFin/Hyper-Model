"""
Thompson Agent
~~~~~~~~~~~~
Agent implementing trading strategies based on Ken Thompson's contributions to 
computer science, particularly in the areas of operating systems (Unix), 
programming languages (B), regular expressions, and computer chess.

Thompson is known for:
1. Co-creating Unix operating system
2. Developing B programming language (predecessor to C)
3. Regular expression pattern matching
4. Thompson NFA (Nondeterministic Finite Automaton)
5. Chess endgame analysis and the "Thompson hack"

This agent models market behavior using:
1. Regular expression-like pattern matching for price sequences
2. Unix-inspired "small is beautiful" philosophy for minimal indicators
3. State machine transitions for regime detection
4. Thompson sampling for exploration vs. exploitation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
import math
import re
from collections import defaultdict, deque  

from ..agent import Agent

logger = logging.getLogger(__name__)

class ThompsonAgent(Agent):
    """
    Trading agent based on Ken Thompson's computing principles.
    
    Parameters
    ----------
    pattern_length : int, default=8
        Length of price patterns for regex-like matching
    state_memory : int, default=10
        Memory length for state transitions
    exploration_rate : float, default=0.2
        Rate of exploration vs. exploitation (Thompson sampling)
    pipe_depth : int, default=3
        Number of processing stages in the "pipeline"
    kernel_window : int, default=5
        Window size for kernel operations
    """
    
    def __init__(
        self,
        pattern_length: int = 8,
        state_memory: int = 10,
        exploration_rate: float = 0.2,
        pipe_depth: int = 3,
        kernel_window: int = 5
    ):
        self.pattern_length = pattern_length
        self.state_memory = state_memory
        self.exploration_rate = exploration_rate
        self.pipe_depth = pipe_depth
        self.kernel_window = kernel_window
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.pattern_library = {}
        self.state_machine = {}
        self.price_symbols = []
        self.arm_rewards = {'long': [0], 'short': [0], 'neutral': [0]}
        
    def _encode_price_patterns(self, prices: np.ndarray) -> str:
        """
        Encode price movements into a string for regex-like pattern matching
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        str
            Encoded price pattern
        """
        if len(prices) < 3:
            return ""
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Convert to simplified price movement alphabet
        # u: large up, U: small up, d: large down, D: small down, s: sideways
        
        # Define thresholds for movement classification
        mean_abs_return = np.mean(np.abs(returns))
        up_threshold = mean_abs_return * 0.5
        down_threshold = -up_threshold
        large_threshold = mean_abs_return * 1.5
        
        # Encode price movements
        symbols = []
        for ret in returns:
            if ret > large_threshold:
                symbols.append('u')  # Large up move
            elif ret > up_threshold:
                symbols.append('U')  # Small up move
            elif ret < -large_threshold:
                symbols.append('d')  # Large down move
            elif ret < down_threshold:
                symbols.append('D')  # Small down move
            else:
                symbols.append('s')  # Sideways
                
        # Store in internal state
        self.price_symbols = symbols
        
        # Return as string
        pattern = ''.join(symbols[-self.pattern_length:])
        
        return pattern
    
    def _thompson_regex_match(self, pattern: str, library: Dict[str, List[float]]) -> Tuple[bool, Optional[str], float]:
        """
        Match current pattern against the pattern library using regex-like matching
        
        Parameters
        ----------
        pattern : str
            Current price pattern
        library : dict
            Pattern library with historical outcomes
            
        Returns
        -------
        tuple
            (match_found, matched_pattern, confidence)
        """
        if not pattern or not library:
            return False, None, 0.0
            
        best_match = None
        best_match_score = 0.0
        best_match_length = 0
        
        # Try to find exact matches first
        for lib_pattern, outcomes in library.items():
            if lib_pattern == pattern:
                avg_outcome = np.mean(outcomes)
                return True, lib_pattern, min(1.0, abs(avg_outcome) * 2)
        
        # If no exact match, try regex-like matching
        for lib_pattern, outcomes in library.items():
            # Skip patterns that are too short
            if len(lib_pattern) < 4:
                continue
                
            # Calculate similarity score
            match_length = 0
            for i in range(min(len(pattern), len(lib_pattern))):
                if pattern[i] == lib_pattern[i]:
                    match_length += 1
                else:
                    # Allow wildcards
                    # '.' can match any single movement
                    if lib_pattern[i] == '.':
                        match_length += 0.8  # Lower weight for wildcard matches
            
            # Check for repetition patterns with + symbol
            # For example, "u+" matches one or more consecutive "u"
            processed_lib_pattern = lib_pattern
            for char in set(lib_pattern):
                if char + '+' in lib_pattern:
                    # Check if consecutive characters match the repetition
                    rep_char = char
                    rep_pattern = rep_char + '+'
                    rep_index = processed_lib_pattern.index(rep_pattern)
                    
                    # Count consecutive occurrences in the actual pattern
                    count = 0
                    for i in range(rep_index, min(len(pattern), len(processed_lib_pattern))):
                        if i < len(pattern) and pattern[i] == rep_char:
                            count += 1
                        else:
                            break
                    
                    # Add match score based on repetition length
                    match_length += count * 0.9  # Slightly lower weight for repetitions
            
            # Normalize by pattern length
            match_score = match_length / max(len(pattern), len(lib_pattern))
            
            # Check if it's the best match so far
            if match_score > best_match_score:
                best_match_score = match_score
                best_match = lib_pattern
                best_match_length = match_length
        
        # Only return match if score is high enough
        if best_match_score >= 0.7:
            # Calculate confidence based on match score and outcomes
            avg_outcome = np.mean(library[best_match])
            confidence = best_match_score * min(1.0, abs(avg_outcome) * 2)
            return True, best_match, confidence
        
        return False, None, 0.0
    
    def _build_state_machine(self, price_series: np.ndarray) -> Dict:
        """
        Build a state machine model of price transitions (like NFA)
        
        Parameters
        ----------
        price_series : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            State machine representation
        """
        if len(price_series) < self.state_memory + 5:
            return {}
            
        # Calculate returns
        returns = np.diff(price_series) / price_series[:-1]
        
        # Define states based on return quantiles
        states = []
        num_states = 5  # 5 states: strong down, down, neutral, up, strong up
        
        # Calculate quantiles
        if len(returns) > num_states:
            quantiles = np.linspace(0, 1, num_states+1)[1:-1]  # 3 inner quantiles
            breakpoints = np.quantile(returns, quantiles)
        else:
            # Default breakpoints if not enough data
            breakpoints = [-0.01, -0.001, 0.001, 0.01]
        
        # Assign states
        for ret in returns:
            state = 0  # Default: strong down
            for i, bp in enumerate(breakpoints):
                if ret > bp:
                    state = i + 1
            states.append(state)
        
        # Build state transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        
        # Count transitions
        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            transitions[current][next_state] += 1
            
        # Convert to probabilities
        prob_transitions = {}
        
        for current, nexts in transitions.items():
            total = sum(nexts.values())
            prob_transitions[current] = {next_state: count / total 
                                      for next_state, count in nexts.items()}
            
        # Add metadata to the state machine
        state_machine = {
            'transitions': prob_transitions,
            'breakpoints': breakpoints,
            'current_state': states[-1] if states else 2,  # Default to neutral
            'state_history': states[-self.state_memory:] if len(states) >= self.state_memory else states
        }
        
        return state_machine
    
    def _unix_pipe_process(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        Process price data through Unix-like pipe stages
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        float
            Processed signal
        """
        if len(prices) < 20:
            return 0.0
            
        # Stage 1: Calculate simple returns (like 'cat' command - reading input)
        returns = np.diff(prices) / prices[:-1]
        
        # Stage 2: Filter outliers (like 'grep' command - filtering)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        filtered_returns = returns[np.abs(returns - mean_ret) <= 2 * std_ret]
        
        if len(filtered_returns) < 10:
            filtered_returns = returns  # Fallback if too many were filtered
        
        # Stage 3: Moving average (like 'sed' command - transforming)
        kernel_size = min(self.kernel_window, len(filtered_returns))
        moving_avgs = []
        
        for i in range(len(filtered_returns) - kernel_size + 1):
            window = filtered_returns[i:i+kernel_size]
            moving_avgs.append(np.mean(window))
        
        # Stage 4: Directional change detection (like 'awk' command - processing)
        direction_changes = np.diff(np.sign(moving_avgs))
        
        # Stage 5: Final aggregation (like 'sort | uniq -c' command - counting)
        pos_changes = np.sum(direction_changes > 0)
        neg_changes = np.sum(direction_changes < 0)
        total_changes = len(direction_changes)
        
        # Calculate normalized difference (like final output redirection)
        if total_changes > 0:
            signal = (pos_changes - neg_changes) / total_changes
        else:
            signal = 0.0
            
        # Add volatility information if volumes available
        if volumes is not None and len(volumes) > 20:
            # Avoid division by zero
            vol_changes = np.diff(volumes) / np.maximum(volumes[:-1], 1e-10)
            vol_ma = np.mean(vol_changes[-10:])
            
            # Adjust signal based on volume trend
            signal *= (1.0 + np.sign(vol_ma) * min(1.0, abs(vol_ma) * 5))
            
        return np.clip(signal, -1.0, 1.0)
    
    def _thompson_sampling(self, context: Dict) -> str:
        """
        Apply Thompson sampling for action selection
        
        Parameters
        ----------
        context : dict
            Current market context
            
        Returns
        -------
        str
            Selected action
        """
        # Define actions
        actions = ['long', 'short', 'neutral']
        
        # Calculate Beta distribution parameters for each action
        # Alpha = successes + 1, Beta = failures + 1 (add 1 to avoid division by zero)
        beta_params = {}
        
        for action in actions:
            rewards = self.arm_rewards.get(action, [0])
            # Success count is positive rewards
            successes = sum(1 for r in rewards if r > 0)
            # Failure count is negative rewards
            failures = sum(1 for r in rewards if r <= 0)
            
            # Beta distribution parameters
            alpha = successes + 1
            beta = failures + 1
            
            beta_params[action] = (alpha, beta)
            
        # Sample from Beta distributions
        samples = {}
        for action, (alpha, beta) in beta_params.items():
            # Simple approximation of beta sampling to avoid scipy dependency
            # Note: In a full implementation, would use scipy.stats.beta.rvs(alpha, beta)
            uniform_samples = [np.random.random() for _ in range(10)]
            # Transform samples with a simplified beta transformation
            beta_samples = [s ** (1 / alpha) * (1 - s) ** (1 / beta) for s in uniform_samples]
            samples[action] = np.mean(beta_samples)
            
        # Apply exploration rate
        if np.random.random() < self.exploration_rate:
            # Exploration: choose randomly
            chosen_action = np.random.choice(actions)
        else:
            # Exploitation: choose action with highest sample value
            chosen_action = max(samples.items(), key=lambda x: x[1])[0]
            
        return chosen_action
    
    def _update_rewards(self, action: str, price_change: float) -> None:
        """
        Update reward history for Thompson sampling
        
        Parameters
        ----------
        action : str
            Action taken
        price_change : float
            Resulting price change
        """
        # Calculate reward based on action and price change
        reward = 0.0
        
        if action == 'long':
            reward = price_change  # Positive when price goes up
        elif action == 'short':
            reward = -price_change  # Positive when price goes down
        elif action == 'neutral':
            reward = 1.0 - abs(price_change) * 10  # Positive when price unchanged
            
        # Add to reward history
        if action in self.arm_rewards:
            self.arm_rewards[action].append(reward)
            # Keep history capped
            if len(self.arm_rewards[action]) > 100:
                self.arm_rewards[action] = self.arm_rewards[action][-100:]
        else:
            self.arm_rewards[action] = [reward]
            
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and build Thompson-inspired models
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.pattern_length + 10:
            self.is_fitted = False
            return
            
        try:
            # Extract price and volume data
            prices = historical_df['close'].values
            volumes = historical_df['volume'].values if 'volume' in historical_df.columns else None
            
            # 1. Regex pattern matching
            current_pattern = self._encode_price_patterns(prices)
            
            # Build pattern library if needed
            if not self.pattern_library or len(self.pattern_library) < 50:
                # Process historical segments
                for i in range(len(prices) - self.pattern_length - 5):
                    segment = prices[i:i+self.pattern_length+5]
                    pattern = self._encode_price_patterns(segment[:-5])
                    
                    # Calculate outcome (5-step forward return)
                    outcome = (segment[-1] / segment[self.pattern_length-1]) - 1
                    
                    # Add to library
                    if pattern not in self.pattern_library:
                        self.pattern_library[pattern] = []
                    self.pattern_library[pattern].append(outcome)
            
            # 2. State machine construction
            self.state_machine = self._build_state_machine(prices)
            
            # 3. Apply Unix-like pipe processing
            pipe_signal = self._unix_pipe_process(prices, volumes)
            
            # 4. Pattern matching
            match_found, matched_pattern, match_confidence = self._thompson_regex_match(
                current_pattern, self.pattern_library
            )
            
            pattern_signal = 0.0
            if match_found and matched_pattern in self.pattern_library:
                outcomes = self.pattern_library[matched_pattern]
                pattern_signal = np.sign(np.mean(outcomes)) * match_confidence
            
            # 5. State machine prediction
            state_signal = 0.0
            if self.state_machine and 'transitions' in self.state_machine:
                current_state = self.state_machine.get('current_state', 2)  # Default to neutral
                transitions = self.state_machine['transitions']
                
                if current_state in transitions:
                    # Find most likely next state
                    next_state = max(transitions[current_state].items(), key=lambda x: x[1])[0]
                    
                    # Generate signal based on state transition
                    state_signal = (next_state - 2) / 2  # Map from [0,4] to [-1,1]
                    
                    # Weight by transition probability
                    prob = transitions[current_state][next_state]
                    state_signal *= prob
            
            # 6. Thompson sampling for exploration/exploitation
            context = {
                'pattern_signal': pattern_signal,
                'state_signal': state_signal,
                'pipe_signal': pipe_signal
            }
            
            chosen_action = self._thompson_sampling(context)
            
            # Convert action to signal
            if chosen_action == 'long':
                thompson_signal = 1.0
            elif chosen_action == 'short':
                thompson_signal = -1.0
            else:  # neutral
                thompson_signal = 0.0
                
            # Check recent price changes to update rewards
            if len(prices) >= 2:
                recent_change = (prices[-1] / prices[-2]) - 1
                # Update rewards for the last action we took
                self._update_rewards(chosen_action, recent_change)
            
            # 7. Combine signals
            combined_signal = (
                pattern_signal * 0.4 +
                state_signal * 0.3 +
                pipe_signal * 0.2 +
                thompson_signal * 0.1
            )
            
            self.latest_signal = np.clip(combined_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Thompson Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Thompson's computing principles
        
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
        return "Thompson Agent" 
    
    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Thompson sampling principles.
        
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
            logger.error(f"ValueError in Thompson strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Thompson strategy: {str(e)}")
            return 0.0000