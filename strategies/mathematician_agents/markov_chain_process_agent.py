"""
Markov Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Andrey Markov's principles of
Markov chains and stochastic processes.

This agent models price movements as a Markov process, where future states
depend only on the current state and not the path taken to reach it. The agent
discovers transition probabilities between market states and uses these to
predict future movements.

Concepts employed:
1. Markov chains for state transitions
2. Hidden Markov models for regime detection
3. Transition probability matrices 
4. Markov decision processes for optimal action selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class MarkovAgent:
    """
    Trading agent based on Markov chain principles.
    
    Parameters
    ----------
    lookback_window : int, default=100
        Window size for training the Markov model
    state_count : int, default=5
        Number of market states to model
    state_features : List[str], default=['return', 'volatility']
        Features to use for state classification
    transition_smoothing : float, default=0.05
        Smoothing factor for transition probabilities (0 to 1)
    prediction_horizon : int, default=5
        Steps ahead to predict
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        state_count: int = 5,
        state_features: List[str] = ['return', 'volatility'],
        transition_smoothing: float = 0.05,
        prediction_horizon: int = 5
    ):
        self.lookback_window = lookback_window
        self.state_count = state_count
        self.state_features = state_features
        self.transition_smoothing = transition_smoothing
        self.prediction_horizon = prediction_horizon
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Initialize state transition matrix
        self.transition_matrix = np.ones((state_count, state_count)) / state_count
        
        # Track state histories
        self.state_history = []
        self.state_returns = np.zeros(state_count)
        
    def _extract_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for state identification
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with 'close' and other columns
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with extracted features
        """
        # Make a copy to avoid warnings
        df_copy = df.copy()
        
        # Calculate price returns
        df_copy['return'] = df_copy['close'].pct_change()
        
        # Calculate logarithmic returns
        df_copy['log_return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        
        # Calculate volatility (rolling standard deviation of returns)
        df_copy['volatility'] = df_copy['return'].rolling(window=20).std()
        
        # Calculate volume features if volume is available
        if 'volume' in df_copy.columns:
            # Normalized volume
            df_copy['volume_norm'] = df_copy['volume'] / df_copy['volume'].rolling(window=20).mean()
            
            # Volume delta
            df_copy['volume_delta'] = df_copy['volume'].pct_change()
            
        # Calculate range
        if all(col in df_copy.columns for col in ['high', 'low']):
            df_copy['range'] = (df_copy['high'] - df_copy['low']) / df_copy['close']
        
        # Momentum indicators
        df_copy['momentum_5'] = df_copy['close'].pct_change(periods=5)
        df_copy['momentum_10'] = df_copy['close'].pct_change(periods=10)
        
        # Calculate available features based on state_features list
        available_features = [f for f in self.state_features if f in df_copy.columns]
        
        # Ensure we have at least one feature
        if not available_features:
            available_features = ['return']
            
        return df_copy[available_features].dropna()
    
    def _classify_states(self, features: pd.DataFrame) -> np.ndarray:
        """
        Classify market states using clustering
        
        Parameters
        ----------
        features : pandas.DataFrame
            Feature data for clustering
            
        Returns
        -------
        numpy.ndarray
            Array of state classifications
        """
        if len(features) < self.state_count:
            # Not enough data points to cluster
            return np.zeros(len(features), dtype=int)
            
        # Normalize features for clustering
        norm_features = (features - features.mean()) / features.std().clip(lower=1e-8)
        
        # Cluster into states
        kmeans = KMeans(n_clusters=self.state_count, random_state=42, n_init=10)
        states = kmeans.fit_predict(norm_features.values)
        
        return states
    
    def _calculate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Calculate state transition probability matrix
        
        Parameters
        ----------
        states : numpy.ndarray
            Array of state classifications
            
        Returns
        -------
        numpy.ndarray
            State transition matrix
        """
        n = self.state_count
        transitions = np.zeros((n, n))
        
        # Count transitions
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i+1]
            transitions[from_state, to_state] += 1
            
        # Apply smoothing (add small value to prevent zero probabilities)
        transitions += self.transition_smoothing
        
        # Convert to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = transitions / row_sums
        
        # Smooth with previous transition matrix for stability
        if hasattr(self, 'transition_matrix'):
            transition_matrix = (0.8 * transition_matrix + 
                                0.2 * self.transition_matrix)
            
        return transition_matrix
    
    def _calculate_state_returns(self, states: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Calculate average returns for each state
        
        Parameters
        ----------
        states : numpy.ndarray
            Array of state classifications
        returns : numpy.ndarray
            Array of corresponding returns
            
        Returns
        -------
        numpy.ndarray
            Average return for each state
        """
        state_returns = np.zeros(self.state_count)
        
        for state in range(self.state_count):
            state_mask = states == state
            if np.any(state_mask):
                state_returns[state] = returns[state_mask].mean()
                
        return state_returns
    
    def _forecast_probabilities(self, current_state: int) -> np.ndarray:
        """
        Forecast state probabilities using Markov chain
        
        Parameters
        ----------
        current_state : int
            The current market state
            
        Returns
        -------
        numpy.ndarray
            Probability distribution over states after prediction_horizon steps
        """
        # Initial state probability (one-hot encoded)
        state_probs = np.zeros(self.state_count)
        state_probs[current_state] = 1.0
        
        # Propagate through the transition matrix
        for _ in range(self.prediction_horizon):
            state_probs = state_probs @ self.transition_matrix
            
        return state_probs
    
    def _calculate_expected_return(self, state_probs: np.ndarray) -> float:
        """
        Calculate expected return given state probabilities
        
        Parameters
        ----------
        state_probs : numpy.ndarray
            Probability distribution over states
            
        Returns
        -------
        float
            Expected return
        """
        return np.sum(state_probs * self.state_returns)
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate Markov model
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.lookback_window:
            self.is_fitted = False
            return
            
        try:
            # Extract features for state identification
            features_df = self._extract_state_features(historical_df)
            
            # Need more data than self.state_count for proper clustering
            if len(features_df) < self.state_count * 3:
                self.is_fitted = False
                return
                
            # Extract returns column for later use
            returns = features_df['return'].values if 'return' in features_df.columns else np.zeros(len(features_df))
            
            # Classify states
            states = self._classify_states(features_df)
            
            # Calculate transition probabilities
            self.transition_matrix = self._calculate_transition_matrix(states)
            
            # Calculate average return for each state
            self.state_returns = self._calculate_state_returns(states, returns)
            
            # Store current state
            self.current_state = states[-1]
            
            # Update state history
            self.state_history.append(self.current_state)
            if len(self.state_history) > self.lookback_window:
                self.state_history = self.state_history[-self.lookback_window:]
                
            # Forecast future state probabilities
            future_probs = self._forecast_probabilities(self.current_state)
            
            # Calculate expected return
            expected_return = self._calculate_expected_return(future_probs)
            
            # Calculate persistence probability (probability of staying in current state)
            persistence = self.transition_matrix[self.current_state, self.current_state]
            
            # Generate signal based on expected return and state persistence
            # Scale signal by expected return (assuming typical returns are small)
            signal_scale = 20.0  # Scaling factor to convert expected return to signal
            base_signal = expected_return * signal_scale
            
            # Adjust signal based on state persistence and current state returns
            if persistence > 0.5:  # High persistence means current trend likely continues
                current_return = self.state_returns[self.current_state]
                persistence_boost = (persistence - 0.5) * 2  # Scale to [0, 1]
                base_signal = base_signal + np.sign(current_return) * persistence_boost * 0.3
                
            # Clip signal to valid range
            self.latest_signal = np.clip(base_signal, -1.0, 1.0)
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Markov Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Markov chain predictions
        
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
        return "Markov Agent" 