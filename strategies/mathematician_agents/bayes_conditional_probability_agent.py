"""
Bayes Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Thomas Bayes's principles of
conditional probability and Bayesian inference.

This agent applies Bayes' theorem to update trading beliefs as new data arrives,
combining prior market beliefs with the likelihood of current price action to
generate posterior probabilities that drive trading decisions.

Concepts employed:
1. Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
2. Prior and posterior probability distributions
3. Bayesian updating with sequential data
4. Bayesian model comparison for regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class BayesAgent:
    """
    Trading agent based on Bayesian statistical principles.
    
    Parameters
    ----------
    lookback_window : int, default=60
        Window size for statistical calculations
    prior_bull_probability : float, default=0.5
        Initial probability assigned to bullish market (between 0 and 1)
    sensitivity : float, default=0.3
        How quickly the model updates probabilities (between 0 and 1)
    regime_count : int, default=3
        Number of market regimes to model (e.g., bullish, bearish, sideways)
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        prior_bull_probability: float = 0.5,
        sensitivity: float = 0.3,
        regime_count: int = 3
    ):
        self.lookback_window = lookback_window
        self.prior_bull_probability = prior_bull_probability
        self.sensitivity = sensitivity
        self.regime_count = regime_count
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Initialize regime probabilities
        self.regime_probs = np.ones(regime_count) / regime_count
        
        # Initialize return distribution parameters for each regime
        self.regime_params = {
            i: {'mean': 0.0, 'std': 0.01} for i in range(regime_count)
        }
        
        # State for Bayesian updating
        self.posterior_bull = prior_bull_probability
        
    def _calculate_likelihood(self, returns: np.ndarray, regime_idx: int) -> np.ndarray:
        """
        Calculate likelihood of observing returns given a specific regime
        
        Parameters
        ----------
        returns : numpy.ndarray
            Array of price returns
        regime_idx : int
            Index of the regime to calculate likelihood for
            
        Returns
        -------
        numpy.ndarray
            Array of likelihood values
        """
        mean = self.regime_params[regime_idx]['mean']
        std = self.regime_params[regime_idx]['std']
        
        # Normal PDF as likelihood function
        likelihoods = stats.norm.pdf(returns, loc=mean, scale=std)
        
        return likelihoods
    
    def _update_regime_parameters(self, returns: np.ndarray) -> None:
        """
        Update distribution parameters for each regime using Bayesian updating
        
        Parameters
        ----------
        returns : numpy.ndarray
            Array of price returns
        """
        # For the bullish regime: positive mean
        self.regime_params[0]['mean'] = max(0.0001, np.mean(returns[returns > 0]))
        self.regime_params[0]['std'] = max(0.001, np.std(returns[returns > 0]))
        
        # For the bearish regime: negative mean
        self.regime_params[1]['mean'] = min(-0.0001, np.mean(returns[returns < 0]))
        self.regime_params[1]['std'] = max(0.001, np.std(returns[returns < 0]))
        
        # For neutral/sideways regime: mean close to zero
        self.regime_params[2]['mean'] = 0.0
        self.regime_params[2]['std'] = max(0.001, np.std(returns) / 2)
        
    def _bayesian_update(self, returns: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Apply Bayes' theorem to update market beliefs
        
        Parameters
        ----------
        returns : numpy.ndarray
            Array of price returns
            
        Returns
        -------
        tuple
            (posterior_bull_probability, regime_probabilities)
        """
        # Update distribution parameters for each regime
        self._update_regime_parameters(returns)
        
        # Get the latest return for updating
        latest_return = returns[-1]
        
        # Calculate likelihood of this return under bullish and bearish hypotheses
        p_return_given_bull = stats.norm.pdf(
            latest_return, 
            loc=self.regime_params[0]['mean'],
            scale=self.regime_params[0]['std']
        )
        
        p_return_given_bear = stats.norm.pdf(
            latest_return, 
            loc=self.regime_params[1]['mean'],
            scale=self.regime_params[1]['std']
        )
        
        # Apply Bayes' theorem for binary classification
        prior_bear = 1.0 - self.posterior_bull
        
        # P(bull|return) = P(return|bull) * P(bull) / [P(return|bull) * P(bull) + P(return|bear) * P(bear)]
        denominator = (p_return_given_bull * self.posterior_bull + 
                      p_return_given_bear * prior_bear)
        
        if denominator > 0:
            posterior_bull = (p_return_given_bull * self.posterior_bull) / denominator
        else:
            posterior_bull = self.posterior_bull  # No update if denominator is zero
            
        # Adjust sensitivity of the update
        posterior_bull = (self.sensitivity * posterior_bull + 
                         (1 - self.sensitivity) * self.posterior_bull)
        
        # Update regime probabilities (multi-class version)
        likelihoods = np.array([
            self._calculate_likelihood(latest_return, i).item() 
            for i in range(self.regime_count)
        ])
        
        # Prior * likelihood
        unnormalized_posterior = self.regime_probs * likelihoods
        
        # Normalize to get posterior probabilities
        sum_posterior = np.sum(unnormalized_posterior)
        if sum_posterior > 0:
            regime_probs = unnormalized_posterior / sum_posterior
        else:
            regime_probs = self.regime_probs  # No update if sum is zero
            
        return posterior_bull, regime_probs
    
    def _extract_returns_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract return features for Bayesian analysis
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with 'close' column
            
        Returns
        -------
        dict
            Dictionary with return features
        """
        # Make a copy to avoid warnings
        df_copy = df.copy()
        
        # Calculate simple returns
        df_copy['returns'] = df_copy['close'].pct_change()
        
        # Calculate log returns (more theoretically appropriate for Bayesian analysis)
        df_copy['log_returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        
        # Apply lookback window
        recent_data = df_copy.iloc[-self.lookback_window:].copy()
        
        # Calculate volatility (standard deviation of returns)
        rolling_vol = recent_data['returns'].rolling(window=min(20, len(recent_data))).std()
        
        # Calculate direction features
        recent_data['direction'] = np.sign(recent_data['returns'])
        
        # Calculate streak lengths (consecutive up or down days)
        direction = recent_data['direction'].values
        streak = np.zeros_like(direction)
        
        for i in range(1, len(direction)):
            if direction[i] == direction[i-1] and direction[i] != 0:
                streak[i] = streak[i-1] + 1
            elif direction[i] != 0:
                streak[i] = 1
                
        recent_data['streak'] = streak
        
        return {
            'returns': recent_data['returns'].dropna().values,
            'log_returns': recent_data['log_returns'].dropna().values,
            'volatility': rolling_vol.dropna().values,
            'direction': direction,
            'streak': streak
        }
    
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
            # Extract features for Bayesian analysis
            features = self._extract_returns_features(historical_df)
            
            # Apply Bayesian updating
            self.posterior_bull, self.regime_probs = self._bayesian_update(features['returns'])
            
            # Generate trading signal from Bayesian probabilities
            
            # 1. Use posterior bull probability (scaled to [-1, 1])
            bull_signal = (self.posterior_bull - 0.5) * 2
            
            # 2. Use regime probabilities (bullish - bearish)
            regime_signal = self.regime_probs[0] - self.regime_probs[1]
            
            # 3. Incorporate trend strength based on streaks
            recent_streak = features['streak'][-1] if len(features['streak']) > 0 else 0
            recent_direction = features['direction'][-1] if len(features['direction']) > 0 else 0
            streak_factor = min(0.3, recent_streak * 0.1)  # Cap at 0.3
            streak_signal = recent_direction * streak_factor
            
            # Combine signals (with weights)
            combined_signal = (bull_signal * 0.4 + 
                              regime_signal * 0.4 + 
                              streak_signal * 0.2)
            
            # Clip to [-1, 1] range
            self.latest_signal = np.clip(combined_signal, -1.0, 1.0)
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Bayes Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Bayesian inference
        
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
        return "Bayes Agent" 