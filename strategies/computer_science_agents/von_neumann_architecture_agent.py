"""
Von Neumann Agent
~~~~~~~~~~~~~~~
Agent implementing trading strategies based on John von Neumann's pioneering work in computer
science, game theory, cellular automata, and the stored-program computer architecture.

Von Neumann made fundamental contributions to computing including:
1. The von Neumann architecture (stored-program computer)
2. Game theory and the minimax theorem
3. Self-replicating cellular automata
4. Monte Carlo methods

This agent models market behavior as a game-theoretic system with:
1. Minimax optimization for risk-reward balancing
2. Cellular automata for pattern detection
3. Monte Carlo simulations for probability estimation
4. Self-organizing systems for market regime detection

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
from typing import Dict, List, Optional, Tuple, Union
import logging
import math
from collections import deque
from scipy import stats

from ..agent import Agent

logger = logging.getLogger(__name__)

class VonNeumannAgent(Agent):
    """
    Trading agent based on John von Neumann's mathematical and computing principles.
    
    Parameters
    ----------
    game_window : int, default=30
        Window size for game-theoretic analysis
    monte_carlo_sims : int, default=100
        Number of Monte Carlo simulations
    automata_rule : int, default=30
        Cellular automata rule number (0-255)
    self_organization_threshold : float, default=0.3
        Threshold for detecting self-organizing regimes
    minimax_depth : int, default=3
        Depth of minimax search for decision making
    """
    
    def __init__(
        self,
        game_window: int = 30,
        monte_carlo_sims: int = 100,
        automata_rule: int = 30,
        self_organization_threshold: float = 0.3,
        minimax_depth: int = 3
    ):
        self.game_window = game_window
        self.monte_carlo_sims = monte_carlo_sims
        self.automata_rule = automata_rule
        self.self_organization_threshold = self_organization_threshold
        self.minimax_depth = minimax_depth
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.market_state = {}
        self.simulated_paths = []
        self.regime_history = deque(maxlen=30)
    
    def _game_theory_analysis(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        Apply game theory to model market behavior as strategic interactions
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        float
            Game-theoretic signal
        """
        if len(prices) < self.game_window:
            return 0.0
            
        # Use recent window
        recent_prices = prices[-self.game_window:]
        recent_returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Check for invalid returns
        if len(recent_returns) == 0 or np.all(np.isnan(recent_returns)):
            return 0.0
        
        # Remove NaN values from returns
        recent_returns = recent_returns[~np.isnan(recent_returns)]
        
        if len(recent_returns) == 0:
            return 0.0
        
        # Define "market" and "trader" as two players in a zero-sum game
        # Market moves: price up, price down, price unchanged
        # Trader moves: buy, sell, hold
        
        # Create payoff matrix based on recent market behavior
        payoff_matrix = np.zeros((3, 3))  # [trader_move, market_move]
        
        # Fill payoff matrix based on returns distribution
        up_returns = recent_returns[recent_returns > 0]
        down_returns = recent_returns[recent_returns < 0]
        flat_returns = recent_returns[recent_returns == 0]
        
        # Expected values for different market moves
        up_ev = np.mean(up_returns) if len(up_returns) > 0 else 0.001
        down_ev = np.mean(down_returns) if len(down_returns) > 0 else -0.001
        flat_ev = 0.0
        
        # Handle NaN values in expected values
        if np.isnan(up_ev):
            up_ev = 0.001
        if np.isnan(down_ev):
            down_ev = -0.001
        
        # Probabilities of market moves
        p_up = len(up_returns) / len(recent_returns) if len(recent_returns) > 0 else 0.33
        p_down = len(down_returns) / len(recent_returns) if len(recent_returns) > 0 else 0.33
        p_flat = len(flat_returns) / len(recent_returns) if len(recent_returns) > 0 else 0.34
        
        # Payoff matrix [trader_move, market_move]
        # Trader moves: 0=buy, 1=hold, 2=sell
        # Market moves: 0=up, 1=flat, 2=down
        
        # If trader buys
        payoff_matrix[0, 0] = up_ev      # Trader buys, market goes up: win
        payoff_matrix[0, 1] = 0          # Trader buys, market flat: neutral
        payoff_matrix[0, 2] = down_ev    # Trader buys, market goes down: loss
        
        # If trader holds
        payoff_matrix[1, 0] = 0          # Trader holds, market goes up: neutral
        payoff_matrix[1, 1] = 0          # Trader holds, market flat: neutral
        payoff_matrix[1, 2] = 0          # Trader holds, market goes down: neutral
        
        # If trader sells
        payoff_matrix[2, 0] = -up_ev     # Trader sells, market goes up: loss
        payoff_matrix[2, 1] = 0          # Trader sells, market flat: neutral
        payoff_matrix[2, 2] = -down_ev   # Trader sells, market goes down: win
        
        # Apply minimax to find optimal strategy
        trader_strategy = self._minimax_solve(payoff_matrix, [p_up, p_flat, p_down])
        
        # Calculate expected signal based on optimal strategy
        signal = trader_strategy[0] - trader_strategy[2]  # buy probability - sell probability
        
        # Ensure signal is valid
        if np.isnan(signal) or np.isinf(signal):
            signal = 0.0
        
        return np.clip(signal, -1.0, 1.0)
    
    def _minimax_solve(self, payoff_matrix: np.ndarray, market_probs: List[float]) -> List[float]:
        """
        Solve game using minimax principle
        
        Parameters
        ----------
        payoff_matrix : numpy.ndarray
            Payoff matrix for the game
        market_probs : list
            Probability distribution of market moves
            
        Returns
        -------
        list
            Optimal trader strategy (probabilities)
        """
        # Validate inputs
        if np.any(np.isnan(payoff_matrix)) or np.any(np.isnan(market_probs)):
            return [0.33, 0.34, 0.33]  # Default equal strategy
        
        # Simple case: determine dominant strategy based on expected value
        ev_buy = np.sum(payoff_matrix[0, :] * market_probs)
        ev_hold = np.sum(payoff_matrix[1, :] * market_probs)
        ev_sell = np.sum(payoff_matrix[2, :] * market_probs)
        
        # Handle NaN values in expected values
        if np.isnan(ev_buy):
            ev_buy = 0.0
        if np.isnan(ev_hold):
            ev_hold = 0.0
        if np.isnan(ev_sell):
            ev_sell = 0.0
        
        # If one strategy dominates
        if ev_buy > ev_hold and ev_buy > ev_sell:
            return [0.8, 0.1, 0.1]  # Mostly buy
        elif ev_sell > ev_hold and ev_sell > ev_buy:
            return [0.1, 0.1, 0.8]  # Mostly sell
        elif ev_hold > ev_buy and ev_hold > ev_sell:
            return [0.1, 0.8, 0.1]  # Mostly hold
            
        # Mixed strategy case
        # Normalize to ensure non-negative values for probability calculation
        adjusted_evs = [ev_buy - min(ev_buy, ev_hold, ev_sell) + 0.01,
                        ev_hold - min(ev_buy, ev_hold, ev_sell) + 0.01,
                        ev_sell - min(ev_buy, ev_hold, ev_sell) + 0.01]
        
        # Normalize to get probabilities
        total = sum(adjusted_evs)
        if total <= 0:
            return [0.33, 0.34, 0.33]  # Default equal strategy
            
        strategy = [ev / total for ev in adjusted_evs]
        
        # Validate strategy
        if any(np.isnan(s) or np.isinf(s) for s in strategy):
            return [0.33, 0.34, 0.33]  # Default equal strategy
        
        return strategy
    
    def _cellular_automata(self, prices: np.ndarray) -> float:
        """
        Apply cellular automata to detect price patterns
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Cellular automata based signal
        """
        if len(prices) < 20:
            return 0.0
            
        # Create binary representation of price movements
        returns = np.diff(prices) / prices[:-1]
        
        # Check for invalid returns
        if len(returns) == 0 or np.all(np.isnan(returns)):
            return 0.0
        
        # Remove NaN values from returns
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return 0.0
        
        # Convert to binary state: 1 for up, 0 for down or flat
        binary_state = (returns > 0).astype(int)
        
        # Apply elementary cellular automaton rule
        # Convert rule number to binary
        rule_binary = [int(x) for x in np.binary_repr(self.automata_rule, width=8)]
        
        # Run cellular automaton for several generations
        num_cells = len(binary_state)
        if num_cells < 3:  # Need at least 3 cells for neighborhood
            return 0.0
            
        generations = min(5, num_cells // 4)
        if generations < 1:
            generations = 1
        
        automaton_states = np.zeros((generations + 1, num_cells))
        automaton_states[0, :] = binary_state
        
        for gen in range(1, generations + 1):
            for i in range(1, num_cells - 1):
                # Get neighborhood configuration
                left = automaton_states[gen-1, i-1]
                center = automaton_states[gen-1, i]
                right = automaton_states[gen-1, i+1]
                
                # Calculate rule index
                idx = int(left * 4 + center * 2 + right)
                
                # Apply rule
                automaton_states[gen, i] = rule_binary[7 - idx]
        
        # Analyze final generation pattern
        final_gen = automaton_states[-1, :]
        
        # Calculate Shannon entropy of the pattern
        values, counts = np.unique(final_gen, return_counts=True)
        if len(values) == 0:
            return 0.0
            
        probs = counts / len(final_gen)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Handle NaN entropy
        if np.isnan(entropy) or np.isinf(entropy):
            entropy = 0.0
        
        # Normalize entropy to [0, 1]
        max_entropy = 1.0  # Maximum entropy for binary state
        normalized_entropy = entropy / max_entropy
        
        # Calculate majority of states in final generation
        majority = 1 if np.mean(final_gen) > 0.5 else -1
        
        # Signal strength based on pattern clarity (inverse of entropy)
        strength = 1.0 - normalized_entropy
        
        # Final signal
        signal = majority * strength
        
        # Ensure signal is valid
        if np.isnan(signal) or np.isinf(signal):
            signal = 0.0
        
        return np.clip(signal, -1.0, 1.0)
    
    def _monte_carlo_simulation(self, prices: np.ndarray, returns: np.ndarray) -> float:
        """
        Use Monte Carlo simulation to estimate future price probabilities
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        returns : numpy.ndarray
            Array of historical returns
            
        Returns
        -------
        float
            Monte Carlo based signal
        """
        if len(returns) < 20:
            return 0.0
            
        # Estimate return distribution parameters
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Handle edge case where sigma is 0 or very small
        if sigma < 1e-10:
            return 0.0
            
        # Handle NaN values
        if np.isnan(mu) or np.isnan(sigma):
            return 0.0
        
        # Current price
        current_price = prices[-1]
        
        # Run Monte Carlo simulations
        forecast_horizon = 5  # Days
        price_paths = np.zeros((self.monte_carlo_sims, forecast_horizon + 1))
        
        # Initialize all paths with current price
        price_paths[:, 0] = current_price
        
        # Generate paths
        for path in range(self.monte_carlo_sims):
            for day in range(1, forecast_horizon + 1):
                # Sample return from historical distribution
                if np.random.random() < 0.7:
                    # Sample from fitted normal distribution
                    r = np.random.normal(mu, sigma)
                else:
                    # Sample from historical returns to capture fat tails
                    r = np.random.choice(returns)
                    
                # Apply return to get next price
                price_paths[path, day] = price_paths[path, day-1] * (1 + r)
        
        # Store simulated paths for internal state
        self.simulated_paths = price_paths
        
        # Calculate probability estimates
        final_prices = price_paths[:, -1]
        prob_up = np.mean(final_prices > current_price)
        prob_down = np.mean(final_prices < current_price)
        
        # Expected value
        expected_return = np.mean(final_prices / current_price - 1)
        
        # Handle NaN values in calculations
        if np.isnan(prob_up) or np.isnan(prob_down) or np.isnan(expected_return):
            return 0.0
        
        # Calculate signal based on probability differential and expected return
        signal = (prob_up - prob_down) * 0.7 + np.sign(expected_return) * min(1.0, abs(expected_return) * 10) * 0.3
        
        # Ensure signal is a valid number
        if np.isnan(signal) or np.isinf(signal):
            signal = 0.0
            
        return np.clip(signal, -1.0, 1.0)
    
    def _self_organizing_system(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        Model market as a self-organizing system
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        float
            Self-organization based signal
        """
        if len(prices) < 25:
            return 0.0
            
        # Identify market regimes using self-organization principles
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate key metrics for regime identification
        volatility = np.std(returns[-20:])
        mean_return = np.mean(returns[-20:])
        
        # Volume analysis if available
        volume_trend = 0.0
        if volumes is not None and len(volumes) >= 30:
            recent_vol = volumes[-20:]
            # Check if volume data has sufficient variance for correlation
            if np.std(recent_vol) > 1e-10:
                try:
                    corr_matrix = np.corrcoef(np.arange(len(recent_vol)), recent_vol)
                    if not np.isnan(corr_matrix[0, 1]):
                        volume_trend = corr_matrix[0, 1]
                except:
                    volume_trend = 0.0
            
        # Autocorrelation as a measure of trend persistence
        autocorr = 0.0
        if len(returns) >= 21:
            try:
                returns_lag0 = returns[-21:-1]
                returns_lag1 = returns[-20:]
                
                # Check if both arrays have sufficient variance
                if np.std(returns_lag0) > 1e-10 and np.std(returns_lag1) > 1e-10:
                    corr_matrix = np.corrcoef(returns_lag0, returns_lag1)
                    if not np.isnan(corr_matrix[0, 1]):
                        autocorr = corr_matrix[0, 1]
            except:
                autocorr = 0.0
        
        # Handle NaN values in autocorr
        if np.isnan(autocorr):
            autocorr = 0.0
            
        # Define regimes
        # 1. Trending Market: Low to moderate volatility, persistent returns
        # 2. Mean-Reverting Market: Moderate volatility, negative autocorrelation
        # 3. Chaotic Market: High volatility, low predictability
        # 4. Stable Market: Low volatility, low directional bias
        
        regime_scores = {
            'trending': min(1.0, max(0.0, 0.5 + autocorr * 2)) * (1.0 - min(1.0, volatility * 20)),
            'mean_reverting': min(1.0, max(0.0, 0.5 - autocorr * 2)) * min(1.0, volatility * 10),
            'chaotic': min(1.0, volatility * 30),
            'stable': max(0.0, 1.0 - volatility * 20) * (1.0 - abs(autocorr))
        }
        
        # Ensure all regime scores are valid numbers
        for regime in regime_scores:
            if np.isnan(regime_scores[regime]) or np.isinf(regime_scores[regime]):
                regime_scores[regime] = 0.0
        
        # Determine dominant regime
        dominant_regime = max(regime_scores, key=regime_scores.get)
        regime_strength = regime_scores[dominant_regime]
        
        # Store for internal tracking
        self.regime_history.append(dominant_regime)
        
        # Generate signal based on regime
        signal = 0.0
        
        if dominant_regime == 'trending':
            # In trending markets, follow the trend
            signal = np.sign(mean_return) * regime_strength
        elif dominant_regime == 'mean_reverting':
            # In mean-reverting markets, go against recent move
            signal = -np.sign(mean_return) * regime_strength
        elif dominant_regime == 'chaotic':
            # In chaotic markets, reduce exposure
            signal = 0.0
        elif dominant_regime == 'stable':
            # In stable markets, use volume signals if available
            if abs(volume_trend) > 1e-10:
                signal = np.sign(volume_trend) * regime_strength
            else:
                signal = np.sign(mean_return) * regime_strength * 0.5
        
        # Ensure signal is a valid number
        if np.isnan(signal) or np.isinf(signal):
            signal = 0.0
            
        return np.clip(signal, -1.0, 1.0)
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data to learn von Neumann-inspired patterns
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        # Need at least game_window + 10 for proper analysis
        min_required = max(self.game_window + 10, 40)
        if len(historical_df) < min_required:
            self.is_fitted = False
            return
            
        try:
            # Extract price and volume data
            prices = historical_df['close'].values
            volumes = historical_df['volume'].values if 'volume' in historical_df.columns else None
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            
            # Apply von Neumann's concepts with individual error handling
            
            # 1. Game Theory Analysis
            try:
                game_signal = self._game_theory_analysis(prices, volumes)
                if np.isnan(game_signal) or np.isinf(game_signal):
                    game_signal = 0.0
            except Exception as e:
                logger.warning(f"Game theory analysis failed: {e}")
                game_signal = 0.0
            
            # 2. Cellular Automata
            try:
                automata_signal = self._cellular_automata(prices)
                if np.isnan(automata_signal) or np.isinf(automata_signal):
                    automata_signal = 0.0
            except Exception as e:
                logger.warning(f"Cellular automata analysis failed: {e}")
                automata_signal = 0.0
            
            # 3. Monte Carlo Simulation
            try:
                monte_carlo_signal = self._monte_carlo_simulation(prices, returns)
                if np.isnan(monte_carlo_signal) or np.isinf(monte_carlo_signal):
                    monte_carlo_signal = 0.0
            except Exception as e:
                logger.warning(f"Monte Carlo simulation failed: {e}")
                monte_carlo_signal = 0.0
            
            # 4. Self-Organizing Systems
            try:
                self_org_signal = self._self_organizing_system(prices, volumes)
                if np.isnan(self_org_signal) or np.isinf(self_org_signal):
                    self_org_signal = 0.0
            except Exception as e:
                logger.warning(f"Self-organizing system analysis failed: {e}")
                self_org_signal = 0.0
            
            # Combine signals with weights based on the strength of each approach
            combined_signal = (
                game_signal * 0.3 +
                automata_signal * 0.2 +
                monte_carlo_signal * 0.3 +
                self_org_signal * 0.2
            )
            
            # Final validation
            if np.isnan(combined_signal) or np.isinf(combined_signal):
                combined_signal = 0.0
            
            self.latest_signal = np.clip(combined_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Von Neumann Agent fit: {e}")
            self.latest_signal = 0.0
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on von Neumann's computing principles
        
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
        return "Von Neumann Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using von Neumann's architectural principles.
        
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
            logger.error(f"ValueError in Von Neumann strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Von Neumann strategy: {str(e)}")
            return 0.0000 