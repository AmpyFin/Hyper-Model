"""
Knuth Agent
~~~~~~~~~~
Agent implementing trading strategies based on Donald Knuth's work on algorithm design,
analysis of algorithms, and his comprehensive study of computational techniques in
"The Art of Computer Programming".

Knuth's contributions to computer science include algorithm analysis, optimization
techniques, combinatorial algorithms, and the concept of "premature optimization".

This agent models market analysis using Knuth's algorithmic principles:
1. Algorithmic Efficiency: Analyzing market data with optimal computational techniques
2. Big-O Analysis: Identifying growth patterns in price/volume trends
3. Combinatorial Pattern Matching: Finding significant price patterns
4. Balanced Search Trees: Maintaining balanced risk-reward profiles

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
from collections import deque, defaultdict

from ..agent import Agent

logger = logging.getLogger(__name__)

class KnuthAgent(Agent):
    """
    Trading agent based on Donald Knuth's algorithmic principles.
    
    Parameters
    ----------
    analysis_window : int, default=120
        Window size for algorithm analysis
    pattern_complexity : int, default=3
        Complexity level for pattern matching (1=simple, 5=complex)
    optimization_level : int, default=2
        Level of algorithmic optimization
    tree_balancing_factor : float, default=0.6
        Factor controlling risk-reward balance
    combinatorial_depth : int, default=4
        Depth of combinatorial pattern search
    """
    
    def __init__(
        self,
        analysis_window: int = 120,
        pattern_complexity: int = 3,
        optimization_level: int = 2,
        tree_balancing_factor: float = 0.6,
        combinatorial_depth: int = 4
    ):
        self.analysis_window = analysis_window
        self.pattern_complexity = pattern_complexity
        self.optimization_level = optimization_level
        self.tree_balancing_factor = tree_balancing_factor
        self.combinatorial_depth = combinatorial_depth
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.pattern_library = {}
        self.growth_rates = {}
        self.search_tree = {}
        
    def _analyze_algorithmic_efficiency(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze the efficiency of various market indicators
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            Dictionary of efficiency metrics
        """
        if len(prices) < self.analysis_window:
            return {}
            
        # Use recent window
        recent_prices = prices[-self.analysis_window:]
        recent_vols = volumes[-self.analysis_window:] if volumes is not None else None
        
        # Calculate returns and metrics
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Define algorithmic efficiency metrics
        metrics = {}
        
        # 1. Linear scan efficiency (O(n))
        metrics['linear_momentum'] = np.sum(returns[-20:]) * 20
        
        # 2. Logarithmic search efficiency (O(log n))
        # Simulate a binary search process by comparing first/last half returns
        mid = len(returns) // 2
        first_half = np.sum(returns[:mid])
        second_half = np.sum(returns[mid:])
        metrics['log_momentum'] = np.sign(second_half - first_half) * np.log2(1 + abs(second_half - first_half))
        
        # 3. Quadratic complexity (O(n^2)) - correlation matrix density
        if len(returns) >= 30:
            slices = [returns[i:i+10] for i in range(0, 30, 10)]
            corr_count = 0
            total = 0
            
            for i in range(len(slices)):
                for j in range(i+1, len(slices)):
                    corr = np.corrcoef(slices[i], slices[j])[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.7:
                        corr_count += 1
                    total += 1
            
            metrics['quadratic_density'] = corr_count / total if total > 0 else 0
        
        # 4. Divide-and-conquer efficiency
        # Compare recursive subdivisions of return series
        metrics['divide_conquer'] = self._recursive_analysis(returns, 0, len(returns)-1, 2)
        
        # 5. Greedy algorithm efficiency
        # Cumulative optimal path of positive returns
        greedy_path = 0
        running_sum = 0
        for ret in returns:
            running_sum = max(0, running_sum + ret)
            greedy_path = max(greedy_path, running_sum)
        metrics['greedy_efficiency'] = min(1.0, greedy_path)
        
        # Include volume-weighted metrics if available
        if recent_vols is not None:
            vol_returns = returns * recent_vols[1:] / np.mean(recent_vols[1:])
            metrics['volume_weighted_momentum'] = np.sum(vol_returns[-20:]) / 20
        
        return metrics
    
    def _recursive_analysis(self, returns: np.ndarray, start: int, end: int, depth: int) -> float:
        """
        Perform recursive analysis of returns using divide-and-conquer approach
        
        Parameters
        ----------
        returns : numpy.ndarray
            Array of returns
        start : int
            Start index
        end : int
            End index
        depth : int
            Current recursion depth
            
        Returns
        -------
        float
            Recursion score
        """
        if depth <= 0 or end - start < 3:
            return np.sum(returns[start:end+1]) / (end - start + 1)
            
        mid = (start + end) // 2
        left = self._recursive_analysis(returns, start, mid, depth - 1)
        right = self._recursive_analysis(returns, mid + 1, end, depth - 1)
        
        # Combine results with weight based on which half is more "efficient"
        if abs(left) > abs(right):
            return 0.7 * left + 0.3 * right
        else:
            return 0.3 * left + 0.7 * right
    
    def _big_o_growth_analysis(self, prices: np.ndarray) -> Dict[str, Tuple[str, float]]:
        """
        Analyze growth patterns in price data using Big-O notation concepts
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Dictionary of growth patterns and rates
        """
        if len(prices) < self.analysis_window:
            return {}
            
        # Calculate returns at different timescales
        returns = {}
        windows = [1, 3, 5, 10, 20, 50]
        
        for window in windows:
            if window < len(prices):
                # Calculate returns over different window sizes
                rets = np.array([
                    (prices[i] - prices[i-window]) / prices[i-window]
                    for i in range(window, len(prices))
                ])
                returns[window] = rets
        
        # Analyze growth patterns
        growth_patterns = {}
        
        # Check if data fits different growth patterns (O(1), O(log n), O(n), O(n log n), O(n^2))
        for window in returns:
            if len(returns[window]) < 30:
                continue
                
            # Recent returns for this window
            recent = returns[window][-30:]
            
            # Detect patterns
            # 1. Constant growth O(1)
            constant_score = 1.0 - np.std(recent) / (np.mean(np.abs(recent)) + 1e-10)
            
            # 2. Logarithmic growth O(log n)
            x = np.arange(1, len(recent) + 1)
            log_fit = np.polyfit(np.log(x), recent, 1)[0]
            log_score = log_fit
            
            # 3. Linear growth O(n)
            linear_fit = np.polyfit(x, recent, 1)[0]
            linear_score = linear_fit
            
            # 4. Superlinear growth O(n log n)
            nlogn_fit = np.polyfit(x * np.log(x), recent, 1)[0]
            nlogn_score = nlogn_fit
            
            # 5. Quadratic growth O(n^2)
            quad_fit = np.polyfit(x**2, recent, 1)[0]
            quad_score = quad_fit
            
            # Find most likely growth pattern
            scores = {
                'O(1)': constant_score,
                'O(log n)': log_score,
                'O(n)': linear_score,
                'O(n log n)': nlogn_score,
                'O(n^2)': quad_score
            }
            
            # Get dominant pattern and its score
            dominant_pattern = max(scores.items(), key=lambda x: abs(x[1]))
            growth_patterns[f'window_{window}'] = dominant_pattern
            
        return growth_patterns
    
    def _knuth_morris_pratt_pattern(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Use KMP algorithm (developed by Knuth and others) for pattern matching
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Dictionary of pattern match strengths
        """
        if len(prices) < 20:
            return {}
            
        # Convert price series to a discrete "string" for pattern matching
        # Use direction changes as the alphabet
        returns = np.diff(prices) / prices[:-1]
        directions = np.sign(returns).astype(int)
        
        # Add 1 to make all values non-negative (0, 1, 2 instead of -1, 0, 1)
        directions = directions + 1
        
        # Find patterns of different lengths
        pattern_matches = {}
        
        for pattern_length in range(3, 3 + self.pattern_complexity):
            if len(directions) < 2 * pattern_length:
                continue
                
            # Look for repeating patterns in the recent price movements
            for i in range(len(directions) - 2 * pattern_length):
                pattern = directions[i:i+pattern_length]
                pattern_str = ''.join(map(str, pattern))
                
                # Use KMP-like prefix table to find matches
                matches = []
                start_pos = i + pattern_length  # Look for matches after this pattern
                
                j = start_pos
                while j <= len(directions) - pattern_length:
                    candidate = directions[j:j+pattern_length]
                    if np.array_equal(pattern, candidate):
                        matches.append(j)
                        j += pattern_length  # Skip ahead
                    else:
                        j += 1
                        
                if matches:
                    # Calculate what typically happens after this pattern
                    next_moves = []
                    for pos in [i] + matches:
                        if pos + pattern_length < len(directions):
                            next_dir = directions[pos + pattern_length]
                            next_moves.append(next_dir)
                    
                    if next_moves:
                        # Predict based on most common next move
                        counts = np.bincount(next_moves)
                        most_common = np.argmax(counts)
                        confidence = counts[most_common] / len(next_moves)
                        
                        # Convert back to -1, 0, 1 direction
                        signal_dir = most_common - 1
                        
                        # Add to pattern library with confidence
                        pattern_matches[pattern_str] = signal_dir * confidence
        
        # Maintain pattern library
        self.pattern_library = pattern_matches
        
        # If current ending matches a known pattern, extract the signal
        current_pattern = None
        if len(directions) >= self.pattern_complexity + 2:
            current_pattern = ''.join(map(str, directions[-(self.pattern_complexity + 2):-2]))
            
        signal = 0.0
        if current_pattern in pattern_matches:
            signal = pattern_matches[current_pattern]
            
        result = {'current_pattern_signal': signal}
        result.update({k: v for k, v in pattern_matches.items() if abs(v) > 0.7})
        
        return result
    
    def _balanced_tree_analysis(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        Use balanced search tree concepts for risk-reward optimization
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        float
            Balance-based signal
        """
        if len(prices) < 30:
            return 0.0
            
        # Create a simplified balanced tree with keys representing price levels
        # and values representing frequency/volume at those levels
        recent_prices = prices[-min(self.analysis_window, len(prices)):]
        
        # Create price bins (tree nodes)
        min_price = np.min(recent_prices)
        max_price = np.max(recent_prices)
        range_size = max_price - min_price
        
        # Skip if price range is too small
        if range_size < 1e-6:
            return 0.0
            
        # Number of bins based on complexity
        num_bins = 10 + 2 * self.pattern_complexity
        bin_size = range_size / num_bins
        
        # Create tree (dictionary of price bins)
        tree = {i: 0 for i in range(num_bins)}
        
        # Fill tree with price frequency counts
        for price in recent_prices:
            bin_idx = min(num_bins - 1, int((price - min_price) / bin_size))
            tree[bin_idx] += 1
            
        # If volumes are available, weight by volume
        if volumes is not None:
            recent_volumes = volumes[-min(self.analysis_window, len(volumes)):]
            if len(recent_volumes) == len(recent_prices):
                tree = {i: 0 for i in range(num_bins)}
                for price, volume in zip(recent_prices, recent_volumes):
                    bin_idx = min(num_bins - 1, int((price - min_price) / bin_size))
                    tree[bin_idx] += volume
        
        # Calculate tree balance
        total_weight = sum(tree.values())
        weighted_sum = sum(idx * weight for idx, weight in tree.items())
        centroid = weighted_sum / total_weight if total_weight > 0 else num_bins / 2
        
        # Calculate current price position
        current_bin = min(num_bins - 1, int((prices[-1] - min_price) / bin_size))
        
        # Calculate balance (how far current price is from centroid)
        # Normalize to [-1, 1] range
        balance = (current_bin - centroid) / (num_bins / 2)
        
        # Apply tree balancing factor
        return -balance * self.tree_balancing_factor
    
    def _combinatorial_search(self, returns: np.ndarray) -> float:
        """
        Use combinatorial search techniques to find optimal trading patterns
        
        Parameters
        ----------
        returns : numpy.ndarray
            Array of returns
            
        Returns
        -------
        float
            Combinatorial search signal
        """
        if len(returns) < 10:
            return 0.0
            
        # Get recent returns
        recent = returns[-min(50, len(returns)):]
        
        # Generate all possible combinations of trading rules up to combinatorial_depth
        best_score = 0.0
        best_signal = 0.0
        
        # Simple rules based on ma crossovers with different windows
        windows = [2, 3, 5, 8, 13]  # Fibonacci-inspired windows
        
        # Limit search space based on optimization level
        max_rules = self.optimization_level * self.combinatorial_depth
        rule_count = 0
        
        for w1 in windows:
            if w1 >= len(recent) or rule_count >= max_rules:
                break
                
            ma1 = np.convolve(recent, np.ones(w1)/w1, mode='valid')
            
            for w2 in windows:
                if w1 >= w2 or w2 >= len(recent) or rule_count >= max_rules:
                    continue
                    
                rule_count += 1
                ma2 = np.convolve(recent, np.ones(w2)/w2, mode='valid')
                
                # Trim series to same length
                min_len = min(len(ma1), len(ma2))
                ma1_trim = ma1[-min_len:]
                ma2_trim = ma2[-min_len:]
                
                # Crossover rule
                signals = np.sign(ma1_trim - ma2_trim)
                
                # Calculate performance
                # Shift signals by 1 to align with next return
                aligned_returns = recent[-(min_len-1):]
                aligned_signals = signals[:-1]
                
                if len(aligned_returns) == len(aligned_signals):
                    performance = np.sum(aligned_signals * aligned_returns)
                    sharpe = performance / (np.std(aligned_returns) + 1e-10)
                    
                    if abs(sharpe) > abs(best_score):
                        best_score = sharpe
                        best_signal = signals[-1]  # Current signal from best rule
        
        # "Premature optimization is the root of all evil" - Knuth
        # Add a noise term to avoid overfitting
        noise = np.random.normal(0, 0.1) if self.optimization_level < 3 else 0
        
        return np.clip(best_signal + noise, -1.0, 1.0)
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data using Knuth's algorithmic principles
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < 30:
            self.is_fitted = False
            return
            
        try:
            # Extract price and volume data
            prices = historical_df['close'].values
            volumes = historical_df['volume'].values if 'volume' in historical_df.columns else None
            returns = np.diff(prices) / prices[:-1]
            
            # Apply Knuth's algorithmic techniques
            
            # 1. Algorithm Efficiency Analysis
            efficiency_metrics = self._analyze_algorithmic_efficiency(prices, volumes)
            
            # 2. Big-O Growth Analysis
            growth_patterns = self._big_o_growth_analysis(prices)
            self.growth_rates = growth_patterns
            
            # 3. KMP Pattern Matching
            pattern_signals = self._knuth_morris_pratt_pattern(prices)
            
            # 4. Balanced Tree Analysis
            balance_signal = self._balanced_tree_analysis(prices, volumes)
            
            # 5. Combinatorial Search
            combinatorial_signal = self._combinatorial_search(returns)
            
            # Combine signals with different weights based on their algorithmic complexity
            signals = []
            weights = []
            
            # Linear complexity (O(n)) signals
            if 'linear_momentum' in efficiency_metrics:
                signals.append(np.clip(efficiency_metrics['linear_momentum'], -1, 1))
                weights.append(0.15)
                
            # Logarithmic complexity (O(log n)) signals
            if 'log_momentum' in efficiency_metrics:
                signals.append(np.clip(efficiency_metrics['log_momentum'], -1, 1))
                weights.append(0.10)
                
            # Pattern matching signal (moderate complexity)
            if 'current_pattern_signal' in pattern_signals:
                signals.append(pattern_signals['current_pattern_signal'])
                weights.append(0.25)
                
            # Tree balance signal (balanced tree complexity)
            signals.append(balance_signal)
            weights.append(0.20)
            
            # Combinatorial search signal (highest complexity)
            signals.append(combinatorial_signal)
            weights.append(0.30)
            
            # Normalize weights
            if sum(weights) > 0:
                weights = [w / sum(weights) for w in weights]
                
            # Weighted average of signals
            self.latest_signal = sum(s * w for s, w in zip(signals, weights))
            self.latest_signal = np.clip(self.latest_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Knuth Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on algorithmic analysis
        
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
        return "Knuth Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Knuth's algorithm analysis principles.
        
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
            logger.error(f"ValueError in Knuth strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Knuth strategy: {str(e)}")
            return 0.0000 