"""
Torvalds Agent
~~~~~~~~~~~~
Agent implementing trading strategies based on Linus Torvalds' contributions to 
computer science, particularly the Linux kernel, Git version control system, and 
his approach to open-source software development.

Linus Torvalds is known for:
1. Creating and maintaining the Linux kernel
2. Developing Git, a distributed version control system
3. Promoting decentralized and distributed development models
4. Pragmatic and meritocratic approach to software development
5. Focus on modularity, stability, and backward compatibility

This agent models market behavior using:
1. Git-like branching and merging of trading strategies
2. Linux kernel-inspired modularity and subsystems
3. Distributed consensus mechanisms for signal generation
4. Pragmatic "show me the code" (results-based) model evaluation
5. Continuous integration of market signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import math
from collections import defaultdict, deque
import hashlib

from ..agent import Agent

logger = logging.getLogger(__name__)

class TorvaldsAgent(Agent):
    """
    Trading agent based on Linus Torvalds' software development principles.
    
    Parameters
    ----------
    branch_count : int, default=3
        Number of strategy branches to maintain
    subsystem_count : int, default=5
        Number of market subsystems to analyze
    kernel_window : int, default=50
        Window size for core market analysis
    merge_threshold : float, default=0.6
        Threshold for merging strategy branches
    commit_interval : int, default=5
        Interval for committing market state
    """
    
    def __init__(
        self,
        branch_count: int = 3,
        subsystem_count: int = 5,
        kernel_window: int = 50,
        merge_threshold: float = 0.6,
        commit_interval: int = 5
    ):
        self.branch_count = branch_count
        self.subsystem_count = subsystem_count
        self.kernel_window = kernel_window
        self.merge_threshold = merge_threshold
        self.commit_interval = commit_interval
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.subsystems = {}
        self.branches = {}
        self.commit_history = deque(maxlen=100)
        self.current_stable_version = "0.0.1"
        self.merge_conflicts = {}
        self.linux_version = 0.01  # Starts at 0.01 like the original Linux
        
    def _calculate_hash(self, data: np.ndarray) -> str:
        """
        Calculate a Git-like hash for data (like Git SHA)
        
        Parameters
        ----------
        data : numpy.ndarray
            Market data to hash
            
        Returns
        -------
        str
            Hash string
        """
        # Convert data to bytes and calculate SHA1 hash
        data_bytes = data.tobytes()
        sha1 = hashlib.sha1(data_bytes).hexdigest()
        return sha1[:10]  # Return first 10 chars like Git's short hash
        
    def _commit_state(self, prices: np.ndarray, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Commit current market state (like Git commit)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Current price data
        state : dict
            Current market state
            
        Returns
        -------
        dict
            Commit object
        """
        # Calculate hash of current data
        data_hash = self._calculate_hash(prices)
        
        # Create commit object
        commit = {
            'hash': data_hash,
            'timestamp': len(self.commit_history),
            'state': state,
            'parent': self.commit_history[-1]['hash'] if self.commit_history else None
        }
        
        self.commit_history.append(commit)
        return commit
        
    def _initialize_subsystems(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize market subsystems (like Linux kernel subsystems)
        
        Returns
        -------
        dict
            Initialized subsystems
        """
        subsystems = {
            'mm': {  # Memory Management (price levels)
                'name': 'Price Levels',
                'maintainer': 'technical_analysis',
                'function': self._subsystem_price_levels,
                'weight': 0.2
            },
            'sched': {  # Scheduler (market timing)
                'name': 'Market Timing',
                'maintainer': 'time_series_analysis',
                'function': self._subsystem_market_timing,
                'weight': 0.3
            },
            'net': {  # Networking (market connectedness)
                'name': 'Market Connectivity',
                'maintainer': 'correlation_analysis',
                'function': self._subsystem_market_connectivity,
                'weight': 0.15
            },
            'fs': {  # File System (market structure)
                'name': 'Market Structure',
                'maintainer': 'pattern_recognition',
                'function': self._subsystem_market_structure,
                'weight': 0.2
            },
            'io': {  # Input/Output (volume analysis)
                'name': 'Volume Analysis',
                'maintainer': 'flow_of_funds',
                'function': self._subsystem_volume_analysis,
                'weight': 0.15
            }
        }
        
        return subsystems
        
    def _create_branches(self) -> Dict[str, Dict[str, Any]]:
        """
        Create strategy branches (like Git branches)
        
        Returns
        -------
        dict
            Strategy branches
        """
        branches = {
            'master': {  # Main branch (stable)
                'name': 'Stable Trading Strategy',
                'signals': deque(maxlen=100),
                'last_commit': None,
                'performance': 0.0
            },
            'testing': {  # Testing branch (cutting edge)
                'name': 'Testing Trading Strategy',
                'signals': deque(maxlen=100),
                'last_commit': None,
                'performance': 0.0
            },
            'experimental': {  # Experimental branch (risky)
                'name': 'Experimental Trading Strategy',
                'signals': deque(maxlen=100),
                'last_commit': None,
                'performance': 0.0
            }
        }
        
        return branches
        
    def _subsystem_price_levels(self, prices: np.ndarray) -> float:
        """
        Analyze price levels (like memory management in Linux)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Price level signal
        """
        if len(prices) < 20:
            return 0.0
            
        # Identify support and resistance levels
        window = min(len(prices), self.kernel_window)
        recent_prices = prices[-window:]
        
        # Calculate key levels
        support = np.percentile(recent_prices, 20)
        resistance = np.percentile(recent_prices, 80)
        mid_point = (support + resistance) / 2
        
        current_price = prices[-1]
        
        # Generate signal based on price position
        if current_price < support:
            # Below support - potential for mean reversion up
            signal = 0.5
        elif current_price > resistance:
            # Above resistance - potential for mean reversion down
            signal = -0.5
        else:
            # In the middle range - weaker signal
            normalized_position = (current_price - support) / (resistance - support) if resistance > support else 0.5
            signal = 0.5 - normalized_position  # 0.5 at support, -0.5 at resistance
        
        # Amplify signal if close to levels
        distance_to_support = abs(current_price - support) / current_price
        distance_to_resistance = abs(current_price - resistance) / current_price
        
        if distance_to_support < 0.01:  # Within 1% of support
            signal = max(signal, 0.7)
        elif distance_to_resistance < 0.01:  # Within 1% of resistance
            signal = min(signal, -0.7)
            
        return signal
        
    def _subsystem_market_timing(self, prices: np.ndarray) -> float:
        """
        Analyze market timing (like scheduler in Linux)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Market timing signal
        """
        if len(prices) < 30:
            return 0.0
            
        # Calculate short and long-term momentum
        short_returns = prices[-5] / prices[-10] - 1 if len(prices) >= 10 else 0
        medium_returns = prices[-10] / prices[-20] - 1 if len(prices) >= 20 else 0
        long_returns = prices[-20] / prices[-30] - 1 if len(prices) >= 30 else 0
        
        # Calculate timing signal using a scheduler-like approach
        # Prioritize different timeframes based on their "urgency"
        
        # Calculate "priority" of each timeframe (like Linux's CFS scheduler)
        short_priority = abs(short_returns) * 3  # Higher weight to short term
        medium_priority = abs(medium_returns) * 2
        long_priority = abs(long_returns)
        
        # Normalize priorities
        total_priority = short_priority + medium_priority + long_priority
        if total_priority == 0:
            return 0.0
            
        short_weight = short_priority / total_priority
        medium_weight = medium_priority / total_priority
        long_weight = long_priority / total_priority
        
        # Calculate weighted signal
        signal = (
            np.sign(short_returns) * short_weight +
            np.sign(medium_returns) * medium_weight +
            np.sign(long_returns) * long_weight
        )
        
        return signal
        
    def _subsystem_market_connectivity(self, prices: np.ndarray) -> float:
        """
        Analyze market connectivity (like networking in Linux)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Market connectivity signal
        """
        if len(prices) < 20:
            return 0.0
            
        # Calculate autocorrelation at different lags (like network packets)
        max_lag = min(10, len(prices) // 3)
        correlations = []
        
        for lag in range(1, max_lag + 1):
            if len(prices) <= lag:
                continue
                
            # Calculate correlation between price and lagged price
            corr = np.corrcoef(prices[:-lag], prices[lag:])[0, 1]
            if np.isfinite(corr):
                correlations.append((lag, corr))
                
        if not correlations:
            return 0.0
            
        # Find strongest correlation
        strongest_lag, strongest_corr = max(correlations, key=lambda x: abs(x[1]))
        
        # Generate signal based on correlation pattern
        signal = strongest_corr  # Use correlation directly as signal
        
        # Apply "network congestion control" - reduce signal when market is choppy
        if len(prices) >= 20:
            # Calculate "packet loss" (noise in the market)
            returns = np.diff(prices) / prices[:-1]
            noise_level = np.std(returns[-10:]) / np.std(returns[-20:]) if len(returns) >= 20 else 1.0
            
            # Adjust signal based on noise level (like TCP congestion control)
            if noise_level > 1.5:  # High noise, reduce signal
                signal *= 0.5
            elif noise_level < 0.5:  # Low noise, amplify signal
                signal *= 1.2
                
        return np.clip(signal, -1.0, 1.0)
        
    def _subsystem_market_structure(self, prices: np.ndarray) -> float:
        """
        Analyze market structure (like file system in Linux)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Market structure signal
        """
        if len(prices) < 30:
            return 0.0
            
        # Detect market "directory structure" (trends, ranges, etc.)
        window = min(len(prices), self.kernel_window)
        
        # Calculate simple moving averages (like file system layers)
        sma_fast = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        sma_medium = np.mean(prices[-20:]) if len(prices) >= 20 else sma_fast
        sma_slow = np.mean(prices[-30:]) if len(prices) >= 30 else sma_medium
        
        # Determine market structure based on MA relationships
        # Like file system hierarchy in Linux
        
        trending_up = sma_fast > sma_medium > sma_slow
        trending_down = sma_fast < sma_medium < sma_slow
        mean_reverting = (sma_fast < sma_medium and sma_medium > sma_slow) or \
                        (sma_fast > sma_medium and sma_medium < sma_slow)
        
        # Generate signal based on identified structure
        signal = 0.0
        if trending_up:
            signal = 0.7  # Strong uptrend
        elif trending_down:
            signal = -0.7  # Strong downtrend
        elif mean_reverting:
            # Mean-reverting - signal depends on price vs medium MA
            signal = -0.5 * np.sign(prices[-1] - sma_medium)
            
        # Check for pattern consistency (like file system consistency check)
        if len(prices) >= 40:
            prev_sma_fast = np.mean(prices[-20:-10]) if len(prices) >= 20 else prices[-11]
            prev_sma_medium = np.mean(prices[-30:-10]) if len(prices) >= 30 else prev_sma_fast
            
            # If structure changed recently, reduce signal conviction
            structure_changed = (prev_sma_fast > prev_sma_medium) != (sma_fast > sma_medium)
            if structure_changed:
                signal *= 0.5
                
        return signal
        
    def _subsystem_volume_analysis(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        Analyze volume (like I/O in Linux)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        float
            Volume analysis signal
        """
        if volumes is None or len(volumes) < 10 or len(prices) < 10:
            return 0.0
            
        # Calculate recent volume changes
        recent_volumes = volumes[-10:]
        avg_volume = np.mean(recent_volumes)
        latest_volume = volumes[-1]
        
        # Calculate price change
        price_change = prices[-1] / prices[-2] - 1 if len(prices) >= 2 else 0
        
        # Volume spike detection (like I/O spike in Linux)
        volume_spike = latest_volume > 1.5 * avg_volume
        
        # Calculate price-volume correlation (recent)
        if len(prices) >= 11 and len(volumes) >= 11:  # Need at least 11 points to get 10 diffs
            try:
                # Make sure we have enough data for both arrays
                price_changes = np.diff(prices[-11:]) / prices[-11:-1]
                volume_changes = np.diff(volumes[-11:]) / volumes[-11:-1]
            except (IndexError, ValueError):
                # Fallback if array lengths don't match or other errors
                min_len = min(len(prices), len(volumes))
                if min_len > 2:
                    safe_len = min_len - 1  # Ensure we have enough elements for diff
                    price_changes = np.diff(prices[-safe_len:]) / prices[-safe_len:-1]
                    volume_changes = np.diff(volumes[-safe_len:]) / volumes[-safe_len:-1]
                else:
                    price_changes = np.array([0.0])
                    volume_changes = np.array([0.0])
            
            if len(price_changes) == len(volume_changes) and len(price_changes) > 1:
                pv_correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                if not np.isfinite(pv_correlation):
                    pv_correlation = 0
            else:
                pv_correlation = 0
        else:
            pv_correlation = 0
            
        # Generate signal based on volume analysis
        signal = 0.0
        
        # Volume spike with price movement
        if volume_spike:
            signal = np.sign(price_change) * 0.5
            
        # Price-volume correlation factor
        signal += pv_correlation * 0.3
        
        # Volume trend factor
        if len(volumes) >= 5:
            volume_trend = np.mean(volumes[-3:]) / np.mean(volumes[-5:-2]) - 1
            signal += np.sign(volume_trend) * min(abs(volume_trend) * 2, 0.2)
            
        return np.clip(signal, -1.0, 1.0)
        
    def _kernel_integration(self, subsystem_signals: Dict[str, float]) -> Dict[str, float]:
        """
        Integrate subsystem signals (like Linux kernel integration)
        
        Parameters
        ----------
        subsystem_signals : dict
            Signals from different subsystems
            
        Returns
        -------
        dict
            Integrated signals for different branches
        """
        # Weights for each subsystem in different branches
        branch_subsystem_weights = {
            'master': {  # Stable branch - balanced weights
                'mm': 0.2,
                'sched': 0.25,
                'net': 0.15,
                'fs': 0.2,
                'io': 0.2
            },
            'testing': {  # Testing branch - more weight on timing and structure
                'mm': 0.15,
                'sched': 0.3,
                'net': 0.15,
                'fs': 0.25,
                'io': 0.15
            },
            'experimental': {  # Experimental branch - more weight on recent signals
                'mm': 0.1,
                'sched': 0.35,
                'net': 0.2,
                'fs': 0.15,
                'io': 0.2
            }
        }
        
        # Calculate integrated signal for each branch
        branch_signals = {}
        
        for branch_name, weights in branch_subsystem_weights.items():
            signal = 0.0
            weight_sum = 0.0
            
            for subsystem, weight in weights.items():
                if subsystem in subsystem_signals:
                    signal += subsystem_signals[subsystem] * weight
                    weight_sum += weight
                    
            if weight_sum > 0:
                signal /= weight_sum
                
            branch_signals[branch_name] = signal
            
        return branch_signals
        
    def _git_merge_branches(self, branch_signals: Dict[str, float], prices: np.ndarray) -> float:
        """
        Merge strategy branches (like Git merge)
        
        Parameters
        ----------
        branch_signals : dict
            Signals from different branches
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Merged signal
        """
        # Check branch performance (like Git's "merge worthiness")
        # Simple performance metric: direction accuracy
        for branch_name, branch in self.branches.items():
            if branch_name in branch_signals:
                # Add current signal to branch history
                branch['signals'].append(branch_signals[branch_name])
                
                # Calculate performance if we have enough history
                if len(branch['signals']) >= 5 and len(prices) >= 5:
                    correct_directions = 0
                    total_predictions = 0
                    
                    for i in range(min(len(branch['signals'])-1, len(prices)-1, 5)):
                        signal = branch['signals'][-(i+2)]
                        actual_change = prices[-(i+1)] - prices[-(i+2)]
                        
                        # Check if direction was correct
                        if (signal > 0 and actual_change > 0) or (signal < 0 and actual_change < 0):
                            correct_directions += 1
                            
                        total_predictions += 1
                        
                    if total_predictions > 0:
                        branch['performance'] = correct_directions / total_predictions
                
        # Merge strategy (like Git's merge strategies)
        
        # 1. Fast-forward merge: If one branch is clearly better, use it
        best_branch = max(self.branches.items(), key=lambda x: x[1]['performance'])
        best_performance = best_branch[1]['performance']
        
        if best_performance > self.merge_threshold:
            # Fast-forward merge - use best branch
            merged_signal = branch_signals[best_branch[0]]
            
            # Record merge
            self.merge_conflicts[best_branch[0]] = 0
            return merged_signal
            
        # 2. Three-way merge: Combine branches with conflict resolution
        # Check for conflicts (significantly different signals)
        has_conflicts = False
        for branch1, signal1 in branch_signals.items():
            for branch2, signal2 in branch_signals.items():
                if branch1 != branch2 and abs(signal1 - signal2) > 1.0:
                    has_conflicts = True
                    # Record conflict
                    if branch1 not in self.merge_conflicts:
                        self.merge_conflicts[branch1] = 0
                    self.merge_conflicts[branch1] += 1
                    
        if has_conflicts:
            # Conflict resolution: weight by branch performance
            total_performance = sum(branch['performance'] for branch in self.branches.values())
            
            if total_performance > 0:
                merged_signal = sum(
                    branch_signals[branch_name] * self.branches[branch_name]['performance'] / total_performance
                    for branch_name in branch_signals
                    if branch_name in self.branches
                )
            else:
                # Equal weights if no performance data
                merged_signal = sum(branch_signals.values()) / len(branch_signals)
        else:
            # No conflicts - simple average
            merged_signal = sum(branch_signals.values()) / len(branch_signals)
            
        return merged_signal
        
    def _update_version(self, prices: np.ndarray, final_signal: float) -> None:
        """
        Update agent version (like Linux version numbering)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        final_signal : float
            Final trading signal
        """
        # Update version based on changes and stability
        
        # Calculate version components
        major = int(self.linux_version)
        minor = int((self.linux_version - major) * 100)
        
        # Determine if a new version should be released
        # Like Linux kernel development, major changes happen less frequently
        
        # Check if we need a minor version bump (like Linux point releases)
        commit_count = len(self.commit_history)
        if commit_count % 10 == 0 and commit_count > 0:
            minor += 1
            
            # Reset for overflow
            if minor >= 100:
                major += 1
                minor = 0
        
        # Check if we need a major version bump (significant changes)
        # This would happen if performance dramatically improves or market regime changes
        
        # Detect major market regime change
        if len(prices) >= 50:
            recent_vol = np.std(prices[-20:])
            prev_vol = np.std(prices[-50:-20])
            
            # Major volatility change could trigger version bump
            if (recent_vol > prev_vol * 2) or (recent_vol < prev_vol * 0.5):
                major += 1
                minor = 0
                
        # Update version
        self.linux_version = major + minor / 100
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data using Torvalds' software development principles
        
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
            
            # Initialize subsystems if needed
            if not self.subsystems:
                self.subsystems = self._initialize_subsystems()
                
            # Initialize branches if needed
            if not self.branches:
                self.branches = self._create_branches()
                
            # 1. Run subsystem analysis
            subsystem_signals = {}
            
            for subsystem_name, subsystem in self.subsystems.items():
                if subsystem_name == 'io' and volumes is not None:
                    # Volume analysis requires volume data
                    signal = subsystem['function'](prices, volumes)
                else:
                    signal = subsystem['function'](prices)
                    
                subsystem_signals[subsystem_name] = signal
                
            # 2. Kernel integration (combine subsystem signals for each branch)
            branch_signals = self._kernel_integration(subsystem_signals)
            
            # 3. Git-like branch merging
            merged_signal = self._git_merge_branches(branch_signals, prices)
            
            # 4. Commit state if necessary
            if len(self.commit_history) % self.commit_interval == 0:
                state = {
                    'subsystem_signals': subsystem_signals,
                    'branch_signals': branch_signals,
                    'merged_signal': merged_signal
                }
                
                self._commit_state(prices, state)
                
            # 5. Update version number
            self._update_version(prices, merged_signal)
            
            # Store final signal
            self.latest_signal = np.clip(merged_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Torvalds Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Torvalds' software principles
        
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
        return "Torvalds Agent" 

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Torvalds' distributed system principles.
        
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
            logger.error(f"ValueError in Torvalds strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Torvalds strategy: {str(e)}")
            return 0.0000