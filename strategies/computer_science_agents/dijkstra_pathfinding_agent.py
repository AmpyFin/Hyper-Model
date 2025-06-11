"""
Dijkstra Agent
~~~~~~~~~~~~~
Agent implementing trading strategies based on Edsger Dijkstra's principles of
algorithm design, shortest path finding, and structured programming.

Dijkstra is best known for his shortest path algorithm, but also contributed to
semaphore primitives, deadlock prevention, and the GOTO-less structured programming.

This agent models market decision-making as a shortest path problem:
1. Path Optimization: Finding optimal entry/exit points with minimal risk
2. State Transition Control: Using semaphore-like mechanisms to control trading decisions
3. Structured Decision Tree: Implementing a GOTO-less hierarchical decision process
4. Deadlock Prevention: Avoiding trapped positions through proactive risk management

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
from typing import Dict, List, Optional, Tuple, Set, Union
import logging
import heapq
from collections import defaultdict, deque
import math

from ..agent import Agent

logger = logging.getLogger(__name__)

class DijkstraAgent(Agent):
    """
    Trading agent based on Dijkstra's algorithmic principles.
    
    Parameters
    ----------
    lookback_window : int, default=42
        Window size for market graph construction
    risk_weight : float, default=0.7
        Weight given to risk vs. reward in path optimization
    state_threshold : float, default=0.25
        Threshold for state transition significance
    structure_levels : int, default=3
        Number of levels in the structured decision tree
    deadlock_sensitivity : float, default=1.5
        Sensitivity for deadlock detection
    """
    
    def __init__(
        self,
        lookback_window: int = 42,
        risk_weight: float = 0.7,
        state_threshold: float = 0.25,
        structure_levels: int = 3,
        deadlock_sensitivity: float = 1.5
    ):
        super().__init__()
        self.lookback_window = lookback_window
        self.risk_weight = risk_weight
        self.state_threshold = state_threshold
        self.structure_levels = structure_levels
        self.deadlock_sensitivity = deadlock_sensitivity
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.market_graph = {}
        self.current_state = 'neutral'
        self.semaphore = 0  # Controls state transitions (0=neutral, +n=bullish, -n=bearish)
        self.deadlock_indicators = {}
        
    def _construct_market_graph(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict:
        """
        Construct a market graph with nodes as price states and edges as transitions
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            Market graph representation
        """
        if len(prices) < self.lookback_window:
            return {}
            
        # Use recent window
        recent_prices = prices[-self.lookback_window:]
        price_mean = np.mean(recent_prices)
        price_std = np.std(recent_prices)
        
        # Define price bins (states)
        bins = [
            'extreme_low',  # < -2 std
            'very_low',     # -2 to -1 std
            'low',          # -1 to -0.5 std
            'slight_low',   # -0.5 to -0.25 std
            'neutral',      # -0.25 to 0.25 std
            'slight_high',  # 0.25 to 0.5 std
            'high',         # 0.5 to 1 std
            'very_high',    # 1 to 2 std
            'extreme_high'  # > 2 std
        ]
        
        # Map prices to states
        z_scores = (recent_prices - price_mean) / price_std
        states = []
        
        for z in z_scores:
            if z < -2.0:
                states.append('extreme_low')
            elif z < -1.0:
                states.append('very_low')
            elif z < -0.5:
                states.append('low')
            elif z < -0.25:
                states.append('slight_low')
            elif z < 0.25:
                states.append('neutral')
            elif z < 0.5:
                states.append('slight_high')
            elif z < 1.0:
                states.append('high')
            elif z < 2.0:
                states.append('very_high')
            else:
                states.append('extreme_high')
        
        # Build graph - nodes are states, edges are transitions with weights
        graph = {bin: {} for bin in bins}
        
        # Calculate transition frequencies and associated returns
        for i in range(1, len(states)):
            from_state = states[i-1]
            to_state = states[i]
            price_change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            
            # Add volume weight if available
            volume_weight = 1.0
            if volumes is not None and i < len(volumes):
                vol_z = (volumes[i] - np.mean(volumes)) / np.std(volumes)
                volume_weight = 1.0 + 0.2 * min(2.5, max(-2.5, vol_z))
            
            # Calculate edge weight - negative for rewards (we want shortest path)
            # Lower weight = more attractive transition
            reward = -price_change * volume_weight if price_change > 0 else 0
            risk = abs(price_change) * volume_weight if price_change < 0 else 0
            
            # Combined weight (Dijkstra minimizes weight, so use negative for reward)
            weight = self.risk_weight * risk - (1 - self.risk_weight) * reward
            
            # Add or update edge
            if to_state in graph[from_state]:
                graph[from_state][to_state].append(weight)
            else:
                graph[from_state][to_state] = [weight]
                
        # Average the weights for multiple occurrences of the same transition
        for from_state in graph:
            for to_state in list(graph[from_state].keys()):
                if graph[from_state][to_state]:
                    graph[from_state][to_state] = np.mean(graph[from_state][to_state])
                    
        return graph
    
    def _dijkstra_shortest_path(self, graph: Dict, start: str, targets: List[str]) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Apply Dijkstra's algorithm to find optimal paths in the market graph
        
        Parameters
        ----------
        graph : dict
            Market graph representation
        start : str
            Starting state
        targets : list
            Target states
            
        Returns
        -------
        tuple
            (distances, predecessors)
        """
        # Initialize distances with infinity and predecessors with None
        distances = {node: float('infinity') for node in graph}
        predecessors = {node: None for node in graph}
        distances[start] = 0
        
        # Priority queue with (distance, node)
        pq = [(0, start)]
        visited = set()
        
        while pq:
            # Get node with min distance
            current_distance, current_node = heapq.heappop(pq)
            
            # If it's already visited, skip
            if current_node in visited:
                continue
                
            # Mark as visited
            visited.add(current_node)
            
            # Stop if we've visited all target nodes
            if all(target in visited for target in targets):
                break
                
            # Check all neighbors
            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight
                
                # If we found a shorter path, update
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
                    
        return distances, predecessors
    
    def _semaphore_control(self, current_state: str, target_state: str, returns: np.ndarray) -> int:
        """
        Implement Dijkstra's semaphore control for state transitions
        
        Parameters
        ----------
        current_state : str
            Current market state
        target_state : str
            Target market state
        returns : numpy.ndarray
            Recent returns
            
        Returns
        -------
        int
            Semaphore value
        """
        # Start with previous semaphore value
        semaphore = self.semaphore
        
        # Calculate state numeric values for comparison
        state_values = {
            'extreme_low': -4,
            'very_low': -3,
            'low': -2,
            'slight_low': -1,
            'neutral': 0,
            'slight_high': 1,
            'high': 2,
            'very_high': 3,
            'extreme_high': 4
        }
        
        current_value = state_values.get(current_state, 0)
        target_value = state_values.get(target_state, 0)
        
        # Direction of desired transition
        direction = 1 if target_value > current_value else -1 if target_value < current_value else 0
        
        # If no state change, decay semaphore toward neutral
        if direction == 0:
            if semaphore > 0:
                semaphore -= 1
            elif semaphore < 0:
                semaphore += 1
            return semaphore
            
        # Recent return momentum - affects semaphore increment
        recent_momentum = np.sum(np.sign(returns[-min(5, len(returns)):]))
        
        # Increment semaphore based on direction and momentum
        if direction > 0 and recent_momentum > 0:
            # Moving up with positive momentum
            semaphore += min(2, abs(target_value - current_value))
        elif direction < 0 and recent_momentum < 0:
            # Moving down with negative momentum
            semaphore -= min(2, abs(target_value - current_value))
        elif direction > 0:
            # Moving up against momentum
            semaphore += 1
        elif direction < 0:
            # Moving down against momentum
            semaphore -= 1
            
        # Limit semaphore range
        semaphore = max(-5, min(5, semaphore))
        
        return semaphore
    
    def _structured_decision_tree(self, graph: Dict, current_state: str, returns: np.ndarray) -> Tuple[str, float]:
        """
        Implement a structured decision tree (GOTO-less) for market decisions
        
        Parameters
        ----------
        graph : dict
            Market graph representation
        current_state : str
            Current market state
        returns : numpy.ndarray
            Recent returns
            
        Returns
        -------
        tuple
            (recommended state, confidence)
        """
        if not graph or current_state not in graph:
            return 'neutral', 0.0
            
        # Level 1: Identify potential target states
        bullish_targets = ['high', 'very_high', 'extreme_high']
        bearish_targets = ['low', 'very_low', 'extreme_low']
        neutral_targets = ['slight_low', 'neutral', 'slight_high']
        
        # Find shortest paths to each target category
        _, bull_paths = self._dijkstra_shortest_path(graph, current_state, bullish_targets)
        _, bear_paths = self._dijkstra_shortest_path(graph, current_state, bearish_targets)
        
        # Level 2: Analyze path characteristics
        bull_path_exists = any(bull_paths[target] is not None for target in bullish_targets)
        bear_path_exists = any(bear_paths[target] is not None for target in bearish_targets)
        
        # Calculate recent volatility and trend
        if len(returns) >= 10:
            recent_volatility = np.std(returns[-10:])
            recent_trend = np.mean(returns[-10:]) / recent_volatility if recent_volatility > 0 else 0
        else:
            recent_volatility = 0
            recent_trend = 0
        
        # Level 3: Make structured decision
        if self.structure_levels >= 3 and (bull_path_exists or bear_path_exists):
            # More complex decision with state changes
            
            # Bullish case
            if bull_path_exists and (
                (self.semaphore > 0 and recent_trend > 0) or
                (self.semaphore >= 3) or
                (recent_trend > self.state_threshold * self.deadlock_sensitivity)
            ):
                # Find best bullish target
                best_target = min(
                    [t for t in bullish_targets if bull_paths[t] is not None],
                    key=lambda t: self._path_score(graph, current_state, t, bull_paths)
                )
                confidence = min(1.0, 0.5 + 0.1 * self.semaphore + 0.2 * recent_trend)
                return best_target, confidence
                
            # Bearish case
            elif bear_path_exists and (
                (self.semaphore < 0 and recent_trend < 0) or
                (self.semaphore <= -3) or
                (recent_trend < -self.state_threshold * self.deadlock_sensitivity)
            ):
                # Find best bearish target
                best_target = min(
                    [t for t in bearish_targets if bear_paths[t] is not None],
                    key=lambda t: self._path_score(graph, current_state, t, bear_paths)
                )
                confidence = min(1.0, 0.5 - 0.1 * self.semaphore - 0.2 * recent_trend)
                return best_target, confidence
        
        # Level 2 fallback: Simpler trend-based decision
        if self.structure_levels >= 2:
            if recent_trend > self.state_threshold:
                return 'slight_high', min(0.7, 0.4 + recent_trend)
            elif recent_trend < -self.state_threshold:
                return 'slight_low', min(0.7, 0.4 - recent_trend)
        
        # Level 1 fallback: Stay in current region
        return 'neutral', 0.3
    
    def _path_score(self, graph: Dict, start: str, end: str, predecessors: Dict[str, str]) -> float:
        """
        Calculate the score of a path in the graph
        
        Parameters
        ----------
        graph : dict
            Market graph representation
        start : str
            Starting state
        end : str
            Target state
        predecessors : dict
            Predecessors from Dijkstra's algorithm
            
        Returns
        -------
        float
            Path score (lower is better)
        """
        # Reconstruct path
        path = []
        current = end
        while current != start and current is not None:
            path.append(current)
            current = predecessors[current]
            
        if current is None:  # Path not found
            return float('infinity')
            
        path.append(start)
        path.reverse()
        
        # Calculate path score
        score = 0
        for i in range(1, len(path)):
            from_state = path[i-1]
            to_state = path[i]
            
            if to_state in graph[from_state]:
                score += graph[from_state][to_state]
                
        return score
    
    def _deadlock_detection(self, prices: np.ndarray, current_state: str) -> float:
        """
        Detect potential deadlocks (trapped market positions)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        current_state : str
            Current market state
            
        Returns
        -------
        float
            Deadlock indicator (-1.0 to 1.0, where 0 = no deadlock)
        """
        if len(prices) < 30:
            return 0.0
            
        # Calculate indicators for potential deadlock situations
        returns = np.diff(prices) / prices[:-1]
        
        # 1. Decreasing volatility - potential energy buildup
        volatilities = [np.std(returns[max(0, i-10):i]) for i in range(10, len(returns))]
        vol_trend = 0.0
        if len(volatilities) >= 10:
            vol_diff = (volatilities[-1] / volatilities[-10]) - 1
            vol_trend = -1.0 if vol_diff < -0.3 else 0.0  # Declining volatility
            
        # 2. Narrowing price range (consolidation)
        price_ranges = []
        for i in range(10, len(prices), 5):
            window = prices[max(0, i-10):i]
            price_ranges.append(np.max(window) - np.min(window))
            
        range_trend = 0.0
        if len(price_ranges) >= 2:
            range_diff = (price_ranges[-1] / price_ranges[0]) - 1
            range_trend = -1.0 if range_diff < -0.2 else 0.0  # Narrowing range
            
        # 3. Divergence from larger trend
        short_trend = np.mean(returns[-10:]) if len(returns) >= 10 else 0
        long_trend = np.mean(returns[-30:]) if len(returns) >= 30 else short_trend
        
        trend_divergence = 0.0
        if short_trend * long_trend < 0:  # Opposite signs
            trend_divergence = np.sign(short_trend)  # Direction of short-term divergence
            
        # 4. State persistence (staying in same state too long)
        state_persistence = 0.0
        if current_state in ['extreme_low', 'extreme_high']:
            state_persistence = -0.5  # Potentially trapped at extreme
        elif current_state in ['very_low', 'very_high']:
            state_persistence = -0.3  # Potentially trapped at edge
            
        # Combine indicators
        deadlock_indicator = (
            vol_trend * 0.3 +
            range_trend * 0.2 +
            trend_divergence * 0.3 +
            state_persistence * 0.2
        )
        
        # Scale based on sensitivity
        deadlock_indicator *= self.deadlock_sensitivity
        
        return np.clip(deadlock_indicator, -1.0, 1.0)
    
    def _translate_to_signal(self, target_state: str, confidence: float, deadlock: float) -> float:
        """
        Translate state recommendation to a trading signal
        
        Parameters
        ----------
        target_state : str
            Target market state
        confidence : float
            Confidence in the target state
        deadlock : float
            Deadlock indicator
            
        Returns
        -------
        float
            Trading signal (-1.0 to 1.0)
        """
        # State values mapping
        state_values = {
            'extreme_low': -1.0,
            'very_low': -0.8,
            'low': -0.6,
            'slight_low': -0.3,
            'neutral': 0.0,
            'slight_high': 0.3,
            'high': 0.6,
            'very_high': 0.8,
            'extreme_high': 1.0
        }
        
        base_signal = state_values.get(target_state, 0.0)
        
        # Apply confidence
        signal = base_signal * confidence
        
        # Apply deadlock adjustment - move away from deadlocks
        if deadlock != 0:
            # If we detect a deadlock, it suggests a potential breakout in that direction
            signal = signal * 0.7 + deadlock * 0.3
            
        return np.clip(signal, -1.0, 1.0)
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data to find the shortest path trading strategy
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.lookback_window:
            self.is_fitted = False
            return
            
        try:
            # Extract price and volume data
            prices = historical_df['close'].values
            volumes = historical_df['volume'].values if 'volume' in historical_df.columns else None
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            
            # Construct market graph
            self.market_graph = self._construct_market_graph(prices, volumes)
            
            # Determine current state
            if len(prices) > 0:
                price_mean = np.mean(prices[-self.lookback_window:])
                price_std = np.std(prices[-self.lookback_window:])
                latest_z = (prices[-1] - price_mean) / price_std
                
                # Map to state
                if latest_z < -2.0:
                    current_state = 'extreme_low'
                elif latest_z < -1.0:
                    current_state = 'very_low'
                elif latest_z < -0.5:
                    current_state = 'low'
                elif latest_z < -0.25:
                    current_state = 'slight_low'
                elif latest_z < 0.25:
                    current_state = 'neutral'
                elif latest_z < 0.5:
                    current_state = 'slight_high'
                elif latest_z < 1.0:
                    current_state = 'high'
                elif latest_z < 2.0:
                    current_state = 'very_high'
                else:
                    current_state = 'extreme_high'
                    
                self.current_state = current_state
                
                # Update semaphore
                if len(returns) > 0:
                    target_state, confidence = self._structured_decision_tree(
                        self.market_graph, current_state, returns
                    )
                    self.semaphore = self._semaphore_control(current_state, target_state, returns)
                    
                    # Detect deadlocks
                    deadlock = self._deadlock_detection(prices, current_state)
                    
                    # Generate signal
                    self.latest_signal = self._translate_to_signal(target_state, confidence, deadlock)
                    
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Dijkstra Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Dijkstra's algorithm principles
        
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
        return "Dijkstra Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Dijkstra's pathfinding principles.
        
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
            logger.error(f"ValueError in Dijkstra strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Dijkstra strategy: {str(e)}")
            return 0.0000 