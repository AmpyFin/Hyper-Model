"""
Lamport Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Leslie Lamport's work on distributed
systems, particularly his contributions to consensus algorithms, logical clocks,
and fault tolerance.

Leslie Lamport is known for:
1. Lamport Clocks - Logical timestamps for distributed events
2. Paxos Algorithm - Consensus in distributed systems
3. Temporal Logic - Formal specification of concurrent systems
4. Byzantine Fault Tolerance - Handling malicious failures
5. State Machine Replication - Maintaining consistent state

This agent models market behavior using:
1. Logical clock ordering of market events
2. Consensus-based signal generation
3. Temporal logic for market state analysis
4. Fault-tolerant signal processing
5. Replicated state machines for market modeling

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
from collections import defaultdict, deque
import bisect

from ..agent import Agent

logger = logging.getLogger(__name__)

class LamportAgent(Agent):
    """
    Trading agent based on Leslie Lamport's distributed systems principles.
    
    Parameters
    ----------
    logical_clock_interval : int, default=5
        Clock tick interval for logical time
    consensus_threshold : float, default=0.7
        Threshold for reaching consensus
    temporal_window : int, default=25
        Window size for temporal logic analysis
    replication_factor : int, default=3
        Number of replicated market views
    failure_tolerance : float, default=0.3
        Degree of fault tolerance (0-1)
    """
    
    def __init__(
        self,
        logical_clock_interval: int = 5,
        consensus_threshold: float = 0.7,
        temporal_window: int = 25,
        replication_factor: int = 3,
        failure_tolerance: float = 0.3
    ):
        self.logical_clock_interval = logical_clock_interval
        self.consensus_threshold = consensus_threshold
        self.temporal_window = temporal_window
        self.replication_factor = replication_factor
        self.failure_tolerance = failure_tolerance
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.logical_clock = 0
        self.event_history = deque(maxlen=100)
        self.state_machines = []
        self.consensus_history = deque(maxlen=20)
        self.temporal_formulas = {}
        
    def _increment_logical_clock(self) -> int:
        """
        Increment the logical clock (similar to Lamport's logical clocks)
        
        Returns
        -------
        int
            Updated logical clock value
        """
        self.logical_clock += 1
        return self.logical_clock
        
    def _record_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a timestamped event in the event history
        
        Parameters
        ----------
        event_type : str
            Type of event
        data : dict
            Event data
            
        Returns
        -------
        dict
            Timestamped event
        """
        event = {
            'timestamp': self._increment_logical_clock(),
            'type': event_type,
            'data': data
        }
        
        self.event_history.append(event)
        return event
        
    def _lamport_timestamp_ordering(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Order events by Lamport timestamp
        
        Parameters
        ----------
        events : list
            List of events
            
        Returns
        -------
        list
            Ordered events
        """
        return sorted(events, key=lambda e: e['timestamp'])
        
    def _create_replicated_views(self, prices: np.ndarray) -> List[np.ndarray]:
        """
        Create replicated market views (like replicated state machines)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        list
            List of replicated market views
        """
        if len(prices) < 10:  # Reduced minimum requirement
            return [prices]
            
        replicated_views = []
        
        # Base view is the original prices
        replicated_views.append(prices)
        
        # Create different views by applying transformations
        if self.replication_factor >= 2 and len(prices) >= 5:
            # View 2: Smoothed prices
            try:
                smoothed = np.convolve(prices, np.ones(5)/5, mode='valid')
                if len(smoothed) > 0:
                    replicated_views.append(smoothed)
            except Exception:
                pass
            
        if self.replication_factor >= 3 and len(prices) >= 15:
            # View 3: Detrended prices
            try:
                detrended = prices.copy()
                window = min(len(prices) // 2, 20)  # Use smaller window
                if window > 2:
                    x = np.arange(window)
                    for i in range(window, len(prices)):
                        segment = prices[i-window:i]
                        if len(segment) == window:
                            slope, intercept = np.polyfit(x, segment, 1)
                            trend = slope * x + intercept
                            detrended[i-window:i] = segment - trend + segment.mean()
                            
                    replicated_views.append(detrended)
            except Exception:
                pass
            
        if self.replication_factor >= 4 and len(prices) >= 20:
            # View 4: Normalized by volatility
            try:
                normalized = prices.copy()
                window = min(len(prices) // 3, 15)  # Use smaller window
                for i in range(window, len(prices)):
                    segment = prices[i-window:i]
                    mean = np.mean(segment)
                    std = np.std(segment)
                    if std > 0:
                        normalized[i-window:i] = (segment - mean) / std
                        
                replicated_views.append(normalized)
            except Exception:
                pass
            
        if self.replication_factor >= 5 and len(prices) >= 25:
            # View 5: Leading indicator (momentum-shifted)
            try:
                momentum_shifted = prices.copy()
                window = min(len(prices) // 3, 15)  # Use smaller window
                returns = np.diff(prices) / prices[:-1]
                
                for i in range(window, len(prices)-1):
                    # Adjust future price estimate based on momentum
                    momentum = np.mean(returns[i-window:i])
                    if np.isfinite(momentum):
                        momentum_shifted[i+1] = prices[i] * (1 + momentum)
                        
                replicated_views.append(momentum_shifted)
            except Exception:
                pass
            
        return replicated_views
        
    def _paxos_consensus(self, signals: List[float]) -> Tuple[float, bool]:
        """
        Apply Paxos-like consensus algorithm to trading signals
        
        Parameters
        ----------
        signals : list
            List of trading signals from different sources
            
        Returns
        -------
        tuple
            (consensus_signal, consensus_reached)
        """
        if not signals:
            return 0.0, False
            
        # Phase 1: Prepare (collect proposals)
        proposals = {}
        for i, signal in enumerate(signals):
            # Discretize signal to identify similar proposals
            discretized = round(signal * 10) / 10
            if discretized not in proposals:
                proposals[discretized] = []
            proposals[discretized].append(i)
            
        # Phase 2: Accept (find majority)
        max_votes = 0
        accepted_signal = 0.0
        
        for signal, voters in proposals.items():
            if len(voters) > max_votes:
                max_votes = len(voters)
                accepted_signal = signal
                
        # Check if consensus threshold is reached
        consensus_reached = max_votes >= len(signals) * self.consensus_threshold
        
        # Phase 3: Learn (inform all replicas of the result)
        consensus_signal = accepted_signal
        
        return consensus_signal, consensus_reached
        
    def _temporal_logic_analysis(self, prices: np.ndarray) -> Dict[str, bool]:
        """
        Apply temporal logic to market analysis
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Temporal formula evaluations
        """
        if len(prices) < self.temporal_window:
            return {}
            
        results = {}
        
        # Define temporal logic formulas inspired by Lamport's TLA+
        # Formula 1: Eventually(price > moving_avg) in the last window
        window = min(self.temporal_window, len(prices))
        ma = np.mean(prices[-window:])
        results['eventually_above_ma'] = any(prices[-window:] > ma)
        
        # Formula 2: Always(price > support) in the last window
        support = np.min(prices[-window:]) * 1.01  # 1% above minimum
        results['always_above_support'] = all(prices[-window:] > support)
        
        # Formula 3: Until(price_increasing, resistance_break)
        if len(prices) >= window + 5:
            increasing = all(prices[i] <= prices[i+1] for i in range(-window, -1))
            resistance = np.max(prices[-window-5:-window]) * 0.99  # 1% below previous max
            resistance_break = prices[-1] > resistance
            results['increasing_until_breakout'] = increasing and resistance_break
            
        # Formula 4: Next(momentum_continues)
        if len(prices) >= 10:
            current_momentum = prices[-1] - prices[-5]
            previous_momentum = prices[-5] - prices[-10]
            results['momentum_continues'] = (np.sign(current_momentum) == np.sign(previous_momentum)) and \
                                          (abs(current_momentum) >= abs(previous_momentum) * 0.8)
                                          
        # Formula 5: Leads_to(high_volume, price_change)
        # This would require volume data, so we approximate
        if len(prices) >= 15:
            high_movement = abs(prices[-10] - prices[-15]) > np.std(prices[-15:])
            subsequent_movement = abs(prices[-1] - prices[-5]) > np.std(prices[-15:])
            results['large_move_leads_to_followthrough'] = high_movement and subsequent_movement
            
        return results
        
    def _fault_tolerant_state_machine(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        Implement fault-tolerant state machine for market regimes
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            State machine output
        """
        if len(prices) < 20:
            return {'state': 'unknown', 'confidence': 0.0, 'signal': 0.0}
            
        # Define states
        states = ['bull_trend', 'bear_trend', 'bull_correction', 'bear_correction', 'sideways']
        
        # Calculate features for state determination
        returns = np.diff(prices) / prices[:-1]
        
        # Short-term trend
        short_slope, _ = np.polyfit(range(10), prices[-10:], 1) if len(prices) >= 10 else (0, 0)
        short_trend = np.sign(short_slope)
        
        # Medium-term trend
        med_slope, _ = np.polyfit(range(20), prices[-20:], 1) if len(prices) >= 20 else (0, 0)
        med_trend = np.sign(med_slope)
        
        # Volatility
        volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0
        
        # State determination (fault-tolerant by using multiple indicators)
        state_votes = {state: 0 for state in states}
        
        # Rule 1: Trend direction
        if short_trend > 0 and med_trend > 0:
            state_votes['bull_trend'] += 2
        elif short_trend < 0 and med_trend < 0:
            state_votes['bear_trend'] += 2
        elif short_trend < 0 and med_trend > 0:
            state_votes['bull_correction'] += 2
        elif short_trend > 0 and med_trend < 0:
            state_votes['bear_correction'] += 2
            
        # Rule 2: Volatility
        # Safe reshaping that works with any length of returns
        if len(returns) >= 5:
            # Calculate the number of complete groups of 5
            num_groups = len(returns) // 5
            if num_groups > 0:
                # Reshape only complete groups
                reshaped_returns = returns[:num_groups*5].reshape(num_groups, 5)
                avg_volatility = np.mean(np.std(reshaped_returns, axis=1))
                
                if volatility > avg_volatility:
                    if short_trend > 0:
                        state_votes['bull_trend'] += 1
                    else:
                        state_votes['bear_trend'] += 1
                else:
                    state_votes['sideways'] += 1
            else:
                # Not enough data for proper comparison
                state_votes['sideways'] += 1
        else:
            # Not enough data
            state_votes['sideways'] += 1
            
        # Rule 3: Recent performance
        recent_return = prices[-1] / prices[-10] - 1 if len(prices) >= 10 else 0
        if abs(recent_return) < 0.01:  # Less than 1% move
            state_votes['sideways'] += 2
        elif recent_return > 0.03:  # More than 3% up
            state_votes['bull_trend'] += 1
        elif recent_return < -0.03:  # More than 3% down
            state_votes['bear_trend'] += 1
            
        # Find state with most votes
        current_state = max(state_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence (normalized vote count)
        total_votes = sum(state_votes.values())
        confidence = state_votes[current_state] / total_votes if total_votes > 0 else 0
        
        # Generate state-dependent signal
        if current_state == 'bull_trend':
            signal = 0.8
        elif current_state == 'bear_trend':
            signal = -0.8
        elif current_state == 'bull_correction':
            signal = 0.3
        elif current_state == 'bear_correction':
            signal = -0.3
        else:  # sideways
            signal = 0.0
            
        # Adjust signal by confidence
        signal *= confidence
        
        return {
            'state': current_state,
            'confidence': confidence,
            'signal': signal,
            'votes': state_votes
        }
        
    def _sequential_consistency_check(self, signals: List[float], events: List[Dict[str, Any]]) -> float:
        """
        Check for sequential consistency among signals and adjust accordingly
        
        Parameters
        ----------
        signals : list
            List of trading signals
        events : list
            List of market events
            
        Returns
        -------
        float
            Consistency-adjusted signal
        """
        if not signals or len(signals) < 2:
            return signals[0] if signals else 0.0
            
        # Order events by timestamp
        ordered_events = self._lamport_timestamp_ordering(events)
        
        # Extract event signals if available
        event_signals = []
        for event in ordered_events:
            if 'signal' in event.get('data', {}):
                event_signals.append(event['data']['signal'])
                
        # Combine all signals
        all_signals = signals + event_signals
        
        if not all_signals:
            return 0.0
            
        # Check for sequential consistency (signals shouldn't change drastically)
        consistent = True
        for i in range(1, len(all_signals)):
            if abs(all_signals[i] - all_signals[i-1]) > 1.0:  # Large jump
                consistent = False
                break
                
        # If consistent, simple average is fine
        if consistent:
            return np.mean(all_signals)
            
        # If inconsistent, use median to filter outliers
        return np.median(all_signals)
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data using Lamport's distributed systems principles
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        # Reduce minimum requirement to be more flexible
        min_required = max(self.temporal_window, 25)
        if len(historical_df) < min_required:
            self.is_fitted = False
            return
            
        try:
            # Extract price data
            prices = historical_df['close'].values
            
            # Initialize components
            signals_from_views = []
            events = []
            
            # 1. Create replicated market views
            try:
                replicated_views = self._create_replicated_views(prices)
            except Exception as e:
                logger.warning(f"Replicated views creation failed: {e}")
                replicated_views = [prices]  # Fallback to single view
            
            # 2. Record key events for each view
            for view_id, view_prices in enumerate(replicated_views):
                try:
                    # Skip views that are too short
                    if len(view_prices) < 10:
                        continue
                        
                    # Detect significant events in this view
                    returns = np.diff(view_prices) / view_prices[:-1] if len(view_prices) > 1 else np.array([])
                    
                    # Event: Significant price move
                    if len(returns) >= 5:
                        recent_return = np.sum(returns[-5:])
                        if abs(recent_return) > 0.03:  # More than 3% move
                            event = self._record_event('significant_move', {
                                'view_id': view_id,
                                'return': recent_return,
                                'signal': np.sign(recent_return) * 0.5
                            })
                            events.append(event)
                            
                    # Event: Trend change
                    if len(view_prices) >= 20:
                        short_ma = np.mean(view_prices[-10:])
                        long_ma = np.mean(view_prices[-20:])
                        
                        prev_short_ma = np.mean(view_prices[-11:-1])
                        prev_long_ma = np.mean(view_prices[-21:-1])
                        
                        cross_up = prev_short_ma < prev_long_ma and short_ma > long_ma
                        cross_down = prev_short_ma > prev_long_ma and short_ma < long_ma
                        
                        if cross_up or cross_down:
                            event = self._record_event('ma_cross', {
                                'view_id': view_id,
                                'direction': 'up' if cross_up else 'down',
                                'signal': 0.7 if cross_up else -0.7
                            })
                            events.append(event)
                            
                    # Calculate trading signal for this view
                    window = min(len(view_prices), self.temporal_window)
                    if window >= 10:
                        # Simple strategy: short-term vs long-term momentum
                        short_momentum = view_prices[-1] / view_prices[-5] - 1 if len(view_prices) >= 5 else 0
                        long_momentum = view_prices[-1] / view_prices[-10] - 1 if len(view_prices) >= 10 else 0
                        
                        signal = (short_momentum * 0.6 + long_momentum * 0.4) * 10  # Scale to typical [-1, 1] range
                        signal = np.clip(signal, -1.0, 1.0)
                        
                        if np.isfinite(signal):
                            signals_from_views.append(signal)
                            
                except Exception as e:
                    logger.warning(f"Error processing view {view_id}: {e}")
                    continue
                        
            # 3. Apply temporal logic analysis
            try:
                temporal_results = self._temporal_logic_analysis(prices)
                
                # Convert temporal results to numeric signals
                temporal_signal = 0.0
                if temporal_results:
                    # Bullish conditions
                    bullish_conditions = [
                        temporal_results.get('eventually_above_ma', False),
                        temporal_results.get('increasing_until_breakout', False),
                        temporal_results.get('momentum_continues', False) and prices[-1] > prices[-5]
                    ]
                    
                    # Bearish conditions
                    bearish_conditions = [
                        not temporal_results.get('always_above_support', True),
                        temporal_results.get('momentum_continues', False) and prices[-1] < prices[-5]
                    ]
                    
                    # Calculate signal from conditions
                    if len(bullish_conditions) > 0 or len(bearish_conditions) > 0:
                        temporal_signal = (sum(bullish_conditions) - sum(bearish_conditions)) / max(len(bullish_conditions), len(bearish_conditions))
                        if np.isfinite(temporal_signal):
                            signals_from_views.append(temporal_signal)
            except Exception as e:
                logger.warning(f"Temporal logic analysis failed: {e}")
                
            # 4. Apply fault-tolerant state machine
            try:
                fsm_result = self._fault_tolerant_state_machine(prices)
                if 'signal' in fsm_result and np.isfinite(fsm_result['signal']):
                    signals_from_views.append(fsm_result['signal'])
            except Exception as e:
                logger.warning(f"Fault-tolerant state machine failed: {e}")
            
            # 5. Ensure we have at least one signal
            if not signals_from_views:
                # Simple momentum fallback
                if len(prices) >= 10:
                    recent_returns = np.diff(prices[-10:]) / prices[-11:-1]
                    clean_returns = recent_returns[np.isfinite(recent_returns)]
                    if len(clean_returns) > 0:
                        momentum = np.mean(clean_returns)
                        if np.isfinite(momentum):
                            signals_from_views.append(np.sign(momentum) * min(0.3, abs(momentum) * 10))
                
                if not signals_from_views:
                    signals_from_views.append(0.0)
            
            # 6. Reach consensus on trading signal
            try:
                consensus_signal, consensus_reached = self._paxos_consensus(signals_from_views)
                if not np.isfinite(consensus_signal):
                    consensus_signal = 0.0
                    consensus_reached = False
            except Exception as e:
                logger.warning(f"Paxos consensus failed: {e}")
                consensus_signal = np.mean(signals_from_views) if signals_from_views else 0.0
                consensus_reached = False
            
            # 7. Apply sequential consistency check if needed
            try:
                if not consensus_reached:
                    final_signal = self._sequential_consistency_check(signals_from_views, events)
                else:
                    final_signal = consensus_signal
                    
                if not np.isfinite(final_signal):
                    final_signal = np.mean(signals_from_views) if signals_from_views else 0.0
            except Exception as e:
                logger.warning(f"Sequential consistency check failed: {e}")
                final_signal = consensus_signal
                
            # Record consensus outcome
            try:
                consensus_event = self._record_event('consensus', {
                    'signal': final_signal,
                    'reached': consensus_reached,
                    'participant_count': len(signals_from_views)
                })
                
                self.consensus_history.append(consensus_event)
            except Exception:
                pass  # Non-critical
                
            # Store final signal
            if np.isfinite(final_signal):
                self.latest_signal = np.clip(final_signal, -1.0, 1.0)
            else:
                self.latest_signal = 0.0
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Lamport Agent fit: {e}")
            self.latest_signal = 0.0
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Lamport's distributed systems principles
        
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
        return "Lamport Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Lamport's consensus principles.
        
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
            logger.error(f"ValueError in Lamport strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Lamport strategy: {str(e)}")
            return 0.0000 