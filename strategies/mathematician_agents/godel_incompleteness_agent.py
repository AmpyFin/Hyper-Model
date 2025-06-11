"""
Godel Agent
~~~~~~~~~~
Agent implementing trading strategies based on Kurt Gödel's principles of
mathematical logic, incompleteness theorems, and formal systems analysis.

This agent models markets as formal logical systems, where:
1. Incompleteness: No single market indicator can capture all profitable opportunities
2. Consistency Analysis: Markets tend toward logical consistency over time
3. Self-reference: Market dynamics reference and influence themselves
4. Paradox Detection: Identifying market paradoxes (contradictions) as trading signals

The agent uses these concepts to detect market regime shifts and anticipate
when current trading systems will fail to describe all market behaviors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class GodelAgent:
    """
    Trading agent based on Gödel's mathematical logic principles.
    
    Parameters
    ----------
    consistency_window : int, default=20
        Window size for evaluating market consistency
    paradox_threshold : float, default=0.6
        Threshold for identifying market paradoxes
    recursion_depth : int, default=3
        Depth of recursive market analysis
    indicator_weights : dict, default=None
        Weights for different indicator types
    formal_system_checks : list, default=None
        List of formal system checks to perform
    """
    
    def __init__(
        self,
        consistency_window: int = 20,
        paradox_threshold: float = 0.6,
        recursion_depth: int = 3,
        indicator_weights: Optional[Dict[str, float]] = None,
        formal_system_checks: Optional[List[str]] = None
    ):
        self.consistency_window = consistency_window
        self.paradox_threshold = paradox_threshold
        self.recursion_depth = recursion_depth
        
        # Default indicator weights
        self.indicator_weights = indicator_weights or {
            'trend': 0.3,
            'volatility': 0.2,
            'momentum': 0.25,
            'volume': 0.25
        }
        
        # Default formal system checks
        self.formal_system_checks = formal_system_checks or [
            'trend_consistency',
            'volatility_consistency',
            'volume_price_consistency',
            'momentum_consistency'
        ]
        
        self.latest_signal = 0.0
        self.is_fitted = False
        self.formal_system_states = {}
        
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate basic indicators for formal system analysis
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with OHLCV columns
            
        Returns
        -------
        dict
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Trend indicators
        indicators['sma_fast'] = df['close'].rolling(window=10).mean().values
        indicators['sma_slow'] = df['close'].rolling(window=30).mean().values
        
        # Volatility indicators
        if len(df) >= 20:
            indicators['volatility'] = df['close'].rolling(window=20).std().values
            indicators['volatility_change'] = np.concatenate([[0], np.diff(indicators['volatility'])])
        else:
            indicators['volatility'] = np.zeros(len(df))
            indicators['volatility_change'] = np.zeros(len(df))
        
        # Momentum indicators
        indicators['roc'] = df['close'].pct_change(periods=10).values
        
        # Volume indicators
        if 'volume' in df.columns:
            indicators['volume'] = df['volume'].values
            indicators['volume_sma'] = df['volume'].rolling(window=10).mean().values
            indicators['volume_change'] = df['volume'].pct_change().values
        else:
            # If volume is not available, use placeholders
            indicators['volume'] = np.ones(len(df))
            indicators['volume_sma'] = np.ones(len(df))
            indicators['volume_change'] = np.zeros(len(df))
        
        return indicators
    
    def _check_consistency(self, indicators: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Check for logical consistency in the market's formal system
        
        Parameters
        ----------
        indicators : dict
            Dictionary of calculated indicators
            
        Returns
        -------
        dict
            Dictionary of consistency scores for different aspects
        """
        n = len(indicators['sma_fast'])
        if n < self.consistency_window:
            return {check: 0.0 for check in self.formal_system_checks}
            
        consistency_scores = {}
        
        # Get recent window
        recent_idx = slice(-self.consistency_window, None)
        
        # 1. Trend consistency - are trends behaving logically?
        # When fast MA > slow MA, prices should be rising more often than falling
        if 'trend_consistency' in self.formal_system_checks:
            # Check if trend direction matches relative MA positions
            ma_diff = indicators['sma_fast'][recent_idx] - indicators['sma_slow'][recent_idx]
            price_diff = np.diff(np.concatenate([[0], indicators['sma_fast'][recent_idx]]))
            
            # Logical consistency: positive MA diff should correspond to positive price changes
            consistent_points = sum((ma_diff > 0) == (price_diff > 0))
            consistency_scores['trend_consistency'] = consistent_points / len(ma_diff)
        
        # 2. Volatility consistency - is volatility behaving logically with price changes?
        if 'volatility_consistency' in self.formal_system_checks:
            # Absolute price changes
            abs_price_change = np.abs(np.diff(np.concatenate([[0], indicators['sma_fast'][recent_idx]])))
            volatility = indicators['volatility'][recent_idx]
            
            # Logical consistency: higher volatility should correspond to larger absolute price changes
            vol_corr = np.corrcoef(abs_price_change, volatility)[0, 1] if len(volatility) > 1 else 0
            consistency_scores['volatility_consistency'] = (vol_corr + 1) / 2  # Scale to [0, 1]
        
        # 3. Volume-price consistency - is volume confirming price moves?
        if 'volume_price_consistency' in self.formal_system_checks:
            # Absolute price changes
            abs_price_change = np.abs(np.diff(np.concatenate([[0], indicators['sma_fast'][recent_idx]])))
            volume = indicators['volume'][recent_idx]
            
            # Logical consistency: higher volume should correspond to larger price changes
            vol_price_corr = np.corrcoef(abs_price_change, volume)[0, 1] if len(volume) > 1 else 0
            consistency_scores['volume_price_consistency'] = (vol_price_corr + 1) / 2  # Scale to [0, 1]
        
        # 4. Momentum consistency - is momentum leading price in a logical way?
        if 'momentum_consistency' in self.formal_system_checks:
            roc = indicators['roc'][recent_idx]
            future_returns = np.roll(roc, -5)  # Future returns (5 periods ahead)
            future_returns[-5:] = 0  # Mask out rolled values
            
            # Logical consistency: momentum should lead returns
            mom_corr = np.corrcoef(roc[:-5], future_returns[:-5])[0, 1] if len(roc) > 5 else 0
            consistency_scores['momentum_consistency'] = (mom_corr + 1) / 2  # Scale to [0, 1]
        
        return consistency_scores
    
    def _detect_paradoxes(self, consistency_scores: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Use Gödel's principles to detect market paradoxes
        
        Parameters
        ----------
        consistency_scores : dict
            Dictionary of consistency scores for different aspects
            
        Returns
        -------
        tuple
            (paradox_exists, paradox_strength, paradox_type)
        """
        # No paradoxes if insufficient data
        if not consistency_scores:
            return False, 0.0, ""
            
        # Find most inconsistent aspect
        min_score = min(consistency_scores.values())
        min_aspect = min(consistency_scores.items(), key=lambda x: x[1])[0]
        
        # Consider it a paradox if consistency is below threshold
        paradox_exists = min_score < (1 - self.paradox_threshold)
        
        # Paradox strength is how far below threshold
        paradox_strength = max(0, (1 - self.paradox_threshold) - min_score)
        
        return paradox_exists, paradox_strength, min_aspect
    
    def _recursive_analysis(self, df: pd.DataFrame, depth: int = 0) -> Dict[str, float]:
        """
        Perform recursive analysis of market system (inspired by Gödel's recursion)
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data
        depth : int, default=0
            Current recursion depth
            
        Returns
        -------
        dict
            Dictionary of recursive insights
        """
        if depth >= self.recursion_depth or len(df) < 20:
            return {'signal': 0.0, 'confidence': 0.0}
            
        # At each level, analyze a different timeframe
        # Level 0: Original data
        # Level 1: 2-period aggregation
        # Level 2: 4-period aggregation
        # etc.
        if depth > 0:
            # Aggregate data for this recursion level
            period = 2 ** depth
            agg_df = df.copy()
            agg_df['close'] = df['close'].rolling(window=period).mean().values
            if 'volume' in df.columns:
                agg_df['volume'] = df['volume'].rolling(window=period).sum().values
            
            # Remove NaN values
            agg_df = agg_df.dropna()
            
            if len(agg_df) < 20:
                return {'signal': 0.0, 'confidence': 0.0}
        else:
            agg_df = df
            
        # Calculate indicators for this level
        indicators = self._calculate_indicators(agg_df)
        
        # Check consistency
        consistency_scores = self._check_consistency(indicators)
        
        # Detect paradoxes
        has_paradox, paradox_strength, paradox_type = self._detect_paradoxes(consistency_scores)
        
        # Recursive call to next level
        next_level = self._recursive_analysis(df, depth + 1)
        
        # Generate signal for this level
        signal = 0.0
        confidence = 0.0
        
        if has_paradox:
            # Paradoxes suggest potential regime change
            # Direction depends on recent trend
            recent_trend = np.mean(np.diff(agg_df['close'].values[-10:]))
            
            # A paradox suggests the opposite of the recent trend
            signal = -np.sign(recent_trend) * paradox_strength
            confidence = paradox_strength
            
        # Combine with deeper levels (with decreasing weight)
        level_weight = 1.0 / (2 ** depth) if depth > 0 else 0.5
        next_level_weight = 1.0 - level_weight
        
        combined_signal = signal * level_weight + next_level['signal'] * next_level_weight
        combined_confidence = confidence * level_weight + next_level['confidence'] * next_level_weight
        
        return {
            'signal': combined_signal,
            'confidence': combined_confidence,
            'paradox_type': paradox_type if has_paradox else next_level.get('paradox_type', '')
        }
    
    def _self_reference_analysis(self, df: pd.DataFrame) -> float:
        """
        Apply Gödel's self-reference concept to market analysis
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data
            
        Returns
        -------
        float
            Self-reference signal
        """
        if len(df) < 30:
            return 0.0
            
        # Self-reference in markets: current price patterns referring to past patterns
        # We'll implement this by checking if current patterns exist in past data
        
        # Create a simple pattern representation of recent price changes
        recent_changes = np.sign(np.diff(df['close'].values[-10:]))
        pattern = ''.join(['1' if c > 0 else ('0' if c < 0 else 'N') for c in recent_changes])
        
        # Look for this pattern in historical data
        all_changes = np.sign(np.diff(df['close'].values))
        pattern_instances = []
        
        for i in range(len(all_changes) - len(recent_changes) + 1):
            hist_pattern = ''.join(['1' if c > 0 else ('0' if c < 0 else 'N') for c in all_changes[i:i+len(recent_changes)]])
            if hist_pattern == pattern:
                pattern_instances.append(i)
        
        # If no matches found
        if len(pattern_instances) <= 1:
            return 0.0
            
        # Analyze what typically happened after this pattern historically
        next_changes = []
        for idx in pattern_instances[:-1]:  # Exclude the most recent instance
            pos = idx + len(recent_changes)
            if pos < len(all_changes):
                next_changes.append(all_changes[pos])
        
        if not next_changes:
            return 0.0
            
        # Calculate probability of up move after this pattern
        prob_up = sum(1 for c in next_changes if c > 0) / len(next_changes)
        
        # Convert to signal: (prob_up - 0.5) * 2 scales to [-1, 1]
        return (prob_up - 0.5) * 2
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        min_required_bars = max(30, self.consistency_window * 2)
        if len(historical_df) < min_required_bars:
            self.is_fitted = False
            return
            
        try:
            # Make a copy of the dataframe
            df = historical_df.copy()
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Check consistency
            consistency_scores = self._check_consistency(indicators)
            
            # Store states
            self.formal_system_states = consistency_scores
            
            # Detect paradoxes
            has_paradox, paradox_strength, paradox_type = self._detect_paradoxes(consistency_scores)
            
            # Perform recursive analysis
            recursive_results = self._recursive_analysis(df)
            
            # Perform self-reference analysis
            self_reference_signal = self._self_reference_analysis(df)
            
            # Combine signals
            
            # 1. Paradox-based signal (50%)
            paradox_signal = 0.0
            if has_paradox:
                # Paradoxes suggest potential regime change
                # Direction depends on recent trend
                recent_trend = np.mean(np.diff(df['close'].values[-10:]))
                # A paradox suggests the opposite of the recent trend
                paradox_signal = -np.sign(recent_trend) * paradox_strength
            
            # 2. Recursive analysis signal (30%)
            recursive_signal = recursive_results['signal']
            
            # 3. Self-reference signal (20%)
            
            # Combined final signal
            self.latest_signal = (
                paradox_signal * 0.5 +
                recursive_signal * 0.3 +
                self_reference_signal * 0.2
            )
            
            # Ensure signal is in [-1, 1] range
            self.latest_signal = np.clip(self.latest_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Godel Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Gödel's mathematical logic principles
        
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
        return "Godel Agent" 