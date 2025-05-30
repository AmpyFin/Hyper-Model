"""
Backus Agent
~~~~~~~~~~
Agent implementing trading strategies based on John Backus's work on functional
programming, formal language theory, and the development of FORTRAN.

John Backus is known for:
1. Creating FORTRAN (first high-level programming language)
2. Backus-Naur Form (BNF) for describing formal languages
3. Function-level programming paradigm
4. FP programming language
5. Pioneering work in computer language design

This agent models market behavior using:
1. Functional composition of trading signals
2. Grammar-based pattern recognition
3. FORTRAN-inspired numerical analysis
4. Function-level market state processing
5. Formal language theory for market patterns

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
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import math
from collections import defaultdict, deque

from ..agent import Agent

logger = logging.getLogger(__name__)

class BackusAgent(Agent):
    """
    Trading agent based on John Backus's programming paradigms.
    
    Parameters
    ----------
    grammar_depth : int, default=4
        Depth of BNF pattern grammar
    fortran_window : int, default=20
        Window size for FORTRAN-inspired calculations
    fp_composition_level : int, default=3
        Level of functional compositions
    algebra_precision : float, default=0.01
        Precision for algebraic equivalence
    syntax_complexity : int, default=2
        Complexity of syntax patterns (1-3)
    """
    
    def __init__(
        self,
        grammar_depth: int = 4,
        fortran_window: int = 20,
        fp_composition_level: int = 3,
        algebra_precision: float = 0.01,
        syntax_complexity: int = 2
    ):
        self.grammar_depth = grammar_depth
        self.fortran_window = fortran_window
        self.fp_composition_level = fp_composition_level
        self.algebra_precision = algebra_precision
        self.syntax_complexity = syntax_complexity
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.bnf_patterns = {}
        self.function_pipeline = []
        self.derivation_tree = {}
        
    def _define_bnf_grammar(self) -> Dict:
        """
        Define a BNF-like grammar for market patterns
        
        Returns
        -------
        dict
            Grammar rules
        """
        # Define grammar rules based on complexity level
        grammar = {
            'trend': ['uptrend', 'downtrend', 'sideways'],
            'volatility': ['high_vol', 'low_vol', 'normal_vol'],
            'volume': ['high_volume', 'low_volume', 'normal_volume'],
            'pattern': ['trend volatility', 'trend volume', 'volatility volume', 'trend volatility volume'],
            'timing': ['early', 'middle', 'late', 'indeterminate'],
            'signal': ['pattern timing', 'timing pattern']
        }
        
        # Add more complex grammar elements at higher complexity levels
        if self.syntax_complexity >= 2:
            grammar['candle'] = ['bullish_candle', 'bearish_candle', 'doji', 'hammer', 'shooting_star']
            grammar['sequence'] = ['candle candle', 'candle candle candle', 'trend candle', 'candle trend']
            grammar['pattern'].append('sequence volatility')
            grammar['pattern'].append('sequence volume')
            
        if self.syntax_complexity >= 3:
            grammar['momentum'] = ['accelerating', 'decelerating', 'steady']
            grammar['divergence'] = ['positive_div', 'negative_div', 'no_div']
            grammar['complex'] = ['momentum divergence', 'divergence trend', 'momentum trend volatility']
            grammar['signal'] = ['complex timing', 'pattern complex', 'complex pattern timing']
            
        return grammar
    
    def _parse_market_state(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, str]:
        """
        Parse current market state into BNF terminals
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            Parsed market state mapped to grammar terminals
        """
        if len(prices) < self.fortran_window:
            return {}
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Use recent window
        recent_returns = returns[-self.fortran_window:]
        recent_prices = prices[-self.fortran_window:]
        
        # Parse trend
        mean_return = np.mean(recent_returns)
        trend_threshold = np.std(recent_returns) * 0.25
        
        if mean_return > trend_threshold:
            trend = 'uptrend'
        elif mean_return < -trend_threshold:
            trend = 'downtrend'
        else:
            trend = 'sideways'
            
        # Parse volatility
        volatility = np.std(recent_returns)
        if len(returns) > self.fortran_window * 2:
            historical_vol = np.std(returns[:-self.fortran_window])
            
            if volatility > historical_vol * 1.5:
                vol_state = 'high_vol'
            elif volatility < historical_vol * 0.5:
                vol_state = 'low_vol'
            else:
                vol_state = 'normal_vol'
        else:
            # Default if not enough history
            vol_state = 'normal_vol'
            
        # Parse volume if available
        volume_state = 'normal_volume'  # Default
        if volumes is not None and len(volumes) >= self.fortran_window:
            recent_volumes = volumes[-self.fortran_window:]
            mean_volume = np.mean(recent_volumes)
            
            if len(volumes) > self.fortran_window * 2:
                historical_volume = np.mean(volumes[-2*self.fortran_window:-self.fortran_window])
                
                if mean_volume > historical_volume * 1.3:
                    volume_state = 'high_volume'
                elif mean_volume < historical_volume * 0.7:
                    volume_state = 'low_volume'
                else:
                    volume_state = 'normal_volume'
                    
        # Parse candle patterns if complexity > 1
        candle_state = None
        if self.syntax_complexity >= 2 and len(prices) >= 2:
            # Simple OHLC approximation from close prices
            recent_changes = np.diff(recent_prices)
            last_change = recent_prices[-1] - recent_prices[-2]
            
            if abs(last_change) < np.mean(np.abs(recent_changes)) * 0.2:
                candle_state = 'doji'
            elif last_change > 0 and last_change > np.percentile(recent_changes, 75):
                candle_state = 'bullish_candle'
            elif last_change < 0 and last_change < np.percentile(recent_changes, 25):
                candle_state = 'bearish_candle'
            elif last_change > 0:
                # Simplified hammer detection
                if np.min(recent_prices[-3:]) == recent_prices[-2]:
                    candle_state = 'hammer'
            elif last_change < 0:
                # Simplified shooting star detection
                if np.max(recent_prices[-3:]) == recent_prices[-2]:
                    candle_state = 'shooting_star'
                    
        # Parse momentum and divergence if complexity > 2
        momentum_state = None
        divergence_state = None
        
        if self.syntax_complexity >= 3:
            # Momentum: acceleration/deceleration of trend
            if len(recent_returns) >= 10:
                first_half_return = np.mean(recent_returns[:len(recent_returns)//2])
                second_half_return = np.mean(recent_returns[len(recent_returns)//2:])
                
                if second_half_return > first_half_return * 1.5:
                    momentum_state = 'accelerating'
                elif second_half_return < first_half_return * 0.5:
                    momentum_state = 'decelerating'
                else:
                    momentum_state = 'steady'
                    
            # Divergence: price vs. momentum
            if len(recent_returns) >= 15:
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                return_trend = np.polyfit(range(len(recent_returns)), recent_returns, 1)[0]
                
                if price_trend > 0 and return_trend < 0:
                    divergence_state = 'negative_div'
                elif price_trend < 0 and return_trend > 0:
                    divergence_state = 'positive_div'
                else:
                    divergence_state = 'no_div'
                    
        # Determine cycle timing
        if len(returns) > self.fortran_window * 3:
            # Estimate cycle using autocorrelation
            corr_values = []
            max_lag = min(self.fortran_window, len(returns) // 3)
            
            for lag in range(1, max_lag + 1):
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                if not np.isnan(corr):
                    corr_values.append((lag, abs(corr)))
                    
            # Find dominant cycle length
            if corr_values:
                dominant_cycle = max(corr_values, key=lambda x: x[1])[0]
                
                # Calculate where we are in the cycle
                position_in_cycle = len(returns) % dominant_cycle
                cycle_portion = position_in_cycle / dominant_cycle
                
                if cycle_portion < 0.33:
                    timing = 'early'
                elif cycle_portion < 0.66:
                    timing = 'middle'
                else:
                    timing = 'late'
            else:
                timing = 'indeterminate'
        else:
            timing = 'indeterminate'
            
        # Construct market state dictionary
        market_state = {
            'trend': trend,
            'volatility': vol_state,
            'volume': volume_state,
            'timing': timing
        }
        
        if candle_state:
            market_state['candle'] = candle_state
            
        if momentum_state:
            market_state['momentum'] = momentum_state
            
        if divergence_state:
            market_state['divergence'] = divergence_state
            
        return market_state
    
    def _derive_patterns(self, grammar: Dict, market_state: Dict[str, str], start_symbol: str = 'signal', depth: int = 0) -> Dict:
        """
        Derive patterns using grammar rules in BNF style
        
        Parameters
        ----------
        grammar : dict
            BNF grammar rules
        market_state : dict
            Current market state
        start_symbol : str, default='signal'
            Starting non-terminal symbol
        depth : int, default=0
            Current recursion depth
            
        Returns
        -------
        dict
            Derivation tree
        """
        if depth >= self.grammar_depth:
            return {}
            
        derivation = {}
        
        # Base case: start symbol is a terminal that exists in market state
        if start_symbol in market_state:
            return {start_symbol: market_state[start_symbol]}
            
        # Recursive case: expand non-terminal using grammar rules
        if start_symbol in grammar:
            expansions = []
            
            for rule in grammar[start_symbol]:
                # Split rule into symbols
                symbols = rule.split()
                
                # Derive each symbol
                valid_expansion = True
                expansion = {}
                
                for symbol in symbols:
                    sub_derivation = self._derive_patterns(
                        grammar, market_state, symbol, depth + 1
                    )
                    
                    if not sub_derivation:
                        valid_expansion = False
                        break
                        
                    expansion[symbol] = sub_derivation
                    
                if valid_expansion:
                    expansions.append(expansion)
                    
            if expansions:
                derivation[start_symbol] = expansions
                
        return derivation
    
    def _functional_map(self, data: np.ndarray, f: Callable) -> np.ndarray:
        """
        Apply function to data (map in functional programming)
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data
        f : callable
            Function to apply
            
        Returns
        -------
        numpy.ndarray
            Transformed data
        """
        return np.array([f(x) for x in data])
    
    def _functional_filter(self, data: np.ndarray, predicate: Callable) -> np.ndarray:
        """
        Filter data by predicate (filter in functional programming)
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data
        predicate : callable
            Predicate function
            
        Returns
        -------
        numpy.ndarray
            Filtered data
        """
        return np.array([x for x in data if predicate(x)])
    
    def _functional_reduce(self, data: np.ndarray, f: Callable, initial: float) -> float:
        """
        Reduce data using a function (reduce in functional programming)
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data
        f : callable
            Binary function
        initial : float
            Initial value
            
        Returns
        -------
        float
            Reduced value
        """
        result = initial
        for x in data:
            result = f(result, x)
        return result
    
    def _compose_functions(self, functions: List[Callable]) -> Callable:
        """
        Compose multiple functions (function composition in FP)
        
        Parameters
        ----------
        functions : list
            List of functions to compose
            
        Returns
        -------
        callable
            Composed function
        """
        def composed(x):
            result = x
            for f in reversed(functions):  # Apply from right to left
                result = f(result)
            return result
        return composed
    
    def _fortran_numerical_analysis(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        Perform FORTRAN-inspired numerical analysis
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        float
            Numerical signal
        """
        # Reduce minimum requirement
        min_required = max(self.fortran_window, 15)
        if len(prices) < min_required:
            return 0.0
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Clean returns
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 10:
            return 0.0
        
        # Define functional programming transformations
        # These are inspired by FORTRAN's numerical methods but using functional style
        
        # First, create some base functions
        def smooth(x): return np.mean(x[-5:]) if len(x) >= 5 else np.mean(x)
        def momentum(x): return x[-1] - x[0] if len(x) > 1 else 0
        def normalize(x): return x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 0
        def square(x): return x * x
        def sign_extract(x): return np.sign(x)
        
        # Create function pipeline based on composition level
        pipeline = []
        
        # Level 1: Basic transformations
        pipeline.append(lambda data: self._functional_map(data, lambda x: x * 2 if abs(x) < 0.01 else x))
        
        if self.fp_composition_level >= 2:
            # Level 2: Add filtering and more transformations
            pipeline.append(lambda data: self._functional_filter(
                data, lambda x: abs(x) <= np.std(data) * 3 if np.std(data) > 0 else True
            ))
            
        if self.fp_composition_level >= 3:
            # Level 3: Add reduction and more complex transformations
            pipeline.append(lambda data: np.array([
                self._functional_reduce(data[-min(len(data), 5):], lambda a, b: a * 0.8 + b * 0.2, data[-1])
            ]) if len(data) > 0 else np.array([0.0]))
            
        # Apply pipeline to different segments of the data
        # This mimics FORTRAN's procedural approach but in a functional way
        recent_returns = returns[-min(len(returns), self.fortran_window):]
        processed_segments = []
        
        segment_size = max(3, len(recent_returns) // 5)
        
        for i in range(0, len(recent_returns), segment_size):
            segment = recent_returns[i:i+segment_size]
            
            # Skip segments that are too short
            if len(segment) < 2:
                continue
                
            try:
                # Process segment through the pipeline
                processed = segment
                for process in pipeline:
                    processed = process(processed)
                    
                    # Ensure processed is not empty
                    if len(processed) == 0:
                        processed = np.array([0.0])
                        break
                
                # Take average of processed values
                if len(processed) > 0:
                    segment_value = np.mean(processed)
                    if np.isfinite(segment_value):
                        processed_segments.append(segment_value)
                        
            except Exception:
                # If processing fails, use simple mean
                segment_mean = np.mean(segment)
                if np.isfinite(segment_mean):
                    processed_segments.append(segment_mean)
                
        # Calculate final signal
        if processed_segments:
            # Calculate weighted average, giving more weight to recent segments
            weights = np.linspace(0.5, 1.0, len(processed_segments))
            weighted_sum = np.sum(np.array(processed_segments) * weights)
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                signal = weighted_sum / total_weight
            else:
                signal = 0.0
            
            # Ensure signal is finite and in range [-1, 1]
            if np.isfinite(signal):
                signal = np.clip(signal, -1.0, 1.0)
            else:
                signal = 0.0
        else:
            # Fallback: simple momentum
            if len(recent_returns) >= 5:
                momentum = np.mean(recent_returns[-5:])
                if np.isfinite(momentum):
                    signal = np.sign(momentum) * min(0.5, abs(momentum) * 10)
                else:
                    signal = 0.0
            else:
                signal = 0.0
            
        return signal
    
    def _algebra_of_signals(self, derivation_tree: Dict, numerical_signal: float) -> float:
        """
        Apply Backus's "algebra of programs" concept to combine signals
        
        Parameters
        ----------
        derivation_tree : dict
            BNF derivation tree
        numerical_signal : float
            Signal from numerical analysis
            
        Returns
        -------
        float
            Combined signal
        """
        if not derivation_tree:
            return numerical_signal
            
        # Extract terminal values from derivation tree
        terminal_values = {}
        
        def extract_terminals(tree, path=''):
            if isinstance(tree, dict):
                for key, value in tree.items():
                    new_path = f"{path}.{key}" if path else key
                    extract_terminals(value, new_path)
            elif isinstance(tree, list):
                for i, item in enumerate(tree):
                    new_path = f"{path}[{i}]"
                    extract_terminals(item, new_path)
            elif isinstance(tree, str):
                terminal_values[path] = tree
                
        extract_terminals(derivation_tree)
        
        # Convert terminal values to numerical signals
        term_signals = {}
        
        for path, value in terminal_values.items():
            # Basic mapping of terminal values to signals
            if 'uptrend' in value or 'bullish' in value or 'positive' in value:
                term_signals[path] = 1.0
            elif 'downtrend' in value or 'bearish' in value or 'negative' in value:
                term_signals[path] = -1.0
            elif 'high_vol' in value or 'high_volume' in value:
                term_signals[path] = 0.5 * np.sign(numerical_signal)  # Amplify existing signal
            elif 'low_vol' in value or 'low_volume' in value:
                term_signals[path] = 0.2 * np.sign(numerical_signal)  # Reduce signal
            elif 'accelerating' in value:
                term_signals[path] = 0.8 * np.sign(numerical_signal)
            elif 'decelerating' in value:
                term_signals[path] = -0.2 * np.sign(numerical_signal)
            elif 'early' in value:
                term_signals[path] = 0.3
            elif 'middle' in value:
                term_signals[path] = 0.1
            elif 'late' in value:
                term_signals[path] = -0.3
            else:
                term_signals[path] = 0.0
                
        # Combine signals using algebraic operations
        if term_signals:
            # Calculate weighted average
            values = list(term_signals.values())
            signal = sum(values) / len(values)
            
            # Apply transformations based on pattern structure
            if any('divergence' in path for path in terminal_values):
                # Divergence patterns often signal reversals
                signal = -signal * 1.2
                
            if any('doji' in value for value in terminal_values.values()):
                # Doji suggests indecision - reduce signal
                signal *= 0.5
                
            if any('hammer' in value or 'shooting_star' in value for value in terminal_values.values()):
                # Potential reversal patterns
                signal = -np.sign(numerical_signal) * 0.8
                
            # Add numerical signal component
            signal = 0.7 * signal + 0.3 * numerical_signal
            
            # Ensure signal is in range [-1, 1]
            signal = np.clip(signal, -1.0, 1.0)
        else:
            signal = numerical_signal
            
        return signal
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data with Backus-inspired programming techniques
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        # Reduce minimum requirement to be more flexible
        min_required = max(self.fortran_window + 5, 25)
        if len(historical_df) < min_required:
            self.is_fitted = False
            return
            
        try:
            # Extract price and volume data
            prices = historical_df['close'].values
            volumes = historical_df['volume'].values if 'volume' in historical_df.columns else None
            
            # Initialize signal components
            numerical_signal = 0.0
            grammar_signal = 0.0
            
            # 1. Define BNF grammar
            try:
                grammar = self._define_bnf_grammar()
            except Exception:
                grammar = {}
            
            # 2. Parse current market state
            try:
                market_state = self._parse_market_state(prices, volumes)
            except Exception:
                market_state = {}
            
            # 3. Derive patterns using BNF grammar
            try:
                if grammar and market_state:
                    derivation_tree = self._derive_patterns(grammar, market_state)
                    self.derivation_tree = derivation_tree
                else:
                    derivation_tree = {}
                    self.derivation_tree = {}
            except Exception:
                derivation_tree = {}
                self.derivation_tree = {}
            
            # 4. Perform FORTRAN-inspired numerical analysis
            try:
                numerical_signal = self._fortran_numerical_analysis(prices, volumes)
                if not np.isfinite(numerical_signal):
                    numerical_signal = 0.0
            except Exception:
                numerical_signal = 0.0
            
            # 5. Apply Backus's algebra of programs to combine signals
            try:
                if derivation_tree:
                    grammar_signal = self._algebra_of_signals(derivation_tree, numerical_signal)
                    if not np.isfinite(grammar_signal):
                        grammar_signal = 0.0
                else:
                    grammar_signal = numerical_signal
            except Exception:
                grammar_signal = numerical_signal
            
            # 6. Combine signals with fallback
            if abs(grammar_signal) > 1e-10:
                self.latest_signal = grammar_signal
            elif abs(numerical_signal) > 1e-10:
                self.latest_signal = numerical_signal
            else:
                # Simple momentum fallback
                if len(prices) >= 10:
                    recent_returns = np.diff(prices[-10:]) / prices[-11:-1]
                    clean_returns = recent_returns[np.isfinite(recent_returns)]
                    if len(clean_returns) > 0:
                        momentum = np.mean(clean_returns)
                        if np.isfinite(momentum):
                            self.latest_signal = np.sign(momentum) * min(0.3, abs(momentum) * 10)
                        else:
                            self.latest_signal = 0.0
                    else:
                        self.latest_signal = 0.0
                else:
                    self.latest_signal = 0.0
            
            # Ensure signal is finite and in range
            if np.isfinite(self.latest_signal):
                self.latest_signal = np.clip(self.latest_signal, -1.0, 1.0)
            else:
                self.latest_signal = 0.0
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Backus Agent fit: {e}")
            self.latest_signal = 0.0
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Backus's programming principles
        
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
        return "Backus Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Backus' functional programming principles.
        
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
            logger.error(f"ValueError in Backus strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Backus strategy: {str(e)}")
            return 0.0000 