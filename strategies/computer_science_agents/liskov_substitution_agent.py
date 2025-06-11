"""
Liskov Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Barbara Liskov's contributions to
computer science, particularly the Liskov Substitution Principle (LSP), abstract data types,
and her work on distributed computing and fault tolerance.

Barbara Liskov is known for:
1. Liskov Substitution Principle - a fundamental principle of object-oriented design
2. Abstract data types and data abstraction
3. CLU programming language with support for data abstraction
4. Thor object-oriented database system
5. Argus distributed programming language with focus on fault tolerance

This agent models market behavior using:
1. Type hierarchies with substitutability constraints
2. Abstract market state representations
3. Fault tolerance principles for noisy market data
4. Behavioral subtyping for regime classification

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
from typing import Dict, List, Optional, Tuple, Union, Any, Protocol, TypeVar
import logging
import math
from collections import defaultdict, deque
from abc import ABC, abstractmethod

from ..agent import Agent

logger = logging.getLogger(__name__)

# Define type variables and protocols (inspired by Liskov's type systems)
MarketData = TypeVar('MarketData')
Signal = TypeVar('Signal', bound=float)

class MarketState(Protocol):
    """Protocol defining market state interface"""
    def is_valid(self) -> bool: pass
    def get_signal(self) -> Signal: pass

class LiskovAgent(Agent):
    """
    Trading agent based on Barbara Liskov's computer science principles.
    
    Parameters
    ----------
    substitution_threshold : float, default=0.7
        Threshold for determining valid type substitutions
    abstraction_level : int, default=3
        Level of market data abstraction (1-5)
    fault_tolerance : float, default=0.5
        Degree of tolerance for market anomalies
    behavioral_window : int, default=30
        Window size for behavioral analysis
    inheritance_depth : int, default=3
        Depth of market regime inheritance hierarchy
    """
    
    def __init__(
        self,
        substitution_threshold: float = 0.7,
        abstraction_level: int = 3,
        fault_tolerance: float = 0.5,
        behavioral_window: int = 30,
        inheritance_depth: int = 3
    ):
        self.substitution_threshold = substitution_threshold
        self.abstraction_level = abstraction_level
        self.fault_tolerance = fault_tolerance
        self.behavioral_window = behavioral_window
        self.inheritance_depth = inheritance_depth
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.market_types = {}
        self.type_history = deque(maxlen=30)
        self.behavior_specs = {}
        self.abstractions = {}
        
    # Abstract Base Classes (inspired by Liskov's work on abstraction)
    class AbstractMarketRegime(ABC):
        """Abstract base class for market regimes"""
        
        @abstractmethod
        def calculate_signal(self, data: np.ndarray) -> float:
            """Calculate trading signal based on regime"""
            pass
            
        @abstractmethod
        def is_compatible(self, data: np.ndarray) -> bool:
            """Check if data is compatible with regime"""
            pass
            
    # Concrete subclasses implementing different market regimes
    class TrendingRegime(AbstractMarketRegime):
        """Trending market regime"""
        
        def calculate_signal(self, data: np.ndarray) -> float:
            """Trend-following signal generation"""
            try:
                if len(data) < 10:
                    return 0.0
                    
                # Calculate moving averages
                short_ma = np.mean(data[-5:])
                long_ma = np.mean(data[-10:])
                
                # Generate trend following signal
                return float(np.sign(short_ma - long_ma))
            except:
                return 0.0
            
        def is_compatible(self, data: np.ndarray) -> bool:
            """Check if market is trending"""
            try:
                if len(data) < 20:
                    return False
                    
                # Calculate trend strength using linear regression slope
                x = np.arange(len(data[-20:]))
                slope, _, _, _, _ = np.polyfit(x, data[-20:], 1, full=True)
                
                return abs(slope[0]) > 0.001 * np.mean(data[-20:])
            except:
                return False
            
    class MeanRevertingRegime(AbstractMarketRegime):
        """Mean-reverting market regime"""
        
        def calculate_signal(self, data: np.ndarray) -> float:
            """Mean reversion signal generation"""
            try:
                if len(data) < 10:
                    return 0.0
                    
                # Calculate z-score
                mean = np.mean(data[-10:])
                std = np.std(data[-10:])
                
                if std == 0:
                    return 0.0
                    
                z_score = (data[-1] - mean) / std
                
                # Generate mean reversion signal (inverse of z-score)
                return float(-np.clip(z_score, -1.0, 1.0))
            except:
                return 0.0
            
        def is_compatible(self, data: np.ndarray) -> bool:
            """Check if market is mean-reverting"""
            if len(data) < 20:
                return False
                
            # Check for mean reversion using autocorrelation
            returns = np.diff(data) / data[:-1]
            if len(returns) < 10:
                return False
                
            try:
                # Ensure we have enough data for autocorrelation
                if len(returns) < 2:
                    return False
                    
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                
                # Handle NaN case
                if np.isnan(autocorr):
                    return False
                    
                return autocorr < -0.2  # Negative autocorrelation suggests mean reversion
            except:
                return False
            
    class VolatilityRegime(AbstractMarketRegime):
        """Volatility-based market regime"""
        
        def calculate_signal(self, data: np.ndarray) -> float:
            """Volatility-based signal generation"""
            try:
                if len(data) < 15:
                    return 0.0
                    
                # Calculate current vs historical volatility
                current_vol = np.std(np.diff(data[-5:]) / data[-6:-1])
                historical_vol = np.std(np.diff(data[-15:-5]) / data[-16:-6])
                
                if historical_vol == 0:
                    return 0.0
                    
                vol_ratio = current_vol / historical_vol
                
                # Generate volatility-based signal
                if vol_ratio > 1.5:
                    return 0.0  # High volatility - neutral position
                elif vol_ratio < 0.5:
                    return float(0.5 * np.sign(data[-1] - data[-2]))  # Low volatility - follow short-term trend
                else:
                    return 0.0
            except:
                return 0.0
                
        def is_compatible(self, data: np.ndarray) -> bool:
            """Check if market is in a distinct volatility regime"""
            try:
                if len(data) < 20:
                    return False
                    
                # Check for significant volatility change
                returns = np.diff(data) / data[:-1]
                
                if len(returns) < 15:
                    return False
                    
                recent_vol = np.std(returns[-5:])
                prev_vol = np.std(returns[-15:-5])
                
                if prev_vol == 0:
                    return False
                    
                return abs(recent_vol / prev_vol - 1) > 0.3  # 30% change in volatility
            except:
                return False
            
    def _create_abstraction(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create abstract data representations of market data
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Abstract data representations
        """
        if len(prices) < 20:
            return {}
            
        abstractions = {}
        
        try:
            # Level 1: Basic moving averages
            if self.abstraction_level >= 1:
                abstractions['ma_short'] = np.convolve(prices, np.ones(5)/5, mode='valid')
                abstractions['ma_long'] = np.convolve(prices, np.ones(20)/20, mode='valid')
                
            # Level 2: Add returns and volatility
            if self.abstraction_level >= 2:
                returns = np.diff(prices) / prices[:-1]
                abstractions['returns'] = returns
                abstractions['volatility'] = np.array([np.std(returns[max(0, i-10):i+1]) 
                                                   for i in range(len(returns))])
                                                   
            # Level 3: Add momentum and mean reversion metrics
            if self.abstraction_level >= 3:
                if len(prices) >= 20:
                    momentum = np.array([prices[i] / prices[i-10] - 1 for i in range(10, len(prices))])
                    abstractions['momentum'] = momentum
                    
                    # Mean reversion metric: z-score (simplified)
                    try:
                        ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
                        std20 = np.array([np.std(prices[i-20:i]) for i in range(20, len(prices))])
                        
                        # Ensure arrays have the same shape before subtraction
                        min_len = min(len(ma20), len(std20), len(prices[20:]))
                        if min_len > 0:
                            z_score = (prices[20:20+min_len] - ma20[:min_len]) / (std20[:min_len] + 1e-8)
                            abstractions['z_score'] = z_score
                        else:
                            abstractions['z_score'] = np.array([0.0])
                    except:
                        abstractions['z_score'] = np.array([0.0])
                    
            # Level 4: Add regime indicators
            if self.abstraction_level >= 4:
                if len(prices) >= 30:
                    try:
                        # Trend indicator
                        x = np.arange(30)
                        trend_indicator = []
                        
                        for i in range(30, len(prices)):
                            window = prices[i-30:i]
                            slope, _, _, _, _ = np.polyfit(x, window, 1, full=True)
                            trend_indicator.append(slope[0] * 30 / np.mean(window))  # Normalized slope
                            
                        abstractions['trend_indicator'] = np.array(trend_indicator) if trend_indicator else np.array([0.0])
                        
                        # Mean reversion indicator (autocorrelation)
                        returns = abstractions.get('returns', np.diff(prices) / prices[:-1])
                        autocorr = []
                        
                        for i in range(20, len(returns)):
                            window = returns[i-20:i]
                            if len(window) >= 2:
                                try:
                                    ac = np.corrcoef(window[:-1], window[1:])[0, 1]
                                    if not np.isnan(ac):
                                        autocorr.append(ac)
                                    else:
                                        autocorr.append(0.0)
                                except:
                                    autocorr.append(0.0)
                            else:
                                autocorr.append(0.0)
                            
                        abstractions['autocorrelation'] = np.array(autocorr) if autocorr else np.array([0.0])
                    except:
                        abstractions['trend_indicator'] = np.array([0.0])
                        abstractions['autocorrelation'] = np.array([0.0])
                    
            # Level 5: Add complex derivatives
            if self.abstraction_level >= 5:
                if len(prices) >= 40 and 'returns' in abstractions:
                    try:
                        returns = abstractions['returns']
                        
                        # Skew and kurtosis over rolling window
                        skew = []
                        kurt = []
                        
                        for i in range(20, len(returns)):
                            window = returns[i-20:i]
                            if len(window) > 0 and np.std(window) > 0:
                                # Simple skew approximation
                                s = np.mean((window - np.mean(window))**3) / (np.std(window)**3 + 1e-8)
                                skew.append(s)
                                
                                # Simple kurtosis approximation
                                k = np.mean((window - np.mean(window))**4) / (np.std(window)**4 + 1e-8) - 3
                                kurt.append(k)
                            else:
                                skew.append(0.0)
                                kurt.append(0.0)
                                
                        abstractions['skew'] = np.array(skew) if skew else np.array([0.0])
                        abstractions['kurtosis'] = np.array(kurt) if kurt else np.array([0.0])
                    except:
                        abstractions['skew'] = np.array([0.0])
                        abstractions['kurtosis'] = np.array([0.0])
        
        except Exception as e:
            # If any error occurs, return minimal abstractions
            logger.error(f"Error in abstraction creation: {e}")
            abstractions = {'returns': np.array([0.0]), 'volatility': np.array([0.0])}
        
        return abstractions
        
    def _check_substitution_principle(self, base_regime: AbstractMarketRegime, 
                                     derived_regime: AbstractMarketRegime,
                                     data: np.ndarray) -> bool:
        """
        Apply Liskov Substitution Principle to market regimes
        
        Parameters
        ----------
        base_regime : AbstractMarketRegime
            Base market regime
        derived_regime : AbstractMarketRegime
            Derived market regime
        data : numpy.ndarray
            Market data
            
        Returns
        -------
        bool
            True if substitution is valid
        """
        # LSP check 1: If base is compatible, derived should be compatible
        if base_regime.is_compatible(data) and not derived_regime.is_compatible(data):
            return False
            
        # LSP check 2: Signal from derived should not contradict base too strongly
        if base_regime.is_compatible(data) and derived_regime.is_compatible(data):
            base_signal = base_regime.calculate_signal(data)
            derived_signal = derived_regime.calculate_signal(data)
            
            # Check if signals don't contradict beyond threshold
            if abs(base_signal - derived_signal) > 2 - self.substitution_threshold:
                return False
                
        return True
        
    def _apply_fault_tolerance(self, market_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply fault tolerance to handle anomalies in market data
        
        Parameters
        ----------
        market_data : dict
            Dictionary of market data arrays
            
        Returns
        -------
        dict
            Cleaned market data
        """
        cleaned_data = {}
        
        for key, data in market_data.items():
            if len(data) < 3:
                cleaned_data[key] = data
                continue
                
            # Apply fault tolerance based on level
            if self.fault_tolerance > 0:
                # Remove outliers
                mean = np.mean(data)
                std = np.std(data)
                
                # Threshold based on fault tolerance setting
                threshold = 3.0 / self.fault_tolerance
                
                # Replace outliers with interpolated values
                cleaned = np.copy(data)
                outliers = np.where(np.abs(data - mean) > threshold * std)[0]
                
                for idx in outliers:
                    if idx > 0 and idx < len(data) - 1:
                        # Replace with average of neighbors
                        cleaned[idx] = (cleaned[idx-1] + cleaned[idx+1]) / 2
                    elif idx == 0:
                        cleaned[idx] = cleaned[idx+1]
                    else:  # idx == len(data) - 1
                        cleaned[idx] = cleaned[idx-1]
                        
                cleaned_data[key] = cleaned
            else:
                cleaned_data[key] = data
                
        return cleaned_data
        
    def _determine_market_type(self, prices: np.ndarray, abstractions: Dict[str, np.ndarray]) -> str:
        """
        Determine the current market type using type hierarchy
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        abstractions : dict
            Dictionary of market data abstractions
            
        Returns
        -------
        str
            Market type label
        """
        # Create regime instances
        trending_regime = self.TrendingRegime()
        mean_reverting_regime = self.MeanRevertingRegime()
        volatility_regime = self.VolatilityRegime()
        
        # Check compatibility
        is_trending = trending_regime.is_compatible(prices)
        is_mean_reverting = mean_reverting_regime.is_compatible(prices)
        is_volatility_regime = volatility_regime.is_compatible(prices)
        
        # Check LSP relationships
        trend_mean_valid = self._check_substitution_principle(
            trending_regime, mean_reverting_regime, prices
        )
        
        mean_trend_valid = self._check_substitution_principle(
            mean_reverting_regime, trending_regime, prices
        )
        
        # Determine market type based on compatibility and LSP
        if is_trending and not is_mean_reverting:
            return "strong_trend"
        elif is_mean_reverting and not is_trending:
            return "strong_mean_reversion"
        elif is_trending and is_mean_reverting:
            if trend_mean_valid and not mean_trend_valid:
                return "trend_with_pullbacks"
            elif mean_trend_valid and not trend_mean_valid:
                return "range_with_bias"
            elif trend_mean_valid and mean_trend_valid:
                return "mixed_regime"
            else:
                return "regime_transition"
        elif is_volatility_regime:
            return "volatility_regime"
        else:
            return "undefined_regime"
            
    def _behavioral_subtyping(self, market_type: str, prices: np.ndarray, 
                             abstractions: Dict[str, np.ndarray]) -> float:
        """
        Generate signal based on behavioral subtyping principles
        
        Parameters
        ----------
        market_type : str
            Current market type
        prices : numpy.ndarray
            Array of price values
        abstractions : dict
            Dictionary of market data abstractions
            
        Returns
        -------
        float
            Trading signal
        """
        signal = 0.0
        
        try:
            # Create regime instances
            trending_regime = self.TrendingRegime()
            mean_reverting_regime = self.MeanRevertingRegime()
            volatility_regime = self.VolatilityRegime()
            
            # Generate signals based on market type (behavioral subtyping)
            if market_type == "strong_trend":
                signal = trending_regime.calculate_signal(prices)
            elif market_type == "strong_mean_reversion":
                signal = mean_reverting_regime.calculate_signal(prices)
            elif market_type == "trend_with_pullbacks":
                # Combine signals with emphasis on trend
                trend_signal = trending_regime.calculate_signal(prices)
                reversion_signal = mean_reverting_regime.calculate_signal(prices)
                signal = 0.7 * trend_signal + 0.3 * reversion_signal
            elif market_type == "range_with_bias":
                # Combine signals with emphasis on mean reversion
                trend_signal = trending_regime.calculate_signal(prices)
                reversion_signal = mean_reverting_regime.calculate_signal(prices)
                signal = 0.3 * trend_signal + 0.7 * reversion_signal
            elif market_type == "mixed_regime":
                # Use momentum to decide weights
                if 'momentum' in abstractions and len(abstractions['momentum']) > 0:
                    momentum = abstractions['momentum'][-1]
                    weight = 0.5 + 0.5 * np.clip(momentum, -1.0, 1.0)
                    trend_signal = trending_regime.calculate_signal(prices)
                    reversion_signal = mean_reverting_regime.calculate_signal(prices)
                    signal = weight * trend_signal + (1 - weight) * reversion_signal
                else:
                    # Equal weight if momentum not available
                    signal = 0.5 * trending_regime.calculate_signal(prices) + \
                             0.5 * mean_reverting_regime.calculate_signal(prices)
            elif market_type == "volatility_regime":
                signal = volatility_regime.calculate_signal(prices)
            elif market_type == "regime_transition":
                # Conservative approach during transitions
                signal = 0.0
            else:  # undefined_regime
                # Use simple momentum as fallback
                if len(prices) >= 10:
                    signal = np.sign(prices[-1] - prices[-10]) * 0.3  # Reduced conviction
                else:
                    signal = 0.0
        except Exception as e:
            logger.error(f"Error in behavioral subtyping: {e}")
            signal = 0.0
                
        return signal
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data using Liskov's type system principles
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        # Reduce minimum requirement to be more flexible
        min_required = max(self.behavioral_window, 25)
        if len(historical_df) < min_required:
            self.is_fitted = False
            return
            
        try:
            # Extract price data
            prices = historical_df['close'].values
            
            # Initialize components
            abstractions = {}
            cleaned_abstractions = {}
            market_type = "undefined_regime"
            signal = 0.0
            
            # 1. Create abstract data representations
            try:
                abstractions = self._create_abstraction(prices)
            except Exception as e:
                logger.warning(f"Abstraction creation failed: {e}")
                abstractions = {}
            
            # 2. Apply fault tolerance to abstractions
            try:
                if abstractions:
                    cleaned_abstractions = self._apply_fault_tolerance(abstractions)
                else:
                    cleaned_abstractions = {}
            except Exception as e:
                logger.warning(f"Fault tolerance failed: {e}")
                cleaned_abstractions = abstractions
            
            # 3. Determine current market type using LSP
            try:
                market_type = self._determine_market_type(prices, cleaned_abstractions)
            except Exception as e:
                logger.warning(f"Market type determination failed: {e}")
                market_type = "undefined_regime"
            
            # Store market type in history
            self.type_history.append(market_type)
            
            # 4. Generate signal using behavioral subtyping
            try:
                signal = self._behavioral_subtyping(market_type, prices, cleaned_abstractions)
                if not np.isfinite(signal):
                    signal = 0.0
            except Exception as e:
                logger.warning(f"Behavioral subtyping failed: {e}")
                signal = 0.0
            
            # 5. Fallback signal generation if main signal is zero
            if abs(signal) < 1e-10:
                try:
                    # Simple momentum fallback
                    if len(prices) >= 10:
                        recent_returns = np.diff(prices[-10:]) / prices[-11:-1]
                        clean_returns = recent_returns[np.isfinite(recent_returns)]
                        if len(clean_returns) > 0:
                            momentum = np.mean(clean_returns)
                            if np.isfinite(momentum):
                                signal = np.sign(momentum) * min(0.3, abs(momentum) * 10)
                except Exception:
                    signal = 0.0
            
            # Store abstractions for internal state
            self.abstractions = cleaned_abstractions
            
            # Store the signal
            if np.isfinite(signal):
                self.latest_signal = np.clip(signal, -1.0, 1.0)
            else:
                self.latest_signal = 0.0
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Liskov Agent fit: {e}")
            self.latest_signal = 0.0
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Liskov's type system principles
        
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
        return "Liskov Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Liskov's substitution principles.
        
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
            logger.error(f"ValueError in Liskov strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Liskov strategy: {str(e)}")
            return 0.0000 