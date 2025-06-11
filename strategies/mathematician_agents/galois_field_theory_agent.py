"""
Galois Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Évariste Galois's principles of
group theory, field theory, and symmetry in algebraic structures.

This agent models market movements as transformations within algebraic groups, where:
1. Symmetry Detection: Identifying symmetrical patterns in price movements
2. Group Transformations: Modeling market states as elements in transformation groups
3. Field Extensions: Analyzing market regimes as distinct fields with different rules
4. Invariance: Finding market properties that remain invariant under transformations

The agent applies these concepts to identify market structure and solvable patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal, stats
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class GaloisAgent:
    """
    Trading agent based on Galois's group theory principles.
    
    Parameters
    ----------
    symmetry_window : int, default=20
        Window size for detecting symmetrical patterns
    transformation_levels : int, default=3
        Number of transformation levels to analyze
    field_threshold : float, default=0.6
        Threshold for field boundary detection
    invariant_check_period : int, default=10
        Period for checking invariant properties
    group_size : int, default=5
        Size of price groups to analyze for transformations
    """
    
    def __init__(
        self,
        symmetry_window: int = 20,
        transformation_levels: int = 3,
        field_threshold: float = 0.6,
        invariant_check_period: int = 10,
        group_size: int = 5
    ):
        self.symmetry_window = symmetry_window
        self.transformation_levels = transformation_levels
        self.field_threshold = field_threshold
        self.invariant_check_period = invariant_check_period
        self.group_size = group_size
        self.latest_signal = 0.0
        self.is_fitted = False
        self.group_properties = {}
        
    def _detect_price_symmetry(self, prices: np.ndarray) -> Tuple[float, List[int]]:
        """
        Detect symmetry patterns in price movements
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        tuple
            (symmetry score, list of symmetry points)
        """
        n = len(prices)
        if n < self.symmetry_window * 2:
            return 0.0, []
            
        # Time reversal symmetry
        symmetry_scores = np.zeros(n - self.symmetry_window)
        for i in range(self.symmetry_window, n):
            # Forward window
            forward = prices[i-self.symmetry_window:i]
            
            # Try different transformation levels
            max_symmetry = 0.0
            
            for level in range(1, self.transformation_levels + 1):
                # Window size for this level
                window_size = self.symmetry_window // level
                
                if window_size < 3:
                    continue
                    
                for j in range(level):
                    # Extract windows at this level
                    start_idx = i - self.symmetry_window + j * window_size
                    end_idx = start_idx + window_size
                    
                    if end_idx > n:
                        break
                        
                    window = prices[start_idx:end_idx]
                    
                    # Time-reversed window
                    reversed_window = window[::-1]
                    
                    # Calculate correlation with time-reversed version
                    # Handle potential divide-by-zero or NaN in correlations
                    if np.std(window) > 0 and np.std(reversed_window) > 0 and not np.isnan(np.std(window)) and not np.isnan(np.std(reversed_window)):
                        # Safe to compute correlation
                        corr = np.corrcoef(window, reversed_window)[0, 1]
                        symmetry = abs(corr)
                        max_symmetry = max(max_symmetry, symmetry)
                    
            symmetry_scores[i - self.symmetry_window] = max_symmetry
            
        # Detect points of high symmetry
        symmetry_points = []
        for i in range(1, len(symmetry_scores) - 1):
            if symmetry_scores[i] > 0.8 and symmetry_scores[i] > symmetry_scores[i-1] and symmetry_scores[i] > symmetry_scores[i+1]:
                symmetry_points.append(i + self.symmetry_window)
                
        # Overall symmetry score
        overall_symmetry = np.mean(symmetry_scores) if len(symmetry_scores) > 0 else 0.0
        
        return overall_symmetry, symmetry_points
    
    def _identify_group_transformations(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Identify group transformations in price series
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Dictionary of transformation types and their strengths
        """
        n = len(prices)
        if n < self.group_size * 3:
            return {}
            
        # Group the prices into segments
        segments = []
        for i in range(0, n - self.group_size + 1, self.group_size):
            segment = prices[i:i+self.group_size]
            if len(segment) == self.group_size:
                segments.append(segment)
        
        if len(segments) < 3:
            return {}
            
        # Analyze transformations between adjacent segments
        transformations = {}
        
        # 1. Translation (addition/subtraction)
        translation_values = []
        for i in range(len(segments) - 1):
            # Calculate average difference between segments
            diff = np.mean(segments[i+1] - segments[i])
            translation_values.append(diff)
        
        # Check if translations are consistent
        trans_mean = np.mean(translation_values)
        trans_std = np.std(translation_values)
        trans_consistency = 1.0 - min(1.0, trans_std / (abs(trans_mean) + 1e-10))
        transformations['translation'] = trans_consistency
        
        # 2. Scaling (multiplication/division)
        scaling_values = []
        for i in range(len(segments) - 1):
            # Calculate average ratio between segments
            ratio = np.mean(segments[i+1] / (segments[i] + 1e-10))
            scaling_values.append(ratio)
        
        # Check if scalings are consistent
        scale_mean = np.mean(scaling_values)
        scale_std = np.std(scaling_values)
        scale_consistency = 1.0 - min(1.0, scale_std / (abs(scale_mean) + 1e-10))
        transformations['scaling'] = scale_consistency
        
        # 3. Reflection (sign change)
        reflection_values = []
        for i in range(len(segments) - 1):
            # Calculate correlation (negative correlation suggests reflection)
            corr = np.corrcoef(segments[i], segments[i+1])[0, 1]
            reflection_values.append(corr)
        
        # Check if reflections are consistent
        refl_mean = np.mean(reflection_values)
        reflection_consistency = 1.0 if refl_mean < -0.5 else (0.0 if refl_mean > 0.5 else (0.5 - refl_mean) * 2)
        transformations['reflection'] = reflection_consistency
        
        # 4. Rotation (cyclic pattern)
        rotation_values = []
        for i in range(len(segments) - 1):
            # Calculate correlation with shifted segment
            for shift in range(1, self.group_size):
                shifted = np.roll(segments[i], shift)
                corr = np.corrcoef(shifted, segments[i+1])[0, 1]
                rotation_values.append((shift, corr))
        
        # Find strongest rotation
        max_corr = -1
        best_shift = 0
        for shift, corr in rotation_values:
            if corr > max_corr:
                max_corr = corr
                best_shift = shift
        
        rotation_consistency = max(0.0, max_corr)
        transformations['rotation'] = rotation_consistency
        transformations['rotation_shift'] = best_shift
        
        return transformations
    
    def _detect_field_extensions(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, List[int]]:
        """
        Detect field extensions (regime changes) in market data
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            Dictionary of field boundaries
        """
        n = len(prices)
        if n < self.symmetry_window * 2:
            return {'field_boundaries': []}
            
        # Calculate properties that could indicate field changes
        
        # 1. Volatility changes
        volatility = np.zeros(n)
        for i in range(self.symmetry_window, n):
            window = prices[i-self.symmetry_window:i]
            volatility[i] = np.std(window)
        
        # 2. Correlation structure changes
        correlation_change = np.zeros(n)
        for i in range(self.symmetry_window * 2, n):
            # Calculate correlation in two adjacent windows
            window1 = prices[i-2*self.symmetry_window:i-self.symmetry_window]
            window2 = prices[i-self.symmetry_window:i]
            
            # Create lagged series for autocorrelation
            w1 = window1[:-1]
            w1_lag = window1[1:]
            w2 = window2[:-1]
            w2_lag = window2[1:]
            
            # Autocorrelation in each window - handle potential divide by zero
            ac1 = 0
            if len(w1) > 1 and np.std(w1) > 0 and np.std(w1_lag) > 0 and not np.isnan(np.std(w1)) and not np.isnan(np.std(w1_lag)):
                ac1 = np.corrcoef(w1, w1_lag)[0, 1]
                
            ac2 = 0
            if len(w2) > 1 and np.std(w2) > 0 and np.std(w2_lag) > 0 and not np.isnan(np.std(w2)) and not np.isnan(np.std(w2_lag)):
                ac2 = np.corrcoef(w2, w2_lag)[0, 1]
            
            # Change in autocorrelation
            correlation_change[i] = abs(ac2 - ac1)
        
        # 3. Volume pattern changes (if volume data is available)
        volume_change = np.zeros(n)
        if volumes is not None:
            for i in range(self.symmetry_window * 2, n):
                # Volume patterns in two adjacent windows
                vol1 = volumes[i-2*self.symmetry_window:i-self.symmetry_window]
                vol2 = volumes[i-self.symmetry_window:i]
                
                # Normalize to compare patterns, not absolute levels
                std_vol1 = np.std(vol1)
                std_vol2 = np.std(vol2)
                
                # Avoid division by zero or NaN values
                norm_vol1 = (vol1 - np.mean(vol1)) / (std_vol1 if std_vol1 > 0 and not np.isnan(std_vol1) else 1)
                norm_vol2 = (vol2 - np.mean(vol2)) / (std_vol2 if std_vol2 > 0 and not np.isnan(std_vol2) else 1)
                
                # Correlation between volume patterns - handle potential divide by zero
                vol_corr = 0
                if (len(norm_vol1) == len(norm_vol2) and len(norm_vol1) > 1 and 
                    np.std(norm_vol1) > 0 and np.std(norm_vol2) > 0 and
                    not np.isnan(np.std(norm_vol1)) and not np.isnan(np.std(norm_vol2))):
                    vol_corr = np.corrcoef(norm_vol1, norm_vol2)[0, 1]
                
                # Higher means more change (less correlation)
                volume_change[i] = 1 - abs(vol_corr)
        
        # Combine indicators of field changes
        combined_change = volatility_change = np.zeros(n)
        for i in range(self.symmetry_window * 2, n):
            # Detect changes in volatility
            vol_prev = np.mean(volatility[i-self.symmetry_window:i-self.symmetry_window//2])
            vol_curr = np.mean(volatility[i-self.symmetry_window//2:i])
            volatility_change[i] = abs(vol_curr - vol_prev) / (vol_prev if vol_prev > 0 else 1)
            
            # Weighted combination
            combined_change[i] = (
                volatility_change[i] * 0.4 +
                correlation_change[i] * 0.4 +
                (volume_change[i] * 0.2 if volumes is not None else 0)
            )
        
        # Detect field boundaries (points of significant change)
        field_boundaries = []
        for i in range(self.symmetry_window * 2 + 1, n):
            if combined_change[i] > self.field_threshold and combined_change[i] > combined_change[i-1]:
                # Check if this is a local maximum
                if i + 1 < n and combined_change[i] > combined_change[i+1]:
                    field_boundaries.append(i)
        
        return {'field_boundaries': field_boundaries}
    
    def _find_invariants(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Find properties that remain invariant across transformations
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            Dictionary of invariant properties
        """
        n = len(prices)
        if n < self.invariant_check_period * 3:
            return {}
            
        invariants = {}
        
        # Split data into periods
        periods = []
        for i in range(0, n - self.invariant_check_period + 1, self.invariant_check_period):
            period_prices = prices[i:i+self.invariant_check_period]
            if len(period_prices) == self.invariant_check_period:
                period_data = {
                    'prices': period_prices,
                    'volumes': volumes[i:i+self.invariant_check_period] if volumes is not None else None
                }
                periods.append(period_data)
        
        if len(periods) < 3:
            return {}
        
        # Check various potential invariants
        
        # 1. Price-volume relationship (if volume data is available)
        if volumes is not None:
            price_vol_corrs = []
            for period in periods:
                p = period['prices']
                v = period['volumes']
                # Correlation between absolute price changes and volume
                price_changes = np.abs(np.diff(np.concatenate([[p[0]], p])))
                
                # Handle potential divide by zero in correlation calculation
                if (len(price_changes) == len(v) and len(price_changes) > 1 and 
                    np.std(price_changes) > 0 and np.std(v) > 0 and
                    not np.isnan(np.std(price_changes)) and not np.isnan(np.std(v))):
                    corr = np.corrcoef(price_changes, v)[0, 1]
                else:
                    corr = 0
                    
                price_vol_corrs.append(corr)
            
            # Is this relationship invariant?
            pv_std = np.std(price_vol_corrs)
            invariants['price_volume_relationship'] = 1.0 - min(1.0, pv_std)
        
        # 2. Return distribution shape
        return_skews = []
        return_kurts = []
        for period in periods:
            p = period['prices']
            rets = np.diff(p) / p[:-1]
            # Skewness and kurtosis of returns
            skew = stats.skew(rets) if len(rets) > 2 else 0
            kurt = stats.kurtosis(rets) if len(rets) > 2 else 0
            return_skews.append(skew)
            return_kurts.append(kurt)
        
        # Check if distribution shape is invariant
        skew_std = np.std(return_skews)
        kurt_std = np.std(return_kurts)
        invariants['return_skew_invariance'] = 1.0 - min(1.0, skew_std / 2.0)
        invariants['return_kurt_invariance'] = 1.0 - min(1.0, kurt_std / 5.0)
        
        # 3. Fractal dimension (Hurst exponent proxy)
        hurst_exponents = []
        for period in periods:
            p = period['prices']
            if len(p) < 10:
                continue
                
            # Simple proxy for Hurst exponent using variance ratios
            var1 = np.var(np.diff(p))
            var2 = np.var(np.diff(p[::2])) / 2
            
            # H = log(var2/var1) / log(2) + 0.5
            # Avoid division by zero or negative values for logarithm
            if var1 > 0 and var2 > 0:
                h_est = np.log(var2 / var1) / np.log(2) + 0.5
                hurst_exponents.append(h_est)
        
        # Check if Hurst exponent is invariant
        if hurst_exponents:
            hurst_std = np.std(hurst_exponents)
            invariants['fractal_dimension'] = 1.0 - min(1.0, hurst_std / 0.2)
        
        # Average invariance score
        invariants['overall_invariance'] = np.mean(list(invariants.values())) if invariants else 0.0
        
        return invariants
    
    def _generate_galois_signal(
        self, 
        prices: np.ndarray, 
        symmetry_score: float, 
        transformations: Dict[str, float], 
        field_boundaries: List[int],
        invariants: Dict[str, float]
    ) -> float:
        """
        Generate trading signal based on Galois theory principles
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        symmetry_score : float
            Detected symmetry score
        transformations : dict
            Dictionary of transformation types and their strengths
        field_boundaries : list
            List of detected field boundary indices
        invariants : dict
            Dictionary of invariant properties
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        signal = 0.0
        
        # 1. Symmetry-based signal (30%)
        # Higher symmetry score suggests potential trend reversal
        if symmetry_score > 0.3:
            # Recent trend direction
            recent_trend = np.mean(np.diff(prices[-5:]))
            # Symmetry suggests potential reversal
            symmetry_signal = -np.sign(recent_trend) * symmetry_score
        else:
            symmetry_signal = 0.0
            
        # 2. Transformation-based signal (30%)
        transform_signal = 0.0
        if transformations:
            # Translation - consistent movement in one direction
            if 'translation' in transformations:
                trans_consistency = transformations['translation']
                if trans_consistency > 0.7:
                    # Get recent trend
                    segments = []
                    for i in range(0, len(prices) - self.group_size + 1, self.group_size):
                        segment = prices[i:i+self.group_size]
                        if len(segment) == self.group_size:
                            segments.append(segment)
                    
                    if len(segments) >= 2:
                        # Direction of recent translation
                        recent_trans = np.mean(segments[-1] - segments[-2])
                        trans_signal = np.sign(recent_trans) * trans_consistency
                    else:
                        trans_signal = 0.0
                else:
                    trans_signal = 0.0
            
            # Scaling - exponential growth/decay
            scaling_signal = 0.0
            if 'scaling' in transformations:
                scale_consistency = transformations['scaling']
                if scale_consistency > 0.7:
                    # Recent scaling factor
                    segments = []
                    for i in range(0, len(prices) - self.group_size + 1, self.group_size):
                        segment = prices[i:i+self.group_size]
                        if len(segment) == self.group_size:
                            segments.append(segment)
                    
                    if len(segments) >= 2:
                        # Direction of recent scaling
                        recent_scale = np.mean(segments[-1] / (segments[-2] + 1e-10))
                        scaling_signal = np.sign(recent_scale - 1.0) * scale_consistency
                    else:
                        scaling_signal = 0.0
                else:
                    scaling_signal = 0.0
            
            # Combine transformation signals
            transform_signal = (trans_signal * 0.6 + scaling_signal * 0.4)
            
        # 3. Field extension signal (20%)
        field_signal = 0.0
        if field_boundaries and field_boundaries[-1] > len(prices) - 10:
            # Recent field boundary detected - suggests regime change
            # Direction depends on pre/post boundary trend
            if field_boundaries[-1] < len(prices) - 5:
                pre_trend = np.mean(np.diff(prices[field_boundaries[-1]-5:field_boundaries[-1]]))
                post_trend = np.mean(np.diff(prices[field_boundaries[-1]:field_boundaries[-1]+5]))
                
                # If trends differ in direction, follow new trend strongly
                if np.sign(pre_trend) != np.sign(post_trend):
                    field_signal = np.sign(post_trend) * 0.8
                else:
                    # Same direction, but possibly different magnitude
                    field_signal = np.sign(post_trend) * 0.4
        
        # 4. Invariant-based signal (20%)
        invariant_signal = 0.0
        if 'overall_invariance' in invariants:
            overall_invariance = invariants['overall_invariance']
            
            # If market is in a highly invariant state, trend is likely to continue
            # If invariance is breaking down, potential reversal
            recent_trend = np.mean(np.diff(prices[-5:]))
            
            if overall_invariance > 0.7:
                # High invariance - trend continuation
                invariant_signal = np.sign(recent_trend) * overall_invariance
            elif overall_invariance < 0.3:
                # Low invariance - potential reversal
                invariant_signal = -np.sign(recent_trend) * (1.0 - overall_invariance)
        
        # Combine all signals with weights
        signal = (
            symmetry_signal * 0.3 +
            transform_signal * 0.3 +
            field_signal * 0.2 +
            invariant_signal * 0.2
        )
        
        # Ensure signal is in [-1, 1] range
        return np.clip(signal, -1.0, 1.0)
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        min_required_bars = max(self.symmetry_window * 2, self.group_size * 3, self.invariant_check_period * 3)
        if len(historical_df) < min_required_bars:
            self.is_fitted = False
            return
            
        try:
            # Make a copy of the dataframe
            df = historical_df.copy()
            
            # Extract price and volume data
            prices = df['close'].values
            volumes = df['volume'].values if 'volume' in df.columns else None
            
            # 1. Detect price symmetry
            symmetry_score, symmetry_points = self._detect_price_symmetry(prices)
            
            # 2. Identify group transformations
            transformations = self._identify_group_transformations(prices)
            
            # 3. Detect field extensions (regime changes)
            field_data = self._detect_field_extensions(prices, volumes)
            field_boundaries = field_data['field_boundaries']
            
            # 4. Find invariants
            invariants = self._find_invariants(prices, volumes)
            
            # Store group properties
            self.group_properties = {
                'symmetry_score': symmetry_score,
                'symmetry_points': symmetry_points,
                'transformations': transformations,
                'field_boundaries': field_boundaries,
                'invariants': invariants
            }
            
            # Generate signal
            self.latest_signal = self._generate_galois_signal(
                prices, symmetry_score, transformations, field_boundaries, invariants
            )
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Galois Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Galois group theory principles
        
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
        return "Galois Agent" 