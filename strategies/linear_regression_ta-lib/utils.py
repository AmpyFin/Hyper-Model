"""
Utilities for TA-Lib Agents
===========================

Common utilities and base classes for all TA-Lib agents.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def safe_predict_proba(model, X, default_prob=0.5):
    """
    Safely get probability prediction from sklearn model.
    
    This function handles the common "list indices must be integers or slices, not str" 
    error that occurs when predict_proba returns unexpected formats.
    
    Args:
        model: Trained sklearn model with predict_proba method
        X: Features for prediction
        default_prob: Default probability to return if extraction fails
        
    Returns:
        float: Probability of positive class (class 1)
    """
    try:
        # Get prediction probabilities
        prob_result = model.predict_proba(X)
        
        # Handle different return types (array vs list)
        if hasattr(prob_result, 'shape') and len(prob_result.shape) == 2:
            # Standard numpy array format: [[prob_class_0, prob_class_1]]
            prob_up = float(prob_result[0, 1])
        elif isinstance(prob_result, (list, tuple)) and len(prob_result) > 0:
            # Handle potential list format: [[prob_class_0, prob_class_1]]
            if hasattr(prob_result[0], '__len__') and len(prob_result[0]) > 1:
                prob_up = float(prob_result[0][1])
            else:
                prob_up = default_prob
        else:
            # Fallback to default
            prob_up = default_prob
            
        # Ensure result is valid probability
        if np.isnan(prob_up) or np.isinf(prob_up):
            prob_up = default_prob
            
        return prob_up
        
    except (IndexError, KeyError, ValueError, TypeError, AttributeError):
        # Return default probability if any error occurs
        return default_prob


def safe_predict_signal(model, X, default_signal=0.0):
    """
    Safely get trading signal from sklearn model.
    
    Converts probability to signal in range [-1, 1] where:
    - prob > 0.5 gives positive signal (bullish)
    - prob < 0.5 gives negative signal (bearish)
    - prob = 0.5 gives neutral signal
    
    Args:
        model: Trained sklearn model with predict_proba method
        X: Features for prediction
        default_signal: Default signal to return if prediction fails
        
    Returns:
        float: Trading signal in range [-1, 1]
    """
    try:
        prob_up = safe_predict_proba(model, X)
        signal = 2 * prob_up - 1
        
        # Ensure signal is in valid range
        signal = max(-1.0, min(1.0, signal))
        
        return signal
        
    except Exception:
        return default_signal


class BaseAgent:
    """Base class for all TA-Lib agents with robust machine learning setup."""
    
    def __init__(self, max_iter=2000):
        # Use liblinear solver which is more robust for small datasets
        self.m = LogisticRegression(
            max_iter=max_iter, 
            solver='liblinear',  # More robust than default 'lbfgs'
            tol=1e-4,  # Tolerance for stopping criteria
            C=1.0      # Regularization parameter
        )
        self.scaler = StandardScaler()  # For feature scaling
        self.fitted = False
    
    def _scale_features(self, X):
        """Scale features for training or prediction."""
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
        
        # Check if X is a DataFrame or numpy array
        if isinstance(X, pd.DataFrame):
            # Handle empty dataframes
            if X.empty:
                return np.array([])
            # If fitting, learn the scaling parameters
            if not self.fitted:
                return self.scaler.fit_transform(X)
            # Otherwise use existing scaling parameters
            return self.scaler.transform(X)
        else:
            # For numpy arrays
            if len(X) == 0:
                return np.array([])
            
            if not self.fitted:
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)
    
    def replace_inf_with_nan(self, df, columns):
        """Replace infinities with NaN and handle missing values."""
        if columns and len(columns) > 0:
            df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
            df[columns] = df[columns].ffill().bfill()
        return df

class PatternAgent(BaseAgent):
    """Base class for pattern recognition agents with safe boolean operations."""
    
    def safe_and(self, *conditions):
        """Safely perform logical AND on pandas Series with proper NA handling.
        
        Example:
            is_valid = self.safe_and(
                df["close"] > df["open"],
                df["high"] > df["close"],
                df["volume"] > 1000
            )
        """
        if not conditions:
            return pd.Series(True, index=conditions[0].index)
        
        # Start with all True
        result = pd.Series(True, index=conditions[0].index)
        
        # AND each condition carefully
        for cond in conditions:
            # Convert to boolean series with NaN preserved
            if isinstance(cond, (float, int, bool)):
                # Handle scalar values
                scalar_cond = bool(cond)
                result = result & scalar_cond
            else:
                # Handle Series
                cond_filled = cond.fillna(False)
                result = result & cond_filled
                
        return result
    
    def safe_or(self, *conditions):
        """Safely perform logical OR on pandas Series with proper NA handling."""
        if not conditions:
            return pd.Series(False, index=conditions[0].index)
        
        # Start with all False
        result = pd.Series(False, index=conditions[0].index)
        
        # OR each condition carefully
        for cond in conditions:
            # Convert to boolean series with NaN preserved
            if isinstance(cond, (float, int, bool)):
                # Handle scalar values
                scalar_cond = bool(cond)
                result = result | scalar_cond
            else:
                # Handle Series
                cond_filled = cond.fillna(False)
                result = result | cond_filled
                
        return result
    
    def series_compare(self, series1, op, series2):
        """Safely compare two series with NaN handling.
        
        Args:
            series1: First pandas Series
            op: String operator ('>', '<', '>=', '<=', '==', '!=')
            series2: Second pandas Series
            
        Returns:
            Boolean Series with the comparison result
        """
        # Handle NaN values by filling with values that will make the comparison False
        s1 = series1.copy()
        s2 = series2.copy()
        
        if op in ('>', '>='):
            s1 = s1.fillna(-np.inf)
            s2 = s2.fillna(np.inf)
        elif op in ('<', '<='):
            s1 = s1.fillna(np.inf)
            s2 = s2.fillna(-np.inf)
        else:  # == or !=
            # For equality comparisons, NaN should result in False
            s1 = s1.fillna(object())
            s2 = s2.fillna(object())
        
        # Perform the comparison
        if op == '>':
            return s1 > s2
        elif op == '<':
            return s1 < s2
        elif op == '>=':
            return s1 >= s2
        elif op == '<=':
            return s1 <= s2
        elif op == '==':
            return s1 == s2
        elif op == '!=':
            return s1 != s2
        else:
            raise ValueError(f"Unknown operator: {op}")
            
    def is_doji(self, df, i=0, doji_threshold=0.1):
        """Check if a candle is a doji (open ≈ close).
        
        Args:
            df: DataFrame with OHLC data
            i: Shift value (0 = current bar, 1 = previous bar, etc.)
            doji_threshold: Maximum ratio of body/range to be considered a doji
            
        Returns:
            Boolean Series
        """
        o = df["open"].shift(i)
        c = df["close"].shift(i)
        h = df["high"].shift(i)
        l = df["low"].shift(i)
        
        body = (c - o).abs()
        price_range = (h - l)
        
        # Avoid division by zero
        valid_range = price_range > 0
        ratio = pd.Series(np.inf, index=df.index)
        ratio[valid_range] = body[valid_range] / price_range[valid_range]
        
        return ratio <= doji_threshold 