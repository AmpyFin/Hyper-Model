"""
CORREL Agent
============

30-period rolling **Pearson correlation** ρ between *close* and *open*.

Features
--------
* correl (ρ)
* |ρ| slope
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


def _correl(s1: pd.Series, s2: pd.Series, n: int = 30):
    """Calculate correlation with proper handling of edge cases."""
    correlation = s1.rolling(n).corr(s2)
    # Handle infinities and extreme values
    return correlation.replace([np.inf, -np.inf], np.nan)


class CORREL_Agent:
    def __init__(self, period: int = 30):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.scaler = RobustScaler()  # Use robust scaler to handle outliers
        self.fitted = False

    def _feat(self, df):
        """Extract features with comprehensive data cleaning."""
        # Calculate correlation
        r = _correl(df["close"], df["open"], self.n)
        
        # Calculate absolute slope with proper handling of edge cases
        abs_slope = r.abs().diff().replace([np.inf, -np.inf], np.nan)
        
        # Create DataFrame with clean features
        d = df.copy()
        d["corr"] = r
        d["abs_slope"] = abs_slope
        
        # Drop rows with NaN or infinite values in features
        d = d.replace([np.inf, -np.inf], np.nan)
        return d.dropna(subset=["corr", "abs_slope"])

    def fit(self, df):
        """Fit model with clean data and proper error handling."""
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for CORREL_Agent")
            
        X = d[["corr", "abs_slope"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        
        # Scale features to handle outliers
        X_scaled = self.scaler.fit_transform(X)
        
        # Check if data is valid before fitting
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            # Clean any remaining problematic values
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
            
        self.m.fit(X_scaled, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        """Make prediction with proper data validation."""
        if not self.fitted:
            self.fit(historical_df)
            
        last = self._feat(historical_df).iloc[-1:][["corr", "abs_slope"]]
        
        # Handle case where last row might contain NaN/inf
        if last.isnull().any().any() or np.isinf(last.values).any():
            # If invalid data, return neutral prediction
            return 0.0
            
        # Scale features
        last_scaled = self.scaler.transform(last)
        
        return 2 * self.m.predict_proba(last_scaled)[0, 1] - 1
