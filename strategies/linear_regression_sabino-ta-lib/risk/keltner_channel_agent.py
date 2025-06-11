"""
Keltner Channel Agent
=====================

Default parameters
------------------
* Length **20**
* ATR multiplier **2**

Bands
-----
middle = EMA(close, n)  
upper  = middle + m × ATR(n)  
lower  = middle − m × ATR(n)

Features
--------
* **pctB**   = (close − lower) / (upper − lower)   (0…1)
* **width**  = (upper − lower) / middle
* **pctB_slope**  (first diff)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


def _ema(series: pd.Series, span: int):
    """Calculate EMA with proper handling of NaN values."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _atr(df: pd.DataFrame, n: int):
    """Calculate ATR with proper handling of edge cases."""
    # Ensure all inputs are numeric
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    
    # Calculate true range components
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    # Handle NaN and infinite values
    tr1 = tr1.replace([np.inf, -np.inf], np.nan)
    tr2 = tr2.replace([np.inf, -np.inf], np.nan)
    tr3 = tr3.replace([np.inf, -np.inf], np.nan)
    
    # Combine TR components safely
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3})
    tr = tr.max(axis=1)
    
    # Calculate the ATR
    return tr.rolling(n, min_periods=n).mean()


class KeltnerChannel_Agent:
    def __init__(self, length: int = 20, mult: float = 2.0):
        self.n = length
        self.m = mult
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = RobustScaler()  # Use robust scaler for outliers
        self.fitted = False

    def _safe_division(self, a, b, fill_value=0.5):
        """Safely divide series to handle zeros and infinities."""
        result = a / b
        # Replace infinite values with the fill_value or estimated extremes
        result = result.replace([np.inf, -np.inf], np.nan)
        # Fill NaN with a neutral value for percentage calculations
        result = result.fillna(fill_value)
        return result

    def _feat(self, df):
        """Extract features with comprehensive data cleaning."""
        # Ensure all inputs are numeric
        close = pd.to_numeric(df["close"], errors="coerce")
        
        # Calculate Keltner Channel components
        mid = _ema(close, self.n)
        atr = _atr(df, self.n)
        
        # Handle NaNs in calculations
        mid = mid.replace([np.inf, -np.inf], np.nan)
        atr = atr.replace([np.inf, -np.inf], np.nan)
        
        # Calculate bands
        upper = mid + self.m * atr
        lower = mid - self.m * atr
        
        # Calculate pctB safely (position within the channel)
        band_width = upper - lower
        price_from_lower = close - lower
        
        # Use safe division to avoid infinities
        pctB = self._safe_division(price_from_lower, band_width)
        
        # Calculate width as percentage of middle band
        width = self._safe_division(band_width, mid)
        
        # Calculate slope with proper handling
        pctB_slope = pctB.diff().replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Create DataFrame with clean features
        d = df.copy()
        d["pctB"] = pctB
        d["width"] = width
        d["pctB_slope"] = pctB_slope
        
        # Clean up any remaining infinities or NaNs
        d = d.replace([np.inf, -np.inf], np.nan)
        return d.dropna(subset=["pctB", "width", "pctB_slope"])

    def fit(self, df):
        """Fit model with clean data and proper error handling."""
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for KeltnerChannel_Agent")
            
        X = d[["pctB", "width", "pctB_slope"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        
        # Scale features to handle outliers
        X_scaled = self.scaler.fit_transform(X)
        
        # Check if data is valid before fitting
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            # Clean any remaining problematic values
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
            
        self.model.fit(X_scaled, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        """Make prediction with proper data validation."""
        if not self.fitted:
            self.fit(historical_df)
            
        last = self._feat(historical_df).iloc[-1:][["pctB", "width", "pctB_slope"]]
        
        # Handle case where last row might contain NaN/inf
        if last.isnull().any().any() or np.isinf(last.values).any():
            # If invalid data, return neutral prediction
            return 0.0
            
        # Scale features
        last_scaled = self.scaler.transform(last)
        
        return 2 * self.model.predict_proba(last_scaled)[0, 1] - 1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Keltner Channels.
        
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
            # First ensure the model is fitted with the historical data
            if not self.fitted:
                self.fit(historical_df)
            
            # Get current price for prediction
            current_price = historical_df['close'].iloc[-1]
            
            # Generate signal using predict method
            signal = self.predict(current_price=current_price, historical_df=historical_df)
            
            # Ensure signal is properly formatted to 4 decimal places
            return float(round(signal, 4))
            
        except ValueError as e:
            # Handle the case where there's not enough data
            print(f"Warning: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            print(f"Error in KeltnerChannel strategy: {str(e)}")
            return 0.0000
