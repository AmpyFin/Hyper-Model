"""
Ease-of-Movement Agent
======================

Ease of Movement (EoM, EMV)  +  14-period SMA of EoM.

EoMₜ = ((highₜ − lowₜ) / 2 − (highₜ₋₁ − lowₜ₋₁) / 2) /
       (volumeₜ / 1e6)                                          (units neutralised)

Features
--------
* eom_sma   (14-period simple MA of raw EoM)
* eom_slope (first diff of the SMA)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


def _eom(df: pd.DataFrame):
    """Calculate Ease of Movement with robust handling of edge cases."""
    # Ensure all inputs are numeric
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    
    # Calculate midpoint move
    hm = (high + low) / 2
    dist = hm.diff()
    
    # Handle division by zero and very small values
    # Scale volume to reduce extreme values
    box_ratio = (volume / 1_000_000)
    box_ratio = box_ratio.replace(0, np.nan)
    
    # Calculate EoM
    range_hl = high - low
    eom = range_hl * dist / box_ratio
    
    # Replace infinities with NaN
    return eom.replace([np.inf, -np.inf], np.nan)


class EOM_Agent:
    def __init__(self, sma_period: int = 14):
        self.n = sma_period
        self.m = LogisticRegression(max_iter=1000)
        self.scaler = RobustScaler()  # Use robust scaler for outliers
        self.fitted = False

    def _feat(self, df):
        """Extract features with comprehensive data cleaning."""
        # Calculate EoM with proper handling
        eom_raw = _eom(df)
        
        # Ensure we have finite values for the moving average
        eom_raw = eom_raw.replace([np.inf, -np.inf], np.nan)
        
        # Calculate moving average and slope
        sma = eom_raw.rolling(self.n).mean()
        slope = sma.diff()
        
        # Create DataFrame with clean features
        d = df.copy()
        d["eom_sma"] = sma
        d["eom_slope"] = slope
        
        # Drop rows with NaN or infinite values
        d = d.replace([np.inf, -np.inf], np.nan)
        return d.dropna(subset=["eom_sma", "eom_slope"])

    def fit(self, df):
        """Fit model with clean data and proper error handling."""
        d = self._feat(df)
        if len(d) < self.n + 10: 
            raise ValueError("Not enough rows for EOM_Agent")
            
        X = d[["eom_sma", "eom_slope"]][:-1]
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
            
        last = self._feat(historical_df).iloc[-1:][["eom_sma", "eom_slope"]]
        
        # Handle case where last row might contain NaN/inf
        if last.isnull().any().any() or np.isinf(last.values).any():
            # If invalid data, return neutral prediction
            return 0.0
            
        # Scale features
        last_scaled = self.scaler.transform(last)
        
        return 2 * self.m.predict_proba(last_scaled)[0, 1] - 1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Ease of Movement.
        
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
            print(f"Error in EOM strategy: {str(e)}")
            return 0.0000
