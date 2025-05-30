"""
Vortex Indicator Agent
======================

Default window **14**.

VI⁺ = Σ|highₜ − lowₜ₋₁|  / ATRₙ  
VI⁻ = Σ|lowₜ − highₜ₋₁| / ATRₙ

Features
--------
* vi_pos
* vi_neg
* vi_diff   (vi_pos − vi_neg)
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _atr(df, n):
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).sum()


def _vortex(df, n=14):
    hi, lo = df["high"], df["low"]
    vi_pos = (hi - lo.shift()).abs().rolling(n).sum()
    vi_neg = (lo - hi.shift()).abs().rolling(n).sum()
    atrn   = _atr(df, n)
    return vi_pos / atrn, vi_neg / atrn


class Vortex_Agent:
    def __init__(self, period: int = 14):
        self.n = period
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        vpos, vneg = _vortex(df, self.n)
        d = df.copy()
        d["vi_pos"], d["vi_neg"] = vpos, vneg
        d["vi_diff"] = vpos - vneg
        
        # Replace infinite values with NaN
        d["vi_pos"] = d["vi_pos"].replace([np.inf, -np.inf], np.nan)
        d["vi_neg"] = d["vi_neg"].replace([np.inf, -np.inf], np.nan)
        d["vi_diff"] = d["vi_diff"].replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN or infinite values in any of the feature columns
        return d.dropna(subset=["vi_pos", "vi_neg", "vi_diff"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for Vortex_Agent")
        X=d[["vi_pos","vi_neg","vi_diff"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["vi_pos","vi_neg","vi_diff"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Vortex Indicator.
        
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
            print(f"Error in Vortex strategy: {str(e)}")
            return 0.0000
