"""
Mass Index Agent
================

Mass Index (length 9 EMA of (EMA(High-Low, 9) / EMA(EMA(High-Low,9),9))).

We compute the standard **25-period** sum of the EMA ratio.

Features
--------
* mass_idx
* mass_slope
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _ema(s, n): return s.ewm(span=n, adjust=False).mean()


def _mass_index(df, ema_len=9, sum_len=25):
    rng = (df["high"] - df["low"]).abs()
    ema1 = _ema(rng, ema_len)
    ema2 = _ema(ema1, ema_len)
    ratio = ema1 / ema2
    return ratio.rolling(sum_len).sum()


class MassIndex_Agent:
    def __init__(self, ema_len=9, sum_len=25):
        self.e, self.s = ema_len, sum_len
        self.m=LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        mi = _mass_index(df, self.e, self.s)
        d = df.copy()
        d["mass"] = mi
        d["mass_slope"] = mi.diff()
        return d.dropna(subset=["mass","mass_slope"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.s+10: raise ValueError("Not enough rows for MassIndex_Agent")
        X=d[["mass","mass_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["mass","mass_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Mass Index.
        
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
            print(f"Error in Mass Index strategy: {str(e)}")
            return 0.0000
