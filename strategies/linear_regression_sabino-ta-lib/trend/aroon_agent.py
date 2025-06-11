"""
Aroon Agent
===========

Default window **25**.

AroonUp   = 100 × (n − periods since max(high)) / n  
AroonDown = 100 × (n − periods since min(low )) / n

Features
--------
* up
* down
* diff (up − down)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


def _aroon(df, n=25):
    high, low = df["high"], df["low"]
    roll_high = high.rolling(n)
    roll_low = low.rolling(n)

    up = 100 * (n - roll_high.apply(lambda x: n - 1 - np.argmax(x.values), raw=False)) / n
    dn = 100 * (n - roll_low.apply(lambda x: n - 1 - np.argmin(x.values), raw=False)) / n
    return up, dn


class Aroon_Agent:
    def __init__(self, length=25):
        self.n = length
        self.m=LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        up, dn = _aroon(df, self.n)
        d = df.copy()
        d["up"], d["down"] = up / 100.0, dn / 100.0
        d["diff"] = d["up"] - d["down"]
        return d.dropna(subset=["up","down","diff"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for Aroon_Agent")
        X=d[["up","down","diff"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["up","down","diff"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Aroon Indicator.
        
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
            print(f"Error in Aroon strategy: {str(e)}")
            return 0.0000
