"""
CCI Agent
=========

Commodity Channel Index (length **20**, constant 0.015).

CCI = (TypicalPrice − SMA(TP)) / (0.015 × mean deviation)

Features
--------
* cci_scaled   (CCI / 200  → approx −5…+5 → scaled)
* cci_slope
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _cci(df, n=20, c=0.015):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(n).mean()
    md  = (tp - sma).abs().rolling(n).mean()
    return (tp - sma) / (c * md)


class CCI_Agent:
    def __init__(self, length: int = 20):
        self.n = length
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        cci = _cci(df, self.n)
        d = df.copy()
        d["cci_scaled"] = cci / 200.0
        d["cci_slope"] = d["cci_scaled"].diff()
        return d.dropna(subset=["cci_scaled","cci_slope"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for CCI_Agent")
        X=d[["cci_scaled","cci_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["cci_scaled","cci_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Commodity Channel Index.
        
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
            print(f"Error in CCI strategy: {str(e)}")
            return 0.0000
