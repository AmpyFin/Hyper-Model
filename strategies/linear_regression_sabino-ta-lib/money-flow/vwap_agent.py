"""
VWAP Agent
==========

30-period rolling Volume-Weighted Average Price.

    VWAPₜ = Σ( close × volume ) / Σ volume  (over N)

Features
--------
* vwap_diff   (close − vwap) / vwap
* vwap_slope  (first diff of vwap_diff)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _rolling_vwap(df: pd.DataFrame, n: int = 30):
    pv = (df["close"] * df["volume"]).rolling(n).sum()
    vol = df["volume"].rolling(n).sum().replace(0, pd.NA)
    return pv / vol


class VWAP_Agent:
    def __init__(self, period: int = 30):
        self.n = period
        self.m=LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        vwap = _rolling_vwap(df, self.n)
        d = df.copy()
        d["vwap_diff"] = (df["close"] - vwap) / vwap
        d["vwap_slope"] = d["vwap_diff"].diff()
        return d.dropna(subset=["vwap_diff","vwap_slope"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for VWAP_Agent")
        X=d[["vwap_diff","vwap_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["vwap_diff","vwap_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Volume Weighted Average Price.
        
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
            print(f"Error in VWAP strategy: {str(e)}")
            return 0.0000
