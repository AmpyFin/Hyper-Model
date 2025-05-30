"""
DPO Agent
=========

Detrended Price Oscillator (default window **20**).

DPOₜ = closeₜ₋shift − SMA(close, n),
where shift = n/2 + 1.

Features
--------
* dpo_norm   = DPO / close  (dimensionless)
* dpo_slope
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


def _dpo(close: pd.Series, n: int = 20):
    shift = int(n / 2 + 1)
    sma = close.rolling(n).mean()
    return close.shift(shift) - sma


class DPO_Agent:
    def __init__(self, length: int = 20):
        self.n = length
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False

    def _feat(self, df):
        dpo = _dpo(df["close"], self.n)
        d = df.copy()
        d["dpo_norm"] = dpo / df["close"]
        d["dpo_slope"] = d["dpo_norm"].diff()
        return d.dropna(subset=["dpo_norm", "dpo_slope"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for DPO_Agent")
        X = d[["dpo_norm", "dpo_slope"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["dpo_norm", "dpo_slope"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Detrended Price Oscillator.
        
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
            print(f"Error in DPO strategy: {str(e)}")