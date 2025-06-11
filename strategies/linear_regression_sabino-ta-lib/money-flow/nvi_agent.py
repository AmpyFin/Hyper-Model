"""
NVI Agent
=========

Negative Volume Index (NVI) – starts at 1000.

If volumeₜ < volumeₜ₋₁:
    NVIₜ = NVIₜ₋₁ × (1 + Δclose%),
else:
    NVIₜ = NVIₜ₋₁.

Features
--------
* nvi_norm   (NVI / 1000 − 1)          → centred around 0
* nvi_slope  (first diff)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _nvi(df: pd.DataFrame, start: float = 1000.0):
    nvi = [start]
    for i in range(1, len(df)):
        if df["volume"].iat[i] < df["volume"].iat[i - 1]:
            change = df["close"].iat[i] / df["close"].iat[i - 1] - 1
            nvi.append(nvi[-1] * (1 + change))
        else:
            nvi.append(nvi[-1])
    return pd.Series(nvi, index=df.index)


class NVI_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        nv = _nvi(df)
        d = df.copy()
        d["nvi_norm"] = nv / 1000.0 - 1
        d["nvi_slope"] = d["nvi_norm"].diff()
        return d.dropna(subset=["nvi_norm", "nvi_slope"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<60: raise ValueError("Not enough rows for NVI_Agent")
        X=d[["nvi_norm","nvi_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["nvi_norm","nvi_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Negative Volume Index.
        
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
            print(f"Error in NVI strategy: {str(e)}")
            return 0.0000
