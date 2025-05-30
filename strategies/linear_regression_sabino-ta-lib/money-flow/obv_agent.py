"""
OBV Agent
=========

On-Balance Volume:

    OBVₜ = OBVₜ₋₁ ± volumeₜ
           (+ when close rises, − when falls)

Features
--------
* obv_norm   (OBV / cumulative volume)
* obv_slope
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    return (df["volume"] * direction).cumsum()


class OBV_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        obv = _obv(df)
        d = df.copy()
        d["obv_norm"] = obv / df["volume"].cumsum().replace(0, pd.NA)
        d["obv_slope"] = d["obv_norm"].diff()
        return d.dropna(subset=["obv_norm","obv_slope"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for OBV_Agent")
        X=d[["obv_norm","obv_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["obv_norm","obv_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using On-Balance Volume.
        
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
            print(f"Error in OBV strategy: {str(e)}")
            return 0.0000
