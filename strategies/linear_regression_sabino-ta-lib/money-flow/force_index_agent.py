"""
Force Index Agent
=================

Elder's Force Index:

    FIₜ = volumeₜ × (closeₜ − closeₜ₋₁)

We use a 13-period EMA-smoothed force index.

Features
--------
* fi_ema
* fi_slope
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


class ForceIndex_Agent:
    def __init__(self, span: int = 13):
        self.span = span
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        fi_raw = df["volume"] * (df["close"].diff())
        fi_ema = _ema(fi_raw, self.span)
        d = df.copy()
        d["fi"] = fi_ema
        d["fi_slope"] = fi_ema.diff()
        return d.dropna(subset=["fi","fi_slope"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.span+10: raise ValueError("Not enough rows for ForceIndex_Agent")
        X=d[["fi","fi_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["fi","fi_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Force Index.
        
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
            print(f"Error in Force Index strategy: {str(e)}")
            return 0.0000
