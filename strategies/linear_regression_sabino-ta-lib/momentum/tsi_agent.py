"""
TSI Agent
=========

True Strength Index (long 25, short 13).

TSI = 100 × EMA( EMA(delta, s) , l ) / EMA( EMA(|delta|, s), l )

Features
--------
* tsi
* tsi_slope
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _ema(s, n): return s.ewm(span=n, adjust=False).mean()


def _tsi(close, long=25, short=13):
    delta = close.diff()
    num = _ema(_ema(delta, short), long)
    den = _ema(_ema(delta.abs(), short), long)
    return 100 * num / den


class TSI_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        tsi=_tsi(df["close"])
        d=df.copy(); d["tsi"]=tsi/100.0; d["tsi_slope"]=d["tsi"].diff()
        return d.dropna(subset=["tsi","tsi_slope"])
    def fit(self, df):
        d=self._feat(df)
        if len(d)<60: raise ValueError("Not enough rows for TSI_Agent")
        X=d[["tsi","tsi_slope"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["tsi","tsi_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using True Strength Index.
        
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
            print(f"Error in TSI strategy: {str(e)}")
            return 0.0000
