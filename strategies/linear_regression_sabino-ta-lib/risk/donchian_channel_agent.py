"""
Donchian Channel Agent
======================

20-period high/low channel (classic turtle logic).

Bands
-----
upper = rolling max(high, n)  
lower = rolling min(low,  n)  
middle= (upper + lower) / 2

Features
--------
* **pctB**    = (close − lower) / (upper − lower)
* **width**   = (upper − lower) / middle
* **pctB_slope**
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


class DonchianChannel_Agent:
    def __init__(self, length: int = 20):
        self.n = length
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    def _feat(self, df):
        upper = df["high"].rolling(self.n).max()
        lower = df["low"].rolling(self.n).min()
        middle = (upper + lower) / 2

        pctB = (df["close"] - lower) / (upper - lower)
        width = (upper - lower) / middle

        d = df.copy()
        d["pctB"] = pctB
        d["width"] = width
        d["pctB_slope"] = pctB.diff()
        return d.dropna(subset=["pctB", "width", "pctB_slope"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for DonchianChannel_Agent")
        X = d[["pctB", "width", "pctB_slope"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.model.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["pctB", "width", "pctB_slope"]]
        return 2 * self.model.predict_proba(last)[0, 1] - 1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Donchian Channel.
        
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
            print(f"Error in Donchian Channel strategy: {str(e)}")
            return 0.0000
