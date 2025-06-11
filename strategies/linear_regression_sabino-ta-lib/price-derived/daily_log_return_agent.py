"""
Daily Log-Return Agent
======================

    lrₜ = ln(closeₜ / closeₜ₋₁)

Features
--------
* lr
* lr_zscore   (20-bar z-score)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


class DailyLogReturn_Agent:
    def __init__(self, z_len: int = 20):
        self.z = z_len
        self.m = LogisticRegression(max_iter=1000); self.fitted = False

    def _feat(self, df):
        lr = np.log(df["close"] / df["close"].shift())
        lr_z = (lr - lr.rolling(self.z).mean()) / lr.rolling(self.z).std()
        d = df.copy()
        d["lr"], d["lr_z"] = lr, lr_z
        return d.dropna(subset=["lr", "lr_z"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.z + 10:
            raise ValueError("Not enough rows for DailyLogReturn_Agent")
        X, y = d[["lr", "lr_z"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["lr", "lr_z"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Daily Log Return.
        
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
            print(f"Error in Daily Log Return strategy: {str(e)}")
            return 0.0000
