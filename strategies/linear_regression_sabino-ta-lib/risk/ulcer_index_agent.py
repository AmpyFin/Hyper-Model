"""
Ulcer Index Agent
=================

Ulcer Index (UI) – measures downside volatility.

Computation
-----------
Drawdownₜ = (closeₜ − max(close over N)) / max(close over N) × 100  
UIₜ       = √( mean( drawdown² ) over N )

Default window **14**.

Features
--------
* ui        (scaled 0-1 by /100)
* ui_slope
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _ulcer_index(series: pd.Series, n: int = 14):
    rolling_max = series.rolling(n).max()
    drawdown = (series - rolling_max) / rolling_max * 100
    ui = np.sqrt((drawdown ** 2).rolling(n).mean())
    return ui


class UlcerIndex_Agent:
    def __init__(self, period: int = 14):
        self.n = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    def _feat(self, df):
        ui = _ulcer_index(df["close"], self.n)
        d = df.copy()
        d["ui"] = ui / 100.0          # scale
        d["ui_slope"] = d["ui"].diff()
        return d.dropna(subset=["ui", "ui_slope"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for UlcerIndex_Agent")
        X = d[["ui", "ui_slope"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.model.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["ui", "ui_slope"]]
        return 2 * self.model.predict_proba(last)[0, 1] - 1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Ulcer Index.
        
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
            print(f"Error in Ulcer Index strategy: {str(e)}")
            return 0.0000
