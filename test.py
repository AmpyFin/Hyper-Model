#!/usr/bin/env python3
"""
test.py
Import **all** strategies dynamically and score them on recent MSFT data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project-root importable (assumes test.py lives in project root)
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd

from utils.get_historical_data import pull_and_save
from utils.get_price_data import fetch_latest
from strategies import discover   # <-- new helper

TICKER = "REGN"
START  = "2025-03-01"
END    = "2025-05-01"
FREQ   = "1min"


def main() -> None:
    # ------------------------------------------------------------------ #
    # pull data
    print(f"Fetching {FREQ} history for {TICKER} …")
    hist = pull_and_save(TICKER, FREQ, START, END)
    
    # Debug: Check the shape and content of the historical data
    print(f"Historical data shape: {hist.shape if isinstance(hist, pd.DataFrame) else 'Not a DataFrame'}")
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        print(f"Columns: {hist.columns.tolist()}")
        print(f"First few rows:")
        print(hist.head(3))
        print(f"Last few rows:")
        print(hist.tail(3))
    else:
        print("Warning: Historical data is empty or not a DataFrame")
    
    current_price = hist["close"].iloc[-1] if isinstance(hist, pd.DataFrame) and not hist.empty else None
    if current_price is None:
        print("Error: No historical data available. Cannot proceed.")
        return
        
    try:
        fetch_latest([TICKER])                  # prints but we ignore
    except Exception as e:
        print(f"Warning: Could not fetch latest price due to: {e}")
        pass  # network hiccup → keep fallback

    # ------------------------------------------------------------------ #
    # discover & run every strategy
    strategies = discover()
    print(f"Loaded {len(strategies)} strategy classes: {list(strategies)}")

    results = []
    for name, StratCls in strategies.items():
        print(f"Processing strategy: {name}...")
        try:
            agent = StratCls()
            
            # Debug for AtrStochAgent
            if name == "AtrStochAgent":
                print("Debugging AtrStochAgent:")
                # Add features and check their shape before fitting
                features_df = agent._add_features(hist)
                print(f"Features shape after preprocessing: {features_df.shape}")
                print(f"Features columns: {features_df.columns.tolist()}")
                if features_df.empty:
                    print("Warning: Features dataframe is empty after preprocessing")
                    continue
            
            agent.fit(hist)
            score = agent.predict(current_price=current_price,
                                historical_df=hist)
            results.append((name, score))
            print(f"Successfully processed {name} with score: {score:.4f}")
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # ------------------------------------------------------------------ #
    # pretty-print
    if results:
        results.sort(key=lambda x: -x[1])           # descending by score
        print("\nStrategy scores (-1 … +1):")
        for name, score in results:
            if score > 0.5:
                mark = "STRONG BUY"
            elif score <= 0.5 and score >= 0.20:
                mark = "BUY"
            elif score <= 0.20 and score >= -0.20:
                mark = "HOLD"
            elif score < -0.20 and score >= -0.5:
                mark = "SELL"
            else:
                mark = "STRONG SELL"
            print(f"{name:<25} {score:+.3f}   → {mark}")
    else:
        print("No strategy results available.")


if __name__ == "__main__":
    main()
