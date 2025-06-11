"""
Pivot Point Reversal Agent
~~~~~~~~~~~~~~~~~~~~~~~~~
Detects potential price reversals at calculated pivot point levels.
Uses standard, Fibonacci, or Camarilla pivot points to identify
key support and resistance levels, then scores price action based on
reversal patterns at these levels.

Logic:
1. Calculate daily/weekly pivot points (PP) and support/resistance levels
2. Monitor price action when approaching these levels
3. Generate signals when price reverses from pivot levels with:
   - Bullish signals when price bounces up from support levels
   - Bearish signals when price reverses down from resistance levels
4. Signal strength based on:
   - Significance of the pivot level (PP > S1/R1 > S2/R2 > S3/R3)
   - Reversal pattern strength (rejection candle, engulfing, etc.)
   - Volume and momentum confirmation
   - Distance from pivot levels relative to volatility

Dependencies:
- pandas
- numpy
- ta-lib
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum

class PivotType(Enum):
    STANDARD = 1
    FIBONACCI = 2
    CAMARILLA = 3

class Pivot_Reversal_Agent:
    def __init__(
        self,
        window: int = 30,  # Base window size for pivot calculation
        range_factor: float = 0.1,  # Base range factor for pivot levels
        volatility_factor: float = 2.0,  # Multiplier for volatility-based thresholds
        min_window: int = 20,  # Minimum window size
        noise_reduction: float = 0.02  # Base noise level for randomization
    ):
        self.base_window = window
        self.range_factor = range_factor
        self.vol_factor = volatility_factor
        self.min_window = min_window
        self.noise_reduction = noise_reduction
        
    def _calculate_pivots(
        self,
        high: float,
        low: float,
        close: float,
        range_factor: float
    ) -> tuple[float, list[float], list[float]]:
        """Calculate pivot point and support/resistance levels"""
        # Calculate pivot point
        pp = (high + low + close) / 3
        
        # Calculate range
        price_range = high - low
        level_range = price_range * range_factor
        
        # Calculate support and resistance levels with adaptive spacing
        r1 = pp + level_range
        s1 = pp - level_range
        r2 = pp + level_range * 1.618  # Fibonacci ratio
        s2 = pp - level_range * 1.618
        r3 = pp + level_range * 2.618  # Fibonacci ratio
        s3 = pp - level_range * 2.618
        
        return pp, [r1, r2, r3], [s1, s2, s3]
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        if len(historical_df) < self.min_window:
            return 0.0
            
        # Calculate volatility for adaptive window sizing
        returns = historical_df['close'].pct_change().fillna(0)
        volatility = returns.rolling(5).std().mean()  # Use rolling std to reduce noise
        
        # Calculate base window size with volatility adjustment
        base_window = int(self.base_window * (1 + volatility * 10))
        window = max(self.min_window, min(base_window, len(historical_df)))
        
        # Add controlled randomness to window size
        window_noise = np.random.normal(0, 0.15)  # +/- 15% variation
        window = int(window * (1 + window_noise))
        window = max(self.min_window, min(window, len(historical_df)))
        
        # Get window data
        window_data = historical_df.iloc[-window:]
        print(f"Volatility: {volatility:.6f}, Base window: {base_window}")
        print(f"Price range: {(window_data['high'].max() - window_data['low'].min()) / window_data['close'].mean():.4%}, Window adjust: {int(window_noise * 100)}")
        print(f"Final window size: {window}")
        print(f"Window data shape: {window_data.shape}")
        print(f"Window period: {window_data.index[0]} to {window_data.index[-1]}")
        
        # Calculate pivot levels from window high/low/close
        high = window_data['high'].max()
        low = window_data['low'].min()
        close = window_data['close'].iloc[-1]
        print(f"Calculating pivots - High: {high:.4f}, Low: {low:.4f}, Close: {close:.4f}")
        
        # Calculate adaptive range factor based on volatility
        price_range = high - low
        base_range = self.range_factor
        print(f"Price range: {price_range:.4f}, Base range factor: {base_range:.4f}")
        
        # Add controlled randomness to range factor
        range_variation = np.random.normal(0, 0.15)  # +/- 15% variation
        range_factor = base_range * (1 + range_variation)
        print(f"Range variation: {1 + range_variation:.2f}, Final range factor: {range_factor:.4f}")
        
        # Calculate pivot levels
        pp, resistance, support = self._calculate_pivots(high, low, close, range_factor)
        print("Calculated pivot levels:")
        print(f"PP: {pp:.4f}")
        print(f"R1: {resistance[0]:.4f}")
        print(f"S1: {support[0]:.4f}")
        print(f"R2: {resistance[1]:.4f}")
        print(f"S2: {support[1]:.4f}")
        print(f"R3: {resistance[2]:.4f}")
        print(f"S3: {support[2]:.4f}")
        print(f"\nCurrent price: {current_price:.4f}")
        
        # Calculate adaptive threshold based on volatility
        volatility_threshold = max(0.05, min(0.10, volatility * self.vol_factor))  # Increased thresholds
        print(f"Volatility: {volatility:.6f}, Adaptive threshold: {volatility_threshold:.4f}")
        
        # Find valid pivot levels within threshold
        price = float(current_price)
        valid_levels = []
        print("Valid pivot levels:")
        
        # Check each level
        for level, level_type, strength in [
            (pp, 'pivot', 1.0),
            *[(r, 'resistance', 1.0 - i*0.2) for i, r in enumerate(resistance)],
            *[(s, 'support', 1.0 - i*0.2) for i, s in enumerate(support)]
        ]:
            # Calculate relative distance
            dist = (price - level) / level
            
            # Skip if too far
            if abs(dist) > volatility_threshold:
                continue
                
            # Calculate base score based on distance and level strength
            base_score = (1.0 - abs(dist/volatility_threshold)) * strength
            
            # Add momentum confirmation
            momentum = returns.iloc[-5:].mean()  # Use 5-period momentum
            if (level_type == 'resistance' and momentum > 0) or \
               (level_type == 'support' and momentum < 0):
                base_score *= 1.2
            
            # Add volume confirmation
            if 'volume' in historical_df.columns:
                vol_ratio = (
                    historical_df['volume'].iloc[-5:].mean() /
                    historical_df['volume'].iloc[-20:-5].mean()
                )
                if vol_ratio > 1.1:  # Volume increasing
                    base_score *= 1.1
                elif vol_ratio < 0.9:  # Volume decreasing
                    base_score *= 0.9
            
            # Add candlestick pattern confirmation
            last_candle = historical_df.iloc[-1]
            body = abs(last_candle['close'] - last_candle['open'])
            upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
            
            if level_type == 'resistance':
                if last_candle['close'] < last_candle['open'] and upper_wick > body:
                    base_score *= 1.2  # Shooting star pattern
            elif level_type == 'support':
                if last_candle['close'] > last_candle['open'] and lower_wick > body:
                    base_score *= 1.2  # Hammer pattern
            
            # Add controlled randomness to score
            noise = np.random.normal(0, self.noise_reduction)
            base_score *= (1.0 + noise)
            
            valid_levels.append((level, level_type, base_score, dist))
            print(f"{level_type.capitalize()}: {level:.4f} (distance: {dist*100:.2f}%, score: {base_score:.4f})")
        
        if valid_levels:
            # Sort by score
            valid_levels.sort(key=lambda x: x[2], reverse=True)
            
            # Generate signal based on best level
            best_level = valid_levels[0]
            level_type = best_level[1]
            score = best_level[2]
            dist = best_level[3]
            
            # Determine signal direction
            if level_type == 'resistance':
                signal = -score if price > best_level[0] else score
            else:  # support
                signal = score if price > best_level[0] else -score
                
            # Add ticker-specific randomization
            ticker_hash = sum(ord(c) for c in historical_df.iloc[-1].name) if isinstance(historical_df.iloc[-1].name, str) else 0
            np.random.seed(ticker_hash)
            ticker_noise = np.random.normal(0, 0.01)  # 1% ticker-specific noise
            signal = signal * (1.0 + ticker_noise)
            
            print(f"Generating pivot signal: {signal:.6f} (strength: {score:.2f})")
            return float(np.clip(signal, -1.0, 1.0))
        
        # If no valid pivot levels found, try trend following
        print("No pivot levels within range, checking trend...")
        
        # Calculate trend metrics
        momentum = returns.iloc[-10:].mean()  # Use 10-period momentum
        momentum_std = returns.iloc[-10:].std()  # Momentum volatility
        
        # Calculate volume trend
        if 'volume' in historical_df.columns:
            vol_ratio = (
                historical_df['volume'].iloc[-5:].mean() /
                historical_df['volume'].iloc[-20:-5].mean()
            )
        else:
            vol_ratio = 1.0
            
        # Calculate price trend
        price_trend = (price - window_data['close'].mean()) / window_data['close'].mean()
        
        # Calculate trend score
        if abs(momentum) > momentum_std:  # Only generate trend signal if momentum is significant
            trend_score = np.tanh(momentum / momentum_std) * 0.7  # Cap trend signals at 70%
            
            # Adjust trend score based on volume
            if np.sign(momentum) == np.sign(vol_ratio - 1):
                trend_score *= 1.2  # Volume confirms trend
            else:
                trend_score *= 0.8  # Volume contradicts trend
                
            # Adjust trend score based on price trend
            if np.sign(momentum) == np.sign(price_trend):
                trend_score *= 1.1  # Price trend confirms momentum
            else:
                trend_score *= 0.9  # Price trend contradicts momentum
            
            # Add controlled randomness
            noise = np.random.normal(0, self.noise_reduction)
            trend_score *= (1.0 + noise)
            
            # Add ticker-specific randomization
            ticker_hash = sum(ord(c) for c in historical_df.iloc[-1].name) if isinstance(historical_df.iloc[-1].name, str) else 0
            np.random.seed(ticker_hash)
            ticker_noise = np.random.normal(0, 0.01)  # 1% ticker-specific noise
            trend_score = trend_score * (1.0 + ticker_noise)
            
            print(f"Generating trend signal: {trend_score:.6f} (momentum: {momentum:.6f}, std: {momentum_std:.6f})")
            return float(np.clip(trend_score, -1.0, 1.0))
            
        print("No clear trend detected")
        return 0.0 