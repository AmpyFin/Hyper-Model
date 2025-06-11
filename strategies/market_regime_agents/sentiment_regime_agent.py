"""
Sentiment Regime Agent
~~~~~~~~~~~~~~~~~
Analyzes price action patterns to estimate the overall market sentiment regime.
Uses technical signals as a proxy for market participant sentiment without requiring
external sentiment data.

Logic:
1. Analyze price action characteristics that indicate sentiment shifts
2. Evaluate multiple timeframes for sentiment alignment
3. Detect sentiment extremes (greed/fear) through volatility and momentum
4. Monitor gaps, reversals, and intraday ranges for sentiment clues
5. Calculate cumulative sentiment score based on behavioral finance principles

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* Values near +1: Strong bullish sentiment (greed/euphoria)
* Values near 0: Neutral/mixed sentiment
* Values near -1: Strong bearish sentiment (fear/capitulation)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


class SentimentRegimeAgent:
    def __init__(
        self,
        short_period: int = 5,        # Short-term sentiment window
        medium_period: int = 14,      # Medium-term sentiment window
        long_period: int = 30,        # Long-term sentiment window
        rsi_period: int = 14,         # RSI period for overbought/oversold
        volatility_period: int = 20,  # Volatility analysis period
        momentum_period: int = 10,    # Momentum analysis period
        overbought_threshold: float = 70.0,  # RSI threshold for overbought
        oversold_threshold: float = 30.0     # RSI threshold for oversold
    ):
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.rsi_period = rsi_period
        self.volatility_period = volatility_period
        self.momentum_period = momentum_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.latest_signal = 0.0
        self.sentiment_metrics = {}
        
    def _calculate_sentiment_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate various sentiment metrics from price action"""
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        result = {}
        
        # Calculate RSI
        if len(df_copy) >= self.rsi_period + 1:
            # Get price changes
            delta = df_copy['close'].diff()
            
            # Separate gains and losses
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss  # Convert to positive values
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df_copy['rsi'] = 100 - (100 / (1 + rs))
            
            # Get latest RSI
            result['rsi'] = df_copy['rsi'].iloc[-1]
            result['rsi_trend'] = df_copy['rsi'].iloc[-self.short_period] - df_copy['rsi'].iloc[-self.short_period]
            
            # RSI-based sentiment
            # Scale RSI from [0, 100] to [-1, 1]
            result['rsi_sentiment'] = (df_copy['rsi'].iloc[-1] - 50) / 50
            
            # Check for overbought/oversold conditions
            result['overbought'] = df_copy['rsi'].iloc[-1] > self.overbought_threshold
            result['oversold'] = df_copy['rsi'].iloc[-1] < self.oversold_threshold
        
        # Calculate price momentum
        periods = [self.short_period, self.medium_period, self.long_period]
        for period in periods:
            if len(df_copy) >= period:
                # Calculate rate of change
                df_copy[f'roc_{period}'] = df_copy['close'].pct_change(periods=period)
                
                # Get latest value
                result[f'roc_{period}'] = df_copy[f'roc_{period}'].iloc[-1]
                
                # Convert to sentiment score [-1, 1]
                # Cap at ±30% to prevent extreme readings
                max_move = 0.3  # 30% move
                result[f'momentum_sentiment_{period}'] = np.clip(
                    result[f'roc_{period}'] / max_move, -1.0, 1.0)
        
        # Volume analysis
        if len(df_copy) >= self.medium_period and 'volume' in df_copy.columns:
            # Volume trend
            df_copy['vol_ma'] = df_copy['volume'].rolling(window=self.medium_period).mean()
            result['volume_ratio'] = df_copy['volume'].iloc[-1] / df_copy['vol_ma'].iloc[-1]
            
            # Up/down volume
            df_copy['up_day'] = df_copy['close'] > df_copy['close'].shift(1)
            up_vol = df_copy.loc[df_copy['up_day'], 'volume'].iloc[-self.medium_period:].mean()
            down_vol = df_copy.loc[~df_copy['up_day'], 'volume'].iloc[-self.medium_period:].mean()
            
            if down_vol > 0:
                result['up_down_vol_ratio'] = up_vol / down_vol
                
                # Convert to sentiment [-1, 1]
                # 1.0 = equal up/down volume
                # >1.0 = more up volume (bullish)
                # <1.0 = more down volume (bearish)
                vol_sentiment = (result['up_down_vol_ratio'] - 1.0)
                result['volume_sentiment'] = np.clip(vol_sentiment, -1.0, 1.0)
        
        # Volatility analysis
        if len(df_copy) >= self.volatility_period:
            # Calculate historical volatility (annualized standard deviation of returns)
            df_copy['returns'] = df_copy['close'].pct_change()
            df_copy['hv'] = df_copy['returns'].rolling(window=self.volatility_period).std() * np.sqrt(252)
            
            # Get latest volatility
            result['volatility'] = df_copy['hv'].iloc[-1]
            
            # Volatility trend
            if len(df_copy) >= self.volatility_period * 2:
                prior_vol = df_copy['hv'].iloc[-self.volatility_period*2:-self.volatility_period].mean()
                recent_vol = df_copy['hv'].iloc[-self.volatility_period:].mean()
                
                if prior_vol > 0:
                    result['volatility_change'] = (recent_vol - prior_vol) / prior_vol
                    
                    # Rising volatility in down market = fear (negative sentiment)
                    # Rising volatility in up market = greed/FOMO (positive sentiment)
                    if 'roc_medium_period' in result:
                        price_direction = np.sign(result['roc_medium_period'])
                        vol_intensity = min(abs(result['volatility_change']), 1.0)
                        result['volatility_sentiment'] = price_direction * vol_intensity
        
        # Gap analysis
        if len(df_copy) >= self.short_period:
            # Calculate overnight gaps
            df_copy['gap'] = (df_copy['open'] - df_copy['close'].shift(1)) / df_copy['close'].shift(1)
            
            # Recent gaps
            recent_gaps = df_copy['gap'].iloc[-self.short_period:]
            result['avg_gap'] = recent_gaps.mean()
            result['gap_sentiment'] = np.clip(result['avg_gap'] * 20, -1.0, 1.0)  # Scale for sentiment
            
            # Count positive/negative gaps
            pos_gaps = (recent_gaps > 0.002).sum()  # 0.2% threshold for significant gap
            neg_gaps = (recent_gaps < -0.002).sum()
            
            if pos_gaps + neg_gaps > 0:
                result['gap_direction'] = (pos_gaps - neg_gaps) / (pos_gaps + neg_gaps)
        
        # Price action characteristics
        if len(df_copy) >= self.medium_period:
            # Calculate candle body ratio (close-open) / (high-low)
            df_copy['body_ratio'] = abs(df_copy['close'] - df_copy['open']) / (df_copy['high'] - df_copy['low'])
            
            # Average recent body ratio
            result['avg_body_ratio'] = df_copy['body_ratio'].iloc[-self.medium_period:].mean()
            
            # Calculate upper and lower shadows
            df_copy['upper_shadow'] = (df_copy['high'] - df_copy[['open', 'close']].max(axis=1)) / df_copy['close']
            df_copy['lower_shadow'] = (df_copy[['open', 'close']].min(axis=1) - df_copy['low']) / df_copy['close']
            
            # Average recent shadows
            result['avg_upper_shadow'] = df_copy['upper_shadow'].iloc[-self.medium_period:].mean()
            result['avg_lower_shadow'] = df_copy['lower_shadow'].iloc[-self.medium_period:].mean()
            
            # Shadow ratio sentiment
            # More upper shadow = selling pressure (bearish)
            # More lower shadow = buying pressure (bullish)
            if result['avg_lower_shadow'] + result['avg_upper_shadow'] > 0:
                shadow_ratio = (result['avg_lower_shadow'] - result['avg_upper_shadow']) / (
                    result['avg_lower_shadow'] + result['avg_upper_shadow'])
                result['shadow_sentiment'] = shadow_ratio
        
        # Consecutive move analysis
        if len(df_copy) >= self.medium_period:
            # Calculate consecutive up/down days
            df_copy['up_day'] = df_copy['close'] > df_copy['close'].shift(1)
            
            # Count current streak
            current_direction = df_copy['up_day'].iloc[-1]
            streak = 1
            for i in range(2, min(self.medium_period, len(df_copy))):
                if df_copy['up_day'].iloc[-i] == current_direction:
                    streak += 1
                else:
                    break
                    
            # Convert to sentiment
            direction = 1 if current_direction else -1
            # Scale using log to prevent extreme values for long streaks
            streak_intensity = min(np.log(streak + 1) / np.log(10), 1.0)
            result['streak_sentiment'] = direction * streak_intensity
            
        return result
    
    def _evaluate_sentiment_regime(self, metrics: Dict) -> float:
        """
        Evaluate the sentiment regime based on calculated metrics
        Returns a sentiment score from -1 (extreme fear) to +1 (extreme greed)
        """
        # Default to neutral if insufficient data
        if not metrics:
            return 0.0
        
        # Collect all sentiment signals
        sentiment_signals = []
        weights = []
        
        # RSI sentiment (if available)
        if 'rsi_sentiment' in metrics:
            sentiment_signals.append(metrics['rsi_sentiment'])
            weights.append(0.15)  # 15% weight
            
        # Momentum sentiment across timeframes
        for period in ['short', 'medium', 'long']:
            period_value = getattr(self, f'{period}_period')
            if f'momentum_sentiment_{period_value}' in metrics:
                # Give higher weight to medium-term, lower to short and long
                if period == 'medium':
                    weight = 0.20  # 20% weight
                else:
                    weight = 0.10  # 10% weight each
                    
                sentiment_signals.append(metrics[f'momentum_sentiment_{period_value}'])
                weights.append(weight)
                
        # Volume sentiment
        if 'volume_sentiment' in metrics:
            sentiment_signals.append(metrics['volume_sentiment'])
            weights.append(0.10)  # 10% weight
            
        # Volatility sentiment
        if 'volatility_sentiment' in metrics:
            sentiment_signals.append(metrics['volatility_sentiment'])
            weights.append(0.15)  # 15% weight
            
        # Gap sentiment
        if 'gap_sentiment' in metrics:
            sentiment_signals.append(metrics['gap_sentiment'])
            weights.append(0.10)  # 10% weight
            
        # Shadow sentiment (price action)
        if 'shadow_sentiment' in metrics:
            sentiment_signals.append(metrics['shadow_sentiment'])
            weights.append(0.05)  # 5% weight
            
        # Streak sentiment
        if 'streak_sentiment' in metrics:
            sentiment_signals.append(metrics['streak_sentiment'])
            weights.append(0.05)  # 5% weight
            
        # Calculate weighted sentiment score
        if sentiment_signals:
            sentiment_score = sum(signal * weight for signal, weight in zip(sentiment_signals, weights)) / sum(weights)
        else:
            sentiment_score = 0.0
            
        # Ensure the score is within [-1, 1] range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        return sentiment_score
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify the current sentiment regime
        """
        # Need enough bars for calculation
        required_bars = max(self.short_period, self.medium_period, self.long_period, 
                           self.rsi_period, self.volatility_period)
        if len(historical_df) < required_bars:
            self.latest_signal = 0.0
            return
        
        # Calculate sentiment metrics
        self.sentiment_metrics = self._calculate_sentiment_metrics(historical_df)
        
        # Evaluate sentiment regime
        self.latest_signal = self._evaluate_sentiment_regime(self.sentiment_metrics)
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict the sentiment regime signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate bullish sentiment (greed/euphoria)
          * Negative values indicate bearish sentiment (fear/capitulation)
          * Values near zero indicate neutral sentiment
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Sentiment Regime Agent" 