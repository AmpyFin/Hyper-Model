# Dictionary mapping strategy names to their ideal periods (in days).
# The ideal period represents the optimal timeframe for each strategy to analyze market data.


registry = {
    # Computer Science Agents - Optimized based on algorithmic concepts and data requirements
    "BernersLeeAgent": 10,          # Semantic web analysis: 10 days for hyperlink depth traversal
    "MooreAgent": 21,               # Technology adoption cycles: 3-week periods align with innovation diffusion
    "ThompsonAgent": 14,            # Exploration/exploitation balance: 2 weeks for pattern recognition
    "HopperAgent": 20,              # Compiler optimization cycles: 20 days for lexical analysis
    "TorvaldsAgent": 21,            # Distributed systems consensus: 3-week periods for branch/merge cycles
    "LiskovAgent": 15,              # Behavioral substitution: 15 days for inheritance pattern analysis
    "LamportAgent": 14,             # Distributed consensus algorithms: 2 weeks for logical clock synchronization
    "HammingAgent": 14,             # Error correction patterns: 2-week cycles for distance calculations
    "RitchieAgent": 10,             # Systems programming: 10 days for memory management patterns
    "BackusAgent": 15,              # Grammar analysis: 15 days for functional programming patterns
    "ShannonAgent": 21,             # Information theory: 3-week periods for entropy and channel capacity
    "VonNeumannAgent": 21,          # Game theory and self-organization: 3-week Monte Carlo cycles
    "DijkstraAgent": 21,            # Graph algorithms: 3-week periods for pathfinding optimization
    
    # Mathematician Agents - Based on mathematical convergence and statistical requirements
    "BayesAgent": 42,               # Bayesian updating: 6 weeks for probability convergence
    "BernoulliAgent": 33,           # Bernoulli trials: 33 days for inequality validation with sufficient samples
    "EulerAgent": 19,               # Polynomial fitting: 19 days for Eulerian path analysis
    "FermatAgent": 8,               # Local extrema: 8 days for tangent analysis and optimization
    "FibonacciAgent": 89,           # Fibonacci retracements: 89 days (Fibonacci number) for level validation
    "FourierAgent": 128,            # FFT analysis: 128 days (power of 2) for frequency decomposition
    "GaloisAgent": 12,              # Field theory: 12 days for symmetry transformation analysis
    "GaussAgent": 42,               # Normal distribution: 6 weeks for statistical significance
    "GodelAgent": 13,               # Incompleteness theory: 13 days for paradox and consistency detection
    "HilbertAgent": 33,             # Hilbert spaces: 33 days for orthogonal projection analysis
    "KolmogorovAgent": 19,          # Complexity theory: 19 days for entropy and randomness measurement
    "LagrangeAgent": 42,            # Optimization: 6 weeks for Lagrange multiplier convergence
    "LaplaceAgent": 28,             # Transform analysis: 4 weeks for differential equation solutions
    "MarkovAgent": 84,              # Markov chains: 12 weeks for state transition equilibrium
    "NewtonAgent": 6,               # Calculus-based motion: 6 days for velocity/acceleration analysis
    "PascalAgent": 42,              # Combinatorics: 6 weeks for probability triangle convergence
    "RamanujanAgent": 19,           # Number theory: 19 days for partition function analysis
    "RiemannAgent": 13,             # Complex analysis: 13 days for zeta function critical strip
    "TuringAgent": 84,              # Computation theory: 12 weeks for pattern recognition convergence
    
    # Risk Agents - Optimized for volatility and drawdown measurement
    "KeltnerChannel_Agent": 55,     # 55-day period: 2*20 + 15 buffer for default length=20
    "UlcerIndex_Agent": 40,         # 40-day period: 2*14 + 12 buffer for default period=14  
    "DonchianChannel_Agent": 55,    # 55-day period: 2*20 + 15 buffer for default length=20
    
    # Price-Derived Agents - Based on return calculation requirements
    "CumulativeReturn_Agent": 21,   # 21-day periods optimal for monthly return analysis
    "DailyLogReturn_Agent": 55,     # 55-day period: 2*20 + 15 buffer for default z_len=20
    "DailyReturn_Agent": 55,        # 55-day period: 2*20 + 15 buffer for default z_len=20
    
    # Trend Agents - Optimized for trend identification and momentum
    "Vortex_Agent": 40,             # 40-day period: 2*14 + 12 buffer for default period=14
    "Aroon_Agent": 65,              # 65-day period: 2*25 + 15 buffer for default length=25
    "STC_Agent": 19,                # Schaff Trend Cycle: 19 days optimal for cycle analysis
    "Ichimoku_Agent": 120,          # 120-day period: 2*52 + 16 buffer for default span_b=52
    "KST_Agent": 19,                # Know Sure Thing: 19 days for momentum rate-of-change
    "DPO_Agent": 55,                # 55-day period: 2*20 + 15 buffer for default length=20
    "CCI_Agent": 55,                # 55-day period: 2*20 + 15 buffer for default length=20
    "MassIndex_Agent": 65,          # 65-day period: 2*25 + 15 buffer for default sum_len=25
    "TRIX_Agent": 45,               # 45-day period: 2*15 + 15 buffer for default length=15
    
    # Momentum Agents - Optimized for momentum and oscillator analysis
    "RSI_Agent": 40,                # 40-day period: 2*14 + 12 buffer for default length=14
    "PVO_Agent": 12,                # 12-day fast period for Percentage Volume Oscillator
    "PPO_Agent": 12,                # 12-day fast period for Percentage Price Oscillator
    "ROC_Agent": 40,                # 40-day period: 2*14 + 12 buffer for default length=14
    "AwesomeOscillator_Agent": 34,  # 34-day slow MA for Awesome Oscillator
    "WilliamsR_Agent": 40,          # 40-day period: 2*14 + 12 buffer for default length=14
    "StochOsc_Agent": 14,           # 14-day standard for Stochastic Oscillator
    "UltimateOsc_Agent": 28,        # 28-day longest period for Ultimate Oscillator (uses 7, 14, 28)
    "TSI_Agent": 25,                # 25-day double smoothing for True Strength Index
    "StochRSI_Agent": 14,           # 14-day period for Stochastic RSI
    
    # Money Flow Agents - Optimized for volume and money flow analysis
    "EOM_Agent": 40,                # 40-day period: 2*14 + 12 buffer for default sma_period=14
    "VWAP_Agent": 75,               # 75-day period: 2*30 + 15 buffer for default period=30
    "OBV_Agent": 21,                # 21-day optimal for On-Balance Volume trend analysis
    "NVI_Agent": 252,               # 252-day (1 year) for Negative Volume Index long-term analysis
    "VPT_Agent": 1,                 # Daily calculation for Volume Price Trend
    "ForceIndex_Agent": 40,         # 40-day period: 2*13 + 14 buffer for default span=13
    "CMF_Agent": 55,                # 55-day period: 2*20 + 15 buffer for default period=20
    "ADI_Agent": 1,                 # Daily calculation for Accumulation/Distribution Index
    "MFI_Agent": 40,                # 40-day period: 2*14 + 12 buffer for default period=14
    
    # Market Breadth Agents - Optimized for market-wide analysis
    "StocksAboveMaAgent": 63,       # 63 days (3 months) for moving average penetration analysis
    "ArmsIndexAgent": 10,           # 10-day moving average standard for Arms Index (TRIN)
    "McclellanOscillatorAgent": 19, # 19-day EMA for McClellan Oscillator
    "BreadthDivergenceAgent": 21,   # 21-day periods for meaningful divergence detection
    "UpDownVolumeAgent": 10,        # 10-day moving average for volume ratio smoothing
    "BreadthThrustAgent": 10,       # 10-day standard for Zweig Breadth Thrust
    "NewHighsLowsAgent": 10,        # 10-day moving average for new highs/lows ratio
    "AdvanceDeclineAgent": 10,      # 10-day moving average for advance/decline line
    
    # Market Regime Agents - Optimized for regime identification
    "MarketPhaseAgent": 63,         # 63 days (3 months) for Wyckoff phase identification
    "MarketCycleAgent": 189,        # 189 days (9 months) for full market cycle analysis
    "RangeDetectionAgent": 21,      # 21 days for range-bound market identification
    "SentimentRegimeAgent": 42,     # 42 days (6 weeks) for sentiment shift detection
    "MomentumRegimeAgent": 21,      # 21 days for momentum regime transitions
    "MeanReversionRegimeAgent": 42, # 42 days for mean reversion cycle identification
    "VolatilityRegimeAgent": 21,    # 21 days for volatility regime detection
    "TrendStrengthAgent": 21,       # 21 days for trend strength evaluation
    
    # TA-Lib Momentum Indicators - Standard periods for technical analysis
    "ADX_Agent": 14,                # 14-day standard for Average Directional Movement Index
    "ADXR_Agent": 14,               # 14-day standard for ADX Rating
    "APO_Agent": 12,                # 12-day fast period for Absolute Price Oscillator
    "AROON_Agent": 14,              # 14-day standard for Aroon indicator
    "AROONOSC_Agent": 14,           # 14-day standard for Aroon Oscillator
    "BOP_Agent": 1,                 # Single bar calculation for Balance Of Power
    "CCI_Agent": 20,                # 20-day standard for Commodity Channel Index
    "CMO_Agent": 14,                # 14-day standard for Chande Momentum Oscillator
    "DX_Agent": 14,                 # 14-day standard for Directional Movement Index
    "MACD_Agent": 12,               # 12-day fast EMA for Moving Average Convergence Divergence
    "MACDEXT_Agent": 12,            # 12-day fast EMA for MACD Extended
    "MACDFIX_Agent": 12,            # 12-day fast EMA for MACD Fix
    "MINUS_DI_Agent": 14,           # 14-day standard for Minus Directional Indicator
    "MINUS_DM_Agent": 14,           # 14-day standard for Minus Directional Movement
    "MOM_Agent": 10,                # 10-day standard for Momentum
    "PLUS_DI_Agent": 14,            # 14-day standard for Plus Directional Indicator
    "PLUS_DM_Agent": 14,            # 14-day standard for Plus Directional Movement
    "ROCP_Agent": 10,               # 10-day standard for Rate of Change Percentage
    "ROCR_Agent": 10,               # 10-day standard for Rate of Change Ratio
    "ROCR100_Agent": 10,            # 10-day standard for Rate of Change Ratio 100 scale
    "SAREXT_Agent": 1,              # Daily calculation for Parabolic SAR Extended
    "STOCH_Agent": 14,              # 14-day %K for Stochastic
    "STOCHF_Agent": 14,             # 14-day %K for Stochastic Fast
    "ULTOSC_Agent": 28,             # 28-day longest period for Ultimate Oscillator (uses 7, 14, 28)
    "WILLR_Agent": 14,              # 14-day standard for Williams %R
    
    # TA-Lib Overlap Studies - Optimized for moving average and trend analysis
    "BBANDS_Agent": 20,             # 20-day standard for Bollinger Bands
    "DEMA_Agent": 21,               # 21-day optimal for Double Exponential Moving Average
    "EMA_Agent": 21,                # 21-day optimal for Exponential Moving Average
    "HTTL_Agent": 7,                # 7-day optimal for Hilbert Transform Trendline
    "KAMA_Agent": 20,               # 20-day optimal for Kaufman Adaptive Moving Average
    "MA_Agent": 21,                 # 21-day optimal for Moving Average
    "MAMA_Agent": 1,                # Daily adaptive calculation for MESA Adaptive MA
    "MAVP_Agent": 7,                # 7-day variable period for Moving Average Variable Period
    "MIDPOINT_Agent": 14,           # 14-day standard for MidPoint calculation
    "SAR_Agent": 1,                 # Daily calculation for Parabolic SAR
    "T3_Agent": 5,                  # 5-day optimal for Triple Exponential Moving Average T3
    "TEMA_Agent": 21,               # 21-day optimal for Triple Exponential Moving Average
    "TRIMA_Agent": 21,              # 21-day optimal for Triangular Moving Average
    "WMA_Agent": 21,                # 21-day optimal for Weighted Moving Average
    
    # TA-Lib Volatility Indicators - Standard periods for volatility measurement
    "ATR_Agent": 14,                # 14-day standard for Average True Range
    "NATR_Agent": 14,               # 14-day standard for Normalized Average True Range
    "TRANGE_Agent": 1,              # Single bar calculation for True Range
    
    # TA-Lib Cycle Indicators - Optimized for Hilbert Transform analysis
    "HT_DCPERIOD_Agent": 7,         # 7-day optimal for Hilbert Transform Dominant Cycle Period
    "HT_DCPHASE_Agent": 7,          # 7-day optimal for Hilbert Transform Dominant Cycle Phase
    "HT_PHASOR_Agent": 7,           # 7-day optimal for Hilbert Transform Phasor Components
    "HT_SINE_Agent": 7,             # 7-day optimal for Hilbert Transform SineWave
    "HT_TRENDMODE_Agent": 7,        # 7-day optimal for Hilbert Transform Trend vs Cycle Mode
    
    # TA-Lib Price Transform - Single bar calculations
    "AVGPRICE_Agent": 1,            # Single bar calculation for Average Price
    "MEDPRICE_Agent": 1,            # Single bar calculation for Median Price
    "TYPPRICE_Agent": 1,            # Single bar calculation for Typical Price
    "WCLPRICE_Agent": 1,            # Single bar calculation for Weighted Close Price
    
    # TA-Lib Volume Indicators - Optimized for volume analysis
    "AD_Agent": 1,                  # Daily calculation for Chaikin A/D Line
    "ADOSC_Agent": 3,               # 3-day fast period for Chaikin A/D Oscillator
    
    # TA-Lib Statistic Functions - Optimized for statistical analysis
    "BETA_Agent": 21,               # 21-day optimal for Beta coefficient calculation
    "CORREL_Agent": 21,             # 21-day optimal for Pearson Correlation Coefficient
    "LINEARREG_Agent": 30,          # 30-day standard for Linear Regression (matches agent default)
    "LINEARREGANGLE_Agent": 14,     # 14-day standard for Linear Regression Angle
    "LINEARREGINTERCEPT_Agent": 14, # 14-day standard for Linear Regression Intercept
    "LINEARREGSLOPE_Agent": 14,     # 14-day standard for Linear Regression Slope
    "STDDEV_Agent": 20,             # 20-day optimal for Standard Deviation
    "TSF_Agent": 30,                # 30-day standard for Time Series Forecast (matches agent default)
    "VAR_Agent": 20,                # 20-day optimal for Variance
    
    # TA-Lib Pattern Recognition - Based on candlestick pattern requirements
    "CDL2CROWS_Agent": 2,           # 2-day pattern for Two Crows
    "CDL3BLACKCROWS_Agent": 3,      # 3-day pattern for Three Black Crows
    "CDL3INSIDE_Agent": 3,          # 3-day pattern for Three Inside Up/Down
    "CDL3LINESTRIKE_Agent": 4,      # 4-day pattern for Three-Line Strike
    "CDL3OUTSIDE_Agent": 3,         # 3-day pattern for Three Outside Up/Down
    "CDL3STARSINSOUTH_Agent": 3,    # 3-day pattern for Three Stars In The South
    "CDL3WHITESOLDIERS_Agent": 3,   # 3-day pattern for Three Advancing White Soldiers
    "CDLABANDONEDBABY_Agent": 3,    # 3-day pattern for Abandoned Baby
    "CDLADVANCEBLOCK_Agent": 3,     # 3-day pattern for Advance Block
    "CDLBELTHOLD_Agent": 1,         # 1-day pattern for Belt-hold
    "CDLBREAKAWAY_Agent": 5,        # 5-day pattern for Breakaway
    "CDLCLOSINGMARUBOZU_Agent": 1,  # 1-day pattern for Closing Marubozu
    "CDLCONCEALBABYSWALL_Agent": 4, # 4-day pattern for Concealing Baby Swallow
    "CDLCOUNTERATTACK_Agent": 2,    # 2-day pattern for Counterattack
    "CDLDARKCLOUDCOVER_Agent": 2,   # 2-day pattern for Dark Cloud Cover
    "CDLDOJI_Agent": 1,             # 1-day pattern for Doji
    "CDLDOJISTAR_Agent": 2,         # 2-day pattern for Doji Star
    "CDLDRAGONFLYDOJI_Agent": 1,    # 1-day pattern for Dragonfly Doji
    "CDLENGULFING_Agent": 2,        # 2-day pattern for Engulfing Pattern
    "CDLEVENINGDOJISTAR_Agent": 3,  # 3-day pattern for Evening Doji Star
    "CDLEVENINGSTAR_Agent": 3,      # 3-day pattern for Evening Star
    "CDLGAPSIDESIDEWHITE_Agent": 3, # 3-day pattern for Gap Side-by-Side White Lines
    "CDLGRAVESTONEDOJI_Agent": 1,   # 1-day pattern for Gravestone Doji
    "CDLHAMMER_Agent": 1,           # 1-day pattern for Hammer
    "CDLHANGINGMAN_Agent": 1,       # 1-day pattern for Hanging Man
    "CDLHARAMI_Agent": 2,           # 2-day pattern for Harami Pattern
    "CDLHARAMICROSS_Agent": 2,      # 2-day pattern for Harami Cross Pattern
    "CDLHIGHWAVE_Agent": 1,         # 1-day pattern for High-Wave Candle
    "CDLHIKKAKE_Agent": 3,          # 3-day pattern for Hikkake Pattern
    "CDLHIKKAKEMOD_Agent": 3,       # 3-day pattern for Modified Hikkake Pattern
    "CDLHOMINGPIGEON_Agent": 2,     # 2-day pattern for Homing Pigeon
    "CDLIDENTICAL3CROWS_Agent": 3,  # 3-day pattern for Identical Three Crows
    "CDLINNECK_Agent": 2,           # 2-day pattern for In-Neck Pattern
    "CDLINVERTEDHAMMER_Agent": 1,   # 1-day pattern for Inverted Hammer
    "CDLKICKING_Agent": 2,          # 2-day pattern for Kicking
    "CDLKICKINGBYLENGTH_Agent": 2,  # 2-day pattern for Kicking by Length
    "CDLLADDERBOTTOM_Agent": 5,     # 5-day pattern for Ladder Bottom
    "CDLLONGLEGGEDDOJI_Agent": 1,   # 1-day pattern for Long Legged Doji
    "CDLLONGLINE_Agent": 1,         # 1-day pattern for Long Line Candle
    "CDLMARUBOZU_Agent": 1,         # 1-day pattern for Marubozu
    "CDLMATCHINGLOW_Agent": 2,      # 2-day pattern for Matching Low
    "CDLMATHOLD_Agent": 5,          # 5-day pattern for Mat Hold
    "CDLMORNINGDOJISTAR_Agent": 3,  # 3-day pattern for Morning Doji Star
    "CDLMORNINGSTAR_Agent": 3,      # 3-day pattern for Morning Star
    "CDLONNECK_Agent": 2,           # 2-day pattern for On-Neck Pattern
    "CDLPIERCING_Agent": 2,         # 2-day pattern for Piercing Pattern
    "CDLRICKSHAWMAN_Agent": 1,      # 1-day pattern for Rickshaw Man
    "CDLRISEFALL3METHODS_Agent": 5, # 5-day pattern for Rising/Falling Three Methods
    "CDLSEPARATINGLINES_Agent": 2,  # 2-day pattern for Separating Lines
    "CDLSHOOTINGSTAR_Agent": 1,     # 1-day pattern for Shooting Star
    "CDLSHORTLINE_Agent": 1,        # 1-day pattern for Short Line Candle
    "CDLSPINNINGTOP_Agent": 1,      # 1-day pattern for Spinning Top
    "CDLSTALLEDPATTERN_Agent": 3,   # 3-day pattern for Stalled Pattern
    "CDLSTICKSANDWICH_Agent": 3,    # 3-day pattern for Stick Sandwich
    "CDLTAKURI_Agent": 1,           # 1-day pattern for Takuri
    "CDLTASUKIGAP_Agent": 3,        # 3-day pattern for Tasuki Gap
    "CDLTHRUSTING_Agent": 2,        # 2-day pattern for Thrusting Pattern
    "CDLTRISTAR_Agent": 3,          # 3-day pattern for Tristar Pattern
    "CDLUNIQUE3RIVER_Agent": 3,     # 3-day pattern for Unique 3 River
    "CDLUPSIDEGAP2CROWS_Agent": 3,  # 3-day pattern for Upside Gap Two Crows
    "CDLXSIDEGAP3METHODS_Agent": 5, # 5-day pattern for Gap Three Methods
    
    # Micro-behavior TA-lib Agents - Optimized for short-term pattern detection
    "CUSUM_Trend_Agent": 1,         # Daily CUSUM for real-time trend change detection
    "MACD_Crossover_Agent": 12,     # 12-day fast EMA for MACD crossover signals
    "Mean_Reversion_Agent": 20,     # 20-day optimal for statistical mean reversion
    "Opening_Range_Break_Agent": 1, # Daily opening range breakout strategy
    "Pivot_Reversal_Agent": 1,      # Daily pivot point calculation and monitoring
    "RSI_Reversal_Agent": 14,       # 14-day RSI for reversal signal detection
    "Support_Resistance_Break_Agent": 21, # 21-day lookback for support/resistance levels
    "VWAP_Pressure_Agent": 1,       # Daily VWAP calculation for intraday pressure
    "Volatility_Squeeze_Agent": 20, # 20-day Bollinger Bands for squeeze detection
    
    # Orderflow Analysis Agents - Optimized for order flow and microstructure
    "AggressiveFlowAgent": 1,       # Daily aggressive flow analysis for immediate signals
    "TapeReadingAgent": 1,          # Daily tape reading for transaction flow analysis
    "VolumeProfileDeltaAgent": 5,   # 5-day volume profile for delta analysis
    "PriceLadderAgent": 1,          # Daily order book ladder analysis
    "LargeOrderAgent": 2,           # 2-day large order impact assessment
    "CumulativeDeltaAgent": 10,     # 10-day cumulative delta for trend analysis
    "OrderFlowImbalanceAgent": 1,   # Daily order flow imbalance detection
    "VolumeDeltaAgent": 2,          # 2-day volume delta trend establishment
    
    # Pattern Recognition Agents - Based on pattern formation timeframes
    "DoubleTopBottomAgent": 84,     # 84 days (12 weeks) for double top/bottom formation
    "TrianglePatternAgent": 42,     # 42 days (6 weeks) optimal for triangle pattern completion
    "FlagPatternAgent": 8,          # 8-day optimal for flag pattern (short-term continuation)
    "HeadShouldersAgent": 84,       # 84 days (12 weeks) for head and shoulders development

    
    # Volume Profile Agents - Optimized for volume analysis
    "MarketProfileAgent": 1,        # Daily Market Profile (TPO) session analysis
    "VolumeBreakoutAgent": 8,       # 8-day optimal for volume breakout confirmation
    "VWAPBandAgent": 1,             # Daily VWAP band calculation from session start
    "VolumeAtPriceAgent": 21,       # 21-day volume at price node development
    
    # Utility/Base Agents - Minimal periods for testing and infrastructure
    "AlwaysBuyAgent": 1,            # Constant signal - minimal period for efficiency
    "AlwaysSellAgent": 1,           # Constant signal - minimal period for efficiency
    "NeutralAgent": 1,              # Constant signal - minimal period for efficiency
    "RandomAgent": 1,               # Random signal - minimal period for efficiency
    "BaseAgent": 1,                 # Base class - minimal period for inheritance
    "PatternAgent": 1,              # Base pattern class - minimal period for inheritance
} 