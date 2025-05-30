"""
Moore Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Gordon Moore's contributions to
computer science and semiconductor technology, particularly his eponymous "Moore's Law"
that predicted the doubling of transistor count every two years.

Gordon Moore is known for:
1. Moore's Law - prediction of exponential growth in computing power
2. Co-founding Intel Corporation
3. Advancing integrated circuit technology
4. Promoting semiconductor scaling and miniaturization
5. Pioneering semiconductor manufacturing processes

This agent models market behavior using:
1. Exponential growth/decay patterns in price action
2. Scaling principles for market trend identification
3. Technological S-curves applied to market cycles
4. Integration of multiple market signals similar to circuit integration
5. Power-efficient signal processing (minimize noise)

Input: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
Output: Signal ∈ [-1.0000, 1.0000] where:
  -1.0000 = Strong sell signal (strong downward trend detected)
  -0.5000 = Weak sell signal (weak downward trend detected)
   0.0000 = Neutral signal (no clear trend)
   0.5000 = Weak buy signal (weak upward trend detected)
   1.0000 = Strong buy signal (strong upward trend detected)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import math
from collections import defaultdict, deque
from scipy import optimize, stats

from ..agent import Agent

logger = logging.getLogger(__name__)

class MooreAgent(Agent):
    """
    Trading agent based on Gordon Moore's semiconductor principles.
    
    Parameters
    ----------
    doubling_period : int, default=18
        Period for analyzing exponential growth/decay (in bars)
    integration_level : int, default=5
        Number of signals to integrate (like IC integration)
    scaling_factor : float, default=1.5
        Scaling factor for exponential pattern detection
    process_efficiency : float, default=0.7
        Efficiency parameter for signal processing
    clock_speed : int, default=3
        Frequency of major signal updates
    """
    
    def __init__(
        self,
        doubling_period: int = 18,
        integration_level: int = 5,
        scaling_factor: float = 1.5,
        process_efficiency: float = 0.7,
        clock_speed: int = 3
    ):
        super().__init__()
        self.doubling_period = doubling_period
        self.integration_level = integration_level
        self.scaling_factor = scaling_factor
        self.process_efficiency = process_efficiency
        self.clock_speed = clock_speed
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.growth_rates = deque(maxlen=20)
        self.transistor_count = 1  # Starting with one "transistor"
        self.chip_generation = 1.0
        self.circuit_components = {}
        self.process_node = 1.0  # Current "process node" (smaller is better)
        
    def _fit_exponential_model(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit exponential growth model to price data (Moore's Law-style)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        tuple
            (growth_rate, fit_quality, prediction)
        """
        if len(prices) < 10:
            return 0.0, 0.0, prices[-1] if len(prices) > 0 else 0.0
            
        # Take logarithm to convert exponential to linear
        log_prices = np.log(prices)
        x = np.arange(len(prices))
        
        # Fit linear model to log prices
        slope, intercept, r_value, _, _ = stats.linregress(x, log_prices)
        
        # Convert back to exponential domain
        growth_rate = np.exp(slope) - 1  # Continuous growth rate
        fit_quality = r_value ** 2  # R-squared
        
        # Predict next value
        next_x = len(prices)
        log_prediction = slope * next_x + intercept
        prediction = np.exp(log_prediction)
        
        return growth_rate, fit_quality, prediction
        
    def _analyze_moore_cycles(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        Analyze growth cycles similar to Moore's Law cycles
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Analysis of growth cycles
        """
        if len(prices) < self.doubling_period * 2:
            return {"growth_rate": 0.0, "cycle_status": "unknown", "confidence": 0.0}
            
        cycles = {}
        
        # Analyze growth rates over different time frames
        for period in [self.doubling_period // 2, self.doubling_period, self.doubling_period * 2]:
            if len(prices) >= period:
                growth, fit_quality, _ = self._fit_exponential_model(prices[-period:])
                cycles[f"{period}_bars"] = {
                    "growth_rate": growth,
                    "fit_quality": fit_quality,
                    "doubling_time": int(np.log(2) / np.log(1 + max(growth, 1e-10))) if growth > 0 else np.inf
                }
                
        # Determine current position in the cycle
        # Using the concept that Moore's Law technologies follow S-curves
        
        # Calculate short-term vs. long-term growth
        short_growth = cycles.get(f"{self.doubling_period // 2}_bars", {}).get("growth_rate", 0)
        long_growth = cycles.get(f"{self.doubling_period * 2}_bars", {}).get("growth_rate", 0)
        
        # Determine cycle phase
        if long_growth > 0.05:  # Significant long-term growth
            if short_growth > long_growth * 1.2:
                cycle_status = "early_exponential"  # Accelerating phase
            elif short_growth > long_growth * 0.8:
                cycle_status = "mid_exponential"  # Steady exponential growth
            else:
                cycle_status = "late_exponential"  # Decelerating but still growing
        elif long_growth < -0.05:  # Significant long-term decline
            if short_growth < long_growth * 1.2:
                cycle_status = "early_decline"  # Accelerating decline
            elif short_growth < long_growth * 0.8:
                cycle_status = "mid_decline"  # Steady decline
            else:
                cycle_status = "late_decline"  # Decelerating decline
        else:
            # Relatively flat market
            cycle_status = "transition"  # Between growth and decline
            
        # Overall growth rate (weighted average)
        growth_weights = {
            f"{self.doubling_period // 2}_bars": 0.2,
            f"{self.doubling_period}_bars": 0.5,
            f"{self.doubling_period * 2}_bars": 0.3
        }
        
        weighted_growth = 0.0
        weight_sum = 0.0
        
        for period, weight in growth_weights.items():
            if period in cycles:
                # Weight by both the assigned weight and fit quality
                period_weight = weight * cycles[period]["fit_quality"]
                weighted_growth += cycles[period]["growth_rate"] * period_weight
                weight_sum += period_weight
                
        if weight_sum > 0:
            overall_growth = weighted_growth / weight_sum
        else:
            overall_growth = 0.0
            
        # Confidence based on fit quality
        confidence = sum(cycle.get("fit_quality", 0) * growth_weights[period] 
                        for period, cycle in cycles.items() if period in growth_weights)
        
        return {
            "growth_rate": overall_growth,
            "cycle_status": cycle_status,
            "cycles": cycles,
            "confidence": confidence
        }
        
    def _calculate_transistor_density(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate market "transistor density" metrics
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            Transistor density metrics
        """
        if len(prices) < 20:
            return {"density": 0.0, "efficiency": 0.0, "heat_dissipation": 0.0}
            
        # Volatility (inverse of density - lower is better)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:])
        
        # Higher density = lower volatility (more efficient price movement)
        density = 1.0 / max(volatility, 0.001)
        
        # Normalize density to a reasonable scale
        density = min(10.0, density) / 10.0
        
        # Calculate "power efficiency" (returns per unit of volatility)
        mean_return = np.mean(np.abs(returns[-20:]))
        efficiency = mean_return / max(volatility, 0.001) if volatility > 0 else 0.0
        efficiency = min(5.0, efficiency) / 5.0  # Normalize
        
        # Calculate "heat dissipation" (how quickly outliers revert)
        if len(returns) >= 20:
            # Identify outlier returns (>2 std dev)
            mean_ret = np.mean(returns[-20:])
            std_ret = np.std(returns[-20:])
            
            outliers = []
            for i in range(len(returns) - 20, len(returns) - 1):
                if abs(returns[i] - mean_ret) > 2 * std_ret:
                    # Check how quickly it reverted
                    next_return = returns[i+1]
                    # Opposite sign indicates reversion
                    if returns[i] * next_return < 0:
                        outliers.append(1.0)
                    else:
                        outliers.append(0.0)
                        
            # Heat dissipation = ability to revert from outliers
            heat_dissipation = np.mean(outliers) if outliers else 0.5
        else:
            heat_dissipation = 0.5
            
        # Volume-based metric if volume data is available
        volume_efficiency = 0.5  # Default
        if volumes is not None and len(volumes) >= 20:
            # Calculate "price movement per unit of volume"
            price_range = np.max(prices[-20:]) - np.min(prices[-20:])
            total_volume = np.sum(volumes[-20:])
            
            if total_volume > 0:
                vol_efficiency = price_range / total_volume
                # Normalize to [0, 1] scale
                volume_efficiency = min(1.0, vol_efficiency * 1000)
                
        return {
            "density": density,
            "efficiency": efficiency,
            "heat_dissipation": heat_dissipation,
            "volume_efficiency": volume_efficiency
        }
        
    def _update_process_node(self, density_metrics: Dict[str, float]) -> float:
        """
        Update the "process node" based on density metrics
        
        Parameters
        ----------
        density_metrics : dict
            Density metrics from _calculate_transistor_density
            
        Returns
        -------
        float
            New process node value (smaller is better)
        """
        # Start with current process node
        current_node = self.process_node
        
        # Calculate target process node based on current density
        # Higher density should result in smaller process node
        density = density_metrics.get("density", 0.0)
        efficiency = density_metrics.get("efficiency", 0.0)
        
        # Combine factors, weighting density more heavily
        scaling_factor = 0.7 * density + 0.3 * efficiency
        
        # Target node shrinks with better density/efficiency
        # The "+ 0.1" ensures we don't get to zero
        target_node = current_node / (scaling_factor * self.scaling_factor + 0.1)
        
        # Process node transitions aren't instant (gradual shift)
        new_node = current_node * 0.8 + target_node * 0.2
        
        # Ensure node doesn't get too small
        new_node = max(0.05, new_node)
        
        return new_node
        
    def _generate_integrated_circuit(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate an "integrated circuit" of market signals
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            Integrated circuit components and signals
        """
        if len(prices) < 30:
            return {"components": {}, "clock_signal": 0.0, "power_signal": 0.0}
            
        # Create various "components" for our market integrated circuit
        components = {}
        
        # Component 1: Trend Processor (central processing unit)
        if len(prices) >= 20:
            # Simple trend calculation
            short_sma = np.mean(prices[-10:])
            med_sma = np.mean(prices[-20:])
            
            if short_sma > med_sma:
                trend_signal = min(1.0, (short_sma / med_sma - 1) * 10)
            else:
                trend_signal = max(-1.0, (short_sma / med_sma - 1) * 10)
                
            components["trend_processor"] = {
                "signal": trend_signal,
                "power_usage": 0.3,  # Relative importance
                "clock_cycles": 1  # Updates every bar
            }
            
        # Component 2: Momentum Cache (fast memory)
        if len(prices) >= 10:
            # Rate of change
            momentum = prices[-1] / prices[-10] - 1
            momentum_signal = np.clip(momentum * 5, -1.0, 1.0)
            
            components["momentum_cache"] = {
                "signal": momentum_signal,
                "power_usage": 0.2,
                "clock_cycles": 2  # Updates every 2 bars
            }
            
        # Component 3: Volatility Controller (power management unit)
        if len(prices) >= 30:
            # Volatility ratio (recent vs longer-term)
            try:
                # Ensure we have enough data for both calculations
                if len(prices) >= 31:  # Need at least 31 points for 30 diffs
                    recent_vol = np.std(np.diff(prices[-10:]) / prices[-11:-1])
                    longer_vol = np.std(np.diff(prices[-30:]) / prices[-31:-1])
                    vol_ratio = recent_vol / longer_vol if longer_vol > 0 else 1.0
                else:
                    # Fallback if we don't have enough data
                    vol_ratio = 1.0
            except (IndexError, ValueError):
                # Fallback if any array issues
                vol_ratio = 1.0
            
            # Signal interpretation: >1 means increasing volatility
            if vol_ratio > 1.2:
                vol_signal = -0.5  # Increasing volatility often precedes drops
            elif vol_ratio < 0.8:
                vol_signal = 0.3  # Decreasing volatility can be bullish
            else:
                vol_signal = 0.0
                
            components["volatility_controller"] = {
                "signal": vol_signal,
                "power_usage": 0.15,
                "clock_cycles": 3  # Updates every 3 bars
            }
            
        # Component 4: Volume Interface (I/O controller)
        if volumes is not None and len(volumes) >= 20 and len(prices) >= 20:
            # Volume trend
            recent_vol_avg = np.mean(volumes[-5:])
            older_vol_avg = np.mean(volumes[-20:-5])
            
            vol_trend = recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1.0
            
            # Price-volume correlation
            if len(prices) >= 21 and len(volumes) >= 21:  # Need at least 21 points to get 20 diffs
                try:
                    price_changes = np.diff(prices[-21:]) / prices[-21:-1]
                    volume_changes = np.diff(volumes[-21:]) / volumes[-21:-1]
                except IndexError:
                    # Fallback if array lengths don't match
                    min_len = min(len(prices), len(volumes))
                    price_changes = np.diff(prices[-min_len:]) / prices[-min_len:-1]
                    volume_changes = np.diff(volumes[-min_len:]) / volumes[-min_len:-1]
                
                if len(price_changes) == len(volume_changes) and len(price_changes) > 1:
                    correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                    if not np.isfinite(correlation):
                        correlation = 0
                else:
                    correlation = 0
            else:
                correlation = 0
                
            # Combine into signal
            volume_signal = correlation * 0.7  # -1 to 1 based on correlation
            
            # Adjust by volume trend
            if vol_trend > 1.3:  # Significantly increasing volume
                volume_signal *= 1.5  # Amplify signal
            elif vol_trend < 0.7:  # Significantly decreasing volume
                volume_signal *= 0.5  # Dampen signal
                
            components["volume_interface"] = {
                "signal": volume_signal,
                "power_usage": 0.15,
                "clock_cycles": 2
            }
            
        # Component 5: Market Memory (longer-term patterns)
        if len(prices) >= self.doubling_period * 2:
            # Identify if current pattern matches exponential growth/decay pattern
            growth_rate, fit_quality, _ = self._fit_exponential_model(prices[-self.doubling_period:])
            
            # Better fit = stronger signal
            memory_signal = growth_rate * min(1.0, fit_quality * 2)
            memory_signal = np.clip(memory_signal * 10, -1.0, 1.0)  # Scale appropriately
            
            components["market_memory"] = {
                "signal": memory_signal,
                "power_usage": 0.2,
                "clock_cycles": 5  # Updates less frequently
            }
            
        # Calculate overall clock signal (synchronization)
        clock_signal = 0.0
        relevant_count = 0
        
        for name, component in components.items():
            # Only include components that should update on this cycle
            if len(self.growth_rates) % component["clock_cycles"] == 0:
                clock_signal += component["signal"]
                relevant_count += 1
                
        if relevant_count > 0:
            clock_signal /= relevant_count
            
        # Calculate weighted power signal (overall direction)
        power_signal = 0.0
        total_power = 0.0
        
        for name, component in components.items():
            power_signal += component["signal"] * component["power_usage"]
            total_power += component["power_usage"]
            
        if total_power > 0:
            power_signal /= total_power
            
        return {
            "components": components,
            "clock_signal": clock_signal,
            "power_signal": power_signal
        }
        
    def _apply_scaling_law(self, circuit: Dict[str, Any], density_metrics: Dict[str, float]) -> float:
        """
        Apply Moore's scaling law to generate final signal
        
        Parameters
        ----------
        circuit : dict
            Integrated circuit from _generate_integrated_circuit
        density_metrics : dict
            Density metrics from _calculate_transistor_density
            
        Returns
        -------
        float
            Final trading signal
        """
        # Extract primary signals
        power_signal = circuit.get("power_signal", 0.0)
        clock_signal = circuit.get("clock_signal", 0.0)
        
        # Extract density metrics
        density = density_metrics.get("density", 0.0)
        efficiency = density_metrics.get("efficiency", 0.0)
        heat_dissipation = density_metrics.get("heat_dissipation", 0.0)
        
        # Apply scaling principles to signal generation
        
        # 1. Main directional component from power signal
        base_signal = power_signal
        
        # 2. Adjust signal strength based on market "chip density"
        # Higher density = stronger signal (more confidence)
        signal_strength = base_signal * (0.5 + 0.5 * density)
        
        # 3. Adjust by "process efficiency" (how efficient price moves are)
        # Higher efficiency = less noise, clearer signal
        process_quality = 0.5 + 0.5 * efficiency
        
        # 4. Apply "die size" scaling (smaller process node = stronger signal)
        # Normalize process node to [0,1] range where 1 is best
        node_factor = 1.0 / (1.0 + self.process_node)
        
        # 5. Apply "heat dissipation" factor
        # Better heat dissipation = better recovery from anomalies
        heat_factor = 0.5 + 0.5 * heat_dissipation
        
        # Calculate final signal with all scaling factors
        scaled_signal = (
            signal_strength * 0.5 +  # Base signal with density scaling
            clock_signal * 0.2 * process_quality +  # Clock signal adjusted by efficiency
            base_signal * 0.2 * node_factor +  # Process node scaling
            base_signal * 0.1 * heat_factor  # Heat dissipation factor
        )
        
        # Apply overall process efficiency parameter
        final_signal = scaled_signal * self.process_efficiency
        
        return np.clip(final_signal, -1.0, 1.0)
        
    def _update_transistor_count(self, growth_rate: float) -> int:
        """
        Update internal "transistor count" based on growth rate
        
        Parameters
        ----------
        growth_rate : float
            Growth rate from _fit_exponential_model
            
        Returns
        -------
        int
            New transistor count
        """
        # Apply growth rate to current transistor count
        # This is conceptually similar to how transistor counts double
        # in Moore's Law every ~2 years
        
        # Limit growth rate to prevent unreasonable values
        capped_growth = np.clip(growth_rate, -0.5, 1.0)
        
        # Update count with growth
        new_count = self.transistor_count * (1 + capped_growth)
        
        # Ensure count stays at least 1
        return max(1, int(new_count))
        
    def _update_chip_generation(self, density: float, growth: float) -> float:
        """
        Update the chip "generation" based on density and growth
        
        Parameters
        ----------
        density : float
            Market density metric
        growth : float
            Growth rate
            
        Returns
        -------
        float
            New chip generation
        """
        # Combine density and growth to determine if we're advancing
        progress_factor = density * 0.5 + max(0, growth) * 0.5
        
        # Need significant progress to advance a generation
        if progress_factor > 0.6:
            # Advance to next generation
            return self.chip_generation + 0.1
        elif progress_factor < 0.2:
            # Regress slightly
            return max(1.0, self.chip_generation - 0.05)
        else:
            # Maintain current generation
            return self.chip_generation
            
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data using Moore's semiconductor scaling principles
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.doubling_period:
            self.is_fitted = False
            return
            
        try:
            # Extract price and volume data
            prices = historical_df['close'].values
            volumes = historical_df['volume'].values if 'volume' in historical_df.columns else None
            
            # 1. Analyze growth cycles (Moore's Law-like patterns)
            cycle_analysis = self._analyze_moore_cycles(prices)
            growth_rate = cycle_analysis["growth_rate"]
            
            # Update growth rate history
            self.growth_rates.append(growth_rate)
            
            # 2. Calculate "transistor density" metrics
            density_metrics = self._calculate_transistor_density(prices, volumes)
            
            # 3. Update process node
            self.process_node = self._update_process_node(density_metrics)
            
            # 4. Generate integrated circuit
            circuit = self._generate_integrated_circuit(prices, volumes)
            
            # 5. Apply Moore's scaling law to generate signal
            signal = self._apply_scaling_law(circuit, density_metrics)
            
            # 6. Update internal state
            self.transistor_count = self._update_transistor_count(growth_rate)
            self.chip_generation = self._update_chip_generation(
                density_metrics["density"], growth_rate
            )
            self.circuit_components = circuit["components"]
            
            # Store final signal
            self.latest_signal = signal
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Moore Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on semiconductor scaling principles
        
        Parameters
        ----------
        current_price : float
            Current asset price
        historical_df : pandas.DataFrame
            Historical price data
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        # Process the data
        self.fit(historical_df)
        
        if not self.is_fitted:
            return 0.0
            
        return self.latest_signal
    
    def __str__(self) -> str:
        return "Moore Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Moore's Law scaling principles.
        
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
            # Process the data using existing workflow
            self.fit(historical_df)
            
            if not self.is_fitted:
                return 0.0000
                
            # Get current price for prediction
            current_price = historical_df['close'].iloc[-1]
            
            # Generate signal using existing predict method
            signal = self.predict(current_price=current_price, historical_df=historical_df)
            
            # Ensure signal is properly formatted to 4 decimal places
            return float(round(signal, 4))
            
        except ValueError as e:
            # Handle the case where there's not enough data
            logger.error(f"ValueError in Moore strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Moore strategy: {str(e)}")
            return 0.0000 