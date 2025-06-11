"""
Ritchie Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Dennis Ritchie's contributions to
computer science, particularly the C programming language, Unix operating system,
and his philosophy of simplicity and efficiency.

Dennis Ritchie is known for:
1. Creating the C programming language
2. Co-creating the Unix operating system with Ken Thompson
3. Developing efficient, portable, and low-level systems programming
4. Focus on minimalist design principles

This agent models market behavior using:
1. C-language inspired pointers/references to key market levels
2. Unix-like modularity and composition principles
3. Memory management concepts (allocation/deallocation of positions)
4. Low-level optimization techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import math
from collections import deque, defaultdict

from ..agent import Agent

logger = logging.getLogger(__name__)

class RitchieAgent(Agent):
    """
    Trading agent based on Dennis Ritchie's programming principles.
    
    Parameters
    ----------
    pointer_sensitivity : float, default=0.02
        Sensitivity for identifying pointer levels
    memory_blocks : int, default=5
        Number of memory blocks for position management
    optimization_level : int, default=2
        Level of signal optimization (0-3)
    unix_pipe_depth : int, default=3
        Depth of data transformation pipeline
    stack_size : int, default=20
        Size of the memory stack for historical signals
    """
    
    def __init__(
        self,
        pointer_sensitivity: float = 0.02,
        memory_blocks: int = 5,
        optimization_level: int = 2,
        unix_pipe_depth: int = 3,
        stack_size: int = 20
    ):
        self.pointer_sensitivity = pointer_sensitivity
        self.memory_blocks = memory_blocks
        self.optimization_level = optimization_level
        self.unix_pipe_depth = unix_pipe_depth
        self.stack_size = stack_size
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state (modeled after C/Unix concepts)
        self.memory_heap = {}  # Store market states
        self.signal_stack = deque(maxlen=stack_size)  # Store recent signals
        self.pointers = {}  # Store references to key price levels
        self.pipes = []  # Store data transformation functions
        self.preprocessor = None  # Data preprocessing function
        
    def _malloc_memory(self, data_size: int) -> Dict:
        """
        Allocate memory for market data (like C's malloc)
        
        Parameters
        ----------
        data_size : int
            Size of data to allocate memory for
            
        Returns
        -------
        dict
            Allocated memory structure
        """
        # Allocate memory blocks based on data size
        block_size = data_size // self.memory_blocks
        if block_size < 5:  # Minimum block size
            block_size = 5
            
        memory = {
            'size': data_size,
            'block_size': block_size,
            'blocks': {},
            'pointers': {},
            'allocated': True
        }
        
        return memory
    
    def _free_memory(self, memory: Dict) -> None:
        """
        Free allocated memory (like C's free)
        
        Parameters
        ----------
        memory : dict
            Memory structure to free
        """
        if memory and memory.get('allocated', False):
            # Mark as deallocated
            memory['allocated'] = False
            memory['blocks'] = {}
            memory['pointers'] = {}
    
    def _identify_pointers(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Identify key price levels as pointers
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        dict
            Key price levels as pointers
        """
        if len(prices) < 20:
            return {}
            
        pointers = {}
        
        # Key pointer types (like variable types in C)
        # 1. Support levels (int *)
        # 2. Resistance levels (const int *)
        # 3. Moving averages (float *)
        # 4. Volatility bands (double *)
        
        # 1. Support levels
        recent_window = min(len(prices), 50)
        recent_prices = prices[-recent_window:]
        
        # Find local minima
        support_levels = []
        for i in range(2, len(recent_prices) - 2):
            if (recent_prices[i] < recent_prices[i-1] and 
                recent_prices[i] < recent_prices[i-2] and
                recent_prices[i] < recent_prices[i+1] and
                recent_prices[i] < recent_prices[i+2]):
                support_levels.append(recent_prices[i])
                
        # Cluster close support levels
        if support_levels:
            clustered_supports = []
            current_cluster = [support_levels[0]]
            
            for level in support_levels[1:]:
                # Check if level is within sensitivity threshold of current cluster average
                cluster_avg = sum(current_cluster) / len(current_cluster)
                if abs(level - cluster_avg) / cluster_avg <= self.pointer_sensitivity:
                    current_cluster.append(level)
                else:
                    # Save current cluster and start a new one
                    clustered_supports.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
                    
            # Add the last cluster
            if current_cluster:
                clustered_supports.append(sum(current_cluster) / len(current_cluster))
                
            # Store support pointers
            for i, level in enumerate(clustered_supports):
                pointers[f'support_{i}'] = level
        
        # 2. Resistance levels
        resistance_levels = []
        for i in range(2, len(recent_prices) - 2):
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i-2] and
                recent_prices[i] > recent_prices[i+1] and
                recent_prices[i] > recent_prices[i+2]):
                resistance_levels.append(recent_prices[i])
                
        # Cluster close resistance levels
        if resistance_levels:
            clustered_resistances = []
            current_cluster = [resistance_levels[0]]
            
            for level in resistance_levels[1:]:
                # Check if level is within sensitivity threshold of current cluster average
                cluster_avg = sum(current_cluster) / len(current_cluster)
                if abs(level - cluster_avg) / cluster_avg <= self.pointer_sensitivity:
                    current_cluster.append(level)
                else:
                    # Save current cluster and start a new one
                    clustered_resistances.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
                    
            # Add the last cluster
            if current_cluster:
                clustered_resistances.append(sum(current_cluster) / len(current_cluster))
                
            # Store resistance pointers
            for i, level in enumerate(clustered_resistances):
                pointers[f'resistance_{i}'] = level
                
        # 3. Moving averages
        ma_lengths = [20, 50, 100, 200]
        for length in ma_lengths:
            if len(prices) >= length:
                ma_value = np.mean(prices[-length:])
                pointers[f'ma_{length}'] = ma_value
                
        # 4. Volatility bands
        if len(prices) >= 20:
            ma_20 = np.mean(prices[-20:])
            std_20 = np.std(prices[-20:])
            
            pointers['upper_band'] = ma_20 + 2 * std_20
            pointers['lower_band'] = ma_20 - 2 * std_20
            
        return pointers
    
    def _create_preprocessor(self) -> callable:
        """
        Create a preprocessing function for data (like a C preprocessor)
        
        Returns
        -------
        callable
            Preprocessing function
        """
        def preprocessor(data):
            # Remove NaN values
            if isinstance(data, np.ndarray):
                return data[~np.isnan(data)]
            return data
            
        return preprocessor
    
    def _build_unix_pipes(self) -> List[callable]:
        """
        Build data transformation functions like Unix pipes
        
        Returns
        -------
        list
            List of transformation functions
        """
        pipes = []
        
        # Pipe 1: Calculate returns
        def calc_returns(prices):
            if len(prices) < 2:
                return np.array([])
            return np.diff(prices) / prices[:-1]
        pipes.append(calc_returns)
        
        # Pipe 2: Remove outliers
        def remove_outliers(returns):
            if len(returns) < 5:
                return returns
            mean = np.mean(returns)
            std = np.std(returns)
            return returns[np.abs(returns - mean) <= 3 * std]
        pipes.append(remove_outliers)
        
        # Additional pipes based on depth
        if self.unix_pipe_depth >= 2:
            # Pipe 3: Calculate momentum
            def calc_momentum(returns):
                if len(returns) < 10:
                    return np.array([0])
                    
                # Simple momentum calculation
                return np.array([np.sum(returns[-min(10, len(returns)):])])
            pipes.append(calc_momentum)
            
        if self.unix_pipe_depth >= 3:
            # Pipe 4: Normalize signal
            def normalize_signal(values):
                if not len(values) or np.max(np.abs(values)) == 0:
                    return np.array([0])
                return values / np.max(np.abs(values))
            pipes.append(normalize_signal)
            
        return pipes
    
    def _dereference_pointer(self, pointer_name: str, current_price: float) -> Tuple[float, float]:
        """
        Get value from a pointer and distance from current price
        
        Parameters
        ----------
        pointer_name : str
            Name of the pointer
        current_price : float
            Current price
            
        Returns
        -------
        tuple
            (pointer_value, relative_distance)
        """
        if pointer_name not in self.pointers:
            return 0.0, 0.0
            
        pointer_value = self.pointers[pointer_name]
        relative_distance = (current_price - pointer_value) / current_price
        
        return pointer_value, relative_distance
    
    def _memory_management(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict:
        """
        Manage memory allocation for market data
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            Memory structure with market data
        """
        # Allocate memory
        memory = self._malloc_memory(len(prices))
        
        # Store data in blocks
        for i in range(0, len(prices), memory['block_size']):
            block_end = min(i + memory['block_size'], len(prices))
            block = {
                'prices': prices[i:block_end],
                'start_idx': i,
                'end_idx': block_end - 1
            }
            
            if volumes is not None and i < len(volumes):
                vol_end = min(i + memory['block_size'], len(volumes))
                block['volumes'] = volumes[i:vol_end]
                
            memory['blocks'][i // memory['block_size']] = block
            
        return memory
    
    def _c_structs_analysis(self, memory: Dict) -> Dict[str, float]:
        """
        Analyze market data using C-like struct concepts
        
        Parameters
        ----------
        memory : dict
            Memory structure with market data
            
        Returns
        -------
        dict
            Analysis results
        """
        if not memory or not memory.get('allocated', False):
            return {}
            
        results = {}
        
        # Analyze each memory block like a C struct
        for block_id, block in memory['blocks'].items():
            if 'prices' not in block or len(block['prices']) < 5:
                continue
                
            # Simple statistics for each block
            prices = block['prices']
            
            # struct block_stats { ... }
            block_stats = {
                'min': np.min(prices),
                'max': np.max(prices),
                'mean': np.mean(prices),
                'std': np.std(prices),
                'median': np.median(prices)
            }
            
            # Calculate returns
            if len(prices) >= 2:
                returns = np.diff(prices) / prices[:-1]
                
                # Add return stats
                block_stats['return_mean'] = np.mean(returns)
                block_stats['return_std'] = np.std(returns)
                
            # Add volume stats if available
            if 'volumes' in block and len(block['volumes']) > 0:
                volumes = block['volumes']
                block_stats['volume_mean'] = np.mean(volumes)
                block_stats['volume_std'] = np.std(volumes)
                
                # Calculate price-volume correlation
                if len(volumes) == len(prices):
                    block_stats['price_volume_corr'] = np.corrcoef(prices, volumes)[0, 1]
                    
            # Store block stats
            results[f'block_{block_id}'] = block_stats
            
        # Aggregate across all blocks
        if memory['blocks']:
            all_prices = np.concatenate([block['prices'] for block in memory['blocks'].values()])
            
            results['global'] = {
                'min': np.min(all_prices),
                'max': np.max(all_prices),
                'mean': np.mean(all_prices),
                'std': np.std(all_prices),
                'range': np.max(all_prices) - np.min(all_prices)
            }
            
        return results
    
    def _optimize_signal(self, signal: float, memory: Dict) -> float:
        """
        Optimize trading signal using C-like optimization techniques
        
        Parameters
        ----------
        signal : float
            Raw trading signal
        memory : dict
            Memory structure with market data
            
        Returns
        -------
        float
            Optimized signal
        """
        if not memory or not memory.get('allocated', False):
            return signal
            
        # Optimization levels are inspired by C compiler optimization flags
        # 0: No optimization
        # 1: Basic optimization
        # 2: More aggressive optimization
        # 3: Maximum optimization
        
        if self.optimization_level == 0:
            return signal
            
        # Get most recent block
        last_block_id = max(memory['blocks'].keys())
        last_block = memory['blocks'][last_block_id]
        
        # Basic optimization (O1)
        # Adjust signal based on recent volatility
        if self.optimization_level >= 1 and len(last_block['prices']) >= 5:
            prices = last_block['prices']
            returns = np.diff(prices) / prices[:-1]
            
            # Scale signal by volatility (lower volatility = stronger signal)
            volatility = np.std(returns)
            if volatility > 0:
                # Dampen signal in high volatility environments
                signal = signal * (0.02 / volatility) if volatility > 0.02 else signal
                
        # More aggressive optimization (O2)
        if self.optimization_level >= 2:
            # Consider trend persistence
            if len(returns) >= 10:
                # Check for trend persistence
                sign_changes = np.sum(np.diff(np.signbit(returns)))
                
                # Fewer sign changes indicate more persistent trend
                persistence = 1.0 - (sign_changes / len(returns))
                
                # Amplify signal in persistent trends
                signal = signal * (1.0 + persistence)
                
        # Maximum optimization (O3)
        if self.optimization_level >= 3 and 'volumes' in last_block:
            volumes = last_block['volumes']
            
            if len(volumes) >= 5 and len(prices) == len(volumes):
                # Volume-weighted signal adjustment
                vol_ratio = volumes[-1] / np.mean(volumes)
                
                # Higher volume reinforces signal
                signal = signal * (1.0 + 0.2 * (vol_ratio - 1.0) * np.sign(signal))
                
        # Ensure signal is in range [-1, 1]
        return np.clip(signal, -1.0, 1.0)
    
    def _process_with_unix_pipes(self, prices: np.ndarray) -> float:
        """
        Process data through Unix-like pipes
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        float
            Processed signal
        """
        if len(prices) < 10 or not self.pipes:
            return 0.0
            
        # Create a copy to avoid modifying the original
        data = np.copy(prices)
        
        # Apply preprocessor first (like CPP)
        if self.preprocessor:
            data = self.preprocessor(data)
            
        # Apply each pipe in sequence
        for pipe in self.pipes:
            data = pipe(data)
            
            # Check if pipe returned valid data
            if len(data) == 0:
                return 0.0
                
        # Final result should be a signal value
        if len(data) == 1:
            signal = float(data[0])
        else:
            # If multiple values, take the last one
            signal = float(data[-1])
            
        # Ensure signal is in range [-1, 1]
        return np.clip(signal, -1.0, 1.0)
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data with Ritchie's programming principles
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < 20:
            self.is_fitted = False
            return
            
        try:
            # Extract price and volume data
            prices = historical_df['close'].values
            volumes = historical_df['volume'].values if 'volume' in historical_df.columns else None
            
            # 1. Create preprocessor if needed
            if not self.preprocessor:
                self.preprocessor = self._create_preprocessor()
                
            # 2. Build Unix pipes if needed
            if not self.pipes:
                self.pipes = self._build_unix_pipes()
                
            # 3. Identify key price levels as pointers
            self.pointers = self._identify_pointers(prices)
            
            # 4. Allocate and manage memory for data
            memory = self._memory_management(prices, volumes)
            self.memory_heap = memory
            
            # 5. Analyze market structures
            struct_analysis = self._c_structs_analysis(memory)
            
            # 6. Process data through Unix pipes
            pipe_signal = self._process_with_unix_pipes(prices)
            
            # 7. Generate pointer-based signals from price levels
            pointer_signal = 0.0
            current_price = prices[-1]
            
            # Check distance to support/resistance pointers
            support_distances = []
            resistance_distances = []
            
            for name, value in self.pointers.items():
                if 'support' in name:
                    _, distance = self._dereference_pointer(name, current_price)
                    support_distances.append(distance)
                elif 'resistance' in name:
                    _, distance = self._dereference_pointer(name, current_price)
                    resistance_distances.append(distance)
                    
            # Generate signal based on proximity to supports/resistances
            if support_distances and resistance_distances:
                # Find closest support and resistance
                closest_support = min(support_distances, key=abs)
                closest_resistance = min(resistance_distances, key=abs)
                
                # Generate signal based on relative distances
                # Positive when closer to support, negative when closer to resistance
                if abs(closest_support) < abs(closest_resistance):
                    # Closer to support - bullish
                    pointer_signal = 0.5 * (1.0 - abs(closest_support) / self.pointer_sensitivity)
                else:
                    # Closer to resistance - bearish
                    pointer_signal = -0.5 * (1.0 - abs(closest_resistance) / self.pointer_sensitivity)
                    
            # Add MA pointer signals
            ma_signals = []
            for name in self.pointers:
                if 'ma_' in name:
                    value, distance = self._dereference_pointer(name, current_price)
                    # Positive when price above MA, negative when below
                    if abs(distance) < self.pointer_sensitivity * 2:
                        ma_signals.append(-np.sign(distance) * (1.0 - abs(distance) / (self.pointer_sensitivity * 2)))
                        
            if ma_signals:
                ma_signal = sum(ma_signals) / len(ma_signals)
                pointer_signal = 0.6 * pointer_signal + 0.4 * ma_signal
                
            # 8. Combine signals
            combined_signal = 0.5 * pipe_signal + 0.5 * pointer_signal
            
            # 9. Optimize signal
            final_signal = self._optimize_signal(combined_signal, memory)
            
            # 10. Store signal in stack
            self.signal_stack.append(final_signal)
            
            # 11. Set latest signal
            self.latest_signal = final_signal
            
            self.is_fitted = True
            
            # Free memory that's no longer needed (like in C)
            # In a real C program, we'd free memory when done
            # self._free_memory(memory)
            
        except Exception as e:
            logger.error(f"Error in Ritchie Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on C and Unix programming principles
        
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
        return "Ritchie Agent" 
    
    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Ritchie's systems programming principles.
        
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
            logger.error(f"ValueError in Ritchie strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Ritchie strategy: {str(e)}")
            return 0.0000
