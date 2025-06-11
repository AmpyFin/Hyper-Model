"""
Berners-Lee Agent
~~~~~~~~~~~~~~
Agent implementing trading strategies based on Sir Tim Berners-Lee's fundamental
contributions to computer science, particularly the World Wide Web, HTTP, HTML,
and semantic web technologies.

Tim Berners-Lee is known for:
1. Inventing the World Wide Web
2. Creating HTTP (Hypertext Transfer Protocol)
3. Developing HTML (Hypertext Markup Language)
4. Championing the Semantic Web and linked data
5. Advocating for net neutrality and the open web

This agent models market behavior using:
1. Hypertext-like linking of related market signals
2. HTTP-inspired request/response trading cycles
3. HTML-like structured representation of market data
4. Semantic web concepts for market information classification
5. Decentralized decision-making based on linked market factors

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
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import logging
import math
from collections import defaultdict, deque
import re
import networkx as nx

from ..agent import Agent

logger = logging.getLogger(__name__)

class BernersLeeAgent(Agent):
    """
    Trading agent based on Tim Berners-Lee's web technologies.
    
    Parameters
    ----------
    hyperlink_depth : int, default=3
        Depth of hyperlinks between market factors
    semantic_classes : int, default=5
        Number of semantic classes for market states
    uri_precision : float, default=0.01
        Precision for identifying unique market states
    http_cycles : int, default=4
        Number of request/response cycles for signal refinement
    web_connectivity : float, default=0.5
        Connectivity parameter for market information network
    """
    
    def __init__(
        self,
        hyperlink_depth: int = 3,
        semantic_classes: int = 5,
        uri_precision: float = 0.01,
        http_cycles: int = 4,
        web_connectivity: float = 0.5
    ):
        self.hyperlink_depth = hyperlink_depth
        self.semantic_classes = semantic_classes
        self.uri_precision = uri_precision
        self.http_cycles = http_cycles
        self.web_connectivity = web_connectivity
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.market_graph = nx.DiGraph()
        self.resource_cache = {}
        self.hyperlink_index = {}
        self.semantic_ontology = {}
        self.http_response_history = deque(maxlen=20)
        
    def _create_market_uri(self, prices: np.ndarray) -> str:
        """
        Create a unique URI for the current market state
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        str
            URI identifying the market state
        """
        if len(prices) < 10:
            return "http://market.state/insufficient_data"
            
        # Calculate key metrics to identify the market state
        # Similar to how URLs identify web resources
        
        # Recent return (part of the path)
        recent_return = prices[-1] / prices[-10] - 1
        return_category = int(recent_return / self.uri_precision) * self.uri_precision
        
        # Volatility (query parameter)
        try:
            # Ensure we have enough data to calculate volatility correctly
            if len(prices) >= 11:
                price_diffs = np.diff(prices[-10:])
                price_bases = prices[-11:-1]  # One more element needed for the base prices
                
                # Ensure arrays have the same shape
                if len(price_diffs) == len(price_bases):
                    volatility = np.std(price_diffs / price_bases)
                else:
                    volatility = 0.0
            else:
                volatility = 0.0
        except (IndexError, ValueError):
            # Fallback if any array issues
            volatility = 0.0
            
        vol_category = int(volatility / self.uri_precision) * self.uri_precision
        
        # Trend strength (another query parameter)
        try:
            x = np.arange(min(10, len(prices)))
            trend_strength = 0.0
            if len(x) >= 2:
                slope, _ = np.polyfit(x, prices[-len(x):], 1)
                trend_strength = slope / prices[-1] * 100  # Normalized slope in percent
        except (IndexError, ValueError):
            trend_strength = 0.0
            
        trend_category = int(trend_strength / self.uri_precision) * self.uri_precision
        
        # Construct the URI similar to a web URL
        uri = f"http://market.state/price/{return_category:.4f}"
        uri += f"?volatility={vol_category:.4f}&trend={trend_category:.4f}"
        
        # Add fragment identifier based on overall market regime
        if trend_strength > 0.5:
            uri += "#strong_uptrend"
        elif trend_strength > 0.1:
            uri += "#mild_uptrend"
        elif trend_strength < -0.5:
            uri += "#strong_downtrend"
        elif trend_strength < -0.1:
            uri += "#mild_downtrend"
        else:
            uri += "#sideways"
            
        return uri
        
    def _html_representation(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Create structured HTML-like representation of market data
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        volumes : numpy.ndarray, optional
            Array of volume values
            
        Returns
        -------
        dict
            HTML-like market representation
        """
        if len(prices) < 20:
            return {}
            
        # Create a structured representation of market data similar to HTML
        # <html>
        #   <head>
        #     <meta name="market_state" content="uptrend">
        #   </head>
        #   <body>
        #     <div class="price_section">...</div>
        #     <div class="volume_section">...</div>
        #   </body>
        # </html>
        
        # Calculate basic metrics
        returns = np.diff(prices) / prices[:-1]
        recent_return = prices[-1] / prices[-10] - 1 if len(prices) >= 10 else 0
        
        # Determine market state with more sensitive thresholds for shorter periods
        # Adjusted thresholds to be more responsive to smaller moves
        if recent_return > 0.02:  # 2% return in 10 days
            market_state = "strong_uptrend"
        elif recent_return > 0.005:  # 0.5% return in 10 days
            market_state = "mild_uptrend"
        elif recent_return < -0.02:  # -2% return in 10 days
            market_state = "strong_downtrend"
        elif recent_return < -0.005:  # -0.5% return in 10 days
            market_state = "mild_downtrend"
        else:
            market_state = "sideways"
            
        # Create the HTML structure
        html = {
            "html": {
                "head": {
                    "meta": {
                        "market_state": market_state,
                        "last_price": prices[-1],
                        "recent_return": recent_return,
                        "volatility": np.std(returns[-20:]) if len(returns) >= 20 else 0
                    }
                },
                "body": {
                    "price_section": {
                        "current": prices[-1],
                        "high": np.max(prices[-20:]),
                        "low": np.min(prices[-20:]),
                        "trend": np.polyfit(range(min(20, len(prices))), prices[-min(20, len(prices)):], 1)[0]
                    }
                }
            }
        }
        
        # Add volume section if available
        if volumes is not None and len(volumes) > 0:
            html["html"]["body"]["volume_section"] = {
                "current": volumes[-1],
                "average": np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1],
                "trend": np.polyfit(range(min(20, len(volumes))), volumes[-min(20, len(volumes)):], 1)[0] if len(volumes) >= 2 else 0
            }
            
        return html
        
    def _extract_hyperlinks(self, html_data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Extract hyperlinks between market factors
        
        Parameters
        ----------
        html_data : dict
            HTML-like market representation
            
        Returns
        -------
        list
            List of (source, target) hyperlinks
        """
        if not html_data or "html" not in html_data:
            return []
            
        hyperlinks = []
        
        # Extract market state from metadata
        market_state = html_data["html"]["head"]["meta"].get("market_state", "unknown")
        
        # Create hyperlinks similar to web page links
        # Source -> Target represents influence or relationship
        
        # Price trend influences market state
        if "price_section" in html_data["html"]["body"]:
            price_data = html_data["html"]["body"]["price_section"]
            trend = price_data.get("trend", 0)
            
            # Create hyperlink: trend -> market_state
            hyperlinks.append(("trend", market_state))
            
            # Price level relative to recent high/low
            current = price_data.get("current", 0)
            high = price_data.get("high", current)
            low = price_data.get("low", current)
            
            if high > low:
                relative_level = (current - low) / (high - low) if high > low else 0.5
                level_state = "overbought" if relative_level > 0.8 else "oversold" if relative_level < 0.2 else "neutral"
                
                # Create hyperlink: price_level -> level_state
                hyperlinks.append(("price_level", level_state))
                
                # Link level_state to market_state
                hyperlinks.append((level_state, market_state))
                
        # Volume influences price trend
        if "volume_section" in html_data["html"]["body"] and "price_section" in html_data["html"]["body"]:
            volume_data = html_data["html"]["body"]["volume_section"]
            volume_trend = volume_data.get("trend", 0)
            volume_state = "increasing" if volume_trend > 0 else "decreasing"
            
            # Create hyperlink: volume_trend -> price_trend
            hyperlinks.append((volume_state, "trend"))
            
        return hyperlinks
        
    def _http_request_response(self, uri: str, html_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate HTTP request/response cycle for market data
        
        Parameters
        ----------
        uri : str
            URI identifying the market state
        html_data : dict
            HTML-like market representation
            
        Returns
        -------
        dict
            HTTP response with market analysis
        """
        # Check if we have a cached response
        if uri in self.resource_cache:
            return self.resource_cache[uri]
            
        # Parse the URI to extract information
        # Similar to how a web server handles HTTP requests
        
        # Extract components from URI
        path_match = re.search(r'/price/([-\d\.]+)', uri)
        volatility_match = re.search(r'volatility=([-\d\.]+)', uri)
        trend_match = re.search(r'trend=([-\d\.]+)', uri)
        fragment_match = re.search(r'#(\w+)', uri)
        
        # Default values
        price_return = 0.0
        volatility = 0.0
        trend = 0.0
        market_regime = "unknown"
        
        # Extract values if matches found
        if path_match:
            try:
                price_return = float(path_match.group(1))
            except ValueError:
                pass
                
        if volatility_match:
            try:
                volatility = float(volatility_match.group(1))
            except ValueError:
                pass
                
        if trend_match:
            try:
                trend = float(trend_match.group(1))
            except ValueError:
                pass
                
        if fragment_match:
            market_regime = fragment_match.group(1)
            
        # Generate response based on the request
        # Similar to how a web server generates HTTP responses
        
        # Base signals for different regimes
        regime_signals = {
            "strong_uptrend": 0.8,
            "mild_uptrend": 0.4,
            "sideways": 0.0,
            "mild_downtrend": -0.4,
            "strong_downtrend": -0.8
        }
        
        # Default response
        response = {
            "status": 200,
            "content_type": "application/json",
            "body": {
                "signal": regime_signals.get(market_regime, 0.0),
                "confidence": 0.5,
                "factors": {}
            }
        }
        
        # Adjust signal based on volatility
        # Higher volatility reduces confidence
        confidence_adjustment = max(0.0, 1.0 - volatility * 5)
        response["body"]["confidence"] *= confidence_adjustment
        
        # Adjust signal based on return/trend alignment
        if (price_return > 0 and trend > 0) or (price_return < 0 and trend < 0):
            # Return and trend align - strengthen signal
            response["body"]["signal"] *= 1.2
            response["body"]["confidence"] *= 1.1
        elif (price_return > 0 and trend < 0) or (price_return < 0 and trend > 0):
            # Return and trend conflict - weaken signal
            response["body"]["signal"] *= 0.8
            response["body"]["confidence"] *= 0.9
            
        # Apply constraints
        response["body"]["signal"] = np.clip(response["body"]["signal"], -1.0, 1.0)
        response["body"]["confidence"] = np.clip(response["body"]["confidence"], 0.1, 1.0)
        
        # Add relevant factors to the response
        response["body"]["factors"] = {
            "return": price_return,
            "volatility": volatility,
            "trend": trend,
            "regime": market_regime
        }
        
        # Cache the response
        self.resource_cache[uri] = response
        
        # Add to history
        self.http_response_history.append(response)
        
        return response
        
    def _build_semantic_ontology(self, html_data: Dict[str, Any], hyperlinks: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Build semantic ontology for market data
        
        Parameters
        ----------
        html_data : dict
            HTML-like market representation
        hyperlinks : list
            List of hyperlinks between market factors
            
        Returns
        -------
        dict
            Semantic market ontology
        """
        # Create a semantic web-like ontology for the market
        # Like RDF triples: Subject-Predicate-Object
        
        ontology = {
            "classes": {},
            "properties": {},
            "instances": {},
            "relations": []
        }
        
        # Define market classes (like semantic web classes)
        ontology["classes"] = {
            "MarketState": {
                "subClassOf": "Thing",
                "properties": ["volatility", "trend", "regime"]
            },
            "PriceLevel": {
                "subClassOf": "Thing",
                "properties": ["value", "relativeLevel"]
            },
            "Volume": {
                "subClassOf": "Thing",
                "properties": ["value", "trend"]
            },
            "Signal": {
                "subClassOf": "Thing",
                "properties": ["value", "confidence"]
            }
        }
        
        # Define properties
        ontology["properties"] = {
            "influences": {
                "domain": "Thing",
                "range": "Thing"
            },
            "partOf": {
                "domain": "Thing",
                "range": "MarketState"
            },
            "hasTrend": {
                "domain": "Thing",
                "range": "float"
            }
        }
        
        # Create instances from HTML data
        if "html" in html_data:
            # Market state instance
            market_state = html_data["html"]["head"]["meta"].get("market_state", "unknown")
            volatility = html_data["html"]["head"]["meta"].get("volatility", 0)
            
            ontology["instances"]["currentMarketState"] = {
                "type": "MarketState",
                "regime": market_state,
                "volatility": volatility
            }
            
            # Price instance
            if "price_section" in html_data["html"]["body"]:
                price_data = html_data["html"]["body"]["price_section"]
                current = price_data.get("current", 0)
                high = price_data.get("high", current)
                low = price_data.get("low", current)
                
                relative_level = (current - low) / (high - low) if high > low else 0.5
                
                ontology["instances"]["currentPrice"] = {
                    "type": "PriceLevel",
                    "value": current,
                    "relativeLevel": relative_level
                }
                
                # Relation: price is part of market state
                ontology["relations"].append({
                    "subject": "currentPrice",
                    "predicate": "partOf",
                    "object": "currentMarketState"
                })
                
            # Volume instance
            if "volume_section" in html_data["html"]["body"]:
                volume_data = html_data["html"]["body"]["volume_section"]
                current_vol = volume_data.get("current", 0)
                vol_trend = volume_data.get("trend", 0)
                
                ontology["instances"]["currentVolume"] = {
                    "type": "Volume",
                    "value": current_vol,
                    "trend": vol_trend
                }
                
                # Relation: volume influences price
                ontology["relations"].append({
                    "subject": "currentVolume",
                    "predicate": "influences",
                    "object": "currentPrice"
                })
                
        # Add relations from hyperlinks
        for source, target in hyperlinks:
            source_id = f"{source}Factor"
            target_id = f"{target}Factor"
            
            # Add source and target as instances if they don't exist
            if source_id not in ontology["instances"]:
                ontology["instances"][source_id] = {
                    "type": "Thing",
                    "name": source
                }
                
            if target_id not in ontology["instances"]:
                ontology["instances"][target_id] = {
                    "type": "Thing",
                    "name": target
                }
                
            # Add influence relation
            ontology["relations"].append({
                "subject": source_id,
                "predicate": "influences",
                "object": target_id
            })
            
        return ontology
        
    def _construct_market_graph(self, hyperlinks: List[Tuple[str, str]], ontology: Dict[str, Any]) -> nx.DiGraph:
        """
        Construct a graph of market factors (like the web)
        
        Parameters
        ----------
        hyperlinks : list
            List of hyperlinks between market factors
        ontology : dict
            Semantic market ontology
            
        Returns
        -------
        networkx.DiGraph
            Directed graph of market factors
        """
        G = nx.DiGraph()
        
        # Add nodes from hyperlinks
        for source, target in hyperlinks:
            if not G.has_node(source):
                G.add_node(source, type="factor")
            if not G.has_node(target):
                G.add_node(target, type="factor")
                
            # Add edge for hyperlink
            G.add_edge(source, target, type="hyperlink")
            
        # Add nodes and edges from ontology
        if "instances" in ontology:
            for instance_id, instance_data in ontology["instances"].items():
                if not G.has_node(instance_id):
                    G.add_node(instance_id, **instance_data)
                    
        if "relations" in ontology:
            for relation in ontology["relations"]:
                subject = relation["subject"]
                predicate = relation["predicate"]
                obj = relation["object"]
                
                if subject in G and obj in G:
                    G.add_edge(subject, obj, type=predicate)
                    
        return G
        
    def _traverse_hyperlinks(self, graph: nx.DiGraph, start_node: str, depth: int = 3) -> Dict[str, float]:
        """
        Traverse hyperlinks to aggregate influence (like web crawling)
        
        Parameters
        ----------
        graph : networkx.DiGraph
            Market factor graph
        start_node : str
            Starting node for traversal
        depth : int, default=3
            Maximum traversal depth
            
        Returns
        -------
        dict
            Influence scores for nodes
        """
        if not graph.has_node(start_node):
            return {}
            
        # Initialize scores
        scores = {node: 0.0 for node in graph.nodes()}
        scores[start_node] = 1.0
        
        # Track visited nodes
        visited = {start_node}
        
        # Queue for BFS traversal
        queue = deque([(start_node, 0)])  # (node, depth)
        
        while queue:
            node, current_depth = queue.popleft()
            
            if current_depth >= depth:
                continue
                
            # Calculate influence factor that decreases with depth
            influence = 1.0 / (current_depth + 1)
            
            # Traverse outgoing links
            for neighbor in graph.successors(node):
                # Update score
                scores[neighbor] += scores[node] * influence
                
                # Add to queue if not visited
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))
                    
        return scores
        
    def _semantic_web_inference(self, ontology: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform semantic web inference on market ontology
        
        Parameters
        ----------
        ontology : dict
            Semantic market ontology
            
        Returns
        -------
        dict
            Inferred market properties
        """
        inferred = {}
        
        # Extract market regime
        market_regime = "unknown"
        market_volatility = 0.0
        
        if "instances" in ontology and "currentMarketState" in ontology["instances"]:
            market_state = ontology["instances"]["currentMarketState"]
            market_regime = market_state.get("regime", "unknown")
            market_volatility = market_state.get("volatility", 0.0)
            
        # Infer market properties based on regime
        if market_regime == "strong_uptrend":
            inferred["bullishSentiment"] = 0.9
            inferred["bearishSentiment"] = 0.1
            inferred["expectedDirection"] = 1.0
        elif market_regime == "mild_uptrend":
            inferred["bullishSentiment"] = 0.7
            inferred["bearishSentiment"] = 0.3
            inferred["expectedDirection"] = 0.5
        elif market_regime == "strong_downtrend":
            inferred["bullishSentiment"] = 0.1
            inferred["bearishSentiment"] = 0.9
            inferred["expectedDirection"] = -1.0
        elif market_regime == "mild_downtrend":
            inferred["bullishSentiment"] = 0.3
            inferred["bearishSentiment"] = 0.7
            inferred["expectedDirection"] = -0.5
        else:  # sideways or unknown
            inferred["bullishSentiment"] = 0.5
            inferred["bearishSentiment"] = 0.5
            inferred["expectedDirection"] = 0.0
            
        # Infer confidence based on volatility
        inferred["confidence"] = max(0.1, 1.0 - market_volatility * 5)
        
        # Infer properties based on ontology relations
        if "relations" in ontology:
            # Count influence relations
            influence_count = defaultdict(int)
            for relation in ontology["relations"]:
                if relation["predicate"] == "influences":
                    influence_count[relation["object"]] += 1
                    
            # Calculate influence centrality
            if influence_count:
                max_influences = max(influence_count.values())
                for obj, count in influence_count.items():
                    inferred[f"centrality_{obj}"] = count / max_influences
                    
        return inferred
        
    def _generate_linked_data_signal(self, graph: nx.DiGraph, ontology: Dict[str, Any], 
                                   inference: Dict[str, float]) -> float:
        """
        Generate trading signal from linked market data
        
        Parameters
        ----------
        graph : networkx.DiGraph
            Market factor graph
        ontology : dict
            Semantic market ontology
        inference : dict
            Inferred market properties
            
        Returns
        -------
        float
            Trading signal
        """
        # Calculate signal components
        
        # 1. Basic signal from expected direction
        base_signal = inference.get("expectedDirection", 0.0)
        
        # 2. Sentiment component
        sentiment_signal = inference.get("bullishSentiment", 0.5) - inference.get("bearishSentiment", 0.5)
        
        # 3. Graph centrality component
        centrality_signals = []
        for key, value in inference.items():
            if key.startswith("centrality_"):
                # Extract factor name
                factor = key.replace("centrality_", "")
                
                # Check if factor is in graph
                if factor in graph:
                    # Get factor attributes
                    attrs = graph.nodes[factor]
                    
                    # Determine factor's contribution to signal
                    if "regime" in attrs:
                        regime = attrs["regime"]
                        if regime in ["strong_uptrend", "mild_uptrend"]:
                            centrality_signals.append(value)
                        elif regime in ["strong_downtrend", "mild_downtrend"]:
                            centrality_signals.append(-value)
                            
        # Average centrality signals
        centrality_signal = np.mean(centrality_signals) if centrality_signals else 0.0
        
        # 4. Find most authoritative nodes (like PageRank)
        try:
            pagerank = nx.pagerank(graph, alpha=0.85)
            
            # Top 3 authoritative nodes
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Calculate authority signal
            authority_signal = 0.0
            authority_sum = 0.0
            
            for node, rank in top_nodes:
                # Get node attributes
                attrs = graph.nodes[node]
                
                if "regime" in attrs:
                    regime = attrs["regime"]
                    if regime == "strong_uptrend":
                        node_signal = 1.0
                    elif regime == "mild_uptrend":
                        node_signal = 0.5
                    elif regime == "strong_downtrend":
                        node_signal = -1.0
                    elif regime == "mild_downtrend":
                        node_signal = -0.5
                    else:
                        node_signal = 0.0
                        
                    authority_signal += node_signal * rank
                    authority_sum += rank
                    
            if authority_sum > 0:
                authority_signal /= authority_sum
            else:
                authority_signal = 0.0
        except:
            # Fallback if PageRank fails
            authority_signal = 0.0
            
        # Combine signal components
        combined_signal = (
            base_signal * 0.4 +
            sentiment_signal * 0.3 +
            centrality_signal * 0.2 +
            authority_signal * 0.1
        )
        
        # Apply confidence adjustment
        confidence = inference.get("confidence", 0.5)
        
        # A lower confidence reduces the amplitude of the signal
        final_signal = combined_signal * confidence
        
        return np.clip(final_signal, -1.0, 1.0)
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data using Berners-Lee's web technology principles
        
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
            
            # 1. Create unique URI for current market state
            uri = self._create_market_uri(prices)
            
            # 2. Create HTML-like representation
            html_data = self._html_representation(prices, volumes)
            
            # 3. Extract hyperlinks from HTML representation
            hyperlinks = self._extract_hyperlinks(html_data)
            
            # 4. Simulate HTTP request/response cycles
            response = None
            for _ in range(self.http_cycles):
                response = self._http_request_response(uri, html_data)
                
                # Update HTML with response data
                if "body" in response and "signal" in response["body"]:
                    html_data["html"]["head"]["meta"]["signal"] = response["body"]["signal"]
                    html_data["html"]["head"]["meta"]["confidence"] = response["body"]["confidence"]
                    
                # Extract updated hyperlinks
                hyperlinks = self._extract_hyperlinks(html_data)
                
            # 5. Build semantic ontology
            ontology = self._build_semantic_ontology(html_data, hyperlinks)
            
            # 6. Construct market graph
            self.market_graph = self._construct_market_graph(hyperlinks, ontology)
            
            # 7. Traverse hyperlinks
            influences = self._traverse_hyperlinks(self.market_graph, "market_state", self.hyperlink_depth)
            
            # 8. Perform semantic inference
            inference = self._semantic_web_inference(ontology)
            
            # 9. Generate signal from linked data
            signal = self._generate_linked_data_signal(self.market_graph, ontology, inference)
            
            # Store final signal
            self.latest_signal = signal
            
            # Store key components
            self.hyperlink_index = dict(enumerate(hyperlinks))
            self.semantic_ontology = ontology
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Berners-Lee Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on web technology principles
        
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
        return "Berners-Lee Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Berners-Lee's web network principles.
        
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
            logger.error(f"ValueError in Berners-Lee strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Berners-Lee strategy: {str(e)}")
            return 0.0000 