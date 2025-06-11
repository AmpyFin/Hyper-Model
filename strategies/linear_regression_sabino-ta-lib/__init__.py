"""
Linear Regression Sabino TA-lib Module

This module contains specialized linear regression strategies based on Sabino's
trading methodologies combined with TA-lib indicators.
"""

import sys
import os
from pathlib import Path

# Add the current directory to sys.path to handle hyphenated directory imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import all agents
AGENTS = {}

try:
    # Risk agents
    from risk.keltner_channel_agent import KeltnerChannel_Agent
    from risk.ulcer_index_agent import UlcerIndex_Agent
    from risk.donchian_channel_agent import DonchianChannel_Agent
    
    AGENTS.update({
        "KeltnerChannel_Agent": KeltnerChannel_Agent,
        "UlcerIndex_Agent": UlcerIndex_Agent,
        "DonchianChannel_Agent": DonchianChannel_Agent,
    })
except ImportError as e:
    print(f"Warning: Could not import risk agents: {e}")

try:
    # Import from price-derived directory (with hyphen)
    price_derived_path = current_dir / "price-derived"
    sys.path.insert(0, str(price_derived_path))
    
    from cumulative_return_agent import CumulativeReturn_Agent
    from daily_log_return_agent import DailyLogReturn_Agent
    from daily_return_agent import DailyReturn_Agent
    
    AGENTS.update({
        "CumulativeReturn_Agent": CumulativeReturn_Agent,
        "DailyLogReturn_Agent": DailyLogReturn_Agent,
        "DailyReturn_Agent": DailyReturn_Agent,
    })
except ImportError as e:
    print(f"Warning: Could not import price-derived agents: {e}")

try:
    # Trend agents
    from trend.vortex_agent import Vortex_Agent
    from trend.aroon_agent import Aroon_Agent
    from trend.stc_agent import STC_Agent
    from trend.ichimoku_agent import Ichimoku_Agent
    from trend.kst_agent import KST_Agent
    from trend.dpo_agent import DPO_Agent
    from trend.cci_agent import CCI_Agent
    from trend.mass_index_agent import MassIndex_Agent
    from trend.trix_agent import TRIX_Agent
    
    AGENTS.update({
        "Vortex_Agent": Vortex_Agent,
        "Aroon_Agent": Aroon_Agent,
        "STC_Agent": STC_Agent,
        "Ichimoku_Agent": Ichimoku_Agent,
        "KST_Agent": KST_Agent,
        "DPO_Agent": DPO_Agent,
        "CCI_Agent": CCI_Agent,
        "MassIndex_Agent": MassIndex_Agent,
        "TRIX_Agent": TRIX_Agent,
    })
except ImportError as e:
    print(f"Warning: Could not import trend agents: {e}")

try:
    # Momentum agents
    from momentum.rsi_agent import RSI_Agent
    from momentum.pvo_agent import PVO_Agent
    from momentum.ppo_agent import PPO_Agent
    from momentum.roc_agent import ROC_Agent
    from momentum.awesome_oscillator_agent import AwesomeOscillator_Agent
    from momentum.williamsr_agent import WilliamsR_Agent
    from momentum.stochosc_agent import StochOsc_Agent
    from momentum.ultimate_oscillator_agent import UltimateOsc_Agent
    from momentum.tsi_agent import TSI_Agent
    from momentum.stochrsi_agent import StochRSI_Agent
    
    AGENTS.update({
        "RSI_Agent": RSI_Agent,
        "PVO_Agent": PVO_Agent,
        "PPO_Agent": PPO_Agent,
        "ROC_Agent": ROC_Agent,
        "AwesomeOscillator_Agent": AwesomeOscillator_Agent,
        "WilliamsR_Agent": WilliamsR_Agent,
        "StochOsc_Agent": StochOsc_Agent,
        "UltimateOsc_Agent": UltimateOsc_Agent,
        "TSI_Agent": TSI_Agent,
        "StochRSI_Agent": StochRSI_Agent,
    })
except ImportError as e:
    print(f"Warning: Could not import momentum agents: {e}")

try:
    # Import from money-flow directory (with hyphen)
    money_flow_path = current_dir / "money-flow"
    sys.path.insert(0, str(money_flow_path))
    
    from adi_agent import ADI_Agent
    from eom_agent import EOM_Agent
    from vwap_agent import VWAP_Agent
    from obv_agent import OBV_Agent
    from nvi_agent import NVI_Agent
    from vpt_agent import VPT_Agent
    from force_index_agent import ForceIndex_Agent
    from cmf_agent import CMF_Agent
    from mfi_agent import MFI_Agent
    
    AGENTS.update({
        "ADI_Agent": ADI_Agent,
        "EOM_Agent": EOM_Agent,
        "VWAP_Agent": VWAP_Agent,
        "OBV_Agent": OBV_Agent,
        "NVI_Agent": NVI_Agent,
        "VPT_Agent": VPT_Agent,
        "ForceIndex_Agent": ForceIndex_Agent,
        "CMF_Agent": CMF_Agent,
        "MFI_Agent": MFI_Agent,
    })
except ImportError as e:
    print(f"Warning: Could not import money-flow agents: {e}")

# Clean up sys.path
if str(current_dir) in sys.path:
    sys.path.remove(str(current_dir))
if str(current_dir / "price-derived") in sys.path:
    sys.path.remove(str(current_dir / "price-derived"))
if str(current_dir / "money-flow") in sys.path:
    sys.path.remove(str(current_dir / "money-flow"))

__all__ = list(AGENTS.keys()) + ["AGENTS"] 