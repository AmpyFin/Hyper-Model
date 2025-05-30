"""
strategies package

Automatic discovery of every *.py file in this directory that
contains at least one class with both `.fit()` and `.predict()` methods.

This module provides a collection of trading strategies, agents, and indicators 
to be used with the AmpyFin platform.

Each submodule provides a different category of strategies.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Dict, Type

# Import standard submodules (without hyphens)
from . import market_regime_agents
from . import pattern_recognition_agents
from . import volume_profile_agents
from . import orderflow_analysis_agents
from . import market_breadth_agents
from . import mathematician_agents
from . import computer_science_agents

# Import hyphenated modules using importlib
try:
    micro_behavior_ta_lib = importlib.import_module("strategies.micro-behavior_ta-lib")
except ImportError:
    micro_behavior_ta_lib = None

try:
    linear_regression_sabino_ta_lib = importlib.import_module("strategies.linear_regression_sabino-ta-lib")
except ImportError:
    linear_regression_sabino_ta_lib = None

try:
    linear_regression_ta_lib = importlib.import_module("strategies.linear_regression_ta-lib")
except ImportError:
    linear_regression_ta_lib = None

__all__ = [
    "discover",
    "market_regime_agents",
    "pattern_recognition_agents",
    "volume_profile_agents",
    "orderflow_analysis_agents",
    "market_breadth_agents",
    "mathematician_agents",
    "computer_science_agents", 
    "micro_behavior_ta_lib",
    "linear_regression_sabino_ta_lib",
    "linear_regression_ta_lib",
]

# ------------------------------------------------------------------ #
# public helper
# ------------------------------------------------------------------ #
def discover() -> Dict[str, Type]:
    """
    Return {class_name: class_object} for all strategy classes found.
    Recursively searches in the main strategies directory and all subdirectories at any depth.
    """
    classes: Dict[str, Type] = {}
    package_path = Path(__file__).parent
    
    # Start recursive discovery from the main strategies directory
    _discover_recursive(package_path, classes)
    
    return classes


def _discover_recursive(directory: Path, classes: Dict[str, Type]) -> None:
    """
    Recursively discovers strategies in directories at any depth.
    """
    # Process Python modules in the current directory
    _discover_in_directory(directory, classes)
    
    # Recursively process all subdirectories
    for subdir in directory.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("_") and not subdir.name.startswith("."):
            _discover_recursive(subdir, classes)


def _discover_in_directory(directory: Path, classes: Dict[str, Type]) -> None:
    """
    Discovers strategies in a specific directory and adds them to the classes dict.
    """
    package_name = directory.relative_to(Path(__file__).parent.parent).as_posix().replace("/", ".")
    
    for mod_info in pkgutil.iter_modules([str(directory)]):
        if mod_info.ispkg or mod_info.name.startswith("_"):
            continue  # skip sub-packages and private files
        
        try:
            module_path = f"{package_name}.{mod_info.name}"
            module = importlib.import_module(module_path)
            new_classes = _extract_strategies(module)
            if new_classes:
                print(f"Found {len(new_classes)} strategies in {module_path}")
            classes.update(new_classes)
        except Exception as e:
            print(f"Warning: Failed to import {mod_info.name} from {directory}: {e}")


# ------------------------------------------------------------------ #
# internal helper
# ------------------------------------------------------------------ #
def _extract_strategies(module: ModuleType) -> Dict[str, Type]:
    """Pick every class that defines both 'fit' and 'predict'."""
    found: Dict[str, Type] = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue  # skip re-exported / imported classes
        if callable(getattr(obj, "fit", None)) and callable(getattr(obj, "predict", None)):
            found[name] = obj
    return found
