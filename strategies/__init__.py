"""
strategies package

Automatic discovery of every *.py file in this directory that
contains at least one class with both `.fit()` and `.predict()` methods.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Dict, Type

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
