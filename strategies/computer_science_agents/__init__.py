"""
Computer Science Agents
~~~~~~~~~~~~~~~~~~~~~
This module implements trading agents based on the theories and principles of notable
computer scientists throughout history, applying their innovations to financial markets.

Each agent models market behavior and makes predictions using concepts from a specific
computer scientist's most influential work.
"""

from typing import Dict, Type

from .hopper_compiler_agent import HopperAgent
from .dijkstra_pathfinding_agent import DijkstraAgent
from .knuth_analysis_algorithms_agent import KnuthAgent
from .von_neumann_architecture_agent import VonNeumannAgent
from .thompson_sampling_agent import ThompsonAgent
from .shannon_information_theory_agent import ShannonAgent
from .backus_functional_agent import BackusAgent
from .ritchie_systems_language_agent import RitchieAgent
from .liskov_substitution_agent import LiskovAgent
from .lamport_consensus_agent import LamportAgent
from .torvalds_distributed_kernel_agent import TorvaldsAgent
from .hamming_error_correction_agent import HammingAgent
from .berners_lee_web_network_agent import BernersLeeAgent
from .moore_scaling_law_agent import MooreAgent

# TODO: Implement the remaining agents
# from .kay_object_oriented_agent import KayAgent
# from .hopcroft_automata_agent import HopcroftAgent
# from .rivest_cryptography_agent import RivestAgent
# from .cerf_internet_protocol_agent import CerfAgent
# from .engelbart_human_interface_agent import EngelbartAgent
# from .brooks_project_mgmt_agent import BrooksAgent
# from .mccarthy_ai_agent import McCarthyAgent

__all__ = [
    "HopperAgent",
    "DijkstraAgent",
    "KnuthAgent",
    "VonNeumannAgent",
    "ThompsonAgent",
    "ShannonAgent",
    "BackusAgent",
    "RitchieAgent",
    "LiskovAgent",
    "LamportAgent",
    "TorvaldsAgent",
    "HammingAgent",
    "BernersLeeAgent", 
    "MooreAgent"
    # TODO: Add the remaining agents when implemented
    # "KayAgent",
    # "HopcroftAgent",
    # "RivestAgent",
    # "CerfAgent",
    # "EngelbartAgent",
    # "BrooksAgent",
    # "McCarthyAgent"
]

AGENTS: Dict[str, Type] = {
    "HopperAgent": HopperAgent,
    "DijkstraAgent": DijkstraAgent,
    "KnuthAgent": KnuthAgent,
    "VonNeumannAgent": VonNeumannAgent,
    "ThompsonAgent": ThompsonAgent,
    "ShannonAgent": ShannonAgent,
    "BackusAgent": BackusAgent,
    "RitchieAgent": RitchieAgent,
    "LiskovAgent": LiskovAgent,
    "LamportAgent": LamportAgent,
    "TorvaldsAgent": TorvaldsAgent,
    "HammingAgent": HammingAgent,
    "BernersLeeAgent": BernersLeeAgent,
    "MooreAgent": MooreAgent
    # TODO: Add the remaining agents when implemented
    # "KayAgent": KayAgent,
    # "HopcroftAgent": HopcroftAgent,
    # "RivestAgent": RivestAgent,
    # "CerfAgent": CerfAgent,
    # "EngelbartAgent": EngelbartAgent,
    # "BrooksAgent": BrooksAgent,
    # "McCarthyAgent": McCarthyAgent
} 