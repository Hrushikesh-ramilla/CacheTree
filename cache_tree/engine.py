"""
Test-Time Compute Engine: Step-Level MCTS with PRM Verification & KV-Cache Recycling.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Tuple, Any

import torch
from .tree import ThoughtTree, StepNode, NodeStatus, UCTConfig
from .prm import ProcessRewardModel, RuleBasedMathPRM, StepVerdict, StepEvaluation
from .recycler import TreeKVCacheManager, TokenRecyclingBuffer, RecyclingStats


@dataclasses.dataclass
class EngineConfig:
    max_search_rounds: int = 15
    branching_factor: int = 3
    max_tree_depth: int = 8
    pruning_threshold: float = 0.20
    c_puct: float = 1.41421356
    early_stop_on_success: bool = True


@dataclasses.dataclass
class CandidateStepProposal:
    thought_text: str
    delta_tokens: List[int]
    prior_prob: float = 1.0
    key_tensor: Optional[torch.Tensor] = None
    value_tensor: Optional[torch.Tensor] = None
    dynamic_cache: Optional[Any] = None


@dataclasses.dataclass
class ReasoningResult:
    problem: str
    best_solution: Optional[str]
    best_trajectory: List[str]
    total_search_rounds: int
    total_nodes_expanded: int
    pruned_branches_count: int
    terminal_solutions_found: int
    recycling_metrics: Dict[str, Any]
    time_elapsed_seconds: float
    tree: Optional[ThoughtTree] = None
    consensus_distribution: Dict[str, float] = dataclasses.field(default_factory=dict)
    consensus_confidence: float = 0.0
    margin_of_victory: float = 0.0

