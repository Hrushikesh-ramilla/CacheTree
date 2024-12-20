"""
Principled Step-Decomposition Monte Carlo Tree Search (MCTS) for Reasoning.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Dict, List, Optional, Tuple, Set


class NodeStatus(enum.Enum):
    ACTIVE = "active"
    PRUNED = "pruned"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_FAILURE = "terminal_failure"


@dataclasses.dataclass
class UCTConfig:
    c_puct: float = 1.41421356  # sqrt(2)
    normalize_q: bool = True
    epsilon: float = 1e-6
    temperature: float = 1.0
    gamma_discount: float = 0.98  # Path-efficiency discount per step


@dataclasses.dataclass
class StepNode:
    node_id: int
    parent_id: Optional[int]
    thought_step: str
    tokens: List[int]
    cumulative_tokens: List[int]
    depth: int = 0
    visits: int = 0
    total_value: float = 0.0
    prior_prob: float = 1.0
    step_reward: float = 0.5
    status: NodeStatus = NodeStatus.ACTIVE
    is_terminal: bool = False
    solution: Optional[str] = None
    children_ids: List[int] = dataclasses.field(default_factory=list)

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def is_pruned(self) -> bool:
        return self.status == NodeStatus.PRUNED

    @property
    def is_expandable(self) -> bool:
        return not self.is_terminal and self.status == NodeStatus.ACTIVE

