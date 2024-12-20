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


class ThoughtTree:
    """
    Step-Level Monte Carlo Tree Search graph.
    Maintains rigorous invariants:
    1. Acyclic tree structure with single root.
    2. Monotonically non-decreasing visit counts.
    3. Numerically stabilized min-max normalized Q-values in [0, 1].
    4. Deterministic argmax UCT selection with zero division safeguards.
    """

    def __init__(self, root_prompt: str, root_tokens: Optional[List[int]] = None, config: Optional[UCTConfig] = None):
        self.config = config or UCTConfig()
        self.nodes: Dict[int, StepNode] = {}
        self.next_node_id: int = 0
        self.q_min: float = 0.0
        self.q_max: float = 1.0

        # Initialize root node
        tokens = root_tokens if root_tokens is not None else []
        root_node = StepNode(
            node_id=self.next_node_id,
            parent_id=None,
            thought_step=root_prompt,
            tokens=tokens,
            cumulative_tokens=tokens,
            depth=0,
            visits=1,
            total_value=0.5,
            prior_prob=1.0,
            step_reward=1.0,
            status=NodeStatus.ACTIVE,
            is_terminal=False,
            solution=None,
        )
        self.nodes[root_node.node_id] = root_node
        self.root_id = root_node.node_id
        self.next_node_id += 1

    def get_node(self, node_id: int) -> StepNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node id {node_id} does not exist in ThoughtTree.")
        return self.nodes[node_id]

    def add_child(
        self,
        parent_id: int,
        thought_step: str,
        delta_tokens: List[int],
        prior_prob: float = 1.0,
        step_reward: float = 0.5,
        is_terminal: bool = False,
        solution: Optional[str] = None,
        status: NodeStatus = NodeStatus.ACTIVE,
    ) -> StepNode:
        parent = self.get_node(parent_id)
        if parent.is_terminal:
            raise ValueError(f"Cannot add child to terminal node {parent_id}.")

        child_id = self.next_node_id
        self.next_node_id += 1

        cumulative = list(parent.cumulative_tokens) + list(delta_tokens)

        child = StepNode(
            node_id=child_id,
            parent_id=parent_id,
            thought_step=thought_step,
            tokens=delta_tokens,
            cumulative_tokens=cumulative,
            depth=parent.depth + 1,
            visits=0,
            total_value=0.0,
            prior_prob=max(1e-4, min(1.0, prior_prob)),
            step_reward=step_reward,
            status=status,
            is_terminal=is_terminal,
            solution=solution,
        )

        self.nodes[child_id] = child
        parent.children_ids.append(child_id)
        return child

