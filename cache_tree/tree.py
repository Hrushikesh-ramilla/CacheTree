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

    def compute_uct(self, child: StepNode, parent_visits: int) -> float:
        """
        Computes normalized Upper Confidence Bound for Trees:
        UCT(s, a) = normalized_Q(s, a) + c_puct * P(s, a) * sqrt(ln(N_parent)) / (1 + N_child)
        """
        if child.is_pruned:
            return -float("inf")

        # Normalize Q value to [0, 1] using tree-wide bounds
        if self.config.normalize_q and self.q_max > self.q_min + self.config.epsilon:
            norm_q = (child.q_value - self.q_min) / (self.q_max - self.q_min + self.config.epsilon)
            norm_q = max(0.0, min(1.0, norm_q))
        else:
            norm_q = child.q_value

        # Exploration bonus
        exploration = (
            self.config.c_puct
            * child.prior_prob
            * (math.sqrt(max(1.0, math.log(max(1, parent_visits)))) / (1.0 + child.visits))
        )

        return norm_q + exploration

    def select_best_child(self, parent_id: int) -> Optional[StepNode]:
        """
        Selects the child with the highest UCT score among active (non-pruned) children.
        Deterministic tie-breaking by highest prior prob.
        """
        parent = self.get_node(parent_id)
        active_children = [
            self.nodes[cid] for cid in parent.children_ids if not self.nodes[cid].is_pruned
        ]
        if not active_children:
            return None

        best_score = -float("inf")
        best_child = None

        for child in active_children:
            score = self.compute_uct(child, parent.visits)
            if score > best_score:
                best_score = score
                best_child = child
            elif math.isclose(score, best_score, rel_tol=1e-9) and best_child is not None:
                # Tie-breaking
                if child.prior_prob > best_child.prior_prob:
                    best_child = child

        return best_child

    def traverse_to_leaf(self) -> Tuple[StepNode, List[int]]:
        """
        Traverses the tree from root using UCT until an unexpanded or terminal leaf is reached.
        Returns the leaf node and the sequence of node_ids traversed.
        """
        path: List[int] = [self.root_id]
        curr = self.get_node(self.root_id)

        while curr.children_ids and not curr.is_terminal and curr.status == NodeStatus.ACTIVE:
            best_child = self.select_best_child(curr.node_id)
            if best_child is None:
                # All children are pruned; curr is effectively a dead-end
                break
            if best_child.visits == 0:
                # Unvisited child, stop here to evaluate/rollout
                curr = best_child
                path.append(curr.node_id)
                break
            curr = best_child
            path.append(curr.node_id)

        return curr, path

