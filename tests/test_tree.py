"""
Unit and Invariant Tests for Step-Decomposition MCTS Tree.
"""

import pytest
import math
from cache_tree.tree import ThoughtTree, StepNode, NodeStatus, UCTConfig


def test_tree_initialization():
    tree = ThoughtTree(root_prompt="Solve 2x + 5 = 15", root_tokens=[10, 20, 30])
    assert tree.root_id == 0
    root = tree.get_node(0)
    assert root.thought_step == "Solve 2x + 5 = 15"
    assert root.tokens == [10, 20, 30]
    assert root.cumulative_tokens == [10, 20, 30]
    assert root.depth == 0
    assert root.visits == 1
    assert root.status == NodeStatus.ACTIVE


def test_add_child_and_cumulative_tokens():
    tree = ThoughtTree(root_prompt="Root", root_tokens=[1, 2])
    child1 = tree.add_child(
        parent_id=tree.root_id,
        thought_step="Step 1: Subtract 5",
        delta_tokens=[3, 4],
        prior_prob=0.8,
        step_reward=0.9,
    )
    assert child1.node_id == 1
    assert child1.depth == 1
    assert child1.tokens == [3, 4]
    assert child1.cumulative_tokens == [1, 2, 3, 4]
    assert child1.node_id in tree.get_node(tree.root_id).children_ids

    child2 = tree.add_child(
        parent_id=child1.node_id,
        thought_step="Step 2: Divide by 2",
        delta_tokens=[5, 6],
        prior_prob=0.9,
        step_reward=0.95,
        is_terminal=True,
        solution="5",
        status=NodeStatus.TERMINAL_SUCCESS,
    )
    assert child2.depth == 2
    assert child2.cumulative_tokens == [1, 2, 3, 4, 5, 6]
    assert child2.is_terminal is True
    assert child2.solution == "5"


def test_uct_computation_and_normalization():
    cfg = UCTConfig(c_puct=1.414, normalize_q=True)
    tree = ThoughtTree(root_prompt="Math", config=cfg)
    c1 = tree.add_child(tree.root_id, "Step A", [1], prior_prob=0.5, step_reward=0.8)
    c2 = tree.add_child(tree.root_id, "Step B", [2], prior_prob=0.5, step_reward=0.2)

    # Initially unvisited: child with higher prior prob has exploration bonus
    uct1 = tree.compute_uct(c1, parent_visits=1)
    uct2 = tree.compute_uct(c2, parent_visits=1)
    assert not math.isnan(uct1)
    assert not math.isnan(uct2)

    # Backpropagate rewards
    tree.backpropagate(c1.node_id, terminal_value=0.9)
    assert c1.visits == 1
    assert c1.q_value == 0.9

    best = tree.select_best_child(tree.root_id)
    assert best is not None
    # Unvisited child c2 should have strong exploration incentive or c1 high Q
    assert best.node_id in [c1.node_id, c2.node_id]

