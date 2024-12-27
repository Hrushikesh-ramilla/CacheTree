"""
Tests for Path-Efficiency Discounting.
"""

import pytest
from cache_tree.tree import ThoughtTree, NodeStatus, UCTConfig


def test_gamma_discounting_in_backpropagation():
    """
    Verifies that deeper paths receive exponentially discounted values:
    V(s_t) = V_terminal * gamma^(leaf_depth - curr_depth).
    """
    gamma = 0.90
    cfg = UCTConfig(gamma_discount=gamma)
    tree = ThoughtTree(root_prompt="Root", config=cfg)

    # Root (depth 0) -> c1 (depth 1) -> c2 (depth 2) -> c3 (depth 3, terminal)
    c1 = tree.add_child(tree.root_id, "1", [1])
    c2 = tree.add_child(c1.node_id, "2", [2])
    c3 = tree.add_child(c2.node_id, "3", [3], is_terminal=True, solution="42")

    # Backpropagate terminal value = 1.0 from c3
    tree.backpropagate(c3.node_id, terminal_value=1.0)

    # c3 (depth 3, dist 0): value = 1.0 * (0.9^0) = 1.0
    assert pytest.approx(c3.total_value, rel=1e-4) == 1.0

    # c2 (depth 2, dist 1): value = 1.0 * (0.9^1) = 0.9
    assert pytest.approx(c2.total_value, rel=1e-4) == 0.90

    # c1 (depth 1, dist 2): value = 1.0 * (0.9^2) = 0.81
    assert pytest.approx(c1.total_value, rel=1e-4) == 0.81

    # root (depth 0, dist 3): initial 0.5 + 1.0 * (0.9^3) = 0.5 + 0.729 = 1.229
    assert pytest.approx(tree.get_node(tree.root_id).total_value, rel=1e-4) == 0.5 + (0.9 ** 3)
