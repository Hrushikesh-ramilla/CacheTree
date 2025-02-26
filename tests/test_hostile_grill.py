"""
Hostile Red-Team Grill and Boundary Stress Test Suite for CacheTree.
Mounts aggressive adversarial attacks demanding proof certificates for defense.
"""

from __future__ import annotations

import math
import pytest
import torch

from cache_tree.tree import ThoughtTree, StepNode, NodeStatus, UCTConfig
from cache_tree.prm import RuleBasedMathPRM, StepVerdict
from cache_tree.recycler import TreeKVCacheManager, TokenRecyclingBuffer
from cache_tree.engine import (
    TestTimeComputeEngine,
    EngineConfig,
    CandidateStepProposal,
)
from cache_tree.proofs import (
    verify_tree_consistency,
    verify_kv_exactness,
    TreeConsistencyCertificate,
    KVExactnessCertificate,
)


class TestHostileTreeAndUCTAttacks:
    """Attacks on MCTS tree invariants and UCT numerical stability."""

    def test_uct_zero_division_resilience(self):
        """Attack: Parent visits 0 or child visits 0. Must never produce NaN or crash."""
        cfg = UCTConfig(c_puct=1.414, normalize_q=True, epsilon=1e-6)
        tree = ThoughtTree(root_prompt="Attack", config=cfg)
        root = tree.get_node(tree.root_id)
        root.visits = 0  # Force zero parent visits

        c1 = tree.add_child(tree.root_id, "Child 1", [1], prior_prob=0.0)
        c1.visits = 0
        c1.total_value = 0.0

        score = tree.compute_uct(c1, parent_visits=0)
        assert not math.isnan(score)
        assert not math.isinf(score)
        assert score >= 0.0

    def test_uct_extreme_negative_q_bounds(self):
        """Attack: Q-values are negative or extreme scale. Normalization must clamp cleanly."""
        tree = ThoughtTree(root_prompt="Attack")
        c1 = tree.add_child(tree.root_id, "Child 1", [1])
        c1.visits = 10
        c1.total_value = -1000.0  # extreme negative Q

        tree.q_min = -100.0
        tree.q_max = 10.0

        score = tree.compute_uct(c1, parent_visits=20)
        assert not math.isnan(score)
        assert not math.isinf(score)
        assert score >= 0.0

    def test_cycle_detection_in_tree_consistency(self):
        """Attack: Adversary crafts a cycle in parent pointers. Proof must detect and reject."""
        tree = ThoughtTree(root_prompt="Root")
        c1 = tree.add_child(tree.root_id, "1", [1])
        c2 = tree.add_child(c1.node_id, "2", [2])
        # Forge cycle: root's parent becomes c2
        tree.get_node(tree.root_id).parent_id = c2.node_id

        cert = verify_tree_consistency(tree)
        assert cert.is_valid is False
        assert cert.is_acyclic is False
        assert cert.is_topologically_ordered is False

    def test_pruned_subtrees_never_resurrected(self):
        """Attack: Attempt to traverse or select children through a pruned branch."""
        tree = ThoughtTree(root_prompt="Root")
        c1 = tree.add_child(tree.root_id, "Bad", [1])
        c1_sub = tree.add_child(c1.node_id, "Sub", [2])
        tree.prune_branch(c1.node_id)

        # compute_uct on pruned child must return -inf
        assert tree.compute_uct(c1, parent_visits=10) == -float("inf")
        # Traversal from root must not pick pruned child
        leaf, path = tree.traverse_to_leaf()
        assert c1.node_id not in path
        assert c1_sub.node_id not in path

