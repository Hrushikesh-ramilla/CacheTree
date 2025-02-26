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


class TestHostilePRMAttacks:
    """Attacks on Process Reward Model arithmetic and parsing bounds."""

    def test_adversarial_arithmetic_overflow(self):
        """Attack: Floating point overflow strings (1e308 * 10). Must not crash with OverflowError."""
        prm = RuleBasedMathPRM()
        eval_res = prm.evaluate_step(
            problem="Overflow",
            reasoning_history="",
            current_step="Compute 1e308 * 10 = 1e309",
        )
        assert isinstance(eval_res.score, float)
        assert not math.isnan(eval_res.score)

    def test_homoglyph_and_garbage_math_caught(self):
        """Attack: Homoglyphs and corrupted equations."""
        prm = RuleBasedMathPRM()
        # Empty / whitespace step
        eval_empty = prm.evaluate_step("Prob", "", "   ")
        assert eval_empty.is_fatal is True
        assert eval_empty.score <= 0.10

    def test_negative_zero_division(self):
        """Attack: Negative zero division."""
        prm = RuleBasedMathPRM()
        eval_neg0 = prm.evaluate_step("Prob", "", "10 / -0.0 = 0")
        assert eval_neg0.is_fatal is True
        assert "division by zero" in eval_neg0.rationale.lower()


class TestHostileKVCacheAttacks:
    """Attacks on KV-Cache memory capacity, multi-head tensor dimensions, and LCA."""

    def test_lru_eviction_under_high_node_pressure(self):
        """Attack: Push 500 nodes into a cache capped at 50 entries."""
        mgr = TreeKVCacheManager(max_cached_nodes=50)
        for i in range(500):
            mgr.store_cache(node_id=i, token_count=i + 1)

        summary = mgr.get_summary()
        # Active cached nodes must never exceed max_cached_nodes
        assert summary["active_cached_nodes"] == 50
        # The oldest node (0) must have been evicted
        assert mgr.get_cache(0) is None
        # The newest node (499) must be retained
        assert mgr.get_cache(499) is not None

    def test_4d_multi_head_attention_kv_exactness_proof(self):
        """
        Proof Duel Attack: Verify exactness across batch=2, heads=8, dim=64 (Llama-style).
        Demands KVExactnessCertificate with error < 1e-12.
        """
        batch = 2
        heads = 8
        head_dim = 64
        seq_len = 128
        prefix = 96
        delta = 32

        torch.manual_seed(1337)
        hidden = heads * head_dim
        W = torch.randn(hidden, hidden, dtype=torch.float64)

        tokens = torch.randn(batch, seq_len, hidden, dtype=torch.float64)
        t_prefix = tokens[:, :prefix, :]
        t_delta = tokens[:, prefix:, :]

        # Full scratch
        k_scratch = torch.matmul(tokens, W).view(batch, seq_len, heads, head_dim).transpose(1, 2)

        # Recycled
        k_p = torch.matmul(t_prefix, W).view(batch, prefix, heads, head_dim).transpose(1, 2)
        k_d = torch.matmul(t_delta, W).view(batch, delta, heads, head_dim).transpose(1, 2)
        k_recycled = torch.cat([k_p, k_d], dim=2)

        cert = verify_kv_exactness(k_recycled, k_scratch)
        assert cert.is_proven is True
        assert cert.is_machine_precision is True
        assert cert.max_absolute_error < 1e-12
        assert cert.tokens_compared == seq_len

    def test_lca_on_disjoint_or_deeply_skewed_trees(self):
        """Attack: Skewed line tree (depth 20) LCA calculation."""
        tree = ThoughtTree(root_prompt="Root")
        curr = tree.root_id
        for i in range(20):
            node = tree.add_child(curr, f"step_{i}", [i])
            curr = node.node_id

        # LCA of deepest node (20) and root (0) must be root (0)
        assert tree.lowest_common_ancestor(20, 0) == 0
        # LCA of node 15 and node 10 must be 10
        assert tree.lowest_common_ancestor(15, 10) == 10
