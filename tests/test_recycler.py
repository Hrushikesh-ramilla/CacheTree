"""
Tests for TreeKVCacheManager and TokenRecyclingBuffer.
"""

import pytest
import torch
from cache_tree.tree import ThoughtTree
from cache_tree.recycler import TreeKVCacheManager, TokenRecyclingBuffer


def test_kv_cache_storage_and_retrieval():
    mgr = TreeKVCacheManager()
    mgr.store_cache(node_id=0, token_count=100)
    entry = mgr.get_cache(node_id=0)
    assert entry is not None
    assert entry.token_count == 100
    assert mgr.get_cache(node_id=999) is None


def test_token_recycling_savings_ratio():
    mgr = TreeKVCacheManager()
    # Root prompt: 100 tokens
    mgr.store_cache(node_id=0, token_count=100)

    # Branch 1: child 1 adds 20 tokens
    mgr.register_step_transition(
        parent_id=0,
        child_id=1,
        parent_token_count=100,
        delta_token_count=20,
    )
    # Branch 2: child 2 adds 20 tokens (sibling from same root)
    mgr.register_step_transition(
        parent_id=0,
        child_id=2,
        parent_token_count=100,
        delta_token_count=20,
    )

    stats = mgr.stats
    # Without recycling: Child 1 needed 120 tokens, Child 2 needed 120 tokens -> Total 240
    assert stats.total_tokens_without_recycling == 240
    # With recycling: Child 1 computed 20, Child 2 computed 20 -> Total 40
    assert stats.tokens_computed_with_recycling == 40
    # Tokens saved: 240 - 40 = 200
    assert stats.tokens_saved == 200
    # Efficiency: 200 / 240 = 83.33%
    assert pytest.approx(stats.recycling_efficiency, rel=1e-3) == (200 / 240)
    assert stats.cache_hits == 2


def test_deep_tree_recycling_efficiency():
    mgr = TreeKVCacheManager()
    tree = ThoughtTree(root_prompt="Root", root_tokens=list(range(50)))
    mgr.store_cache(tree.root_id, token_count=50)

    # Build a balanced binary tree of depth 3
    # Level 1: 2 nodes (delta=10)
    # Level 2: 4 nodes (delta=10)
    current_parents = [tree.root_id]
    for depth in range(1, 4):
        next_parents = []
        for p in current_parents:
            parent_node = tree.get_node(p)
            for branch in range(2):
                c = tree.add_child(p, f"d{depth}_b{branch}", list(range(10)))
                mgr.register_step_transition(
                    parent_id=p,
                    child_id=c.node_id,
                    parent_token_count=len(parent_node.cumulative_tokens),
                    delta_token_count=10,
                )
                next_parents.append(c.node_id)
        current_parents = next_parents

    summary = mgr.get_summary()
    assert summary["total_forward_passes"] == 2 + 4 + 8  # 14
    # With deep prefix sharing, efficiency must be well over 70%
    assert summary["recycling_efficiency_pct"] > 70.0


def test_rewind_to_lca():
    mgr = TreeKVCacheManager()
    tree = ThoughtTree(root_prompt="Root", root_tokens=list(range(20)))

    # Root -> Node 1 -> Node 2
    # Root -> Node 3
    n1 = tree.add_child(tree.root_id, "1", list(range(10)))
    n2 = tree.add_child(n1.node_id, "2", list(range(10)))
    n3 = tree.add_child(tree.root_id, "3", list(range(10)))

    lca_id, preserved_tokens = mgr.rewind_to_lca(
        current_node_id=n2.node_id,
        target_node_id=n3.node_id,
        tree=tree,
    )

    assert lca_id == tree.root_id
    assert preserved_tokens == 20
    assert mgr.active_node_id == tree.root_id


def test_token_recycling_buffer():
    buf = TokenRecyclingBuffer(vocab_size=1000, top_k=3)
    # Observe sequence where 10 is followed by 20 twice and 30 once
    buf.update_from_sequence([10, 20, 10, 20, 10, 30])
    preds = buf.predict_next_tokens(10)
    assert preds[0] == 20
    assert preds[1] == 30
    assert len(preds) == 2
