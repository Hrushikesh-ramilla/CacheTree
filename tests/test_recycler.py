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

