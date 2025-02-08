"""
Tests for live DynamicCache integration in TreeKVCacheManager.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from cache_tree.recycler import (
    TreeKVCacheManager,
    KVCacheEntry,
    clone_dynamic_cache,
)


def test_tree_kv_manager_with_dynamic_cache():
    mgr = TreeKVCacheManager()

    # Create a real DynamicCache with 2 transformer layers
    cache = DynamicCache()
    # Layer 0: (batch=1, heads=4, seq_len=16, dim=32)
    k0 = torch.randn(1, 4, 16, 32)
    v0 = torch.randn(1, 4, 16, 32)
    cache.update(k0, v0, layer_idx=0)

    # Layer 1
    k1 = torch.randn(1, 4, 16, 32)
    v1 = torch.randn(1, 4, 16, 32)
    cache.update(k1, v1, layer_idx=1)

    assert cache.get_seq_length() == 16

    # Store cache at root node (node 0)
    mgr.store_cache(node_id=0, token_count=16, dynamic_cache=cache)

    entry = mgr.get_cache(0)
    assert entry is not None
    assert entry.dynamic_cache is not None
    assert entry.dynamic_cache.get_seq_length() == 16

    # Test clone and crop to length 10
    cropped = mgr.clone_and_crop_dynamic_cache(source_node_id=0, target_length=10)
    assert cropped is not None
    assert cropped.get_seq_length() == 10

    # Original parent cache must remain unmodified (seq_len = 16)
    assert entry.dynamic_cache.get_seq_length() == 16
    assert entry.dynamic_cache.key_cache[0].shape[-2] == 16
    assert cropped.key_cache[0].shape[-2] == 10
