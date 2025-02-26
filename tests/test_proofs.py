"""
Tests for Formal Proof Certificates in CacheTree.
"""

import pytest
import torch
from cache_tree.tree import ThoughtTree, NodeStatus
from cache_tree.proofs import (
    verify_tree_consistency,
    verify_kv_exactness,
    TreeConsistencyCertificate,
    KVExactnessCertificate,
)


def test_tree_consistency_certificate_on_valid_tree():
    tree = ThoughtTree(root_prompt="Proof Test", root_tokens=[1, 2, 3])
    c1 = tree.add_child(tree.root_id, "Step 1", [4, 5])
    c2 = tree.add_child(tree.root_id, "Step 2", [6, 7])
    c1_sub = tree.add_child(c1.node_id, "Step 1.1", [8, 9])

    tree.backpropagate(c1_sub.node_id, terminal_value=0.8)
    tree.backpropagate(c2.node_id, terminal_value=0.4)

    cert = verify_tree_consistency(tree)
    assert isinstance(cert, TreeConsistencyCertificate)
    assert cert.is_valid is True
    assert cert.is_acyclic is True
    assert cert.is_topologically_ordered is True
    assert cert.visit_conservation_holds is True
    assert cert.q_bounds_valid is True
    assert cert.total_nodes == 4


def test_tree_consistency_detects_invalid_q():
    tree = ThoughtTree(root_prompt="Root")
    c1 = tree.add_child(tree.root_id, "Step 1", [1])
    # Manually corrupt total_value to make q_value > 1.0
    c1.visits = 1
    c1.total_value = 5.0  # q = 5.0 > 1.0

    cert = verify_tree_consistency(tree)
    assert cert.is_valid is False
    assert cert.q_bounds_valid is False


def test_kv_exactness_certificate_machine_precision():
    """
    Proves that prefix KV caching matches full recomputation
    to machine precision (< 1e-12) for attention key/value tensors.
    """
    batch_size = 1
    num_heads = 4
    head_dim = 32
    prefix_len = 32
    delta_len = 32
    total_len = prefix_len + delta_len

    # Simulate key/value projection weights W_k, W_v
    hidden_dim = num_heads * head_dim
    torch.manual_seed(42)
    W_k = torch.randn(hidden_dim, hidden_dim, dtype=torch.float64)

    # Input token embeddings
    all_embeddings = torch.randn(batch_size, total_len, hidden_dim, dtype=torch.float64)
    prefix_emb = all_embeddings[:, :prefix_len, :]
    delta_emb = all_embeddings[:, prefix_len:, :]

    # 1. Scratch computation: full sequence passed through W_k
    k_scratch = torch.matmul(all_embeddings, W_k)
    k_scratch = k_scratch.view(batch_size, total_len, num_heads, head_dim).transpose(1, 2)

    # 2. Recycled computation: prefix passed through W_k, cached, and delta concatenated
    k_prefix = torch.matmul(prefix_emb, W_k)
    k_prefix = k_prefix.view(batch_size, prefix_len, num_heads, head_dim).transpose(1, 2)

    k_delta = torch.matmul(delta_emb, W_k)
    k_delta = k_delta.view(batch_size, delta_len, num_heads, head_dim).transpose(1, 2)

    k_recycled = torch.cat([k_prefix, k_delta], dim=2)

    # Verify mathematical exactness
    cert = verify_kv_exactness(k_recycled, k_scratch, tolerance=1e-10)
    assert isinstance(cert, KVExactnessCertificate)
    assert cert.is_proven is True
    assert cert.is_machine_precision is True
    assert cert.max_absolute_error < 1e-12
    assert cert.tokens_compared == total_len


def test_kv_exactness_detects_mismatch():
    t1 = torch.zeros(1, 2, 8, 16)
    t2 = torch.ones(1, 2, 8, 16)
    cert = verify_kv_exactness(t1, t2, tolerance=1e-5)
    assert cert.is_proven is False
    assert cert.max_absolute_error == 1.0
