"""
Tree-Structured KV-Cache Manager & Speculative Token Recycling Buffer.

Eliminates redundant prefix recomputation in multi-path reasoning tree search:
1. TreeKVCacheManager: Reuses key-value tensor states across tree branches.
   When branching or backtracking, crops/shares tensors using Lowest Common Ancestor (LCA).
2. TokenRecyclingBuffer: Maintains empirical token transition adjacency to predict
   and batch speculative candidate tokens within reasoning steps.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch


def generate_sinusoidal_attention_tensors(
    token_count: int,
    num_heads: int = 4,
    head_dim: int = 64,
    start_pos: int = 0,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates non-zero transformer attention key and value representations using
    orthogonal sinusoidal positional features (Vaswani et al. 2017).
    Guarantees non-trivial attention dot-products and strictly positive norms.
    """
    t_count = max(1, token_count)
    pos = torch.arange(start_pos, start_pos + t_count, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, head_dim, 2, dtype=torch.float32).unsqueeze(0)
    div_term = torch.exp(dim * -(math.log(10000.0) / head_dim))
    sinusoid = torch.zeros((t_count, head_dim), dtype=torch.float32)
    sinusoid[:, 0::2] = torch.sin(pos * div_term)
    sinusoid[:, 1::2] = torch.cos(pos * div_term)
    base_repr = sinusoid.unsqueeze(0).unsqueeze(1).repeat(1, num_heads, 1, 1).to(dtype=dtype, device=device)
    k = base_repr.clone()
    v = torch.roll(base_repr, shifts=1, dims=-1)
    return k, v


@dataclasses.dataclass
class RecyclingStats:
    total_forward_passes: int = 0
    total_tokens_without_recycling: int = 0
    tokens_computed_with_recycling: int = 0
    tokens_saved: int = 0
    recycling_efficiency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    rewind_operations: int = 0

    def compute_metrics(self) -> None:
        self.tokens_saved = max(0, self.total_tokens_without_recycling - self.tokens_computed_with_recycling)
        if self.total_tokens_without_recycling > 0:
            self.recycling_efficiency = self.tokens_saved / self.total_tokens_without_recycling
        else:
            self.recycling_efficiency = 0.0


@dataclasses.dataclass
class KVCacheEntry:
    node_id: int
    token_count: int
    key_tensor: Optional[torch.Tensor] = None
    value_tensor: Optional[torch.Tensor] = None
    dynamic_cache: Optional[Any] = None
    last_accessed: float = dataclasses.field(default_factory=time.time)

