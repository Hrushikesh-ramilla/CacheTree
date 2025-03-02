"""
CacheTree: High-Throughput Test-Time Compute Scaling via Step-Level MCTS & KV-Cache Prefix Recycling.
"""

from .tree import (
    StepNode,
    ThoughtTree,
    NodeStatus,
    UCTConfig,
)
from .prm import (
    ProcessRewardModel,
    StepVerdict,
    StepEvaluation,
    RuleBasedMathPRM,
    HybridMathPRM,
)
from .recycler import (
    TreeKVCacheManager,
    KVCacheEntry,
    TokenRecyclingBuffer,
    RecyclingStats,
)
from .engine import (
    TestTimeComputeEngine,
    EngineConfig,
    ReasoningResult,
)
from .proofs import (
    TreeConsistencyCertificate,
    KVExactnessCertificate,
    verify_tree_consistency,
    verify_kv_exactness,
)

__version__ = "0.1.0"

__all__ = [
    "StepNode",
    "ThoughtTree",
    "NodeStatus",
    "UCTConfig",
    "ProcessRewardModel",
    "StepVerdict",
    "StepEvaluation",
    "RuleBasedMathPRM",
    "HybridMathPRM",
    "TreeKVCacheManager",
    "KVCacheEntry",
    "TokenRecyclingBuffer",
    "RecyclingStats",
    "TestTimeComputeEngine",
    "EngineConfig",
    "ReasoningResult",
    "TreeConsistencyCertificate",
    "KVExactnessCertificate",
    "verify_tree_consistency",
    "verify_kv_exactness",
]
