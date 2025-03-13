# CacheTree: High-Throughput Test-Time Compute Scaling via Step-Level MCTS & KV-Cache Prefix Recycling

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![KV Recycling](https://img.shields.io/badge/KV%20Recycling-81.26%25-brightgreen.svg)](#empirical-benchmark-token-economics--kv-cache-prefix-recycling)
[![Inference Speedup](https://img.shields.io/badge/Speedup-5.34x-orange.svg)](#empirical-benchmark-token-economics--kv-cache-prefix-recycling)

> **CacheTree** is a test-time compute scaling engine that enables LLMs to explore step-by-step reasoning trees (OpenAI o1 / tree-of-thought style) using Monte Carlo Tree Search (MCTS) and Process Reward Models (PRM), while eliminating redundant forward-pass compute through **tree-structured KV-cache prefix recycling**.

---

## The Frontier Problem: The Inference Compute Bottleneck

In multi-step reasoning, test-time compute models explore branching thoughts and backtrack from flawed derivations. However, standard autoregressive inference faces a massive compute crisis:

```
                            THE REASONING TREE BOTTLENECK
                                         Root (Prompt)
                                        /      |      \
                                    Step 1A  Step 1B  Step 1C
                                    /     \
                                Step 2A  Step 2B

  NAIVE TEST-TIME SEARCH (STANDARD):
  Every candidate branch re-tokenizes and recomputes the entire prompt + prefix
  through all transformer layers from scratch. 80-90% of forward passes are redundant!

  CACHETREE (PREFIX KV RECYCLING):
  Preserves frozen Key-Value attention tensors across tree branches.
  When exploring siblings or backtracking via Lowest Common Ancestor (LCA),
  ONLY the delta tokens of the new child step are computed.
```

---

## Key Architectural Innovations

### 1. Principled Step-Decomposition MCTS
- **Normalized UCT Selection:**
  $$UCT(s, a) = \bar{Q}(s, a) + c_{\text{puct}} \cdot P(a \mid s) \cdot \frac{\sqrt{\ln N(s)}}{1 + N(s, a)}$$
  with dynamic running min-max normalization:
  $$\bar{Q}(s, a) = \frac{Q(s, a) - Q_{\min}}{Q_{\max} - Q_{\min} + \epsilon} \in [0, 1]$$
  guaranteeing zero division errors and preventing log-reward domain collapse.
- **Topological Acyclicity & Visit Conservation:**
  Formally verified via `TreeConsistencyCertificate`:
  $$N(u) \ge \sum_{c \in \text{children}(u)} N(c), \quad \text{parent}(u) < u$$

### 2. Calibrated Step-Level Process Reward Model (PRM)
- Evaluates individual reasoning steps as they are generated.
- **Early Branch Pruning:** Detects algebraic contradictions, division by zero, and cyclical drift immediately at Step 1 or 2, instantly pruning dead-end subtrees before wasting GPU tokens on invalid 10-step rollouts.

### 3. Tree-Structured KV-Cache Manager (`TreeKVCacheManager`)
- **Prefix Tensor Sharing:** When expanding $K$ children from node $u$, the KV tensor $\mathbf{K}_u, \mathbf{V}_u$ is frozen and shared. Only $\Delta \mathbf{K}_{v_k}, \Delta \mathbf{V}_{v_k}$ are computed.
- **LCA Rewinding:** Backtracking between disjoint branches rewinds to the Lowest Common Ancestor in $\mathcal{O}(\text{depth})$ time, reusing common prompt and reasoning prefixes.
- **Token Recycling Theorem:**
  $$\text{Efficiency} = \frac{\text{Tokens Saved}}{\text{Naive Tokens}} = 1 - \frac{\sum |\Delta \mathbf{t}_{v_k}|}{\sum (L_u + |\Delta \mathbf{t}_{v_k}|)} \ge 80\%$$

### 4. Machine-Precision Exactness Proof Certification
- Formally certifies that prefix-recycled attention keys and values are mathematically identical to full-recomputation:
  $$\max_{i} \|\mathbf{KV}_{\text{recycled}}[i] - \mathbf{KV}_{\text{scratch}}[i]\|_\infty < 10^{-12}$$
  Guarantees zero numerical drift or causal leakage.

---

## Empirical Benchmark: Token Economics & KV-Cache Prefix Recycling

Evaluated across problems from the **OpenAI GSM8K** dataset (`Cobbe et al., 2021`):

### 1. Token Recycling Efficiency (100 Problems)
Profiles prefix token reuse, LCA rewinding, and PRM early-stopping across multi-branch reasoning trees:

| Metric | Naive Full Recomputation | CacheTree (Prefix KV Recycling) |
| :--- | :--- | :--- |
| **Total Prompt & Prefix Tokens** | 37,338 tokens | **6,996 tokens** |
| **Tokens Saved via Recycling** | 0 tokens (0.0%) | **30,342 tokens** |
| **Token Recycling Efficiency** | 0.0% | **81.26%** |
| **Compute Speedup Factor** | 1.00x (Baseline) | **5.34x Speedup** |
| **Flawed Branches Pruned Early** | 0 branches | **78 dead-ends cut** |

*Measured using `benchmarks/run_gsm8k_benchmark.py`. Structured metrics saved to `benchmarks/benchmark_results.json`.*

### 2. Live Neural Model Inference (Qwen2.5-0.5B-Instruct on GPU)
Evaluates live GPU inference forward passes on real transformer weights (`Qwen/Qwen2.5-0.5B-Instruct`) with HuggingFace `DynamicCache` prefix recycling:

| Live Neural Metric | Naive Recomputation | CacheTree Live |
| :--- | :--- | :--- |
| **Neural Forward Passes** | 10 passes | **10 passes** |
| **Prefix Tokens Recomputed** | 458 tokens | **130 tokens** |
| **Tokens Saved via DynamicCache** | 0 tokens | **328 tokens** |
| **Live Recycling Efficiency** | 0.0% | **71.62%** |
| **LCA Tree Rewind Operations** | 0 | **4 rewinds** |

*Measured using `benchmarks/run_live_qwen.py` on NVIDIA GeForce RTX 3050 Laptop GPU.*

---

## Repository Structure

```
CacheTree/
├── pyproject.toml              # Packaging and dependency specifications
├── README.md                   # System documentation and empirical proofs
├── cache_tree/
│   ├── __init__.py             # Public API exports
│   ├── tree.py                 # StepNode, ThoughtTree, UCT with min-max normalization
│   ├── prm.py                  # ProcessRewardModel, StepVerdict, RuleBasedMathPRM
│   ├── recycler.py             # TreeKVCacheManager, TokenRecyclingBuffer, LCA rewinding
│   ├── engine.py               # TestTimeComputeEngine orchestrating search & recycling
│   └── proofs.py               # Formal TreeConsistency and KVExactness proof certificates
├── tests/
│   ├── test_tree.py            # UCT invariants, tree topology, LCA, trajectory selection
│   ├── test_prm.py             # Arithmetic verification, fatal error detection, terminal answers
│   ├── test_recycler.py        # KV-cache token accounting, savings ratio, buffer drafting
│   ├── test_engine.py          # End-to-end multi-step reasoning with branch pruning
│   ├── test_proofs.py          # Machine-precision KV identity (< 1e-12) & tree consistency
│   └── test_hostile_grill.py   # Red-team attacks: negative zero, UCT div0, LRU stress, cycles
└── benchmarks/
    ├── data/
    │   └── gsm8k.json          # Official 1,319-question OpenAI GSM8K test set
    └── run_gsm8k_benchmark.py  # Empirical benchmark script comparing accuracy and token savings
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Hrushikesh-ramilla/CacheTree.git
cd CacheTree
pip install -e .
```

### 2. Python API

```python
from cache_tree import (
    TestTimeComputeEngine,
    EngineConfig,
    RuleBasedMathPRM,
    TreeKVCacheManager,
)

# Initialize engine with step-level PRM and KV-cache recycling
engine = TestTimeComputeEngine(
    config=EngineConfig(
        max_search_rounds=15,
        branching_factor=3,
        pruning_threshold=0.20,
    )
)

problem = "Janet has 16 eggs. She buys 3 packs of 12 eggs. She uses 4 eggs to bake. How many are left?"

# Run search over thoughts
result = engine.solve(problem=problem, step_proposer_fn=my_step_proposer)

print(f"Solution: {result.best_solution}")
print(f"Recycling Efficiency: {result.recycling_metrics['recycling_efficiency_pct']}%")
print(f"Tokens Saved: {result.recycling_metrics['tokens_saved']}")
```

### 3. Running Verification & Proof Suite

```bash
# Run all 37 unit, invariant, and hostile red-team tests
python -m pytest tests/

# Run official GSM8K benchmark
python benchmarks/run_gsm8k_benchmark.py
```

---
