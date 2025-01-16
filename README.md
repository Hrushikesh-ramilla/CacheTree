# CacheTree: High-Throughput Test-Time Compute Scaling via Step-Level MCTS & KV-Cache Prefix Recycling

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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
