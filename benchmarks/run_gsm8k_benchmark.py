"""
Empirical GSM8K Benchmark Runner for CacheTree.
Compares:
1. Baseline Naive Greedy Generation (no search, single path)
2. Naive Unrecycled Tree Search (recomputes full prefix tokens on every branch)
3. CacheTree (Step-Level PRM MCTS with KV-Cache Prefix Recycling)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache_tree.engine import (
    TestTimeComputeEngine,
    EngineConfig,
    CandidateStepProposal,
)
from cache_tree.generator import CalibratedEmpiricalProposer
from cache_tree.prm import RuleBasedMathPRM
from cache_tree.recycler import TreeKVCacheManager
from cache_tree.proofs import verify_tree_consistency, verify_kv_exactness


def parse_numeric_answer(ans_val: Any) -> Optional[float]:
    """Extracts a float from ground truth answer string or number."""
    if isinstance(ans_val, (int, float)):
        return float(ans_val)
    if not isinstance(ans_val, str):
        return None
    # Check #### pattern
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans_val)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            pass
    # Raw float search
    m_num = re.search(r"(-?[\d,]+(?:\.\d+)?)", ans_val)
    if m_num:
        raw = m_num.group(1).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            pass
    return None


def run_benchmark(dataset_path: str, max_items: int = 100, use_live: bool = False) -> Dict[str, Any]:
    eval_mode = "Live Neural (Qwen2.5-0.5B-Instruct)" if use_live else "Simulated Profiling (CalibratedProposer)"
    print("=" * 80)
    print("      CACHETREE EMPIRICAL TEST-TIME REASONING BENCHMARK")
    print(f"      Mode: {eval_mode}")
    print(f"      Dataset: OpenAI GSM8K (Evaluating {max_items} Questions)")
    print("=" * 80)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data[:max_items]

    # Metrics
    naive_correct = 0
    recycler_correct = 0
    total_tokens_unrecycled = 0
    total_tokens_recycled = 0
    total_pruned_branches = 0
    total_search_rounds = 0

    start_bench = time.time()

    model = None
    tokenizer = None
    device = None

    if use_live:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cache_tree.generator import HuggingFaceStepProposer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading weights from {model_name} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        calibrated_gen = None
    else:
        calibrated_gen = CalibratedEmpiricalProposer(failure_rate_on_hard_problems=0.18)

    for idx, item in enumerate(items):
        q = item["question"]
        gt = parse_numeric_answer(item["answer"])
        if gt is None:
            continue

        # 1. Baseline: Naive single-path rollout
        engine_naive = TestTimeComputeEngine(
            config=EngineConfig(max_search_rounds=1, branching_factor=1)
        )
        if use_live:
            proposer_naive = HuggingFaceStepProposer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_step_tokens=32,
                kv_manager=engine_naive.kv_manager,
            )
        else:
            proposer_naive = calibrated_gen

        history_naive: List[str] = []
        for s_idx in range(1, 4):
            cand_list = proposer_naive(q, history_naive, branching_factor=1)
            if not cand_list:
                break
            chosen = cand_list[0].thought_text
            history_naive.append(chosen)
            if "####" in chosen:
                break
        final_cand_text = history_naive[-1] if history_naive else ""
        naive_pred = parse_numeric_answer(final_cand_text)
        if naive_pred is not None and abs(naive_pred - gt) < 1e-4:
            naive_correct += 1

    print(f"Baseline Accuracy: {naive_correct / max(1, len(questions)):.2%}")
