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

        # 2. CacheTree: PRM-guided MCTS with KV-cache recycling
        engine = TestTimeComputeEngine(
            config=EngineConfig(
                max_search_rounds=8,
                branching_factor=2,
                pruning_threshold=0.20,
                early_stop_on_success=True,
            )
        )

        if use_live:
            proposer_mcts = HuggingFaceStepProposer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_step_tokens=32,
                kv_manager=engine.kv_manager,
            )
        else:
            proposer_mcts = calibrated_gen

        res = engine.solve(problem=q, step_proposer_fn=proposer_mcts, initial_tokens=list(range(len(q.split()))))

        pred = parse_numeric_answer(res.best_solution) if res.best_solution else None
        if pred is not None and abs(pred - gt) < 1e-4:
            recycler_correct += 1

        # Accumulate compute metrics
        metrics = res.recycling_metrics
        total_tokens_unrecycled += metrics["total_tokens_without_recycling"]
        total_tokens_recycled += metrics["tokens_computed_with_recycling"]
        total_pruned_branches += res.pruned_branches_count
        total_search_rounds += res.total_search_rounds

    elapsed = time.time() - start_bench
    n = len(items)

    naive_acc = (naive_correct / n) * 100.0
    recycler_acc = (recycler_correct / n) * 100.0
    tokens_saved = max(0, total_tokens_unrecycled - total_tokens_recycled)
    efficiency_pct = (tokens_saved / max(1, total_tokens_unrecycled)) * 100.0
    speedup_ratio = total_tokens_unrecycled / max(1, total_tokens_recycled)

    print("\n" + "-" * 80)
    print("BENCHMARK RESULTS SUMMARY:")
    print("-" * 80)
    print(f"Total GSM8K Problems Evaluated:           {n}")
    print(f"Baseline LLM Accuracy (Single Pass):       {naive_acc:.1f}%")
    print(f"CacheTree Accuracy (PRM MCTS):       {recycler_acc:.1f}%")
    print(f"Absolute Accuracy Improvement:             +{recycler_acc - naive_acc:.1f}%")
    print(f"Total Naive Tokens (Full Re-computation):  {total_tokens_unrecycled:,}")
    print(f"Actual Tokens Computed (With Recycling):   {total_tokens_recycled:,}")
    print(f"Tokens Saved via Prefix KV-Recycling:      {tokens_saved:,}")
    print(f"Empirical Token Recycling Efficiency:      {efficiency_pct:.2f}%")
    print(f"Inference Compute Speedup Factor:          {speedup_ratio:.2f}x")
    print(f"Flawed Derivation Branches Pruned Early:   {total_pruned_branches}")
    print(f"Evaluation Time:                           {elapsed:.2f}s")
    print("=" * 80)

    summary_dict = {
        "dataset": "OpenAI GSM8K (Cobbe et al., 2021)",
        "eval_mode": "neural_live" if use_live else "simulated_profiling",
        "num_questions": n,
        "baseline_accuracy_pct": round(naive_acc, 2),
        "cache_tree_accuracy_pct": round(recycler_acc, 2),
        "accuracy_boost_pct": round(recycler_acc - naive_acc, 2),
        "total_naive_tokens": total_tokens_unrecycled,
        "tokens_computed_with_recycling": total_tokens_recycled,
        "tokens_saved": tokens_saved,
        "recycling_efficiency_pct": round(efficiency_pct, 2),
        "speedup_factor": round(speedup_ratio, 2),
        "pruned_branches_count": total_pruned_branches,
        "elapsed_seconds": round(elapsed, 2),
    }

    out_name = "benchmark_results_live.json" if use_live else "benchmark_results.json"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Structured results written to: {out_path}\n")

    return summary_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CacheTree GSM8K Benchmark")
    parser.add_argument("--max-items", type=int, default=100, help="Max GSM8K questions to test")
    parser.add_argument("--live", action="store_true", help="Run live neural inference using Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    data_file = os.path.join(os.path.dirname(__file__), "data", "gsm8k.json")
    run_benchmark(data_file, max_items=args.max_items, use_live=args.live)
