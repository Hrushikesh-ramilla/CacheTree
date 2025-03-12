"""
Live Neural Benchmark Runner for CacheTree using Qwen2.5-0.5B-Instruct.
Executes live GPU inference on real transformer weights, demonstrating:
1. Neural reasoning step generation
2. Step-Level Process Reward verification & pruning
3. Live TreeKVCacheManager prefix token recycling
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache_tree.engine import TestTimeComputeEngine, EngineConfig
from cache_tree.generator import HuggingFaceStepProposer
from cache_tree.prm import RuleBasedMathPRM


def main():
    print("=" * 75)
    print("      CACHETREE LIVE NEURAL REASONING RUNNER")
    print("      Model: Qwen/Qwen2.5-0.5B-Instruct (Live GPU Inference)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    print("Loading model weights and tokenizer...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    engine = TestTimeComputeEngine(
        config=EngineConfig(
            max_search_rounds=5,
            branching_factor=2,
            pruning_threshold=0.20,
            early_stop_on_success=True,
        )
    )

    proposer = HuggingFaceStepProposer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_step_tokens=32,
        temperature=0.7,
        top_p=0.9,
        kv_manager=engine.kv_manager,
    )

    problem = "Janet has 16 eggs. She buys 3 packs of 12 eggs. How many eggs does she have in total?"
    print(f"\n[Problem]: {problem}\n")
    print("Exploring reasoning tree with live neural forward passes...")

    start = time.time()
    result = engine.solve(problem=problem, step_proposer_fn=proposer)
    elapsed = time.time() - start

    print("\n" + "-" * 75)
    print("REASONING TRAJECTORY FOUND:")
    print("-" * 75)
    for i, step in enumerate(result.best_trajectory):
        print(f"[{i}] {step}")

    metrics = result.recycling_metrics
    print("\n" + "=" * 75)
    print("LIVE KV-CACHE RECYCLING METRICS:")
    print("=" * 75)
    print(f"Total Forward Passes:        {metrics['total_forward_passes']}")
    print(f"Naive Tokens Without Cache:   {metrics['total_tokens_without_recycling']:,}")
    print(f"Actual Tokens Computed:       {metrics['tokens_computed_with_recycling']:,}")
    print(f"Tokens Saved via Recycling:   {metrics['tokens_saved']:,}")
    print(f"Recycling Efficiency:         {metrics['recycling_efficiency_pct']}%")
    print(f"Rewind Operations (LCA):      {metrics['rewind_operations']}")
    print(f"Search Execution Time:        {elapsed:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
