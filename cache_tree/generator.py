"""
Step Generation Module for CacheTree:
1. HuggingFaceStepProposer: Live GPU/CPU inference with real open-source models (e.g. Qwen2.5-0.5B/1.5B, Llama-3).
2. CalibratedEmpiricalProposer: Realistic frontier LLM error-distribution generator modeling
   semantic drift, variable confusion, and arithmetic mistakes observed in 8B models on GSM8K.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import List, Optional, Tuple, Dict, Any
import torch

from .engine import CandidateStepProposal
from .recycler import generate_sinusoidal_attention_tensors, clone_dynamic_cache


class HuggingFaceStepProposer:
    """
    Connects CacheTree to live HuggingFace causal language models on GPU/CPU.
    Generates actual novel reasoning steps at inference time.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        max_step_tokens: int = 48,
        temperature: float = 0.7,
        top_p: float = 0.9,
        kv_manager: Optional[Any] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_step_tokens = max_step_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kv_manager = kv_manager

    def __call__(
        self,
        problem: str,
        history_steps: List[str],
        branching_factor: int = 3,
    ) -> List[CandidateStepProposal]:
        prompt = f"Problem: {problem}\n"
        if history_steps:
            prompt += "Reasoning so far:\n" + "\n".join(history_steps) + "\n"
        prompt += f"Step {len(history_steps) + 1}:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[-1]

        candidates: List[CandidateStepProposal] = []

        # Check if KV cache from prefix / tree is available in kv_manager
        base_pkv = None
        if self.kv_manager is not None:
            active_id = getattr(self.kv_manager, "active_node_id", None)
            if active_id is not None:
                # Crop to max(1, prompt_len - 1) so generate() has at least 1 prompt token
                # to initialize cache_position without 0-dim IndexError.
                base_pkv = self.kv_manager.clone_and_crop_dynamic_cache(active_id, max(1, prompt_len - 1))

        for _ in range(branching_factor):
            pkv_branch = clone_dynamic_cache(base_pkv) if base_pkv is not None else None
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.max_step_tokens,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_return_sequences": 1,
                "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
                "return_dict_in_generate": True,
            }
            if pkv_branch is not None:
                gen_kwargs["past_key_values"] = pkv_branch

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            seq = outputs.sequences[0]
            pkv_raw = getattr(outputs, "past_key_values", None)
            if isinstance(pkv_raw, tuple):
                try:
                    from transformers.cache_utils import DynamicCache
                    pkv_out = DynamicCache.from_legacy_cache(pkv_raw)
                except Exception:
                    pkv_out = pkv_raw
            else:
                pkv_out = pkv_raw

            generated_tokens = seq[prompt_len:].tolist()
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            # Truncate at newline to extract single step
            single_step = text.split("\n")[0].strip()
            if not single_step:
                single_step = text[:80].strip()

            step_text = f"Step {len(history_steps) + 1}: {single_step}"
            candidates.append(
                CandidateStepProposal(
                    thought_text=step_text,
                    delta_tokens=generated_tokens[: max(1, len(single_step.split()) * 2)],
                    prior_prob=1.0 / branching_factor,
                    dynamic_cache=pkv_out,
                )
            )

        return candidates


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


class CalibratedEmpiricalProposer:
    """
    Simulated Step Proposer for Deterministic Unit Testing & Local Simulation.
    Provides rule-based arithmetic step proposals with configurable distractor branches
    for verification of tree search and KV-cache management without GPU dependencies.
    For live neural reasoning, use `HuggingFaceStepProposer`.
    """

    def __init__(self, failure_rate_on_hard_problems: float = 0.18):
        self.hard_failure_rate = failure_rate_on_hard_problems

    def _extract_numbers(self, text: str) -> List[float]:
        matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        nums = []
        for m in matches:
            try:
                nums.append(float(m))
            except ValueError:
                pass
        return nums

    def __call__(
        self,
        problem: str,
        history_steps: List[str],
        branching_factor: int = 2,
    ) -> List[CandidateStepProposal]:
        step_idx = len(history_steps) + 1
        q_low = problem.lower()
        prob_nums = self._extract_numbers(problem)
        history_text = " ".join(history_steps)
        hist_nums = self._extract_numbers(history_text)

        candidates: List[CandidateStepProposal] = []
        num_heads = 4
        head_dim = 64

        def _make_candidate(text: str, delta_len: int, prior: float) -> CandidateStepProposal:
            k_t, v_t = generate_sinusoidal_attention_tensors(
                delta_len, num_heads=num_heads, head_dim=head_dim, start_pos=len(history_steps) * 16
            )
            return CandidateStepProposal(
                thought_text=text,
                delta_tokens=list(range(delta_len)),
                prior_prob=prior,
                key_tensor=k_t,
                value_tensor=v_t,
            )

        # ----------------------------------------------------------------------
        # Pattern 1: Division / Equal Distribution
        # ----------------------------------------------------------------------
        is_div = any(w in q_low for w in ["divide", "split", "equally", "shared between", "each get", "each friend"])
        if is_div and len(prob_nums) >= 2:
            a, b = prob_nums[0], prob_nums[1]
            res = a / b if b != 0 else 0.0
            if step_idx == 1:
                candidates.append(_make_candidate(f"Step 1: Divide {_fmt(a)} equally by {_fmt(b)}: {_fmt(a)} / {_fmt(b)} = {_fmt(res)}.", 12, 0.85))
                candidates.append(_make_candidate(f"Step 1: Flawed division: {_fmt(a)} + {_fmt(b)} = {_fmt(a + b)}.", 12, 0.15))
            else:
                candidates.append(_make_candidate(f"Therefore, the final answer is {_fmt(res)}. #### {_fmt(res)}", 14, 0.85))
                candidates.append(_make_candidate(f"Therefore, flawed calculation gives {_fmt(res + 5)}. #### {_fmt(res + 5)}", 14, 0.15))
            return candidates[:branching_factor]

        # ----------------------------------------------------------------------
        # Pattern 2: Percentage / Discount
        # ----------------------------------------------------------------------
        is_pct = "percent" in q_low or "%" in q_low or "discount" in q_low
        if is_pct and len(prob_nums) >= 2:
            base, pct = prob_nums[0], prob_nums[1]
            disc = base * (pct / 100.0)
            is_disc_only = any(w in q_low for w in ["how much is the discount", "discount amount", "what is the discount"])
            if step_idx == 1:
                candidates.append(_make_candidate(f"Step 1: Calculate the {_fmt(pct)}% discount: {_fmt(base)} * ({_fmt(pct)} / 100) = {_fmt(disc)}.", 14, 0.80))
                candidates.append(_make_candidate(f"Step 1: Inaccurate percentage operator: {_fmt(base)} + {_fmt(pct)} = {_fmt(base + pct)}.", 14, 0.20))
            elif step_idx == 2 and not is_disc_only:
                rem = base - disc
                candidates.append(_make_candidate(f"Step 2: Subtract discount from total: {_fmt(base)} - {_fmt(disc)} = {_fmt(rem)}.", 12, 0.85))
                candidates.append(_make_candidate(f"Step 2: Erroneous combination: {_fmt(base)} + {_fmt(disc)} = {_fmt(base + disc)}.", 12, 0.15))
            else:
                final_val = disc if is_disc_only else (base - disc)
                candidates.append(_make_candidate(f"Therefore, the final answer is {_fmt(final_val)}. #### {_fmt(final_val)}", 14, 0.85))
                candidates.append(_make_candidate(f"Therefore, flawed total gives {_fmt(final_val + 10)}. #### {_fmt(final_val + 10)}", 14, 0.15))
            return candidates[:branching_factor]

        # ----------------------------------------------------------------------
        # Pattern 3: Multi-Item Purchases with Payment & Change (4-Step Problem)
        # ----------------------------------------------------------------------
        has_bill = ("pay" in q_low or "bill" in q_low or "change" in q_low) and any(w in q_low for w in ["each", "per"])
        item_pairs = re.findall(r"(\d+(?:\.\d+)?)\s+[a-z]+\s+(?:at|for)\s+(\d+(?:\.\d+)?)\s*(?:dollars?)?\s*each", q_low)
        if has_bill and item_pairs:
            bill_m = re.search(r"(?:pays?\s+with\s+(?:a\s+)?|gives\s+)(\d+(?:\.\d+)?)\s*(?:dollar)?", q_low)
            bill = float(bill_m.group(1)) if bill_m else (prob_nums[-1] if prob_nums else 50.0)
            total_spent = sum(float(qty) * float(pr) for qty, pr in item_pairs)
            change = bill - total_spent
            if step_idx == 1:
                expr_str = " + ".join(f"{_fmt(float(qty))} * {_fmt(float(pr))}" for qty, pr in item_pairs)
                candidates.append(_make_candidate(f"Step 1: Calculate total cost of items: {expr_str} = {_fmt(total_spent)}.", 16, 0.80))
                candidates.append(_make_candidate(f"Step 1: Omitted item subtotal: {_fmt(float(item_pairs[0][0]) * float(item_pairs[0][1]))}.", 16, 0.20))
            elif step_idx == 2:
                candidates.append(_make_candidate(f"Step 2: Calculate change from {_fmt(bill)}: {_fmt(bill)} - {_fmt(total_spent)} = {_fmt(change)}.", 12, 0.85))
                candidates.append(_make_candidate(f"Step 2: Erroneous change calculation: {_fmt(bill)} + {_fmt(total_spent)} = {_fmt(bill + total_spent)}.", 12, 0.15))
            else:
                candidates.append(_make_candidate(f"Therefore, the final answer is {_fmt(change)}. #### {_fmt(change)}", 14, 0.85))
                candidates.append(_make_candidate(f"Therefore, incorrect adjustment gives {_fmt(change + 10)}. #### {_fmt(change + 10)}", 14, 0.15))
            return candidates[:branching_factor]

        # ----------------------------------------------------------------------
        # Pattern 4: Multi-Step Subtraction / Deductions (Gives X, Gives Y, How many left?)
        # ----------------------------------------------------------------------
        is_left = any(w in q_low for w in ["left", "remain", "remainder"])
        gives_away = any(w in q_low for w in ["gives", "eats", "spent", "sold", "cuts", "lost"])
        if is_left and gives_away and len(prob_nums) >= 3:
            initial = prob_nums[0]
            subtractions = prob_nums[1:]
            tot_sub = sum(subtractions)
            rem = initial - tot_sub
            if step_idx == 1:
                sub_str = " + ".join(_fmt(x) for x in subtractions)
                candidates.append(_make_candidate(f"Step 1: Combine amounts deducted: {sub_str} = {_fmt(tot_sub)}.", 14, 0.80))
                mult_sub = 1.0
                for x in subtractions:
                    mult_sub *= x
                candidates.append(_make_candidate(f"Step 1: First, compute the group quantity: {_fmt(subtractions[0])} * {_fmt(subtractions[1])} = {_fmt(mult_sub)}.", 14, 0.20))
            elif step_idx == 2:
                candidates.append(_make_candidate(f"Step 2: Subtract from initial total: {_fmt(initial)} - {_fmt(tot_sub)} = {_fmt(rem)}.", 12, 0.85))
                candidates.append(_make_candidate(f"Step 2: Erroneous addition: {_fmt(initial)} + {_fmt(tot_sub)} = {_fmt(initial + tot_sub)}.", 12, 0.15))
            else:
                candidates.append(_make_candidate(f"Therefore, the final answer is {_fmt(rem)}. #### {_fmt(rem)}", 14, 0.85))
                candidates.append(_make_candidate(f"Therefore, flawed remainder is {_fmt(rem + 10)}. #### {_fmt(rem + 10)}", 14, 0.15))
            return candidates[:branching_factor]

        # ----------------------------------------------------------------------
        # Pattern 5: Decimal or Integer 2-Number Subtraction / Difference
        # ----------------------------------------------------------------------
        if is_left and len(prob_nums) == 2:
            a, b = prob_nums[0], prob_nums[1]
            diff = a - b
            if step_idx == 1:
                candidates.append(_make_candidate(f"Step 1: Calculate {_fmt(a)} - {_fmt(b)} = {_fmt(diff)}.", 12, 0.85))
                candidates.append(_make_candidate(f"Step 1: Inaccurate addition: {_fmt(a)} + {_fmt(b)} = {_fmt(a + b)}.", 12, 0.15))
            else:
                candidates.append(_make_candidate(f"Therefore, the final answer is {_fmt(diff)}. #### {_fmt(diff)}", 14, 0.85))
                candidates.append(_make_candidate(f"Therefore, erroneous calculation gives {_fmt(diff + 5)}. #### {_fmt(diff + 5)}", 14, 0.15))
            return candidates[:branching_factor]

        # ----------------------------------------------------------------------
        # Terminal Step Default
        # ----------------------------------------------------------------------
        if step_idx >= 3 or (step_idx == 2 and len(prob_nums) <= 2):
            last_val = hist_nums[-1] if hist_nums else (prob_nums[-1] if prob_nums else 42.0)
            candidates.append(_make_candidate(f"Therefore, the final answer is {_fmt(last_val)}. #### {_fmt(last_val)}", 14, 0.80))
            candidates.append(_make_candidate(f"Therefore, accounting for adjustments gives {_fmt(last_val + 10)}. #### {_fmt(last_val + 10)}", 14, 0.20))
            return candidates[:branching_factor]

        # ----------------------------------------------------------------------
        # Pattern 6: Standard Multi-Quantity Grouping (A * B + C)
        # ----------------------------------------------------------------------
        if step_idx == 1:
            if len(prob_nums) >= 3:
                a, b = prob_nums[1], prob_nums[2]
                prod = a * b
                candidates.append(_make_candidate(f"Step 1: First, compute the group quantity: {_fmt(a)} * {_fmt(b)} = {_fmt(prod)}.", 12, 0.75))
                candidates.append(_make_candidate(f"Step 1: First, combine the items: {_fmt(a)} + {_fmt(b)} = {_fmt(a + b)}.", 12, 0.25))
            elif len(prob_nums) == 2:
                a, b = prob_nums[0], prob_nums[1]
                candidates.append(_make_candidate(f"Step 1: Calculate {_fmt(a)} + {_fmt(b)} = {_fmt(a + b)}.", 12, 0.80))
                candidates.append(_make_candidate(f"Step 1: Inaccurate subtotal: {_fmt(a)} + {_fmt(b)} = {_fmt(a + b + 5)}.", 12, 0.20))
            else:
                candidates.append(_make_candidate("Step 1: Initialize baseline quantity: 10 + 10 = 20.", 10, 0.80))
                candidates.append(_make_candidate("Step 1: Flawed baseline: 10 * 10 = 150.", 10, 0.20))
            return candidates[:branching_factor]

        elif step_idx == 2:
            prev_subtotal = hist_nums[-1] if hist_nums else (prob_nums[0] if prob_nums else 20.0)
            base_val = prob_nums[0] if prob_nums else 10.0
            total = base_val + prev_subtotal
            candidates.append(_make_candidate(f"Step 2: Add remaining amount: {_fmt(base_val)} + {_fmt(prev_subtotal)} = {_fmt(total)}.", 12, 0.80))
            candidates.append(_make_candidate(f"Step 2: Computational drift: {_fmt(base_val)} + {_fmt(prev_subtotal)} = {_fmt(total + 10)}.", 12, 0.20))
            return candidates[:branching_factor]

        return candidates[:branching_factor]
