"""
Tests for HybridMathPRM (Symbolic + Neural Logit Process Reward Model).
"""

import pytest
import torch
from cache_tree.prm import (
    HybridMathPRM,
    RuleBasedMathPRM,
    StepVerdict,
    StepEvaluation,
)


class MockNeuralModel:
    """Mock model returning controlled logits for token 'A' (Yes) vs 'B' (No)."""

    def __init__(self, p_yes: float = 0.9):
        self.p_yes = p_yes

    def __call__(self, **kwargs):
        # Return logits where logit(A) vs logit(B) matches p_yes
        # logit_A - logit_B = ln(p_yes / (1 - p_yes))
        logits = torch.zeros((1, 100))
        # Suppose 'A' has token_id 10, 'B' has token_id 20
        logit_diff = torch.logit(torch.tensor(self.p_yes))
        logits[0, 10] = logit_diff
        logits[0, 20] = 0.0

        class Output:
            pass

        out = Output()
        out.logits = logits.unsqueeze(0)  # (1, 1, 100)
        return out


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99

    def __call__(self, text, return_tensors="pt"):
        class Inputs(dict):
            def to(self, device):
                return self

        inp = Inputs()
        inp["input_ids"] = torch.tensor([[1, 2, 3]])
        return inp

    def encode(self, text, add_special_tokens=False):
        if "A" in text:
            return [10]
        if "B" in text:
            return [20]
        return [1]


def test_hybrid_prm_symbolic_short_circuit():
    """Arithmetic contradiction must be rejected immediately without calling neural head."""
    prm = HybridMathPRM(symbolic_prm=RuleBasedMathPRM(fatal_threshold=0.20))
    # Contradiction: 10 + 10 = 25
    res = prm.evaluate_step(
        problem="What is 10 + 10?",
        reasoning_history="",
        current_step="Step 1: 10 + 10 = 25",
    )
    assert res.verdict == StepVerdict.FATAL_ERROR
    assert res.is_fatal is True
    assert res.score <= 0.20


def test_hybrid_prm_neural_pass():
    mock_model = MockNeuralModel(p_yes=0.92)
    mock_tok = MockTokenizer()
    prm = HybridMathPRM(
        model=mock_model,
        tokenizer=mock_tok,
        device="cpu",
        neural_threshold=0.35,
    )

    res = prm.evaluate_step(
        problem="Calculate Janet's eggs",
        reasoning_history="",
        current_step="Step 1: Janet starts with 16 eggs.",
    )
    assert res.verdict == StepVerdict.CORRECT
    assert res.is_fatal is False
    assert res.score >= 0.80
    assert "Neural Logit" in res.rationale


def test_hybrid_prm_neural_rejects_subtle_error():
    mock_model = MockNeuralModel(p_yes=0.15)  # Neural model strongly detects semantic error
    mock_tok = MockTokenizer()
    prm = HybridMathPRM(
        model=mock_model,
        tokenizer=mock_tok,
        device="cpu",
        neural_threshold=0.35,
    )

    res = prm.evaluate_step(
        problem="Janet eggs",
        reasoning_history="",
        current_step="Step 1: Confusing premise: Jane had half as many so she had 32 eggs.",
    )
    # Neural head flags as suspicious because p_yes < 0.35
    assert res.verdict == StepVerdict.SUSPICIOUS
    assert res.score < 0.35
    assert "Neural verifier rejected" in res.rationale
