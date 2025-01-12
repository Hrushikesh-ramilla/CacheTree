"""
Tests for Calibrated Process Reward Model (PRM).
"""

import pytest
from cache_tree.prm import RuleBasedMathPRM, StepVerdict, StepEvaluation


def test_prm_valid_arithmetic():
    prm = RuleBasedMathPRM()
    eval1 = prm.evaluate_step(
        problem="What is 15 + 27?",
        reasoning_history="",
        current_step="First we add 15 + 27 = 42.",
    )
    assert eval1.verdict == StepVerdict.CORRECT
    assert eval1.is_fatal is False
    assert eval1.score >= 0.70


def test_prm_arithmetic_contradiction_fatal():
    prm = RuleBasedMathPRM(fatal_threshold=0.20)
    # Flawed calculation: 10 * 4 = 45
    eval_bad = prm.evaluate_step(
        problem="What is 10 times 4?",
        reasoning_history="",
        current_step="Calculating the product: 10 * 4 = 45",
    )
    assert eval_bad.verdict == StepVerdict.FATAL_ERROR
    assert eval_bad.is_fatal is True
    assert eval_bad.score <= 0.20
    assert "contradiction" in eval_bad.rationale.lower()


def test_prm_division_by_zero_fatal():
    prm = RuleBasedMathPRM()
    eval_div0 = prm.evaluate_step(
        problem="Divide x by y",
        reasoning_history="",
        current_step="Next step we evaluate 12 / 0 = 0",
    )
    assert eval_div0.verdict == StepVerdict.FATAL_ERROR
    assert eval_div0.is_fatal is True
    assert "division by zero" in eval_div0.rationale.lower()


def test_prm_terminal_answer_extraction():
    prm = RuleBasedMathPRM()

    # GSM8K standard format
    eval_gsm = prm.evaluate_step(
        problem="Find x",
        reasoning_history="Step 1: x = 10",
        current_step="Therefore, the total is 10. #### 10",
    )
    assert eval_gsm.is_terminal is True
    assert eval_gsm.extracted_answer == "10"
    assert eval_gsm.score >= 0.90

    # Natural language 'The answer is'
    eval_nl = prm.evaluate_step(
        problem="Find y",
        reasoning_history="",
        current_step="Final Answer: 350",
    )
    assert eval_nl.is_terminal is True
    assert eval_nl.extracted_answer == "350"

    # Boxed answer
    eval_box = prm.evaluate_step(
        problem="Find z",
        reasoning_history="",
        current_step="Thus we have \\boxed{100}",
    )
    assert eval_box.is_terminal is True
    assert eval_box.extracted_answer == "100"


def test_prm_cyclical_drift_detection():
    prm = RuleBasedMathPRM()
    history = "Step 1: Let x be the cost of apples.\nStep 2: Apples cost $5 each."
    # Step 3 repeats Step 2 exactly
    eval_rep = prm.evaluate_step(
        problem="Apple cost",
        reasoning_history=history,
        current_step="Step 2: Apples cost $5 each.",
    )
    assert eval_rep.verdict == StepVerdict.SUSPICIOUS
    assert eval_rep.score < 0.50
    assert "duplicate" in eval_rep.rationale.lower()
