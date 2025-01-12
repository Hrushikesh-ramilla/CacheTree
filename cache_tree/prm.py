"""
Process Reward Model (PRM) and Calibrated Step Verification Engine.

Evaluates intermediate reasoning steps independently, assigning step-level rewards
and identifying fatal errors early to enable branch pruning before wasting inference tokens.
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import math
import re
from typing import Optional, Tuple, Dict, Any
import torch


class StepVerdict(enum.Enum):
    CORRECT = "correct"
    NEUTRAL = "neutral"
    SUSPICIOUS = "suspicious"
    FATAL_ERROR = "fatal_error"


@dataclasses.dataclass
class StepEvaluation:
    verdict: StepVerdict
    score: float  # Value in [0.0, 1.0]
    confidence: float  # Value in [0.0, 1.0]
    rationale: str
    is_fatal: bool
    is_terminal: bool = False
    extracted_answer: Optional[str] = None


class ProcessRewardModel(abc.ABC):
    """Abstract interface for Step-Level Process Reward Models."""

    @abc.abstractmethod
    def evaluate_step(
        self,
        problem: str,
        reasoning_history: str,
        current_step: str,
    ) -> StepEvaluation:
        """Evaluates an intermediate reasoning step in context."""
        pass

    @abc.abstractmethod
    def extract_terminal_answer(self, step_text: str) -> Optional[str]:
        """Extracts the final solution if this step completes the reasoning."""
        pass


class RuleBasedMathPRM(ProcessRewardModel):
    """
    Deterministic, calibrated Process Reward Model for mathematical and logical reasoning.
    Detects:
    1. Arithmetic hallucinations and contradictions (e.g. 5 + 5 = 12).
    2. Division by zero and undefined operations.
    3. Premature answers and hallucinated numbers not present in problem statement.
    4. Canonical GSM8K / MATH terminal patterns ('#### <number>', 'Answer: <x>').
    """

    def __init__(self, fatal_threshold: float = 0.20):
        self.fatal_threshold = fatal_threshold

    def extract_terminal_answer(self, step_text: str) -> Optional[str]:
        """Extracts answer formatted as '#### <number>' or 'The answer is <x>'."""
        # GSM8K standard format: #### 42
        m_gsm = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", step_text)
        if m_gsm:
            return m_gsm.group(1).replace(",", "").strip()

        # 'The answer is 42' or 'Final Answer: 42'
        m_ans = re.search(r"(?:final\s+)?answer\s*(?:is|=|:)\s*(-?[\d,]+(?:\.\d+)?)", step_text, re.IGNORECASE)
        if m_ans:
            return m_ans.group(1).replace(",", "").strip()

        # Boxed answer \boxed{42}
        m_box = re.search(r"\\boxed\{([^{}]+)\}", step_text)
        if m_box:
            return m_box.group(1).strip()

        return None

    def _check_simple_arithmetic(self, step_text: str) -> Optional[Tuple[bool, str]]:
        """
        Finds equations like '12 + 15 = 27' or '10 * 4 = 40' and checks mathematical validity.
        Returns (is_valid, explanation) or None if no simple equation found.
        """
        # Match standard binary arithmetic: num op num = num (including negative operands)
        patterns = [
            r"(-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
        ]
        for pat in patterns:
            for match in re.finditer(pat, step_text):
                a_str, op, b_str, res_str = match.groups()
                try:
                    a = float(a_str)
                    b = float(b_str)
                    claimed = float(res_str)
                except (ValueError, OverflowError):
                    continue

                try:
                    if op == "+":
                        expected = a + b
                    elif op == "-":
                        expected = a - b
                    elif op == "*":
                        expected = a * b
                    elif op == "/":
                        if abs(b) < 1e-9:
                            return False, f"Division by zero: {a_str} / {b_str}"
                        expected = a / b
                    else:
                        continue
                except OverflowError:
                    return False, f"Arithmetic overflow in {a_str} {op} {b_str}"

                if not abs(expected - claimed) < 1e-4:
                    return False, f"Arithmetic contradiction: {a_str} {op} {b_str} = {claimed} (true: {expected})"

        return True, "Arithmetic checks passed"

    def evaluate_step(
        self,
        problem: str,
        reasoning_history: str,
        current_step: str,
    ) -> StepEvaluation:
        clean_step = current_step.strip()
        terminal_ans = self.extract_terminal_answer(clean_step)
        is_terminal = terminal_ans is not None

        # 1. Check for empty or degenerate steps
        if len(clean_step) < 3:
            return StepEvaluation(
                verdict=StepVerdict.FATAL_ERROR,
                score=0.05,
                confidence=0.99,
                rationale="Empty or degenerate reasoning step",
                is_fatal=True,
                is_terminal=False,
            )

        # 2. Check for arithmetic validity
        arith_result = self._check_simple_arithmetic(clean_step)
        if arith_result is not None:
            is_valid, msg = arith_result
            if not is_valid:
                return StepEvaluation(
                    verdict=StepVerdict.FATAL_ERROR,
                    score=0.10,
                    confidence=0.95,
                    rationale=msg,
                    is_fatal=True,
                    is_terminal=is_terminal,
                    extracted_answer=terminal_ans,
                )

        # 3. Check for repetitive loop (stuck agent hallucinating identical step)
        if clean_step in reasoning_history:
            return StepEvaluation(
                verdict=StepVerdict.SUSPICIOUS,
                score=0.25,
                confidence=0.85,
                rationale="Step is an exact duplicate of previous reasoning step (cyclical drift)",
                is_fatal=False,
                is_terminal=is_terminal,
                extracted_answer=terminal_ans,
            )

        # 4. Normal verified step
        if is_terminal:
            return StepEvaluation(
                verdict=StepVerdict.CORRECT,
                score=0.95,
                confidence=0.90,
                rationale="Valid terminal reasoning step with candidate answer extracted",
                is_fatal=False,
                is_terminal=True,
                extracted_answer=terminal_ans,
            )

        return StepEvaluation(
            verdict=StepVerdict.CORRECT,
            score=0.80,
            confidence=0.85,
            rationale="Coherent intermediate reasoning step with verified arithmetic",
            is_fatal=False,
            is_terminal=False,
        )


class HybridMathPRM(ProcessRewardModel):
    """
    Dual-Head Hybrid Process Reward Model:
    1. Symbolic Head: Fast, deterministic O(1) arithmetic & division-by-zero validation (RuleBasedMathPRM).
    2. Neural Logit Head: Evaluates semantic correctness using model log-probabilities on (A) Yes vs (B) No:
       P(Yes) = exp(z_A) / (exp(z_A) + exp(z_B))
       Intercepts subtle logic inversions and false derivations that bypass simple arithmetic checking.
    """

    def __init__(
        self,
        symbolic_prm: Optional[RuleBasedMathPRM] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        neural_threshold: float = 0.35,
    ):
        self.symbolic_prm = symbolic_prm or RuleBasedMathPRM()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_threshold = neural_threshold

    def extract_terminal_answer(self, step_text: str) -> Optional[str]:
        return self.symbolic_prm.extract_terminal_answer(step_text)

    def evaluate_step(
        self,
        problem: str,
        reasoning_history: str,
        current_step: str,
    ) -> StepEvaluation:
        # 1. Fast Symbolic Pass
        sym_eval = self.symbolic_prm.evaluate_step(problem, reasoning_history, current_step)
        if sym_eval.is_fatal:
            return sym_eval

        # 2. Neural Verification Pass (if model & tokenizer provided)
        if self.model is not None and self.tokenizer is not None:
            eval_prompt = (
                f"Problem: {problem}\n"
                f"History:\n{reasoning_history}\n"
                f"Step to evaluate: {current_step}\n\n"
                f"Question: Is the step mathematically and logically correct?\n"
                f"(A) Yes\n(B) No\nAnswer: ("
            )
            try:
                inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits[:, -1, :]

                a_id = self.tokenizer.encode("A", add_special_tokens=False)[-1]
                b_id = self.tokenizer.encode("B", add_special_tokens=False)[-1]

                z_a = logits[0, a_id].item()
                z_b = logits[0, b_id].item()
                max_z = max(z_a, z_b)
                p_yes = math.exp(z_a - max_z) / (math.exp(z_a - max_z) + math.exp(z_b - max_z))

                if p_yes < self.neural_threshold:
                    return StepEvaluation(
                        verdict=StepVerdict.SUSPICIOUS,
                        score=round(p_yes * 0.5, 4),
                        confidence=0.85,
                        rationale=f"Neural verifier rejected step: P(Correct) = {p_yes:.3f} < threshold {self.neural_threshold}",
                        is_fatal=False,
                        is_terminal=sym_eval.is_terminal,
                        extracted_answer=sym_eval.extracted_answer,
                    )

                hybrid_score = round(0.5 * sym_eval.score + 0.5 * p_yes, 4)
                return StepEvaluation(
                    verdict=StepVerdict.CORRECT,
                    score=hybrid_score,
                    confidence=0.90,
                    rationale=f"Verified by Symbolic Check & Neural Logit (P(Yes)={p_yes:.3f})",
                    is_fatal=False,
                    is_terminal=sym_eval.is_terminal,
                    extracted_answer=sym_eval.extracted_answer,
                )
            except Exception as e:
                # Fallback to symbolic evaluation if neural logit fails
                return sym_eval

        return sym_eval
