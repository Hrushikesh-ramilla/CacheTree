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

