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


