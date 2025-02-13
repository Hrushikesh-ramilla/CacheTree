"""
End-to-End Tests for TestTimeComputeEngine.
"""

import pytest
from cache_tree.engine import (
    TestTimeComputeEngine,
    EngineConfig,
    CandidateStepProposal,
    ReasoningResult,
)
from cache_tree.tree import NodeStatus


def mock_step_proposer(problem: str, history: list[str], branching_factor: int):
    """
    Simulates an LLM proposing reasoning steps for:
    'Janet has 16 eggs. She buys 3 packs of 12. She uses 4 to bake. How many left?'
    """
    step_num = len(history) + 1

    if step_num == 1:
        return [
            CandidateStepProposal(
                thought_text="Step 1: 3 packs of 12 eggs gives 3 * 12 = 36 eggs.",
                delta_tokens=list(range(10)),
                prior_prob=0.8,
            ),
            # Flawed candidate (should be pruned by PRM)
            CandidateStepProposal(
                thought_text="Step 1: 3 packs of 12 eggs gives 3 * 12 = 50 eggs.",
                delta_tokens=list(range(10)),
                prior_prob=0.2,
            ),
        ]
    elif step_num == 2:
        return [
            CandidateStepProposal(
                thought_text="Step 2: Total eggs before baking is 16 + 36 = 52 eggs.",
                delta_tokens=list(range(10)),
                prior_prob=0.9,
            ),
            # Flawed candidate (arithmetic error)
            CandidateStepProposal(
                thought_text="Step 2: Total eggs is 16 + 36 = 60 eggs.",
                delta_tokens=list(range(10)),
                prior_prob=0.1,
            ),
        ]
    elif step_num == 3:
        return [
            CandidateStepProposal(
                thought_text="Step 3: After baking, remaining eggs is 52 - 4 = 48 eggs. #### 48",
                delta_tokens=list(range(12)),
                prior_prob=0.95,
            ),
        ]
    else:
        return []


def test_engine_solves_math_problem_with_pruning():
    engine = TestTimeComputeEngine(
        config=EngineConfig(
            max_search_rounds=10,
            branching_factor=2,
            pruning_threshold=0.20,
            early_stop_on_success=True,
        )
    )

    problem = "Janet has 16 eggs. She buys 3 packs of 12 eggs. She uses 4 eggs. How many left?"
    result = engine.solve(
        problem=problem,
        step_proposer_fn=mock_step_proposer,
        initial_tokens=list(range(25)),
    )

    assert isinstance(result, ReasoningResult)
    # Correct solution extracted
    assert result.best_solution == "48"
    # Bad steps were pruned
    assert result.pruned_branches_count >= 2
    # At least one terminal success found
    assert result.terminal_solutions_found >= 1
    # Check recycling metrics
    metrics = result.recycling_metrics
    assert metrics["tokens_saved"] > 0
    assert metrics["recycling_efficiency_pct"] >= 50.0
    # Final node in trajectory is terminal success
    assert result.tree is not None
    best_traj = result.tree.get_best_trajectory()
    assert best_traj[-1].status == NodeStatus.TERMINAL_SUCCESS
