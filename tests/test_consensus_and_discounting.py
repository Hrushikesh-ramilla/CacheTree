"""
Tests for Path-Efficiency Discounting and Weighted Self-Consistency Consensus.
"""

import pytest
from cache_tree.tree import ThoughtTree, NodeStatus, UCTConfig
from cache_tree.engine import (
    TestTimeComputeEngine,
    EngineConfig,
    CandidateStepProposal,
    ReasoningResult,
)


def test_gamma_discounting_in_backpropagation():
    """
    Verifies that deeper paths receive exponentially discounted values:
    V(s_t) = V_terminal * gamma^(leaf_depth - curr_depth).
    """
    gamma = 0.90
    cfg = UCTConfig(gamma_discount=gamma)
    tree = ThoughtTree(root_prompt="Root", config=cfg)

    # Root (depth 0) -> c1 (depth 1) -> c2 (depth 2) -> c3 (depth 3, terminal)
    c1 = tree.add_child(tree.root_id, "1", [1])
    c2 = tree.add_child(c1.node_id, "2", [2])
    c3 = tree.add_child(c2.node_id, "3", [3], is_terminal=True, solution="42")

    # Backpropagate terminal value = 1.0 from c3
    tree.backpropagate(c3.node_id, terminal_value=1.0)

    # c3 (depth 3, dist 0): value = 1.0 * (0.9^0) = 1.0
    assert pytest.approx(c3.total_value, rel=1e-4) == 1.0

    # c2 (depth 2, dist 1): value = 1.0 * (0.9^1) = 0.9
    assert pytest.approx(c2.total_value, rel=1e-4) == 0.90

    # c1 (depth 1, dist 2): value = 1.0 * (0.9^2) = 0.81
    assert pytest.approx(c1.total_value, rel=1e-4) == 0.81

    # root (depth 0, dist 3): initial 0.5 + 1.0 * (0.9^3) = 0.5 + 0.729 = 1.229
    assert pytest.approx(tree.get_node(tree.root_id).total_value, rel=1e-4) == 0.5 + (0.9 ** 3)


def test_weighted_self_consistency_consensus():
    """
    Verifies that TestTimeComputeEngine aggregates multiple terminal solutions,
    weighting answers by cumulative visit-Q confidence to pick the true consensus.
    """
    def multi_path_proposer(problem, history, b_factor):
        step_idx = len(history) + 1
        if step_idx == 1:
            return [
                CandidateStepProposal("Step 1: Path A", [1], prior_prob=0.7),
                CandidateStepProposal("Step 1: Path B", [2], prior_prob=0.3),
            ]
        elif step_idx == 2:
            last = history[-1]
            if "Path A" in last:
                # Path A produces the correct answer '48'
                return [
                    CandidateStepProposal("Step 2A: Answer is 48. #### 48", [3], prior_prob=0.9),
                ]
            else:
                # Path B produces an alternative answer '50'
                return [
                    CandidateStepProposal("Step 2B: Answer is 50. #### 50", [4], prior_prob=0.8),
                ]
        return []

    engine = TestTimeComputeEngine(
        config=EngineConfig(
            max_search_rounds=6,
            branching_factor=2,
            early_stop_on_success=False,  # Allow exploring multiple paths
        )
    )

    res = engine.solve(
        problem="Calculate Janet eggs",
        step_proposer_fn=multi_path_proposer,
        initial_tokens=[1, 2],
    )

    assert isinstance(res, ReasoningResult)
    # The higher-prior / higher-Q path (48) must dominate the consensus
    assert res.best_solution == "48"
    assert "48" in res.consensus_distribution
    assert res.consensus_distribution["48"] > 0.50
    assert res.consensus_confidence > 0.50
    assert res.margin_of_victory >= 0.0
