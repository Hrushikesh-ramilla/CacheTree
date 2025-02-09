"""
Test-Time Compute Engine: Step-Level MCTS with PRM Verification & KV-Cache Recycling.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Tuple, Any

import torch
from .tree import ThoughtTree, StepNode, NodeStatus, UCTConfig
from .prm import ProcessRewardModel, RuleBasedMathPRM, StepVerdict, StepEvaluation
from .recycler import TreeKVCacheManager, TokenRecyclingBuffer, RecyclingStats


@dataclasses.dataclass
class EngineConfig:
    max_search_rounds: int = 15
    branching_factor: int = 3
    max_tree_depth: int = 8
    pruning_threshold: float = 0.20
    c_puct: float = 1.41421356
    early_stop_on_success: bool = True


@dataclasses.dataclass
class CandidateStepProposal:
    thought_text: str
    delta_tokens: List[int]
    prior_prob: float = 1.0
    key_tensor: Optional[torch.Tensor] = None
    value_tensor: Optional[torch.Tensor] = None
    dynamic_cache: Optional[Any] = None


@dataclasses.dataclass
class ReasoningResult:
    problem: str
    best_solution: Optional[str]
    best_trajectory: List[str]
    total_search_rounds: int
    total_nodes_expanded: int
    pruned_branches_count: int
    terminal_solutions_found: int
    recycling_metrics: Dict[str, Any]
    time_elapsed_seconds: float
    tree: Optional[ThoughtTree] = None
    consensus_distribution: Dict[str, float] = dataclasses.field(default_factory=dict)
    consensus_confidence: float = 0.0
    margin_of_victory: float = 0.0


class TestTimeComputeEngine:
    """
    Orchestrates Test-Time Reasoning Search:
    - Explores multiple step-by-step thinking branches using Monte Carlo Tree Search.
    - Scores each step immediately with a Process Reward Model (PRM).
    - Prunes bad derivation paths at Step 2 before wasting tokens.
    - Uses TreeKVCacheManager to recycle prefix tokens across branches.
    """
    __test__ = False

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        prm: Optional[ProcessRewardModel] = None,
        kv_manager: Optional[TreeKVCacheManager] = None,
    ):
        self.config = config or EngineConfig()
        self.prm = prm or RuleBasedMathPRM(fatal_threshold=self.config.pruning_threshold)
        self.kv_manager = kv_manager or TreeKVCacheManager()

    def solve(
        self,
        problem: str,
        step_proposer_fn: Callable[[str, List[str], int], List[CandidateStepProposal]],
        initial_tokens: Optional[List[int]] = None,
    ) -> ReasoningResult:
        """
        Executes test-time search over reasoning steps for a given problem.

        Args:
            problem: The input problem prompt (math question, logic puzzle).
            step_proposer_fn: Callable(problem, current_steps, num_candidates) -> List[CandidateStepProposal].
            initial_tokens: Optional token IDs of the problem prompt.
        """
        start_time = time.time()
        uct_cfg = UCTConfig(c_puct=self.config.c_puct)
        tree = ThoughtTree(root_prompt=problem, root_tokens=initial_tokens, config=uct_cfg)

        root_len = len(initial_tokens) if initial_tokens is not None else max(1, len(problem.split()))
        self.kv_manager.store_cache(node_id=tree.root_id, token_count=root_len)
        if hasattr(step_proposer_fn, "kv_manager"):
            step_proposer_fn.kv_manager = self.kv_manager

        pruned_count = 0
        terminal_solutions = 0
        current_active_node = tree.root_id

        for round_idx in range(self.config.max_search_rounds):
            # 1. Selection: traverse down to a leaf using normalized UCT
            leaf, path = tree.traverse_to_leaf()

            # 2. Rewind KV-Cache to Lowest Common Ancestor
            if current_active_node != leaf.node_id:
                lca_id, preserved = self.kv_manager.rewind_to_lca(
                    current_node_id=current_active_node,
                    target_node_id=leaf.node_id,
                    tree=tree,
                )
                current_active_node = leaf.node_id

            # 3. Handle terminal or pruned leaf
            if leaf.is_terminal:
                val = 1.0 if leaf.status == NodeStatus.TERMINAL_SUCCESS else 0.0
                tree.backpropagate(leaf.node_id, terminal_value=val)
                continue

            if leaf.depth >= self.config.max_tree_depth:
                leaf.status = NodeStatus.TERMINAL_FAILURE
                leaf.is_terminal = True
                tree.backpropagate(leaf.node_id, terminal_value=0.0)
                continue

            # 4. Expansion: Propose candidate next steps
            history_nodes = tree.get_trajectory_from_root(leaf.node_id)
            history_steps = [n.thought_step for n in history_nodes[1:]]  # Exclude root prompt
            history_text = "\n".join(history_steps)

            self.kv_manager.active_node_id = leaf.node_id
            candidates = step_proposer_fn(problem, history_steps, self.config.branching_factor)

            best_round_score = 0.0
            expanded_children = []

            for cand in candidates:
                # PRM evaluates this specific intermediate step
                eval_result: StepEvaluation = self.prm.evaluate_step(
                    problem=problem,
                    reasoning_history=history_text,
                    current_step=cand.thought_text,
                )

                # Determine status based on PRM verdict
                if eval_result.is_fatal or eval_result.score < self.config.pruning_threshold:
                    child_status = NodeStatus.PRUNED
                    is_terminal = True
                    pruned_count += 1
                elif eval_result.is_terminal:
                    child_status = NodeStatus.TERMINAL_SUCCESS
                    is_terminal = True
                    terminal_solutions += 1
                else:
                    child_status = NodeStatus.ACTIVE
                    is_terminal = False

                # Register token recycling transition
                parent_tokens_len = len(leaf.cumulative_tokens)
                delta_tokens_len = len(cand.delta_tokens)
                child = tree.add_child(
                    parent_id=leaf.node_id,
                    thought_step=cand.thought_text,
                    delta_tokens=cand.delta_tokens,
                    prior_prob=cand.prior_prob,
                    step_reward=eval_result.score,
                    is_terminal=is_terminal,
                    solution=eval_result.extracted_answer,
                    status=child_status,
                )

                self.kv_manager.register_step_transition(
                    parent_id=leaf.node_id,
                    child_id=child.node_id,
                    parent_token_count=parent_tokens_len,
                    delta_token_count=delta_tokens_len,
                    key_tensor=cand.key_tensor,
                    value_tensor=cand.value_tensor,
                    dynamic_cache=cand.dynamic_cache,
                )
                expanded_children.append((child, eval_result.score))
                if eval_result.score > best_round_score:
                    best_round_score = eval_result.score

            # 5. Backpropagation
            tree.backpropagate(leaf.node_id, terminal_value=best_round_score)

            # 6. Early stopping if confident terminal solution found
            if self.config.early_stop_on_success and terminal_solutions > 0:
                best_traj = tree.get_best_trajectory()
                if best_traj[-1].status == NodeStatus.TERMINAL_SUCCESS and best_traj[-1].solution is not None:
                    break

        elapsed = time.time() - start_time
        best_trajectory_nodes = tree.get_best_trajectory()
        best_steps = [n.thought_step for n in best_trajectory_nodes]


        best_solution = best_trajectory_nodes[-1].solution if best_trajectory_nodes else None
        return ReasoningResult(
            problem=problem, best_solution=best_solution, best_trajectory=best_steps,
            total_search_rounds=round_idx + 1, total_nodes_expanded=len(tree.nodes),
            pruned_branches_count=pruned_count, terminal_solutions_found=terminal_solutions,
            recycling_metrics=self.kv_manager.get_summary(), time_elapsed_seconds=elapsed, tree=tree,
        )
