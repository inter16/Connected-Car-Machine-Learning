"""Exact policy evaluation with the Bellman expectation linear system."""

from __future__ import annotations

import torch

from gridworld_rl.envs import GridWorld


def exact_uniform_policy_values(env: GridWorld, gamma: float = 1.0) -> torch.Tensor:
    """Solve V = r + gamma P V, with the terminal state held at zero.

    This is a reference solution, not an additional learning algorithm.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    nonterminal_states = [s for s in range(env.n_states) if s != env.terminal_state]
    state_to_row = {s: i for i, s in enumerate(nonterminal_states)}
    p = torch.zeros((len(nonterminal_states), len(nonterminal_states)), dtype=torch.float64)
    rewards = torch.full((len(nonterminal_states),), env.step_reward, dtype=torch.float64)

    for state in nonterminal_states:
        for action in range(env.n_actions):
            successor = env.next_state(state, action)
            if successor != env.terminal_state:
                p[state_to_row[state], state_to_row[successor]] += 1.0 / env.n_actions

    a = torch.eye(len(nonterminal_states), dtype=torch.float64) - gamma * p
    solved = torch.linalg.solve(a, rewards)
    values = torch.zeros(env.n_states, dtype=torch.float64)
    values[nonterminal_states] = solved
    return values


def mean_absolute_error(estimated: torch.Tensor, reference: torch.Tensor) -> float:
    if estimated.shape != reference.shape:
        raise ValueError("estimated and reference must have identical shapes")
    return float(torch.mean(torch.abs(estimated - reference)).item())
