from __future__ import annotations

import torch

from connected_car_rl.envs import GridWorld


def _initial_values(env: GridWorld, initial_value: float) -> torch.Tensor:
    values = torch.full(
        (env.num_states,),
        float(initial_value),
        dtype=env.dtype,
        device=env.device,
    )
    values[env.terminal_state] = 0.0
    return values


def policy_evaluation(
    env: GridWorld,
    *,
    gamma: float = 1.0,
    iterations: int = 1000,
    initial_value: float = 0.0,
    policy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a fixed policy with synchronous Bellman expectation updates.

    For Assignment 2, ``policy`` defaults to the uniform random policy over
    North/South/East/West. Exactly ``iterations`` updates are performed.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")

    if policy is None:
        policy = env.uniform_random_policy()
    else:
        policy = policy.to(device=env.device, dtype=env.dtype)

    expected_shape = (env.num_states, env.num_actions)
    if tuple(policy.shape) != expected_shape:
        raise ValueError(f"policy shape must be {expected_shape}.")

    values = _initial_values(env, initial_value)

    for _ in range(iterations):
        old_values = values

        # values[next_state] has shape [num_states, num_actions].
        q_values = env.reward + gamma * old_values[env.next_state]
        new_values = torch.sum(policy * q_values, dim=1)

        # The value of a terminal state is fixed at zero.
        new_values[env.terminal_state] = 0.0
        values = new_values

    return values


def value_iteration(
    env: GridWorld,
    *,
    gamma: float = 1.0,
    iterations: int = 1000,
    initial_value: float = 0.0,
) -> torch.Tensor:
    """Compute the optimal state-value function with Bellman optimality updates."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")

    values = _initial_values(env, initial_value)

    for _ in range(iterations):
        old_values = values
        q_values = env.reward + gamma * old_values[env.next_state]
        new_values = torch.max(q_values, dim=1).values
        new_values[env.terminal_state] = 0.0
        values = new_values

    return values
