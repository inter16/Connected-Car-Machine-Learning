"""Every-Visit Monte Carlo evaluation of a uniform random policy."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from gridworld_rl.envs import GridWorld


@dataclass(frozen=True, slots=True)
class MCResult:
    values: torch.Tensor
    visit_counts: torch.Tensor
    total_steps: int
    episodes: int


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Calculate G_t for every t, including repeated visits to the same state."""
    returns = torch.empty(len(rewards), dtype=torch.float64)
    running_return = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def train_every_visit_mc(
    env: GridWorld,
    *,
    episodes: int = 50_000,
    gamma: float = 1.0,
    seed: int = 42,
    max_steps_per_episode: int = 100_000,
) -> MCResult:
    """Evaluate pi(a|s)=1/4 with a sample mean over *every* state visit.

    A state can occur several times in one episode: every occurrence is added
    to the numerator and denominator, not just its first occurrence.
    Updates happen after each *complete* episode.
    """
    if episodes <= 0 or max_steps_per_episode <= 0:
        raise ValueError("episodes and max_steps_per_episode must be positive")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")

    rng = random.Random(seed)
    values = torch.zeros(env.n_states, dtype=torch.float64)
    visit_counts = torch.zeros(env.n_states, dtype=torch.int64)
    return_sums = torch.zeros(env.n_states, dtype=torch.float64)
    total_steps = 0

    for episode in range(episodes):
        env.reset()
        states: list[int] = []
        rewards: list[float] = []

        for _ in range(max_steps_per_episode):
            states.append(env.state)  # State S_t, *before* the action
            action = rng.randrange(env.n_actions)
            _, reward, terminated = env.step(action)
            rewards.append(reward)
            if terminated:
                break
        else:
            raise RuntimeError(
                f"Episode {episode + 1} exceeded {max_steps_per_episode} steps. "
                "Refusing to treat a truncated episode as terminal, since "
                "that would bias the Monte Carlo returns."
            )

        total_steps += len(states)
        state_ids = torch.tensor(states, dtype=torch.int64)
        returns = discounted_returns(rewards, gamma)

        # Every occurrence contributes, including duplicate state IDs.
        return_sums.index_add_(0, state_ids, returns)
        visit_counts += torch.bincount(state_ids, minlength=env.n_states)

        observed = visit_counts > 0
        values[observed] = return_sums[observed] / visit_counts[observed]
        # Terminal S15 is not visited as a *nonterminal* S_t: its value is 0.

    return MCResult(values, visit_counts, total_steps, episodes)
