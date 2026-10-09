"""One-step online temporal-difference (TD(0)) policy evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from gridworld_rl.envs import GridWorld


@dataclass(frozen=True, slots=True)
class TDResult:
    values: torch.Tensor
    visit_counts: torch.Tensor
    total_steps: int
    episodes: int
    alpha: float


def td0_update(
    values: torch.Tensor,
    state: int,
    reward: float,
    next_state: int,
    terminated: bool,
    *,
    alpha: float,
    gamma: float,
) -> None:
    """Update V(S_t) *immediately* after observing (S_t, R, S_{t+1})."""
    old_value = float(values[state].item())
    next_value = 0.0 if terminated else float(values[next_state].item())
    target = reward + gamma * next_value
    values[state] = old_value + alpha * (target - old_value)


def train_td0(
    env: GridWorld,
    *,
    episodes: int = 50_000,
    gamma: float = 1.0,
    alpha: float = 0.01,
    seed: int = 42,
    max_steps_per_episode: int = 100_000,
) -> TDResult:
    """Evaluate pi(a|s)=1/4 using an online TD(0) update per transition."""
    if episodes <= 0 or max_steps_per_episode <= 0:
        raise ValueError("episodes and max_steps_per_episode must be positive")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")

    rng = random.Random(seed)
    values = torch.zeros(env.n_states, dtype=torch.float64)
    visit_counts = torch.zeros(env.n_states, dtype=torch.int64)
    total_steps = 0

    for episode in range(episodes):
        state = env.reset()
        for _ in range(max_steps_per_episode):
            action = rng.randrange(env.n_actions)
            next_state, reward, terminated = env.step(action)

            td0_update(
                values,
                state,
                reward,
                next_state,
                terminated,
                alpha=alpha,
                gamma=gamma,
            )
            visit_counts[state] += 1
            total_steps += 1
            if terminated:
                break
            state = next_state
        else:
            raise RuntimeError(
                f"Episode {episode + 1} exceeded {max_steps_per_episode} steps. "
                "Refusing to silently truncate the episode."
            )

    return TDResult(values, visit_counts, total_steps, episodes, alpha)
