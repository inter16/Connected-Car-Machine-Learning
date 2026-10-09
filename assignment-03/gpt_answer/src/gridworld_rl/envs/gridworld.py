"""A deterministic 4x4 grid with a uniformly random *external* policy.

Sources: HW3.pdf (examples 1 and 2); Lecture 3, pp. 8 and 15;
Lecture 2, p. 11 (boundary actions remain at the current state).
The HW3 / Lecture 3 task has S0 as its start state and S15 as its only
terminal state; Lecture 2, pp. 9-11 instead show a distinct two-terminal task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass(slots=True)
class GridWorld:
    size: int = 4
    start_state: int = 0
    terminal_state: int = 15
    step_reward: float = -1.0
    state: int = field(init=False)
    done: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.size != 4 or self.start_state != 0 or self.terminal_state != 15:
            raise ValueError("HW3 uses exactly a 4x4 grid, start S0, goal S15")
        self.reset()

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return len(Action)

    def reset(self) -> int:
        self.state = self.start_state
        self.done = False
        return self.state

    def next_state(self, state: int, action: int | Action) -> int:
        """Pure dynamics: actions into a wall leave the agent in place."""
        if not (0 <= state < self.n_states):
            raise ValueError("Invalid state")
        if state == self.terminal_state:
            raise ValueError("Terminal state has no outgoing actions")
        try:
            movement = Action(action)
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

        row, col = divmod(state, self.size)
        if movement is Action.UP:
            row = max(row - 1, 0)
        elif movement is Action.DOWN:
            row = min(row + 1, self.size - 1)
        elif movement is Action.LEFT:
            col = max(col - 1, 0)
        else:
            col = min(col + 1, self.size - 1)
        return row * self.size + col

    def step(self, action: int | Action) -> tuple[int, float, bool]:
        """Return (next_state, reward, terminated) for one action."""
        if self.done:
            raise RuntimeError("Episode has finished; call reset() before step()")
        self.state = self.next_state(self.state, action)
        self.done = self.state == self.terminal_state
        return self.state, self.step_reward, self.done
