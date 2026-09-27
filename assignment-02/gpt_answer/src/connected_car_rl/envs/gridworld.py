from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GridWorld:
    """Deterministic rectangular GridWorld used in Assignment 2.

    State numbering is row-major:

        s0   s1   s2   s3
        s4   s5   s6   s7
        s8   s9  s10  s11
       s12  s13  s14  s15

    The assignment designates s0 as the initial state and s15 as the
    terminal state. Actions that would leave the grid keep the agent in
    the same state.
    """

    rows: int = 4
    cols: int = 4
    initial_state: int = 0
    terminal_state: int = 15
    step_reward: float = -1.0
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float64

    # Action order: North, South, East, West
    action_deltas: tuple[tuple[int, int], ...] = field(
        default=((-1, 0), (1, 0), (0, 1), (0, -1)), init=False
    )

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.num_states = self.rows * self.cols
        self.num_actions = len(self.action_deltas)

        if not 0 <= self.initial_state < self.num_states:
            raise ValueError("initial_state is outside the grid.")
        if not 0 <= self.terminal_state < self.num_states:
            raise ValueError("terminal_state is outside the grid.")

        self.next_state = self._build_next_state_table()
        self.reward = torch.full(
            (self.num_states, self.num_actions),
            self.step_reward,
            dtype=self.dtype,
            device=self.device,
        )

        # No additional reward/cost is accumulated after termination.
        self.reward[self.terminal_state, :] = 0.0

        self.terminal_mask = torch.zeros(
            self.num_states, dtype=torch.bool, device=self.device
        )
        self.terminal_mask[self.terminal_state] = True

    def _build_next_state_table(self) -> torch.Tensor:
        table = torch.empty(
            (self.num_states, self.num_actions),
            dtype=torch.long,
            device=self.device,
        )

        for state in range(self.num_states):
            if state == self.terminal_state:
                table[state, :] = state
                continue

            row, col = divmod(state, self.cols)
            for action, (d_row, d_col) in enumerate(self.action_deltas):
                next_row = min(max(row + d_row, 0), self.rows - 1)
                next_col = min(max(col + d_col, 0), self.cols - 1)
                table[state, action] = next_row * self.cols + next_col

        return table

    def uniform_random_policy(self) -> torch.Tensor:
        """Return pi(a|s)=1/4 for every action in every non-terminal state."""
        return torch.full(
            (self.num_states, self.num_actions),
            1.0 / self.num_actions,
            dtype=self.dtype,
            device=self.device,
        )

    def as_grid(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() != self.num_states:
            raise ValueError(
                f"Expected {self.num_states} state values, got {values.numel()}."
            )
        return values.reshape(self.rows, self.cols)
