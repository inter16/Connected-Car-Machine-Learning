import torch


class MarkovRewardProcess:
    """Finite-state Markov Reward Process."""

    def __init__(
        self,
        states,
        transition_matrix,
        rewards,
        gamma=0.9,
        seed=None,
    ):
        self.states = list(states)

        self.P = torch.as_tensor(
            transition_matrix,
            dtype=torch.float64,
        )

        self.R = torch.as_tensor(
            rewards,
            dtype=torch.float64,
        )

        self.gamma = float(gamma)
        self.n_states = len(self.states)

        self.state_to_index = {
            state: index
            for index, state in enumerate(self.states)
        }

        self.generator = torch.Generator(device="cpu")

        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

        self._validate()

    def _validate(self):
        """MRP 입력값이 올바른지 검사한다."""

        if len(set(self.states)) != self.n_states:
            raise ValueError("State names must be unique.")

        if self.P.shape != (self.n_states, self.n_states):
            raise ValueError(
                "Transition matrix must have shape "
                f"({self.n_states}, {self.n_states})."
            )

        if self.R.shape != (self.n_states,):
            raise ValueError(
                f"Reward vector must have shape ({self.n_states},)."
            )

        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1.")

        if torch.any(self.P < 0):
            raise ValueError(
                "Transition probabilities cannot be negative."
            )

        row_sums = self.P.sum(dim=1)

        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
        ):
            raise ValueError(
                "Each row of the transition matrix must sum to 1."
            )

    def get_reward(self, state):
        """주어진 상태의 기대 보상을 반환한다."""

        if state not in self.state_to_index:
            raise ValueError(f"Unknown state: {state}")

        state_index = self.state_to_index[state]

        return self.R[state_index]

    def sample_next_state(self, state):
        """현재 상태에서 다음 상태를 전이확률에 따라 샘플링한다."""

        if state not in self.state_to_index:
            raise ValueError(f"Unknown state: {state}")

        state_index = self.state_to_index[state]
        probabilities = self.P[state_index]

        next_index = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=self.generator,
        ).item()

        return self.states[next_index]

    def simulate(self, start_state, num_steps=10):
        """MRP trajectory를 생성한다."""

        if start_state not in self.state_to_index:
            raise ValueError(f"Unknown state: {start_state}")

        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        trajectory = []
        current_state = start_state

        for step in range(num_steps):
            reward = self.get_reward(current_state)
            next_state = self.sample_next_state(current_state)

            trajectory.append(
                {
                    "step": step,
                    "state": current_state,
                    "reward": reward,
                    "next_state": next_state,
                }
            )

            current_state = next_state

        return trajectory

    def discounted_return(self, trajectory):
        """주어진 trajectory의 할인 누적 보상을 계산한다."""

        total_return = torch.tensor(
            0.0,
            dtype=self.R.dtype,
        )

        for t, transition in enumerate(trajectory):
            reward = transition["reward"]
            total_return += (self.gamma ** t) * reward

        return total_return

    def value_function_iterative(
        self,
        tolerance=1e-10,
        max_iterations=10000,
    ):
        """
        Bellman expectation equation을 반복 적용한다.

        V_{k+1} = R + gamma * P @ V_k
        """

        values = torch.zeros(
            self.n_states,
            dtype=self.R.dtype,
        )

        for _ in range(max_iterations):
            new_values = (
                self.R
                + self.gamma * (self.P @ values)
            )

            error = torch.max(
                torch.abs(new_values - values)
            )

            values = new_values

            if error.item() < tolerance:
                return values

        raise RuntimeError(
            "MRP value iteration did not converge within "
            f"{max_iterations} iterations."
        )

    def value_function_exact(self):
        """
        선형방정식으로 MRP의 가치함수를 계산한다.

        (I - gamma * P)V = R
        """

        identity = torch.eye(
            self.n_states,
            dtype=self.P.dtype,
        )

        return torch.linalg.solve(
            identity - self.gamma * self.P,
            self.R,
        )
