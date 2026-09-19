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
        """
        Parameters
        ----------
        states : list
            상태 이름 목록.

        transition_matrix : array-like or torch.Tensor
            P[i, j] = 상태 i에서 상태 j로 이동할 확률.

        rewards : array-like or torch.Tensor
            R[i] = 상태 i에서 얻는 기대 보상.

        gamma : float
            할인율 (0 <= gamma < 1).

        seed : int or None
            난수 시드.
        """

        self.states = list(states)

        self.P = torch.as_tensor(
            transition_matrix,
            dtype=torch.float64,
        )

        self.R = torch.as_tensor(
            rewards,
            dtype=torch.float64,
        )

        self.gamma = gamma

        self.n_states = len(self.states)

        self.state_to_index = {
            state: index
            for index, state in enumerate(self.states)
        }

        self.generator = torch.Generator()

        if seed is not None:
            self.generator.manual_seed(seed)

        self._validate()

    def _validate(self):
        """MRP 입력값이 올바른지 검사한다."""

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
            raise ValueError(
                "gamma must satisfy 0 <= gamma < 1."
            )

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
            raise ValueError(
                f"Unknown state: {state}"
            )

        state_index = self.state_to_index[state]

        return self.R[state_index]

    def sample_next_state(self, state):
        """현재 상태에서 다음 상태를 전이확률에 따라 샘플링한다."""

        if state not in self.state_to_index:
            raise ValueError(
                f"Unknown state: {state}"
            )

        state_index = self.state_to_index[state]

        probabilities = self.P[state_index]

        next_index = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=self.generator,
        ).item()

        return self.states[next_index]

    def simulate(
        self,
        start_state,
        num_steps=10,
    ):
        """MRP에서 하나의 trajectory를 생성한다."""

        if start_state not in self.state_to_index:
            raise ValueError(
                f"Unknown state: {start_state}"
            )

        trajectory = []

        current_state = start_state

        for step in range(num_steps):
            reward = self.get_reward(current_state)

            next_state = self.sample_next_state(
                current_state
            )

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

            total_return += (
                self.gamma ** t
            ) * reward

        return total_return

    def value_function_iterative(
        self,
        tolerance=1e-10,
        max_iterations=10000,
    ):
        """
        Bellman equation을 반복 적용하여 가치함수를 계산한다.

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
                break

        return values

    def value_function_exact(self):
        """
        선형방정식을 풀어 가치함수의 정확한 해를 계산한다.

        (I - gamma * P)V = R
        """

        identity = torch.eye(
            self.n_states,
            dtype=self.P.dtype,
        )

        values = torch.linalg.solve(
            identity - self.gamma * self.P,
            self.R,
        )

        return values