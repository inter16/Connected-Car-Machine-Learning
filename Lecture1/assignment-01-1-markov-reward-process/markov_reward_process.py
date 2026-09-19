import torch


class MarkovRewardProcess:

    def __init__(
        self,
        states,
        transition_matrix,
        rewards,
        gamma,
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

        self.gamma = gamma

        self.n_states = len(self.states)

        self.state_to_index = {
            state: index
            for index, state in enumerate(self.states)
        }

        self.generator = torch.Generator()

        if seed is not None:
            self.generator.manual_seed(seed)


    def get_reward(self, state):
        
        state_index = self.state_to_index[state]

        return self.R[state_index]


    def sample_next_state(self, state):
        
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