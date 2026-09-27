import torch


class MarkovDecisionProcess:

    def __init__(
        self,
        states,
        actions,
        transition_matrix,
        rewards,
        gamma,
    ):
        self.states = list(states)
        self.actions = list(actions)

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
        self.n_actions = len(self.actions)

        self.state_to_index = {
            state: index
            for index, state in enumerate(self.states)
        }

        self.action_to_index = {
            action: index
            for index, action in enumerate(self.actions)
        }


    def q_values_from_values(self, values):

        values = torch.as_tensor(
            values,
            dtype=self.P.dtype,
        )

        if values.shape != (self.n_states,):
            raise ValueError(
                f"values must have shape ({self.n_states},)."
            )

        expected_next_values = torch.matmul(
            self.P,
            values,
        )

        return (
            self.R
            + self.gamma * expected_next_values
        )

    def value_iteration(
        self,
        tolerance=1e-10,
        max_iterations=10000,
    ):

        values = torch.zeros(
            self.n_states,
            dtype=self.P.dtype,
        )

        for iteration in range(1, max_iterations + 1):
            q_values = self.q_values_from_values(values)

            new_values = torch.max(
                q_values,
                dim=1,
            ).values

            error = torch.max(
                torch.abs(new_values - values)
            )

            values = new_values

            if error.item() < tolerance:
                q_values = self.q_values_from_values(values)

                optimal_action_indices = torch.argmax(
                    q_values,
                    dim=1,
                )

                return (
                    values,
                    q_values,
                    optimal_action_indices,
                    iteration,
                )

        raise RuntimeError(
            "Value iteration did not converge within "
            f"{max_iterations} iterations."
        )

    def action_names_from_indices(self, action_indices):

        return [
            self.actions[action_index.item()]
            for action_index in action_indices
        ]
