import torch


class MarkovDecisionProcess:
    """Finite-state, finite-action Markov Decision Process."""

    def __init__(
        self,
        states,
        actions,
        transition_matrix,
        rewards,
        gamma=0.9,
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

        self._validate()

    def _validate(self):
        """MDP 입력값이 올바른지 검사한다."""

        expected_transition_shape = (
            self.n_states,
            self.n_actions,
            self.n_states,
        )

        if self.P.shape != expected_transition_shape:
            raise ValueError(
                "Transition tensor must have shape "
                f"{expected_transition_shape}."
            )

        expected_reward_shape = (
            self.n_states,
            self.n_actions,
        )

        if self.R.shape != expected_reward_shape:
            raise ValueError(
                "Reward tensor must have shape "
                f"{expected_reward_shape}."
            )

        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(
                "gamma must satisfy 0 <= gamma < 1."
            )

        if torch.any(self.P < 0):
            raise ValueError(
                "Transition probabilities cannot be negative."
            )

        row_sums = self.P.sum(dim=2)

        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
        ):
            raise ValueError(
                "Transition probabilities for each "
                "state-action pair must sum to 1."
            )

    def q_values_from_values(self, values):
        """
        현재 value function으로부터 Q(s, a)를 계산한다.

        Q(s, a)
        = R(s, a)
        + gamma * sum_{s'} P(s' | s, a) V(s')
        """

        values = torch.as_tensor(
            values,
            dtype=self.P.dtype,
        )

        expected_next_values = torch.matmul(
            self.P,
            values,
        )

        q_values = (
            self.R
            + self.gamma * expected_next_values
        )

        return q_values

    def value_iteration(
        self,
        tolerance=1e-10,
        max_iterations=10000,
    ):
        """
        Bellman optimality equation을 반복하여
        optimal value function과 optimal policy를 계산한다.

        V_{k+1}(s) = max_a Q_k(s, a)
        """

        values = torch.zeros(
            self.n_states,
            dtype=self.P.dtype,
        )

        for iteration in range(
            1,
            max_iterations + 1,
        ):
            # ------------------------------------------
            # Q(s, a)
            # ------------------------------------------

            q_values = self.q_values_from_values(
                values
            )

            # ------------------------------------------
            # V(s) = max_a Q(s, a)
            # ------------------------------------------

            new_values = torch.max(
                q_values,
                dim=1,
            ).values

            # ------------------------------------------
            # Convergence check
            # ------------------------------------------

            error = torch.max(
                torch.abs(
                    new_values - values
                )
            )

            values = new_values

            if error.item() < tolerance:
                break
        else:
            raise RuntimeError(
                "Value iteration did not converge."
            )

        # 최종 V*를 이용해 Q*를 다시 계산
        optimal_q_values = (
            self.q_values_from_values(values)
        )

        # pi*(s) = argmax_a Q*(s, a)
        optimal_action_indices = torch.argmax(
            optimal_q_values,
            dim=1,
        )

        return (
            values,
            optimal_q_values,
            optimal_action_indices,
            iteration,
        )

    def action_names_from_indices(
        self,
        action_indices,
    ):
        """action index를 action 이름으로 변환한다."""

        return [
            self.actions[index.item()]
            for index in action_indices
        ]





# class MarkovDecisionProcess:
#     """Finite-state, finite-action Markov Decision Process."""

#     def __init__(
#         self,
#         states,
#         actions,
#         transition_matrix,
#         rewards,
#         gamma=0.9,
#     ):
#         self.states = list(states)
#         self.actions = list(actions)

#         self.P = torch.as_tensor(
#             transition_matrix,
#             dtype=torch.float64,
#         )

#         self.R = torch.as_tensor(
#             rewards,
#             dtype=torch.float64,
#         )

#         self.gamma = float(gamma)

#         self.n_states = len(self.states)
#         self.n_actions = len(self.actions)

#         self.state_to_index = {
#             state: index
#             for index, state in enumerate(self.states)
#         }

#         self.action_to_index = {
#             action: index
#             for index, action in enumerate(self.actions)
#         }

#         self._validate()

#     def _validate(self):
#         """MDP 입력값이 올바른지 검사한다."""

#         if len(set(self.states)) != self.n_states:
#             raise ValueError("State names must be unique.")

#         if len(set(self.actions)) != self.n_actions:
#             raise ValueError("Action names must be unique.")

#         expected_transition_shape = (
#             self.n_states,
#             self.n_actions,
#             self.n_states,
#         )

#         if self.P.shape != expected_transition_shape:
#             raise ValueError(
#                 "Transition tensor must have shape "
#                 f"{expected_transition_shape}."
#             )

#         expected_reward_shape = (
#             self.n_states,
#             self.n_actions,
#         )

#         if self.R.shape != expected_reward_shape:
#             raise ValueError(
#                 "Reward tensor must have shape "
#                 f"{expected_reward_shape}."
#             )

#         if not 0.0 <= self.gamma < 1.0:
#             raise ValueError("gamma must satisfy 0 <= gamma < 1.")

#         if torch.any(self.P < 0):
#             raise ValueError(
#                 "Transition probabilities cannot be negative."
#             )

#         row_sums = self.P.sum(dim=2)

#         if not torch.allclose(
#             row_sums,
#             torch.ones_like(row_sums),
#         ):
#             raise ValueError(
#                 "For every state-action pair, transition "
#                 "probabilities must sum to 1."
#             )

#     def q_values_from_values(self, values):
#         """
#         현재 가치함수 V로부터 모든 Q(s, a)를 계산한다.

#         Q(s, a)
#         = R(s, a)
#         + gamma * sum_{s'} P(s' | s, a) V(s')
#         """

#         values = torch.as_tensor(
#             values,
#             dtype=self.P.dtype,
#         )

#         if values.shape != (self.n_states,):
#             raise ValueError(
#                 f"values must have shape ({self.n_states},)."
#             )

#         expected_next_values = torch.matmul(
#             self.P,
#             values,
#         )

#         return (
#             self.R
#             + self.gamma * expected_next_values
#         )

#     def value_iteration(
#         self,
#         tolerance=1e-10,
#         max_iterations=10000,
#     ):
#         """
#         Bellman optimality equation을 반복 적용하여
#         V*, Q*, 최적 deterministic policy를 계산한다.

#         V_{k+1}(s) = max_a Q_k(s, a)
#         """

#         values = torch.zeros(
#             self.n_states,
#             dtype=self.P.dtype,
#         )

#         for iteration in range(1, max_iterations + 1):
#             q_values = self.q_values_from_values(values)

#             new_values = torch.max(
#                 q_values,
#                 dim=1,
#             ).values

#             error = torch.max(
#                 torch.abs(new_values - values)
#             )

#             values = new_values

#             if error.item() < tolerance:
#                 q_values = self.q_values_from_values(values)

#                 optimal_action_indices = torch.argmax(
#                     q_values,
#                     dim=1,
#                 )

#                 return (
#                     values,
#                     q_values,
#                     optimal_action_indices,
#                     iteration,
#                 )

#         raise RuntimeError(
#             "Value iteration did not converge within "
#             f"{max_iterations} iterations."
#         )

#     def policy_mrp_components(self, action_indices):
#         """
#         deterministic policy를 고정하여 MRP의 P_pi, R_pi를 만든다.

#         action_indices[s] = 상태 s에서 선택할 행동 index
#         """

#         action_indices = torch.as_tensor(
#             action_indices,
#             dtype=torch.long,
#         )

#         if action_indices.shape != (self.n_states,):
#             raise ValueError(
#                 "action_indices must have shape "
#                 f"({self.n_states},)."
#             )

#         if torch.any(action_indices < 0) or torch.any(
#             action_indices >= self.n_actions
#         ):
#             raise ValueError(
#                 "action_indices contains an invalid action index."
#             )

#         state_indices = torch.arange(
#             self.n_states,
#             dtype=torch.long,
#         )

#         policy_transition_matrix = self.P[
#             state_indices,
#             action_indices,
#         ]

#         policy_rewards = self.R[
#             state_indices,
#             action_indices,
#         ]

#         return (
#             policy_transition_matrix,
#             policy_rewards,
#         )

#     def action_names_from_indices(self, action_indices):
#         """행동 index tensor를 행동 이름 목록으로 변환한다."""

#         return [
#             self.actions[action_index.item()]
#             for action_index in action_indices
#         ]