import torch

from mdp import MarkovDecisionProcess


def main():
    # --------------------------------------------------
    # 1. Define states
    # --------------------------------------------------

    states = [
        "State_A",
        "State_B",
        "State_C",
    ]

    # --------------------------------------------------
    # 2. Define actions
    # --------------------------------------------------

    actions = [
        "Action_0",
        "Action_1",
    ]

    # --------------------------------------------------
    # 3. Define transition probabilities
    #
    # P[s, a, s']
    #
    # shape:
    # [num_states, num_actions, num_states]
    # --------------------------------------------------

    transition_matrix = torch.tensor(
        [
            # State_A
            [
                [0.7, 0.3, 0.0],  # Action_0
                [0.1, 0.6, 0.3],  # Action_1
            ],

            # State_B
            [
                [0.4, 0.5, 0.1],  # Action_0
                [0.0, 0.2, 0.8],  # Action_1
            ],

            # State_C
            [
                [0.0, 0.4, 0.6],  # Action_0
                [0.2, 0.0, 0.8],  # Action_1
            ],
        ],
        dtype=torch.float64,
    )

    # --------------------------------------------------
    # 4. Define rewards
    #
    # R[s, a]
    #
    # shape:
    # [num_states, num_actions]
    # --------------------------------------------------

    rewards = torch.tensor(
        [
            [5.0, 2.0],  # State_A
            [0.0, 3.0],  # State_B
            [1.0, 4.0],  # State_C
        ],
        dtype=torch.float64,
    )

    # --------------------------------------------------
    # 5. Discount factor
    # --------------------------------------------------

    gamma = 0.9

    # --------------------------------------------------
    # 6. Create MDP
    # --------------------------------------------------

    mdp = MarkovDecisionProcess(
        states=states,
        actions=actions,
        transition_matrix=transition_matrix,
        rewards=rewards,
        gamma=gamma,
    )

    # --------------------------------------------------
    # 7. Value Iteration
    # --------------------------------------------------

    (
        optimal_values,
        optimal_q_values,
        optimal_action_indices,
        num_iterations,
    ) = mdp.value_iteration(
        tolerance=1e-10,
        max_iterations=10000,
    )

    optimal_actions = (
        mdp.action_names_from_indices(
            optimal_action_indices
        )
    )

    # --------------------------------------------------
    # 8. Print V*
    # --------------------------------------------------

    print("Optimal Value Function V*")
    print("-" * 45)

    for state, value in zip(
        states,
        optimal_values,
    ):
        print(
            f"{state:10s}: "
            f"{value.item():12.6f}"
        )

    # --------------------------------------------------
    # 9. Print Q*
    # --------------------------------------------------

    print()
    print("Optimal Action-Value Function Q*")
    print("-" * 45)

    for state_index, state in enumerate(states):
        print(state)

        for action_index, action in enumerate(actions):
            q_value = optimal_q_values[
                state_index,
                action_index,
            ]

            print(
                f"  {action:10s}: "
                f"{q_value.item():12.6f}"
            )

    # --------------------------------------------------
    # 10. Print optimal policy
    # --------------------------------------------------

    print()
    print("Optimal Policy pi*")
    print("-" * 45)

    for state, action in zip(
        states,
        optimal_actions,
    ):
        print(
            f"{state:10s} -> {action}"
        )

    print()
    print(
        f"Value iteration converged in "
        f"{num_iterations} iterations."
    )


if __name__ == "__main__":
    main()





# def main():
#     # --------------------------------------------------
#     # 1. States
#     # --------------------------------------------------

#     states = [
#         "State_A",
#         "State_B",
#         "State_C",
#     ]

#     # --------------------------------------------------
#     # 2. Actions
#     # --------------------------------------------------

#     actions = [
#         "Action_0",
#         "Action_1",
#     ]

#     # --------------------------------------------------
#     # 3. Transition probabilities
#     #
#     # P[s, a, s']
#     #
#     # shape:
#     # [num_states, num_actions, num_states]
#     # --------------------------------------------------

#     transition_matrix = torch.tensor(
#         [
#             # State_A
#             [
#                 [0.7, 0.3, 0.0],  # Action_0
#                 [0.1, 0.6, 0.3],  # Action_1
#             ],

#             # State_B
#             [
#                 [0.4, 0.5, 0.1],  # Action_0
#                 [0.0, 0.2, 0.8],  # Action_1
#             ],

#             # State_C
#             [
#                 [0.0, 0.4, 0.6],  # Action_0
#                 [0.2, 0.0, 0.8],  # Action_1
#             ],
#         ],
#         dtype=torch.float64,
#     )

#     # --------------------------------------------------
#     # 4. Rewards
#     #
#     # R[s, a]
#     #
#     # shape:
#     # [num_states, num_actions]
#     # --------------------------------------------------

#     rewards = torch.tensor(
#         [
#             [5.0, 2.0],  # State_A
#             [0.0, 3.0],  # State_B
#             [1.0, 4.0],  # State_C
#         ],
#         dtype=torch.float64,
#     )

#     # --------------------------------------------------
#     # 5. Discount factor
#     # --------------------------------------------------

#     gamma = 0.9

#     # --------------------------------------------------
#     # 6. Create MDP
#     # --------------------------------------------------

#     mdp = MarkovDecisionProcess(
#         states=states,
#         actions=actions,
#         transition_matrix=transition_matrix,
#         rewards=rewards,
#         gamma=gamma,
#     )

#     # --------------------------------------------------
#     # 7. Bellman optimality equation / Value Iteration
#     # --------------------------------------------------

#     (
#         optimal_values,
#         optimal_q_values,
#         optimal_action_indices,
#         num_iterations,
#     ) = mdp.value_iteration(
#         tolerance=1e-10,
#         max_iterations=10000,
#     )

#     optimal_actions = mdp.action_names_from_indices(
#         optimal_action_indices
#     )

#     # --------------------------------------------------
#     # 8. Print V*
#     # --------------------------------------------------

#     print("Optimal Value Function V*")
#     print("-" * 45)

#     for state, value in zip(
#         states,
#         optimal_values,
#     ):
#         print(
#             f"{state:10s}: "
#             f"{value.item():12.6f}"
#         )

#     # --------------------------------------------------
#     # 9. Print Q*
#     # --------------------------------------------------

#     print()
#     print("Optimal Action-Value Function Q*")
#     print("-" * 45)

#     for state_index, state in enumerate(states):
#         print(state)

#         for action_index, action in enumerate(actions):
#             q_value = optimal_q_values[
#                 state_index,
#                 action_index,
#             ]

#             print(
#                 f"  {action:10s}: "
#                 f"{q_value.item():12.6f}"
#             )

#     # --------------------------------------------------
#     # 10. Print optimal policy
#     # --------------------------------------------------

#     print()
#     print("Optimal Policy pi*")
#     print("-" * 45)

#     for state, action in zip(
#         states,
#         optimal_actions,
#     ):
#         print(
#             f"{state:10s} -> {action}"
#         )

#     print()
#     print(
#         f"Value iteration converged in "
#         f"{num_iterations} iterations."
#     )

#     # --------------------------------------------------
#     # 11. Fix pi* and convert the MDP into an MRP
#     # --------------------------------------------------

#     (
#         policy_transition_matrix,
#         policy_rewards,
#     ) = mdp.policy_mrp_components(
#         optimal_action_indices
#     )

#     optimal_policy_mrp = MarkovRewardProcess(
#         states=states,
#         transition_matrix=policy_transition_matrix,
#         rewards=policy_rewards,
#         gamma=gamma,
#         seed=42,
#     )

#     # --------------------------------------------------
#     # 12. Evaluate pi* as an MRP
#     # --------------------------------------------------

#     mrp_exact_values = (
#         optimal_policy_mrp.value_function_exact()
#     )

#     mrp_iterative_values = (
#         optimal_policy_mrp.value_function_iterative()
#     )

#     # --------------------------------------------------
#     # 13. Verification
#     # --------------------------------------------------

#     print()
#     print("Verification")
#     print("-" * 60)

#     print(
#         f"{'State':10s}"
#         f"{'V*':>15s}"
#         f"{'MRP Exact':>15s}"
#         f"{'MRP Iter':>15s}"
#     )

#     for (
#         state,
#         optimal_value,
#         exact_value,
#         iterative_value,
#     ) in zip(
#         states,
#         optimal_values,
#         mrp_exact_values,
#         mrp_iterative_values,
#     ):
#         print(
#             f"{state:10s}"
#             f"{optimal_value.item():15.6f}"
#             f"{exact_value.item():15.6f}"
#             f"{iterative_value.item():15.6f}"
#         )

#     max_difference = torch.max(
#         torch.abs(
#             optimal_values
#             - mrp_exact_values
#         )
#     )

#     print()
#     print(
#         "V* and V^{pi*} are equal:",
#         torch.allclose(
#             optimal_values,
#             mrp_exact_values,
#             atol=1e-8,
#             rtol=1e-8,
#         ),
#     )

#     print(
#         "Maximum absolute difference:",
#         f"{max_difference.item():.3e}",
#     )


# if __name__ == "__main__":
#     main()
