import torch

from markov_decision_process import MarkovDecisionProcess

def main():

    states = [
        "Class1",
        "Class2",
        "Class3",
        "Facebook",
        "Sleep"
    ]

    actions = [
        "Study",
        "Pub",
        "Facebook",
        "Quit",
        "Sleep"
    ]

    state_transition_matrix = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],

            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],

            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.2, 0.4, 0.4, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],

            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],

            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            
        ],
    )

    rewards = torch.tensor(
        [
            [-2.0, 0.0, -1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    discount_factor = 1.0

    mdp = MarkovDecisionProcess(
        states=states,
        actions=actions,
        transition_matrix=state_transition_matrix,
        rewards=rewards,
        gamma=discount_factor,
    )





    (
        optimal_values,
        optimal_q_values,
        optimal_action_indices,
        num_iterations,
    ) = mdp.value_iteration(
        tolerance=1e-10,
        max_iterations=10000,
    )

    optimal_actions = mdp.action_names_from_indices(
        optimal_action_indices
    )




    # print("Optimal Value Function V*")
    # print("-" * 45)

    # for state, value in zip(
    #     states,
    #     optimal_values,
    # ):
    #     print(
    #         f"{state:10s}: "
    #         f"{value.item():12.6f}"
    #     )





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
