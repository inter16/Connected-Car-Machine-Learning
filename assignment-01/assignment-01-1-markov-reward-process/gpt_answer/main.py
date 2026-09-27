import torch

from mrp import MarkovRewardProcess


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
    # 2. Define transition matrix
    #
    #               Next state
    #               A     B     C
    #
    # Current A    0.1   0.6   0.3
    # Current B    0.4   0.4   0.2
    # Current C    0.0   0.3   0.7
    # --------------------------------------------------

    transition_matrix = torch.tensor(
        [
            [0.1, 0.6, 0.3],
            [0.4, 0.4, 0.2],
            [0.0, 0.3, 0.7],
        ],
        dtype=torch.float64,
    )

    # --------------------------------------------------
    # 3. Define rewards
    # --------------------------------------------------

    rewards = torch.tensor(
        [
            5.0,    # State_A
            -1.0,   # State_B
            2.0,    # State_C
        ],
        dtype=torch.float64,
    )

    # --------------------------------------------------
    # 4. Discount factor
    # --------------------------------------------------

    gamma = 0.9

    # --------------------------------------------------
    # 5. Create MRP
    # --------------------------------------------------

    mrp = MarkovRewardProcess(
        states=states,
        transition_matrix=transition_matrix,
        rewards=rewards,
        gamma=gamma,
        seed=42,
    )

    # --------------------------------------------------
    # 6. Simulate trajectory
    # --------------------------------------------------

    trajectory = mrp.simulate(
        start_state="State_A",
        num_steps=10,
    )

    print("Trajectory")
    print("-" * 60)

    for transition in trajectory:
        print(
            f"Step {transition['step']:2d}: "
            f"{transition['state']} "
            f"-> {transition['next_state']} "
            f"| Reward = {transition['reward'].item():.2f}"
        )

    # --------------------------------------------------
    # 7. Calculate sampled discounted return
    # --------------------------------------------------

    total_return = mrp.discounted_return(
        trajectory
    )

    print()
    print(
        f"Discounted return: "
        f"{total_return.item():.6f}"
    )

    # --------------------------------------------------
    # 8. Calculate value function
    # --------------------------------------------------

    iterative_values = (
        mrp.value_function_iterative()
    )

    exact_values = (
        mrp.value_function_exact()
    )

    # --------------------------------------------------
    # 9. Print results
    # --------------------------------------------------

    print()
    print("Value Function")
    print("-" * 60)

    print(
        f"{'State':10s}"
        f"{'Iterative':>15s}"
        f"{'Exact':>15s}"
    )

    for state, iterative_value, exact_value in zip(
        states,
        iterative_values,
        exact_values,
    ):
        print(
            f"{state:10s}"
            f"{iterative_value.item():15.6f}"
            f"{exact_value.item():15.6f}"
        )


if __name__ == "__main__":
    main()