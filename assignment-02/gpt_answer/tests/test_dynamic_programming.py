import torch

from connected_car_rl.algorithms import policy_evaluation, value_iteration
from connected_car_rl.envs import GridWorld


def test_policy_evaluation_after_1000_iterations() -> None:
    env = GridWorld()
    values = policy_evaluation(env, iterations=1000, initial_value=0.0)

    expected = torch.tensor(
        [
            [-59.42857143, -57.42857143, -54.28571429, -51.71428571],
            [-57.42857143, -54.57142857, -49.71428571, -45.14285714],
            [-54.28571429, -49.71428571, -40.85714286, -30.0],
            [-51.71428571, -45.14285714, -30.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(env.as_grid(values), expected, atol=1e-7)


def test_policy_evaluation_initialization_does_not_change_limit() -> None:
    env = GridWorld()
    from_zero = policy_evaluation(env, iterations=1000, initial_value=0.0)
    from_ten = policy_evaluation(env, iterations=1000, initial_value=10.0)
    assert torch.allclose(from_zero, from_ten, atol=1e-7)


def test_value_iteration_after_1000_iterations() -> None:
    env = GridWorld()
    values = value_iteration(env, iterations=1000)

    expected = torch.tensor(
        [
            [-6.0, -5.0, -4.0, -3.0],
            [-5.0, -4.0, -3.0, -2.0],
            [-4.0, -3.0, -2.0, -1.0],
            [-3.0, -2.0, -1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert torch.equal(env.as_grid(values), expected)
