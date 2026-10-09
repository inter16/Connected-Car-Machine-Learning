import torch
import pytest

from gridworld_rl.agents import train_every_visit_mc, train_td0
from gridworld_rl.agents.every_visit_mc import discounted_returns
from gridworld_rl.agents.td0 import td0_update
from gridworld_rl.envs import GridWorld
from gridworld_rl.utils.analysis import exact_uniform_policy_values


def test_returns_keep_every_occurrence():
    # If S0 occurs at both t=0 and t=2, its two returns are -3 and -1.
    states = torch.tensor([0, 1, 0])
    g = discounted_returns([-1, -1, -1], gamma=1.0)
    total = torch.zeros(2, dtype=torch.float64)
    total.index_add_(0, states, g)
    counts = torch.bincount(states, minlength=2)
    assert g.tolist() == [-3.0, -2.0, -1.0]
    assert float(total[0] / counts[0]) == -2.0
    assert int(counts[0]) == 2


def test_td_update_bootstrap_and_terminal():
    values = torch.tensor([0.0, -10.0, 0.0], dtype=torch.float64)
    td0_update(values, 0, -1.0, 1, False, alpha=0.5, gamma=1.0)
    assert float(values[0]) == -5.5  # 0 + 0.5*(-1 - 10 - 0)
    td0_update(values, 1, -1.0, 2, True, alpha=0.5, gamma=1.0)
    assert float(values[1]) == -5.5  # -10 + 0.5*(-1 - (-10))


def test_exact_solution_has_zero_terminal_value_and_bellman_equation():
    env = GridWorld()
    v = exact_uniform_policy_values(env, gamma=1.0)
    assert v.shape == (16,)
    assert v[-1].item() == 0.0
    assert torch.all(v[:-1] < 0)
    for state in range(15):
        expected = -1.0 + sum(float(v[env.next_state(state, action)])
                              for action in range(4)) / 4.0
        assert v[state].item() == pytest.approx(expected, abs=1e-10)


def test_both_algorithms_return_4_by_4_and_same_seed_reproducibly():
    env = GridWorld()
    a = train_every_visit_mc(env, episodes=100, seed=7)
    b = train_every_visit_mc(env, episodes=100, seed=7)
    assert torch.equal(a.values, b.values)
    assert torch.equal(a.visit_counts, b.visit_counts)
    assert a.values.shape == (16,)
    assert a.values[15] == 0
    assert a.total_steps == int(a.visit_counts.sum())
    td = train_td0(env, episodes=100, seed=7)
    assert td.values.shape == (16,)
    assert td.values[15] == 0
    assert td.total_steps == a.total_steps  # identical random paths
    assert int(td.visit_counts.sum()) == td.total_steps


def test_invalid_parameters():
    with pytest.raises(ValueError):
        train_every_visit_mc(GridWorld(), episodes=0)
    with pytest.raises(ValueError):
        train_td0(GridWorld(), alpha=0)


def test_lecture2_page14_one_terminal_reference_values():
    """Lecture 2, p. 14 matches the same single-terminal task as HW3."""
    expected_rounded = torch.tensor([
        [-59, -57, -54, -52],
        [-57, -55, -50, -45],
        [-54, -50, -41, -30],
        [-52, -45, -30, 0],
    ], dtype=torch.float64)
    actual = exact_uniform_policy_values(GridWorld()).reshape(4, 4)
    assert torch.equal(torch.round(actual), expected_rounded)
