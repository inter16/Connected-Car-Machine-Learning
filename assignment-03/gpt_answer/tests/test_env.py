import pytest

from gridworld_rl.envs import Action, GridWorld


def test_reset_and_wall_action():
    env = GridWorld()
    assert env.reset() == 0
    assert env.step(Action.UP) == (0, -1.0, False)
    assert env.step(Action.LEFT) == (0, -1.0, False)


def test_terminal_and_reset():
    env = GridWorld()
    for action in [Action.DOWN, Action.DOWN, Action.DOWN, Action.RIGHT,
                   Action.RIGHT, Action.RIGHT]:
        next_state, reward, done = env.step(action)
    assert (next_state, reward, done) == (15, -1.0, True)
    with pytest.raises(RuntimeError):
        env.step(Action.UP)
    assert env.reset() == 0
    assert env.done is False


def test_action_validation():
    env = GridWorld()
    with pytest.raises(ValueError):
        env.step(99)


def test_lecture2_wall_transitions_keep_position_with_minus_one_reward():
    """Lecture 2, p. 11: an impossible up move is still sampled and costs -1."""
    env = GridWorld()
    assert env.next_state(1, Action.UP) == 1
    assert env.next_state(3, Action.RIGHT) == 3
    assert env.next_state(12, Action.LEFT) == 12
    assert env.next_state(14, Action.DOWN) == 14
    env.state = 1
    assert env.step(Action.UP) == (1, -1.0, False)
