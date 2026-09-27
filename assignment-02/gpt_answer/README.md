# Connected Car ML - Assignment 2 (PyTorch)

This project implements the two tasks in Assignment 2 using PyTorch:

1. Policy Evaluation on a 4x4 GridWorld with a uniform random policy.
2. Value Iteration on the same deterministic GridWorld.

## Environment assumptions from the assignment

- Grid: 4 x 4, states `s0` ... `s15`
- Initial state: `s0`
- Terminal state: `s15`
- Actions: North, South, East, West
- Reward: `-1` for every non-terminal step
- Discount factor: `gamma = 1.0`
- Out-of-grid action: the agent stays in the same state and receives `-1`
- Policy Evaluation policy: each of the four actions has probability `0.25`
- State transitions are deterministic for a chosen action
- Each algorithm performs exactly 1,000 synchronous Bellman updates by default

## Project structure

```text
assignment2-pytorch/
├── pyproject.toml
├── README.md
├── src/
│   └── connected_car_rl/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   └── gridworld.py
│       └── algorithms/
│           ├── __init__.py
│           └── dynamic_programming.py
├── scripts/
│   ├── policy_evaluation.py
│   └── value_iteration.py
└── tests/
    └── test_dynamic_programming.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

Policy Evaluation, initial values = 0:

```bash
python scripts/policy_evaluation.py --initial-value 0
```

Policy Evaluation, initial values = 10:

```bash
python scripts/policy_evaluation.py --initial-value 10
```

Value Iteration:

```bash
python scripts/value_iteration.py
```

## Expected output after 1,000 iterations

Policy Evaluation (both initial value 0 and 10 converge to the same result):

```text
tensor([[-59.4286, -57.4286, -54.2857, -51.7143],
        [-57.4286, -54.5714, -49.7143, -45.1429],
        [-54.2857, -49.7143, -40.8571, -30.0000],
        [-51.7143, -45.1429, -30.0000,   0.0000]], dtype=torch.float64)
```

Value Iteration:

```text
tensor([[-6., -5., -4., -3.],
        [-5., -4., -3., -2.],
        [-4., -3., -2., -1.],
        [-3., -2., -1.,  0.]], dtype=torch.float64)
```
