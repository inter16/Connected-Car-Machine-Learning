from __future__ import annotations

import argparse

import torch

from connected_car_rl.algorithms import value_iteration
from connected_car_rl.envs import GridWorld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 2 - Value Iteration")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--initial-value", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=4, sci_mode=False)

    env = GridWorld()
    values = value_iteration(
        env,
        gamma=args.gamma,
        iterations=args.iterations,
        initial_value=args.initial_value,
    )

    print(f"Value Iteration ({args.iterations} iterations)")
    print(f"initial_value = {args.initial_value}, gamma = {args.gamma}")
    print(env.as_grid(values))


if __name__ == "__main__":
    main()
