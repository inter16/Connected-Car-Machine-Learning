"""Run MC and/or TD(0), print values, and save screenshot-ready outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from gridworld_rl.agents import train_every_visit_mc, train_td0
from gridworld_rl.envs import GridWorld
from gridworld_rl.utils.analysis import exact_uniform_policy_values, mean_absolute_error
from gridworld_rl.utils.output import format_value_grid, save_value_csv, save_value_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HW3: 4x4 Grid World MC and TD(0)")
    parser.add_argument("--algorithm", choices=("mc", "td0", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.01, help="TD(0) step size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps-per-episode", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG export")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = GridWorld()
    reference = exact_uniform_policy_values(environment, gamma=args.gamma)

    output_lines = [
        "HW3 4x4 Grid World: uniform random actions, reward -1 per step, "
        "border actions remain in place (Lecture 2, p. 11)",
        f"episodes={args.episodes}, gamma={args.gamma}, seed={args.seed}",
    ]

    if args.algorithm in ("mc", "both"):
        mc = train_every_visit_mc(
            environment,
            episodes=args.episodes,
            gamma=args.gamma,
            seed=args.seed,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        output_lines += [
            "", "[1-2] Every-Visit MC: V(s)",
            format_value_grid(mc.values),
            f"total_steps={mc.total_steps}, MAE_vs_exact={mean_absolute_error(mc.values, reference):.4f}",
        ]
        save_value_csv(mc.values, args.output_dir / "mc_values.csv")
        if not args.no_plot:
            save_value_image(mc.values, args.output_dir / "mc_values.png",
                             f"Every-Visit MC, {args.episodes:,} episodes")

    if args.algorithm in ("td0", "both"):
        td = train_td0(
            environment,
            episodes=args.episodes,
            gamma=args.gamma,
            alpha=args.alpha,
            seed=args.seed,
            max_steps_per_episode=args.max_steps_per_episode,
        )
        output_lines += [
            "", "[2-2] TD(0): V(s)",
            format_value_grid(td.values),
            f"alpha={td.alpha}, total_steps={td.total_steps}, "
            f"MAE_vs_exact={mean_absolute_error(td.values, reference):.4f}",
        ]
        save_value_csv(td.values, args.output_dir / "td0_values.csv")
        if not args.no_plot:
            save_value_image(td.values, args.output_dir / "td0_values.png",
                             f"TD(0), alpha={args.alpha}, {args.episodes:,} episodes")

    output_lines += ["", "[Reference] Exact Bellman-policy value (not learned)",
                     format_value_grid(reference)]
    save_value_csv(reference, args.output_dir / "exact_values.csv")
    report = "\n".join(output_lines) + "\n"
    print(report, end="")
    (args.output_dir / "terminal_output.txt").write_text(report, encoding="utf-8")
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
