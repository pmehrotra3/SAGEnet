# Trains CMA-ES on a Gymnasium environment using CMAESModel from cma_model.py,
# logs episode returns/lengths, saves a learning curve plot,
# and appends a final-performance CSV row.

import argparse
import os
import csv
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from cma_model import CMAESModel, EpisodeLogger


def main():
    # -------------------------
    # Command-line arguments
    # -------------------------
    parser = argparse.ArgumentParser(description="Full CMA-ES training script (SB3-like outputs)")
    parser.add_argument("env_name", type=str, help="Gymnasium env name, e.g. CartPole-v1")

    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total env steps to train for")
    parser.add_argument("--render", action="store_true", help="Render the environment (human)")
    parser.add_argument("--moving-avg-window", type=int, default=100, help="Window size for moving average curve")

    # CMA / policy knobs
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial CMA sigma")
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="*",
        default=[64, 64],
        help="Hidden layer sizes, e.g. --hidden-layers 32 32 (default: 64 64)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed")
    parser.add_argument("--n-eval-episodes", type=int, default=1, help="Episodes per candidate evaluation")
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="If set, reseed each eval episode deterministically using --seed",
    )
    parser.add_argument("--verbose", action="store_true", help="Print CMA progress")

    args = parser.parse_args()

    # -------------------------
    # Create model + logger
    # -------------------------
    logger = EpisodeLogger()

    model = CMAESModel(
        env_name=args.env_name,
        hidden_layers=tuple(args.hidden_layers),
        sigma=args.sigma,
        seed=args.seed,
        n_eval_episodes=args.n_eval_episodes,
        deterministic_eval=args.deterministic_eval,
        verbose=args.verbose,
    )

    # Optional render: gym wants render_mode at env creation time.
    # So if --render is passed, we recreate the env with render_mode="human"
    # and overwrite model.env.
    if args.render:
        model.close()
        model.env = gym.make(args.env_name, render_mode="human")

    # -------------------------
    # Train
    # -------------------------
    print("Training (CMA-ES)...")
    model.learn(total_timesteps=args.total_timesteps, logger=logger)
    model.close()
    print("Done.")

    # -------------------------
    # Output directories
    # -------------------------
    out_dir = os.path.join("output", "CMAES", args.env_name)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Plot learning curve
    # -------------------------
    episode_returns = logger.episode_returns
    episode_end_steps = logger.episode_end_steps

    if len(episode_returns) > 0:
        plt.figure(figsize=(12, 6))

        plt.scatter(
            episode_end_steps,
            episode_returns,
            s=10,
            alpha=0.3,
            label="Episode Return",
        )

        w = args.moving_avg_window
        if len(episode_returns) >= w:
            moving_avg = np.convolve(episode_returns, np.ones(w) / w, mode="valid")
            plt.plot(
                episode_end_steps[w - 1 :],
                moving_avg,
                linewidth=2,
                label=f"{w}-Episode Moving Average",
            )

        plt.xlabel("Environment Steps")
        plt.ylabel("Episode Return")
        plt.title(f"{args.env_name} – CMA-ES Training")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plot_path = os.path.join(out_dir, "learning_curve.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")
    else:
        print("No completed episodes recorded (plot skipped).")

    # -------------------------
    # Save CSV summary row
    # -------------------------
    K = args.moving_avg_window
    last_k = episode_returns[-K:] if len(episode_returns) >= K else episode_returns

    mean_last_k = float(np.mean(last_k)) if last_k else float("nan")
    std_last_k = float(np.std(last_k)) if last_k else float("nan")

    csv_path = os.path.join("output", "CMAES", "final_performance.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "env_name",
                "total_timesteps",
                "episodes_completed",
                f"mean_last_{K}",
                f"std_last_{K}",
                "sigma",
                "hidden_layers",
                "seed",
                "n_eval_episodes",
                "deterministic_eval",
            ])
        writer.writerow([
            args.env_name,
            args.total_timesteps,
            len(episode_returns),
            mean_last_k,
            std_last_k,
            args.sigma,
            "-".join(map(str, args.hidden_layers)),
            "" if args.seed is None else args.seed,
            args.n_eval_episodes,
            int(args.deterministic_eval),
        ])

    print(f"Saved CSV row: {csv_path}")
    print(f"Episodes completed: {len(episode_returns)}")
    print(f"Results saved in: {out_dir}")
    print(f"Best score seen: {model.best_score:.3f}")


if __name__ == "__main__":
    main()