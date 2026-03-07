# simple_train_sb3.py
# Minimal SB3 training loop with manual episode tracking.
# No Monitor, no callbacks, no Redis.

import argparse
import os
import csv
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import A2C, PPO, SAC, DQN
from gymnasium.spaces import Discrete, Box


# -------------------------
# Choose algorithm
# -------------------------

ALGOS = {
    "A2C": A2C,
    "PPO": PPO,
    "SAC": SAC,
    "DQN": DQN,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    parser.add_argument("--algo", required=True, choices=ALGOS.keys())
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--moving-avg-window", type=int, default=100)
    args = parser.parse_args()

    # -------------------------
    # Environment
    # -------------------------

    env_kwargs = {"render_mode": "human"} if args.render else {}
    env = gym.make(args.env_name, **env_kwargs)

    # Safety checks for algorithms
    if args.algo == "DQN" and not isinstance(env.action_space, Discrete):
        raise ValueError("DQN requires discrete action space.")
    if args.algo == "SAC" and not isinstance(env.action_space, Box):
        raise ValueError("SAC requires continuous action space.")

    # -------------------------
    # Model
    # -------------------------

    model = ALGOS[args.algo]("MlpPolicy", env, verbose=1)

    # -------------------------
    # Training loop
    # -------------------------

    episode_returns = []
    episode_lengths = []

    total_steps = 0
    episode_return = 0.0
    episode_length = 0

    obs, _ = env.reset()

    while total_steps < args.total_timesteps:
        # Model chooses action
        action, _ = model.predict(obs, deterministic=False)

        # Environment step
        obs, reward, terminated, truncated, _ = env.step(action)

        episode_return += reward
        episode_length += 1
        total_steps += 1

        done = terminated or truncated

        # Tell SB3 model about the transition (this is the "learning")
        model.learn(total_timesteps=1, reset_num_timesteps=False)

        if done:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

            episode_return = 0.0
            episode_length = 0
            obs, _ = env.reset()

    env.close()

    # -------------------------
    # Output directories
    # -------------------------

    out_dir = os.path.join("output", args.algo, args.env_name)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Plot learning curve
    # -------------------------

    if len(episode_returns) > 0:
        episode_steps = np.cumsum(episode_lengths)

        plt.figure(figsize=(12, 6))
        plt.scatter(episode_steps, episode_returns, s=10, alpha=0.3, label="Episode Return")

        w = args.moving_avg_window
        if len(episode_returns) >= w:
            ma = np.convolve(episode_returns, np.ones(w) / w, mode="valid")
            ma_x = episode_steps[w - 1 :]
            plt.plot(ma_x, ma, linewidth=2, label=f"{w}-Episode Moving Average")

        plt.xlabel("Environment Steps")
        plt.ylabel("Episode Return")
        plt.title(f"{args.env_name} – {args.algo}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plot_path = os.path.join(out_dir, f"{args.env_name}_learning_curve.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------------
    # Save final CSV row
    # -------------------------

    K = args.moving_avg_window
    last_k = episode_returns[-K:] if len(episode_returns) >= K else episode_returns

    mean_last_k = float(np.mean(last_k)) if last_k else float("nan")
    std_last_k = float(np.std(last_k)) if last_k else float("nan")

    table_path = os.path.join("output", args.algo, "final_performance.csv")
    file_exists = os.path.exists(table_path)
    os.makedirs(os.path.dirname(table_path), exist_ok=True)

    with open(table_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "env_name",
                "agent",
                "total_timesteps",
                "episodes_completed",
                f"mean_last_{K}",
                f"std_last_{K}",
            ])
        writer.writerow([
            args.env_name,
            args.algo,
            total_steps,
            len(episode_returns),
            mean_last_k,
            std_last_k,
        ])

    print("Done.")
    print(f"Episodes completed: {len(episode_returns)}")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
